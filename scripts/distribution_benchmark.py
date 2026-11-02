"""Frozen, retrospective four-dataset distribution evaluation. No generation or training."""
import argparse
from collections import Counter
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from dataset_catalog import ROOT, CATALOG, core, data, read_json, read_rows, sha, signature, text_at, value_at
from dataset_metrics import assess_label, fidelity
from json_io import write_json

OUT=ROOT/'reports/distribution_benchmark/v1'
SEED=2026110201
MODEL='openai-community/gpt2-large'
REVISION='32b71b12589c2f8d625668d2335a01cac3249519'
KEYS=['bitext','insuranceqa','reliability','mimic_demo']
STOP=set(ENGLISH_STOP_WORDS)
LINKS={'MMD':'https://www.jmlr.org/papers/v13/gretton12a.html',
       'PRDC':'https://github.com/clovaai/generative-evaluation-prdc',
       'SDMetrics':'https://docs.sdv.dev/sdmetrics/data-metrics/quality',
       'MAUVE':'https://krishnap25.github.io/mauve/'}


def rng(name):
    import hashlib
    return random.Random(SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))


def status(phase,**values):
    write_json(OUT/'status.json',{'phase':phase,**values},best_effort=True)
    print(phase,values,flush=True)


def check_inputs(protocol):
    for path,digest in protocol['input_hashes'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Frozen input changed: '+path)


def text_views(key,record):
    if key=='reliability':
        request=json.dumps(record['input'],sort_keys=True,ensure_ascii=False)
        reply=json.dumps(record['expected_output'],sort_keys=True,ensure_ascii=False)
    else:
        request=text_at(record,CATALOG[key]['request']);reply=text_at(record,CATALOG[key]['reply'])
    return {'request':request,'reply':reply,'joint':'REQUEST\n'+request+'\nRESPONSE\n'+reply}


def keyword_text(key,record):
    if key=='mimic_demo':return None
    if key=='reliability':
        return ' '.join([record['input']['customer_message'],*[turn['content'] for turn in record['input']['conversation_history']]])
    return text_at(record,CATALOG[key]['request'])+' '+text_at(record,CATALOG[key]['reply'])


def words(text):
    text=re.sub(r'\{\{.*?\}\}',' ',text.lower())
    return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?",text) if len(w)>=3 and w not in STOP]


def mmd2(x,y,gamma):
    """Non-negative biased V-statistic with an RBF kernel; no accuracy interpretation."""
    return max(0.,float(np.exp(-gamma*cdist(x,x,'sqeuclidean')).mean()+
                        np.exp(-gamma*cdist(y,y,'sqeuclidean')).mean()-
                        2*np.exp(-gamma*cdist(x,y,'sqeuclidean')).mean()))


def median_gamma(x):
    distances=cdist(x,x,'sqeuclidean');v=distances[np.triu_indices(len(x),1)];v=v[v>1e-12]
    if not len(v):raise ValueError('No positive training distances')
    return float(1/np.median(v))


def features_frame(key,records):
    if key=='mimic_demo':return pd.DataFrame([core(key,r) for r in records])
    rows=[]
    for r in records:
        if key=='bitext':
            row={f:r[f] for f in ['intent','category']}
            row.update({'tag_'+tag:tag in r['tags'] for tag in 'BCEILMPQWZ'})
        elif key=='insuranceqa':row={'domain':r['domain']}
        else:
            row={f:r[f] for f in ['category','subcategory','difficulty']}
            row.update({'routing':r['expected_output']['routing_decision'],'priority':r['expected_output']['priority'],
                        'policy_type':r['input']['context'].get('policy_type'),'state':r['input']['context'].get('state'),
                        'action_count':len(r['expected_output']['required_actions']),'history_turns':len(r['input']['conversation_history'])})
        view=text_views(key,r)
        row.update(request_characters=len(view['request']) if key!='reliability' else len(r['input']['customer_message']),
                   reply_characters=len(view['reply']))
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_for(key,frame):
    numerical=set(data(key)['schema']['numeric_fields']) if key=='mimic_demo' else {'request_characters','reply_characters','action_count','history_turns'}
    boolean={'age_topcoded'} if key=='mimic_demo' else {c for c in frame if c.startswith('tag_')}
    return {'tables':{'records':{'columns':{c:{'sdtype':'numerical' if c in numerical else 'boolean' if c in boolean else 'categorical'} for c in frame}}},'relationships':[]}


def categorical_distributions(a,b,metadata):
    def token(value):
        if pd.isna(value):return '<MISSING>'
        if isinstance(value,(bool,np.bool_)):return str(bool(value))
        if isinstance(value,(int,float,np.integer,np.floating)):return format(float(value),'.15g')
        return str(value)
    result={}
    for column,spec in metadata['tables']['records']['columns'].items():
        if spec['sdtype'] not in ['categorical','boolean']:continue
        x=Counter(token(v) for v in a[column]);y=Counter(token(v) for v in b[column])
        result[column]={'reference_counts':dict(x),'generated_counts':dict(y),
            'total_variation':.5*sum(abs(x[v]/len(a)-y[v]/len(b)) for v in x.keys()|y.keys())}
    return result


def definitions():
    base=ROOT/'outputs';clinical=base/'mimic-demo-fidelity-conditioned-v1-windows'
    items={}
    for key in KEYS:
        folder=base/'bitext-4b-full-pass-evaluation-500-windows' if key=='bitext' else ROOT/CATALOG[key]['assessment']
        items[key]={'original_generic':{'raw':str((folder/'tuned_generations.jsonl').relative_to(ROOT)),
                     'adapter':str((base/CATALOG[key]['run']/'adapter').relative_to(ROOT)),
                     'control_fields':[],'description':'Original fine-tuned adapter, generic generation prompt'}}
    for key in ['bitext','reliability']:
        directory=base/'prefix-alignment-v1'/key
        for arm,file in [('matched_generic','baseline_generations.jsonl'),('selected_prefix','controlled_generations.jsonl')]:
            items[key][arm]={'raw':str((directory/file).relative_to(ROOT)),
                'adapter':items[key]['original_generic']['adapter'],
                'control_fields':(['intent','category'] if key=='bitext' else ['category','subcategory','difficulty']) if arm=='selected_prefix' else [],
                'description':'Same adapter, matched prefix comparison; eight pilot outputs reused' if arm=='selected_prefix' else 'Matched generic arm of prefix comparison'}
    for arm,file,adapter in [('old_conditioned','old_conditioned_generations.jsonl',items['mimic_demo']['original_generic']['adapter']),
                              ('selected_candidate','candidate_conditioned_generations.jsonl',str((clinical/'adapter').relative_to(ROOT)))]:
        items['mimic_demo'][arm]={'raw':str((clinical/file).relative_to(ROOT)),'adapter':adapter,
            'control_fields':['primary_diagnosis_code','primary_diagnosis_version','lab missingness schedule'],
            'description':'Identical diagnosis/null conditions; adapter training budgets differ'}
    return items


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        check_inputs(read_json(OUT/'protocol.json'));print('Existing frozen protocol retained.');return
    cohorts=definitions();input_hashes={};summaries={};embedding_inputs={};selections={}
    freeze=read_json(ROOT/'reports/final_experiment_freeze.json')
    snapshot=read_json(ROOT/'reports/assessment_snapshot/snapshot.json')
    for key,item in snapshot['datasets'].items():
        for field,digest in [('records_path','records_sha256'),('metrics_path','metrics_sha256')]:
            assert sha(ROOT/item[field])==freeze['selected_cohorts'][key][digest]
            input_hashes[item[field]]=sha(ROOT/item[field])
    for item in freeze['adapters'].values():
        for file,digest in [('adapter_model.safetensors','weight_sha256'),('adapter_config.json','config_sha256')]:
            p=Path(item['original_path'])/file;assert sha(p)==item[digest];input_hashes[str(p.relative_to(ROOT))]=sha(p)
    for key in KEYS:
        status('prepare',dataset=key)
        source=data(key);train=source['train'];directory=OUT/key;directory.mkdir(exist_ok=True);summaries[key]={}
        for filename in ['train.jsonl','validation.jsonl','schema.json','generation_prompt.txt']:
            path=ROOT/source['spec']['prepared']/filename;input_hashes[str(path.relative_to(ROOT))]=sha(path)
        for arm,item in cohorts[key].items():
            p=ROOT/item['raw'];input_hashes[item['raw']]=sha(p)
            for file in ['adapter_model.safetensors','adapter_config.json']:
                ap=ROOT/item['adapter']/file;input_hashes[str(ap.relative_to(ROOT))]=sha(ap)
            metrics,rows=assess_label(key,read_rows(p),directory/(arm+'_records.jsonl'))
            write_json(directory/(arm+'_core_metrics.json'),metrics);summaries[key][arm]=metrics
        if key=='mimic_demo':continue
        validation=read_rows(ROOT/source['spec']['prepared']/'validation.jsonl')
        n=min(128,len(validation),len(train)//2,*[m['structure_valid'] for m in summaries[key].values()])
        pool=rng(key+'train_pool').sample(range(len(train)),2*n)
        selections[key]={'n':n,'training_reference':pool[:n],'training_control':pool[n:],
                         'validation_control':rng(key+'validation').sample(range(len(validation)),n),'cohorts':{}}
        collections={'training_reference':[train[i] for i in pool[:n]],'training_control':[train[i] for i in pool[n:]],
                     'validation_control':[validation[i] for i in selections[key]['validation_control']]}
        for arm in cohorts[key]:
            rows=read_rows(directory/(arm+'_records.jsonl'))
            sample_name=key+('_matched_prefix_pair' if arm in ['matched_generic','selected_prefix'] else arm)
            ids=sorted(rng(sample_name).sample(range(len(rows)),n))
            selections[key]['cohorts'][arm]={'row_indices':ids,'sample_ids':[rows[i]['sample_id'] for i in ids]}
            collections[arm]=[rows[i]['record'] for i in ids]
        embedding_inputs[key]={}
        for arm,records in collections.items():
            embedding_inputs[key][arm]={role:[text_views(key,r)[role] for r in records] for role in ['request','reply','joint']}
    write_json(OUT/'embedding_inputs.json',embedding_inputs)
    protocol={'created_date':'2026-11-02','seed':SEED,'purpose':'Retrospective descriptive fidelity evaluation of frozen adapter outputs; no generation, training, new selection or untouched-test claim.',
        'cohorts':cohorts,'input_hashes':input_hashes,'embedding_inputs_sha256':sha(OUT/'embedding_inputs.json'),'embedding_selections':selections,
        'embedding':{'feature_model':MODEL,'revision':REVISION,'dtype':'float32','representation':'Last non-padding token, L2 normalized; three separately encoded views: request, reply, joint',
                     'max_tokens':512,'truncation':'right; measured per view','batch_size':4,'no_top_level_prefix_labels_in_reliability_embedding':True},
        'mmd':{'estimator':'biased non-negative MMD squared V-statistic','kernel':'RBF','gamma':'1 / median positive squared distance in pooled disjoint training reference/control features',
               'gamma_multipliers':[0.5,1.,2.],'reference_shuffle_repeats':100,'bands':'5th/95th percentiles of random disjoint equal-size training-pool splits; descriptive finite-pool context, not a pass threshold or confidence interval'},
        'prdc':{'implementation':'released prdc.compute_prdc','nearest_k':[5,10],'scope':'Experimental adaptation to text features; density can exceed one'},
        'sdmetrics':{'report':'QualityReport','features':'Declared labels, language tags, counts and lengths for insurance; original structured columns for MIMIC. No raw free text treated as categorical IDs.',
                     'missingness':'No numeric imputation; missingness evaluated separately','controlled_fields':'Full-report scores include externally supplied columns; per-column ownership is disclosed, not credited as learned behaviour'},
        'keywords':{'tokenization':'lowercase alphabetic words/apostrophes; length >=3; double-brace placeholders removed; sklearn ENGLISH_STOP_WORDS',
                    'stopwords':sorted(STOP),'vocabulary':'100 highest document-frequency source-only content words, lexical tie-break; every other token in explicit OTHER bucket',
                    'scores':['Jensen-Shannon distance base 2, 0..1 lower closer','Total variation, 0..1 lower closer','top100 keyword coverage','out-of-training-vocabulary token rate'],
                    'reference_repeats':100,'scope':'Bitext/QA request+reply; Reliability customer message+history only. No natural-language keywords for structured MIMIC; diagnosis code/version frequencies instead.'},
        'limits':['No universal passing score; no pooled four-dataset accuracy.',
                  'All distribution/copying scores condition on schema-valid outputs; attempted/invalid counts retained.',
                  'Generated cohorts differ in size; only within-dataset embeddings use equal-sized subsets. Full SDMetrics/keyword scores retain original collection sizes.',
                  'One frozen generation cohort per arm; resampling does not establish independent generation-seed robustness.',
                  'Validation controls were used for earlier checkpoint monitoring, not a new untouched final test.',
                  'Training-reference overlap/reuse is explicitly labelled; small-sample scores do not prove equivalence.',
                  'Keyword and embedding similarity do not verify insurance facts or clinical safety.',
                  'Clinical training has 230 admissions from 80 patients in a 100-patient demo, not U.S. population data.'],
        'sources':LINKS}
    write_json(OUT/'protocol.json',protocol);write_json(OUT/'core_summary.json',summaries)
    status('prepared',cohorts=sum(map(len,cohorts.values())),embedding_collections=sum(len(x)*3 for x in embedding_inputs.values()))


def embed():
    protocol=read_json(OUT/'protocol.json');check_inputs(protocol)
    if sha(OUT/'embedding_inputs.json')!=protocol['embedding_inputs_sha256']:raise RuntimeError('Embedding inputs changed')
    lock=ROOT/'reports/assessment_gpu.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel,AutoTokenizer
        path=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
        status('loading_embeddings',device='cuda')
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        model=AutoModel.from_pretrained(path,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
        inputs=read_json(OUT/'embedding_inputs.json');cache=OUT/'features';cache.mkdir(exist_ok=True)
        manifest={'model':MODEL,'revision':REVISION,'model_weights_sha256':sha(path/'model.safetensors'),
                  'input_sha256':sha(OUT/'embedding_inputs.json'),'features':{},'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']}}
        total=sum(len(v)*3 for v in inputs.values());done=0;started=time.time()
        for key,collections in inputs.items():
            manifest['features'][key]={}
            for arm,views in collections.items():
                manifest['features'][key][arm]={}
                for role,texts in views.items():
                    target=cache/f'{key}_{arm}_{role}.npy';lengths=[len(tokenizer.encode(t)) for t in texts]
                    if not target.exists():
                        blocks=[]
                        for i in range(0,len(texts),4):
                            batch=tokenizer(texts[i:i+4],return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda')
                            with torch.inference_mode():
                                hidden=model(**batch).last_hidden_state;end=batch['attention_mask'].sum(1)-1
                                x=hidden[torch.arange(len(end),device='cuda'),end].float()
                                x=x/torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1e-12)
                                blocks.append(x.cpu().numpy())
                            if i%32==0:status('embedding',dataset=key,arm=arm,view=role,records=min(i+4,len(texts)),of=len(texts),completed=done,total=total)
                        np.save(target,np.concatenate(blocks))
                    x=np.load(target);assert x.shape==(len(texts),1280) and np.isfinite(x).all()
                    manifest['features'][key][arm][role]={'path':str(target.relative_to(ROOT)),'sha256':sha(target),'n':len(x),
                        'truncated':sum(n>512 for n in lengths),'max_original_tokens':max(lengths)}
                    done+=1;status('embedding',completed=done,total=total,elapsed_seconds=round(time.time()-started,1))
                    write_json(OUT/'embedding_manifest.partial.json',manifest)
        write_json(OUT/'embedding_manifest.json',manifest);status('embeddings_complete',completed=done,total=total)
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


def sd_report(key,records,reference):
    from sdmetrics.reports import QualityReport
    from sdmetrics.single_column import MissingValueSimilarity,RangeCoverage,CategoryCoverage,BoundaryAdherence,CategoryAdherence
    a=features_frame(key,reference);b=features_frame(key,records);metadata=metadata_for(key,a)
    report=QualityReport();report.generate({'records':a},{'records':b},metadata,verbose=False)
    details={name:json.loads(report.get_details(name).to_json(orient='records')) for name in ['Column Shapes','Column Pair Trends']}
    single={}
    for c in a:
        numeric=metadata['tables']['records']['columns'][c]['sdtype']=='numerical'
        single[c]={'source_missing_rate':float(a[c].isna().mean()),'generated_missing_rate':float(b[c].isna().mean()),
                   'missing_similarity':float(MissingValueSimilarity.compute(a[c],b[c]))}
        if numeric:
            single[c].update(range_coverage=float(RangeCoverage.compute(a[c],b[c])),boundary_adherence=float(BoundaryAdherence.compute(a[c],b[c])))
        else:
            single[c].update(category_coverage=float(CategoryCoverage.compute(a[c],b[c])),category_adherence=float(CategoryAdherence.compute(a[c],b[c])))
    return {'n':len(b),'reference_n':len(a),'quality':float(report.get_score()),'properties':json.loads(report.get_properties().to_json(orient='records')),
            'details':details,'per_column':single,'metadata':metadata,'categorical_distributions':categorical_distributions(a,b,metadata)}


def lexical_profile(key,reference,collections):
    counters=[Counter(words(keyword_text(key,r))) for r in reference]
    total=Counter();df=Counter()
    for counter in counters:total.update(counter)
    for row in counters:df.update(row.keys())
    vocab=[w for w,_ in sorted(df.items(),key=lambda x:(-x[1],x[0]))[:100]]
    def vector(counter):return np.array([counter[w] for w in vocab]+[sum(counter.values())-sum(counter[w] for w in vocab)],dtype=float)
    vectors=np.stack([vector(c) for c in counters]);ref=vector(total);refp=ref/ref.sum();result={}
    for arm,records in collections.items():
        generated=Counter()
        for r in records:generated.update(words(keyword_text(key,r)))
        gen=vector(generated)
        if not gen.sum():raise ValueError('Empty keyword collection: '+key+'/'+arm)
        genp=gen/gen.sum();n=min(len(records),len(reference));randomizer=rng(key+str(n)+'keywords');bands=[]
        for _ in range(100):
            v=vectors[randomizer.sample(range(len(reference)),n)].sum(0);bands.append(float(jensenshannon(refp,v/v.sum(),base=2)))
        table=[{'keyword':word,'source_count':int(ref[i]),'generated_count':int(gen[i]),'source_share':float(refp[i]),'generated_share':float(genp[i]),
                'percentage_point_gap':100*float(genp[i]-refp[i]),'source_document_frequency':df.get(word)} for i,word in enumerate(vocab+['OTHER'])]
        pd.DataFrame(table).to_csv(OUT/key/(arm+'_keywords.csv'),index=False)
        result[arm]={'js_distance_base2':float(jensenshannon(refp,genp,base=2)),'total_variation':float(.5*np.abs(refp-genp).sum()),
            'source_top100_present':sum(generated[w]>0 for w in vocab),'source_top100_count':len(vocab),
            'generated_content_tokens':sum(generated.values()),'out_of_training_vocabulary_token_rate':sum(n for w,n in generated.items() if w not in total)/sum(generated.values()),
            'training_sample_reference_js_p05_p95':[float(v) for v in np.percentile(bands,[5,95])],
            'training_sample_reference_js_draws':bands,
            'reference_overlap':'Samples are drawn from the full training reference; descriptive context, not independent tests or confidence intervals.',
            'top_source_terms':table[:20]}
    return {'vocabulary':vocab,'source_content_tokens':sum(total.values()),'source_vocabulary_size':len(total),'collections':result}


def score():
    from prdc import compute_prdc
    protocol=read_json(OUT/'protocol.json');check_inputs(protocol)
    manifest=read_json(OUT/'embedding_manifest.json')
    if manifest['input_sha256']!=sha(OUT/'embedding_inputs.json'):raise RuntimeError('Features refer to changed inputs')
    output={'protocol_sha256':sha(OUT/'protocol.json'),'embedding_manifest_sha256':sha(OUT/'embedding_manifest.json'),
            'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','sdmetrics','prdc','scikit-learn']},'datasets':{}}
    cores=read_json(OUT/'core_summary.json')
    for key in KEYS:
        status('scoring',dataset=key)
        reference=[core(key,r) for r in data(key)['train']]
        collections={arm:[r['record'] for r in read_rows(OUT/key/(arm+'_records.jsonl'))] for arm in protocol['cohorts'][key]}
        item={'training_rows':len(reference),'cohorts':{},'text_metrics':None,'keywords':None}
        for arm,records in collections.items():
            status('sdmetrics',dataset=key,arm=arm)
            metrics=sd_report(key,records,reference)
            for name,rows in metrics['details'].items():pd.DataFrame(rows).to_csv(OUT/key/(arm+'_'+name.lower().replace(' ','_')+'.csv'),index=False)
            item['cohorts'][arm]={'core':cores[key][arm],'sdmetrics':metrics,'control_fields':protocol['cohorts'][key][arm]['control_fields']}
        if key=='mimic_demo':
            # Keep every admission from each sampled patient, rather than resampling rows independently.
            original=data(key)['train'];groups={g:[core(key,r) for r in original if r['group_id']==g] for g in sorted({r['group_id'] for r in original})}
            bootstrap=rng('clinical_clusters');controls=[];target_n=64
            for repeat in range(20):
                ids=list(groups);bootstrap.shuffle(ids);rows=[];chosen=[]
                for group in ids:
                    rows+=groups[group];chosen.append(group)
                    if len(rows)>=target_n:break
                q=sd_report(key,rows,reference)
                controls.append({'repeat':repeat,'n':len(rows),'patients':len(chosen),'source_group_ids':chosen,'quality':q['quality']})
            item['patient_cluster_reference']={'draws':controls,'quality_p05_p95':np.percentile([r['quality'] for r in controls],[5,95]).tolist(),
                'scope':'Whole-patient training subsets until at least64 admissions; size varies and subsets overlap full training reference. Descriptive context, not confidence intervals or a sample-matched clinical pass threshold.'}
            # Diagnosis codes, not invented narrative keywords.
            for arm,records in collections.items():
                a=Counter(str(r['primary_diagnosis_version'])+':'+str(r['primary_diagnosis_code']) for r in reference)
                b=Counter(str(r['primary_diagnosis_version'])+':'+str(r['primary_diagnosis_code']) for r in records)
                rows=[{'diagnosis_version_code':c,'source_count':a[c],'generated_count':b[c],'source_share':a[c]/len(reference),'generated_share':b[c]/len(records)} for c in sorted(a.keys()|b.keys())]
                pd.DataFrame(rows).to_csv(OUT/key/(arm+'_diagnoses.csv'),index=False)
                item['cohorts'][arm]['diagnosis_total_variation']=.5*sum(abs(r['source_share']-r['generated_share']) for r in rows)
            item['keywords']={'applicable':False,'reason':'No natural-language records; diagnosis code/version frequency tables saved instead. No clinical codebook/factual validation claim.'}
        else:
            item['keywords']=lexical_profile(key,reference,collections);item['text_metrics']={}
            for role in ['request','reply','joint']:
                arrays={}
                for arm,views in manifest['features'][key].items():
                    info=views[role];path=ROOT/info['path'];assert sha(path)==info['sha256'];arrays[arm]=np.load(path).astype(np.float64)
                x=arrays['training_reference'];pool=np.concatenate([x,arrays['training_control']]);gamma=median_gamma(pool)
                randomizer=np.random.default_rng(SEED+KEYS.index(key));bands=[];n=len(x)
                for _ in range(100):
                    order=randomizer.permutation(len(pool));bands.append(mmd2(pool[order[:n]],pool[order[n:]],gamma))
                values={}
                for arm,y in arrays.items():
                    if arm=='training_reference':continue
                    values[arm]={'n':len(y),'mmd2':mmd2(x,y,gamma),'mmd2_gamma_sensitivity':{str(f):mmd2(x,y,gamma*f) for f in [.5,1.,2.]},
                        'prdc':{str(k):{name:float(v) for name,v in compute_prdc(x,y,nearest_k=k).items()} for k in [5,10]},
                        'truncation':manifest['features'][key][arm][role]}
                item['text_metrics'][role]={'reference_n':n,'gamma':gamma,'real_train_split_mmd2_p05_p95':np.percentile(bands,[5,95]).tolist(),
                    'real_train_split_mmd2_draws':bands,
                    'reference_truncation':manifest['features'][key]['training_reference'][role],'collections':values}
        output['datasets'][key]=item;write_json(OUT/key/'results.json',item);write_json(OUT/'results.partial.json',output)
    write_json(OUT/'results.json',output);status('scores_complete')


def report():
    results=read_json(OUT/'results.json');protocol=read_json(OUT/'protocol.json');check_inputs(protocol)
    # Explicit frequency exports also cover categorical schemas that do not carry
    # the older helper's optional categorical_fields declaration (notably Bitext).
    for key,d in results['datasets'].items():
        reference=features_frame(key,[core(key,r) for r in data(key)['train']])
        for arm,c in d['cohorts'].items():
            if 'categorical_distributions' not in c['sdmetrics']:
                generated=features_frame(key,[r['record'] for r in read_rows(OUT/key/(arm+'_records.jsonl'))])
                c['sdmetrics']['categorical_distributions']=categorical_distributions(reference,generated,c['sdmetrics']['metadata'])
        write_json(OUT/key/'results.json',d)
    write_json(OUT/'results.json',results)
    selected={'bitext':'selected_prefix','insuranceqa':'original_generic','reliability':'selected_prefix','mimic_demo':'selected_candidate'}
    labels={k:CATALOG[k]['label'] for k in KEYS}
    def fmt(v):return 'N/A' if v is None else f'{v:.4f}'
    def headline(key,arm):
        d=results['datasets'][key];c=d['cohorts'][arm];tm=d['text_metrics'];kw=d['keywords']
        row={'dataset':key,'cohort':arm,'attempted':c['core']['attempted'],'valid':c['core']['structure_valid'],
                'sdmetrics_quality':c['sdmetrics']['quality'],'joint_mmd2':tm['joint']['collections'][arm]['mmd2'] if tm else None,
                'embedding_n':tm['joint']['reference_n'] if tm else None,'keyword_js_distance':kw['collections'][arm]['js_distance_base2'] if tm else None,
                'whole_record_copy_rate':c['core']['exact_training_record_match_rate']}
        for prop in c['sdmetrics']['properties']:row['sdmetrics_'+prop['Property'].lower().replace(' ','_')]=prop['Score']
        for k in ['5','10']:
            for metric in ['precision','recall','density','coverage']:
                row['joint_prdc_'+metric+'_k'+k]=tm['joint']['collections'][arm]['prdc'][k][metric] if tm else None
        row['keyword_tv']=kw['collections'][arm]['total_variation'] if tm else None
        row['keyword_oov_rate']=kw['collections'][arm]['out_of_training_vocabulary_token_rate'] if tm else None
        row['request_training_match_rate']=c['core']['training_request_match_rate']
        row['reply_or_workflow_training_match_rate']=c['core']['training_reply_match_rate']
        return row
    all_rows=[headline(key,arm) for key in KEYS for arm in results['datasets'][key]['cohorts']]
    pd.DataFrame(all_rows).to_csv(OUT/'scorecard_all_cohorts.csv',index=False)
    pd.DataFrame([headline(k,'original_generic') for k in KEYS]).to_csv(OUT/'scorecard_original_adapters.csv',index=False)
    pd.DataFrame([headline(k,selected[k]) for k in KEYS]).to_csv(OUT/'scorecard_selected_policies.csv',index=False)
    text_rows=[]
    for key,d in results['datasets'].items():
        if not d['text_metrics']:continue
        for role,view in d['text_metrics'].items():
            for arm,r in view['collections'].items():
                for k,pr in r['prdc'].items():text_rows.append({'dataset':key,'view':role,'cohort':arm,'n':r['n'],'nearest_k':k,'mmd2':r['mmd2'],'truncated':r['truncation']['truncated'],**pr})
    pd.DataFrame(text_rows).to_csv(OUT/'text_metrics_all_views_and_controls.csv',index=False)
    lines=['# Four-dataset synthetic-data fidelity benchmark','',
       'Completed 2 November 2026. Retrospective evaluation of saved outputs, not new training, new generation, a fresh final test or an accuracy certificate.','',
       '## The criterion','',
       'Generated collections should resemble their own training source in structured distributions, relationships, text-feature distributions and content-word frequencies, while retaining valid format and exposing copying. Correctness is assessed separately. No universal pass threshold or pooled four-dataset grade is invented.','',
       '## Four original QLoRA adapters','',
       '| Dataset | Valid / attempted | SDMetrics quality up | Joint MMD squared down | Embedding n | Keyword JS distance down |',
       '|---|---:|---:|---:|---:|---:|']
    for k in KEYS:
        r=headline(k,'original_generic');lines.append(f'| {labels[k]} | {r["valid"]}/{r["attempted"]} | {fmt(r["sdmetrics_quality"])} | {fmt(r["joint_mmd2"])} | {r["embedding_n"] or "N/A"} | {fmt(r["keyword_js_distance"])} |')
    lines+=['','SDMetrics is statistical similarity on declared columns, not percent-correct. MMD is a kernel distance in fixed text features, not accuracy. Keyword Jensen-Shannon distance compares source-derived content-word frequencies, not facts. MMD is N/A for structured MIMIC; no artificial text embeddings are forced onto numerical admissions.',
       'Use these for within-dataset comparisons, not to rank datasets whose declared columns and fitted kernel bandwidths differ. In particular, InsuranceQA SDMetrics covers domain and text lengths, not answer meaning. MMD and PRDC share one feature encoder; they are complementary statistics, not independent semantic validation.','',
       '## Selected presentation policies and clinical candidate','',
       '| Dataset | Policy | Valid / attempted | SDMetrics quality up | Joint MMD squared down | Keyword JS distance down |',
       '|---|---|---:|---:|---:|---:|']
    for k in KEYS:
        arm=selected[k];r=headline(k,arm);lines.append(f'| {labels[k]} | {arm} | {r["valid"]}/{r["attempted"]} | {fmt(r["sdmetrics_quality"])} | {fmt(r["joint_mmd2"])} | {fmt(r["keyword_js_distance"])} |')
    lines+=['','InsuranceQA selected is the same original cohort, not a new improvement. Prefix rows use the same original adapters with externally supplied topic labels. The clinical candidate is a separately trained adapter. Original and selected cohort sizes/settings differ; use the matched comparisons in each dataset section, not a causal claim from these two headline tables.','',
       '## PRDC: joint request-response features, selected policies','',
       '| Dataset | n | Precision | Recall | Density | Coverage |','|---|---:|---:|---:|---:|---:|']
    for k in KEYS:
        d=results['datasets'][k]
        if d['text_metrics']:
            q=d['text_metrics']['joint'];p=q['collections'][selected[k]]['prdc']['5']
            lines.append(f'| {labels[k]} | {q["reference_n"]} | '+' | '.join(fmt(p[v]) for v in ['precision','recall','density','coverage'])+' |')
    lines+=['','PRDC is an experimental adaptation to text features, not classification accuracy. Density can exceed one. k=10 sensitivity, individual request/reply views, real-training controls, validation controls and truncation counts are saved in each results.json.','']
    html_sections=[]
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    for key in KEYS:
        d=results['datasets'][key];arm=selected[key];c=d['cohorts'][arm]
        lines+=['## '+labels[key],'',
            'Training reference: '+str(d['training_rows'])+' rows. Selected cohort: '+arm+'.',
            'Models/protocols and source hashes are in protocol.json. Fields supplied externally: '+(', '.join(c['control_fields']) or 'none')+'.','',
            '| Cohort | Valid / attempted | SDMetrics | Joint MMD squared | Keyword JS |','|---|---:|---:|---:|---:|']
        for a in d['cohorts']:
            r=headline(key,a);lines.append(f'| {a} | {r["valid"]}/{r["attempted"]} | {fmt(r["sdmetrics_quality"])} | {fmt(r["joint_mmd2"])} | {fmt(r["keyword_js_distance"])} |')
        if key in ['bitext','reliability']:
            before=headline(key,'matched_generic');after=headline(key,arm)
            lines+=['',f'Matched comparison: SDMetrics {before["sdmetrics_quality"]:.4f} -> {after["sdmetrics_quality"]:.4f}; joint MMD squared {before["joint_mmd2"]:.5f} -> {after["joint_mmd2"]:.5f}; keyword JS {before["keyword_js_distance"]:.4f} -> {after["keyword_js_distance"]:.4f}. These three measured views improved in the prefix comparison; supplied label frequencies are external controls, and other statistics/semantic problems may remain.']
        elif key=='insuranceqa':
            recall=d['text_metrics']['joint']['collections'][arm]['prdc']['5']['recall']
            lines+=['',f'Original generator retained; no new improvement comparison. Joint-feature recall at k5 is {recall:.3f}, indicating incomplete neighbourhood coverage under this representation. SDMetrics here measures domain and text lengths, not insurance-answer correctness. Prior AI review still has15 Pass,4 Unsure,11 Issue out of30.']
        else:
            before=d['cohorts']['old_conditioned']['sdmetrics']['quality'];after=c['sdmetrics']['quality']
            lines+=['',f'Under the same diagnosis/null conditions, clinical SDMetrics increased {before:.4f} -> {after:.4f}. Training budgets differed. Some categories/correlations and range tails remain imperfect; this does not establish clinical correctness or U.S.-population resemblance.']
        lines+=['','### Structured quality and source ranges','',
                '| Property | Score |','|---|---:|',*['| '+p['Property']+' | '+fmt(p['Score'])+' |' for p in c['sdmetrics']['properties']],'',
                '| Numeric/text field | Source min / median / max | Generated min / median / max | Wasserstein down |','|---|---|---|---:|']
        def triple(v):return ' / '.join(f'{v[k]:.3g}' for k in ['min','median','max']) if v else 'N/A'
        for field,v in c['core']['fidelity_vs_training']['numeric'].items():lines.append('| '+field+' | '+triple(v.get('reference'))+' | '+triple(v.get('generated'))+' | '+fmt(v.get('wasserstein'))+' |')
        lines+=['','### Source categorical proportions','',
                '| Field | Total variation down |','|---|---:|']
        categories=c['sdmetrics']['categorical_distributions']
        for field,v in categories.items():lines.append('| '+field+' | '+fmt(v['total_variation'])+' |')
        categorical_rows=[]
        for field,v in categories.items():
            a=v['reference_counts'];b=v['generated_counts'];na=sum(a.values());nb=sum(b.values())
            categorical_rows.extend({'field':field,'value':value,'source_count':a.get(value,0),'generated_count':b.get(value,0),
                                     'source_share':a.get(value,0)/na,'generated_share':b.get(value,0)/nb} for value in sorted(a.keys()|b.keys()))
        pd.DataFrame(categorical_rows).to_csv(OUT/key/'selected_categorical_frequencies.csv',index=False)
        lines+=['','### Copying and validity','',f'Selected valid/source-check counts: {c["core"]["structure_valid"]}/{c["core"]["attempted"]} and {c["core"]["source_constraint_valid"]}/{c["core"]["attempted"]}. Whole-record normalized training match rate: {c["core"]["exact_training_record_match_rate"]:.2%}. Zero exact matches do not establish privacy or absence of paraphrased memorization.','']
        for field,label in [('training_request_match_rate','Request training matches'),('training_reply_match_rate','Reply/workflow training matches')]:
            value=c['core'][field]
            lines.append(label+': '+('N/A' if value is None else f'{value:.2%}')+'.')
        lines+=['Reliability workflow matches concern expected-output label structures, not natural-language support replies.',''] if key=='reliability' else ['']
        graph=go.Figure();caption=''
        if d['text_metrics']:
            q=d['text_metrics']['joint'];m=q['collections'][arm]
            lines+=['### Text-distribution context','',
                f'Joint MMD squared: {m["mmd2"]:.6f}. Real-training pool split p05-p95: {q["real_train_split_mmd2_p05_p95"]}. Validation-control MMD squared: {q["collections"]["validation_control"]["mmd2"]:.6f}.',
                f'Joint text truncated: {m["truncation"]["truncated"]}/{m["n"]}; training reference {q["reference_truncation"]["truncated"]}/{q["reference_n"]}. Only the first512 tokens are represented when truncation occurs. These are finite-cohort feature comparisons, not an equivalence test.','',
                '### Keyword distributions','',
                '| Source content word | Source share | Generated share | Gap, percentage points |','|---|---:|---:|---:|']
            kw=d['keywords']['collections'][arm]
            for r in kw['top_source_terms'][:15]:lines.append(f'| {r["keyword"]} | {r["source_share"]:.2%} | {r["generated_share"]:.2%} | {r["percentage_point_gap"]:+.2f} |')
            lines+=['',f'Top100 source words present: {kw["source_top100_present"]}/{kw["source_top100_count"]}. Keyword TV: {kw["total_variation"]:.4f}; out-of-training-vocabulary token rate: {kw["out_of_training_vocabulary_token_rate"]:.2%}. Size-matched training-subsample keyword JS p05-p95: {kw["training_sample_reference_js_p05_p95"]}. Samples overlap the full reference; this is descriptive context, not a pass band. Full100-word plus OTHER frequency tables are saved as CSV.','']
            rows=kw['top_source_terms'][:15];x=[r['keyword'] for r in rows]
            graph.add_bar(name='Training source',x=x,y=[r['source_share'] for r in rows]);graph.add_bar(name='Selected generated',x=x,y=[r['generated_share'] for r in rows])
            caption='Content-word shares among all retained content tokens. Only top15 words displayed; full table includes OTHER. Reliability uses customer prose/history, not JSON field names or workflow enums.'
        else:
            lines+=['### Clinical diagnosis frequencies and patient-cluster context','',
                'Natural-language keywords, MMD and text PRDC are not applicable: these are structured admissions. Diagnosis-code/version frequency CSVs are saved instead; codes are not translated into invented clinical narratives.',
                f'Selected diagnosis-pair TV: {c["diagnosis_total_variation"]:.4f}. Whole-patient training-subset SDMetrics quality p05-p95: {d["patient_cluster_reference"]["quality_p05_p95"]}. These variable-size subsets overlap the full source; they are not a clinical pass threshold.','']
            rows=pd.read_csv(OUT/key/(arm+'_diagnoses.csv')).sort_values('source_share',ascending=False).head(15)
            graph.add_bar(name='Training source',x=rows['diagnosis_version_code'],y=rows['source_share']);graph.add_bar(name='Selected generated',x=rows['diagnosis_version_code'],y=rows['generated_share'])
            caption='Top15 source diagnosis version/code pairs. Conditions were supplied externally; the frequency match is not independently learned or codebook validation.'
        graph.update_layout(title=labels[key]+' - source vs generated frequencies',barmode='group',template='plotly_white',yaxis_tickformat='.0%',height=420,margin=dict(b=110))
        html_sections.append('<section><h2>'+html.escape(labels[key])+'</h2><p>'+html.escape(caption)+'</p>'+graph.to_html(full_html=False,include_plotlyjs=False)+'</section>')
        (OUT/key/'report.md').write_text('\n'.join(lines[lines.index('## '+labels[key]):])+'\n',encoding='utf-8')
    lines+=['## How to interpret the criterion professionally','',
        'Report structured fidelity, text-feature fidelity and keyword fidelity separately; compare matched arms rather than averaging unrelated scores. No single scalar proves full joint distribution matching. Supplied label/diagnosis/null schedules remain disclosed. Point estimates on small reused cohorts are descriptive; reference bands do not certify equivalence.','',
        'Meaning/correctness remains separate: prior InsuranceQA AI review15 Pass/4 Unsure/11 Issue out of30; selected Reliability45/15/4 out of64; clinical expert correctness is unverified. No new AI judgments or expert labels were produced by this benchmark.','',
        'MAUVE: earlier Bitext scores remain historical exploratory evidence, not recomputed or mixed into this scorecard. Authors recommend thousands of examples per collection; current small cohorts do not support a stronger headline MAUVE claim.','',
        '## Reproduce and inspect','',
        'Stages: distribution_benchmark.py prepare (analysis environment); embed (Windows GPU environment with env_windows.ps1 and HF offline flags); score and report (analysis environment). Existing frozen caches are reused only after input checks. No teacher API, classifier training, generation or weight updates are used.','',
        'JSON: protocol.json, embedding_manifest.json, results.json and per-dataset results.json. CSV: original/selected/all-cohort scorecards, per-column and pairwise SDMetrics details, keyword profiles or diagnosis distributions. index.html provides offline interactive frequency plots.','',
        '### Sources','',*['- ['+name+']('+url+')' for name,url in LINKS.items()],'',*['- '+v for v in protocol['limits']]]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    def html_table(rows):
        frame=pd.DataFrame(rows);frame['dataset']=frame['dataset'].map(labels)
        frame=frame[['dataset','valid','attempted','sdmetrics_quality','joint_mmd2','embedding_n','keyword_js_distance']]
        frame=frame.rename(columns={'dataset':'Dataset','valid':'Valid','attempted':'Attempted','sdmetrics_quality':'SDMetrics (higher)','joint_mmd2':'MMD squared (lower)','embedding_n':'Text sample n','keyword_js_distance':'Keyword JS (lower)'})
        return frame.to_html(index=False,float_format=lambda x:f'{x:.4f}',na_rep='N/A',escape=True)
    original_table=html_table([headline(k,'original_generic') for k in KEYS]);headline_table=html_table([headline(k,selected[k]) for k in KEYS])
    prdc_table=pd.DataFrame([{'Dataset':labels[row['dataset']],'Text n':row['embedding_n'],**{m.capitalize():row['joint_prdc_'+m+'_k5'] for m in ['precision','recall','density','coverage']}} for row in [headline(k,selected[k]) for k in KEYS]]).to_html(index=False,float_format=lambda x:f'{x:.4f}',na_rep='N/A',escape=True)
    document='<!doctype html><meta charset="utf-8"><title>Gemma distribution benchmark</title><style>body{font:16px/1.6 Arial;background:#f1f5f7;color:#233548;margin:0}main{max-width:1200px;margin:auto;padding:28px}section{background:white;padding:24px;border-radius:12px;margin:24px 0}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:10px;border-bottom:1px solid #dde4ea;text-align:left}.scroll{overflow:auto}h1{line-height:1.2}</style><script>'+get_plotlyjs()+'</script><main><h1>How closely does generated data match its source?</h1><p>Fixed retrospective scorecard: structured distributions, text-feature distributions and keywords. Scores are not factual accuracy. No universal passing percentage. Compare versions within a dataset; do not rank datasets by scores on different features.</p><section><h2>The four original QLoRA adapters</h2><div class="scroll">'+original_table+'</div></section><section><h2>Selected presentation policies</h2><div class="scroll">'+headline_table+'</div><p>InsuranceQA is the same retained original cohort. Prefix rows use the original adapters with topic controls. The clinical row uses a separate candidate adapter. Cohort sizes/settings differ; matched controls are in the Markdown report. MIMIC text metrics are N/A. Supplied labels/diagnoses/missingness are not learned frequency gains.</p></section><section><h2>PRDC: text resemblance and variety</h2><div class="scroll">'+prdc_table+'</div><p>Joint request/response features; k=5. These are neighbourhood statistics, not accuracy. Density can exceed one. Individual fields, k=10 sensitivity and controls are saved in text_metrics_all_views_and_controls.csv.</p></section>'+''.join(html_sections)+'<section><h2>Interpretation</h2><p>Lower MMD squared and keyword JS means closer on those representations. Higher SDMetrics means closer declared statistics. Small samples, truncated text, copying and externally controlled fields limit interpretation. See saved report.md and results.json for counts, controls and full details. This standalone file does not alter the demo website.</p></section></main>'
    (OUT/'index.html').write_text(document,encoding='utf-8')
    check_inputs(protocol)
    verification={'frozen_inputs_unchanged':True,'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
        'implemented':['MMD squared/RBF with sensitivity and real/validation controls','released PRDC k5/k10, request/reply/joint','SDMetrics all four sources and all10 cohorts','source-only top100 content keywords plus OTHER; JSD and TV','clinical diagnosis distributions and whole-patient reference subsets'],
        'artifacts':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['verification.json','status.json'] and p.suffix in ['.json','.csv','.md','.html','.npy']}}
    write_json(OUT/'verification.json',verification);status('complete',datasets=4,cohorts=len(all_rows),report=str(OUT/'report.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','embed','score','report']);args=parser.parse_args()
    try:globals()[args.stage]()
    except BaseException as exc:status('failed',stage=args.stage,error=repr(exc));raise
