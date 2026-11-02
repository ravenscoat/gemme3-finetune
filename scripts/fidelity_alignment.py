"""Train-only profiles, balanced pre-generation conditions and honest fidelity scoring."""
from collections import Counter,defaultdict
import bisect
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np

from dataset_catalog import CATALOG,ROOT,core,data,read_json,read_rows,sha,signature,text_at,value_at
from dataset_metrics import assess_label,tv
from json_io import write_json
from mimic_fidelity_pilot import allocate,missing_pattern
from advanced_exploratory_audits import icd_prefix_issue

OUT=ROOT/'outputs/fidelity-alignment-v1'
SIZES={'bitext':156,'insuranceqa':64,'reliability':64,'mimic_demo':64}
ORDER=['mimic_demo','bitext','insuranceqa','reliability']
SEED=2026110142
PROFANITY=re.compile(r'\b(?:fuck\w*|shit\w*|damn)\b',re.I)


def numeric_values(key,row):
    if key=='mimic_demo':return {f:row[f] for f in data(key)['schema']['numeric_fields'] if f!='age'}
    result={'request_characters':len(text_at(row,CATALOG[key]['request']))}
    if key!='reliability':result['reply_characters']=len(text_at(row,CATALOG[key]['reply']))
    return result


def bins_for(key,rows):
    names=numeric_values(key,rows[0]);edges={}
    for field in names:
        values=[numeric_values(key,r)[field] for r in rows if numeric_values(key,r)[field] is not None]
        edges[field]=sorted(set(float(v) for v in np.quantile(values,np.linspace(0,1,11 if key=='mimic_demo' else 5))))
    return edges


def bin_index(value,edges):
    if value is None:return 'missing'
    if value<edges[0]:return 'below_source'
    if value>edges[-1]:return 'above_source'
    return str(max(0,min(len(edges)-2,bisect.bisect_right(edges,value)-1)))


def interval(value,edges,integer=False):
    if value is None:return None
    if len(edges)==1:lo=hi=edges[0]
    else:
        index=max(0,min(len(edges)-2,bisect.bisect_right(edges,value)-1));lo,hi=edges[index:index+2]
    return [math.ceil(lo),math.floor(hi)] if integer else [lo,hi]


def age_band(value):
    return [90,90] if value==90 else [max(21,value//10*10),min(89,value//10*10+9)]


def markers(row):
    request=row['instruction']
    return {'profanity_marker':bool(PROFANITY.search(request)),
            'ends_question_mark':request.rstrip().endswith('?'),'starts_lowercase':request.lstrip()[:1].islower(),
            'reply_placeholder':bool(re.search(r'\{\{[^{}]+\}\}',row['response']))}


def features(key,row,edges):
    paths={'bitext':['intent','category'], 'insuranceqa':['domain'],
        'reliability':['category','subcategory','difficulty','expected_output.intent','expected_output.routing_decision','expected_output.priority',
            'input.context.policy_type','input.context.policy_status','input.context.customer_tier','input.context.time_of_day','input.context.channel','input.context.state'],
        'mimic_demo':data('mimic_demo')['schema']['categorical_fields']}[key]
    result={p:signature(value_at(row,p)) for p in paths}
    for field,value in numeric_values(key,row).items():result[field+'_bin']=bin_index(value,edges[field])
    if key=='bitext':
        result.update({k:str(v) for k,v in markers(row).items()})
        result.update({'tag_'+c:str(c in row['tags']) for c in data(key)['schema']['tag_characters']})
    if key=='mimic_demo':result.update(age_band=str(age_band(row['age'])),lab_null_pattern=missing_pattern(row))
    if key=='reliability':
        result.update(history_turns=str(len(row['input']['conversation_history'])),actions=str(len(row['expected_output']['required_actions'])))
    return result


def balanced_indices(key,rows,n,edges,seed):
    """Balance profiles BEFORE generation; never select generated records by score."""
    rng=random.Random(seed);primary=CATALOG[key]['group'];groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row[primary]].append(i)
    quotas=allocate({k:len(v) for k,v in groups.items()},n,rng)
    selected=[]
    for group,count in quotas.items():selected.extend(rng.sample(groups[group],count))
    all_features=[features(key,r,edges) for r in rows]
    vocabulary=sorted({(field,value) for f in all_features for field,value in f.items()})
    columns={v:i for i,v in enumerate(vocabulary)};matrix=np.zeros((len(rows),len(columns)),dtype=np.int8)
    for i,f in enumerate(all_features):
        for field,value in f.items():matrix[i,columns[(field,value)]]=1
    target=matrix.sum(axis=0,dtype=np.float64)*n/len(rows);counts=matrix[selected].sum(axis=0,dtype=np.int64)
    loss=float(np.abs(counts-target).sum());initial=loss;chosen=set(selected);accepted=0
    for _ in range(12000):
        slot=rng.randrange(n);old=selected[slot];new=rng.choice(groups[rows[old][primary]])
        if new in chosen:continue
        trial=counts+matrix[new]-matrix[old];score=float(np.abs(trial-target).sum())
        if score<loss-1e-9:
            chosen.remove(old);chosen.add(new);selected[slot]=new;counts=trial;loss=score;accepted+=1
    rng.shuffle(selected)
    return selected,{'algorithm':'Largest-remainder primary quotas;12000 deterministic within-stratum swap proposals minimizing training-profile marginal L1 error',
        'primary_quotas':quotas,'initial_L1':initial,'final_L1':loss,'accepted_swaps':accepted,
        'scope':'External train-only balancing; not learned model frequencies, random-sample inference, or post-generation selection'}


def conditions(key,row,edges):
    values=numeric_values(key,row)
    ranges={field:interval(value,edges[field],integer=key!='mimic_demo') for field,value in values.items()}
    if key=='bitext':
        return {'required':{f:row[f] for f in ['intent','category','tags']},'ranges':ranges,'markers':markers(row)}
    if key=='insuranceqa':return {'required':{'domain':row['domain']},'ranges':ranges}
    if key=='reliability':
        required={f:row[f] for f in ['category','subcategory','difficulty']}
        required.update({'input.context':row['input']['context'],
            **{'expected_output.'+f:row['expected_output'][f] for f in ['intent','routing_decision','priority']}})
        return {'required':required,'ranges':ranges,'history_turns':len(row['input']['conversation_history']),
                'required_action_count':len(row['expected_output']['required_actions'])}
    required={f:row[f] for f in data(key)['schema']['categorical_fields']}
    required['primary_diagnosis_code']=row['primary_diagnosis_code']
    ranges['age']=age_band(row['age'])
    return {'required':required,'ranges':ranges,'lab_null_pattern':missing_pattern(row)}


def prompt_for(key,condition):
    if key=='bitext':
        prompt=('Generate synthetic data. Create one new Bitext-style insurance customer-support record as a JSON object '
            'with exactly string fields intent, category, tags, instruction, response. instruction is a new customer request '
            'and response is its matching support reply. Follow the required conditions below. Return no commentary or markdown.')
    else:prompt=data(key)['prompt']
    prompt+='\nRequired field values (dotted paths refer to nested JSON fields): '+json.dumps(condition['required'],ensure_ascii=False)+'.'
    for field,limits in condition['ranges'].items():
        if limits is None:prompt+=' '+field+' must be null.'
        elif key=='mimic_demo':
            lo,hi=limits;prompt+=f' {field} must be a non-null '+('integer' if field=='age' else 'number')+f' between {lo:g} and {hi:g}, inclusive.'
        else:
            title={'request_characters':'customer_message' if key=='reliability' else CATALOG[key]['request'], 'reply_characters':CATALOG[key]['reply']}[field]
            prompt+=f' Aim for {limits[0]} to {limits[1]} characters in {title}, including spaces; write a complete, coherent text.'
    if key=='bitext':
        m=condition['markers'];prompt+=' The request '+('includes a profanity marker such as damn or fucking.' if m['profanity_marker'] else 'contains no profanity.')
        prompt+=' '+('End the request with a question mark.' if m['ends_question_mark'] else 'Do not end the request with a question mark.')
        prompt+=' '+('Start the request with a lowercase letter.' if m['starts_lowercase'] else 'Start the request with an uppercase letter.')
        prompt+=' '+('Use appropriate double-brace template placeholders in the reply.' if m['reply_placeholder'] else 'Use no double-brace placeholders in the reply.')
    if key=='reliability':prompt+=f" Include exactly {condition['history_turns']} conversation_history turns and {condition['required_action_count']} required_actions. Generate a new customer message consistent with the conditions."
    return prompt


def baseline(key):
    if key=='bitext':
        directory=ROOT/'outputs/bitext-4b-full-pass-evaluation-500-windows';path=directory/'tuned_generations.jsonl';p=read_json(directory/'generation_protocol.json')
    elif key=='mimic_demo':
        directory=ROOT/'outputs/mimic-demo-fidelity-conditioned-v1-windows';path=directory/'candidate_generic_generations.jsonl';p=read_json(directory/'comparison_plan.json')
        p={**p,'batch_seed_start':p['seed_start']}
    else:
        directory=ROOT/CATALOG[key]['assessment'];path=directory/'tuned_generations.jsonl';p=read_json(directory/'protocol.json')
    return path,p


def prepare(key):
    source=data(key);rows=[core(key,r) for r in source['train']];n=SIZES[key];directory=OUT/key;directory.mkdir(parents=True,exist_ok=True)
    selected_path=ROOT/('outputs/mimic-demo-fidelity-conditioned-v1-windows/selection.json' if key=='mimic_demo' else CATALOG[key]['selection'])
    selected=read_json(selected_path);assert sha(Path(selected['adapter'])/'adapter_model.safetensors')==selected['adapter_sha256']
    path,bp=baseline(key);raw=read_rows(path)[:n]
    if len(raw)!=n or [r['sample_id'] for r in raw]!=list(range(n)):raise RuntimeError('Unsupported baseline IDs')
    edges=bins_for(key,rows);indices,balance=balanced_indices(key,rows,n,edges,SEED+list(SIZES).index(key))
    planned=[conditions(key,rows[i],edges) for i in indices]
    protocol={'dataset':key,'n':n,'adapter':selected['adapter'],'adapter_sha256':selected['adapter_sha256'],
        'source_train_sha256':sha(ROOT/CATALOG[key]['prepared']/'train.jsonl'),
        'source_prompt_sha256':hashlib.sha256(source['prompt'].encode()).hexdigest(),
        'baseline_path':str(path.relative_to(ROOT)),'baseline_sha256':sha(path),
        'baseline_ids':list(range(n)),'baseline_batch_seed_start':bp['batch_seed_start'],'batch_size':4,
        'temperature':bp['temperature'],'top_p':bp['top_p'],'top_k':bp['top_k'],'context':CATALOG[key]['context'],
        'controller_seed':SEED+list(SIZES).index(key),'source_edges':edges,'balance':balance,
        'selected_source_rows':[source['train'][i].get('source_id',source['train'][i].get('source_line')) for i in indices],
        'conditions':planned,'prompts':[prompt_for(key,c) for c in planned],
        'method':'Externally controlled generation using train-derived joint profiles and coarse value/length intervals; Gemma fills records. Primary-frequency and supplied-range gains are not evidence of learned probabilities.',
        'policy':'Development only. All attempted records retained. No repair, reroll, output selection, retraining, new test scoring or automatic deployment. Same batch size/seeds/temperature; prompts and remaining generation budgets differ.'}
    frozen=directory/'protocol.json'
    if frozen.exists():
        if read_json(frozen)!=protocol:raise RuntimeError('Alignment protocol changed')
    else:
        write_json(frozen,protocol)
        (directory/'baseline_generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw),encoding='utf-8')
        write_json(directory/'baseline_freeze.json',{'protocol_sha256':sha(frozen),'baseline_copy_sha256':sha(directory/'baseline_generations.jsonl')})
        write_json(directory/'progress.json',{'phase':'prepared','completed':0,'total':n})
    return protocol


def score(key):
    directory=OUT/key;p=read_json(directory/'protocol.json');train=[core(key,r) for r in data(key)['train']]
    result={'dataset':key,'protocol_sha256':sha(directory/'protocol.json'),'arms':{},'method':p['method'],'policy':p['policy']}
    for label in ['baseline','controlled']:
        path=directory/(label+'_generations.jsonl')
        if not path.exists() or len(read_rows(path))!=p['n']:continue
        m,rows=assess_label(key,read_rows(path),directory/(label+'_records.jsonl'));records=[r['record'] for r in rows]
        m['feature_fidelity']={}
        f_source=[features(key,r,p['source_edges']) for r in train];f_generated=[features(key,r,p['source_edges']) for r in records]
        for field in f_source[0]:
            actual=[r[field] for r in f_generated];ref=[r[field] for r in f_source]
            m['feature_fidelity'][field]={'TV':tv(actual,ref),'generated_counts':dict(Counter(actual)),'source_counts':dict(Counter(ref))}
        if key=='bitext':
            for field in ['intent','category','tags']:m['fidelity_vs_training']['categorical'][field]={'total_variation':tv([r[field] for r in records],[r[field] for r in train])}
        if key=='mimic_demo':m['prefix_version_flags']=sum(icd_prefix_issue(r) is not None for r in records)
        if label=='controlled':
            exact=ranges=both=0
            for wrapper in rows:
                c=p['conditions'][wrapper['sample_id']];r=wrapper['record'];a=all(value_at(r,field)==value for field,value in c['required'].items())
                values=numeric_values(key,r)
                if key=='mimic_demo':values['age']=r['age']
                b=all(values[field] is None if limits is None else values[field] is not None and limits[0]-1e-5<=values[field]<=limits[1]+1e-5 for field,limits in c['ranges'].items())
                exact+=a;ranges+=b;both+=a and b
            m['condition_compliance']={'required_values':exact,'all_ranges':ranges,'both':both,'attempts':p['n'],'range_tolerance':1e-5}
        m['raw_sha256']=sha(path);result['arms'][label]=m
    write_json(directory/'metrics.json',result);return result


if __name__=='__main__':
    for key in ORDER:
        p=prepare(key);score(key);print(key,p['n'],'planned; profile L1',round(p['balance']['initial_L1'],2),'->',round(p['balance']['final_L1'],2),flush=True)
