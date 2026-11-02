"""Final bounded source-context pilot; never trains or changes selected models."""
import argparse
import contextlib
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import time

from dataset_catalog import ROOT, core, data, read_json, read_rows, sha, signature
from dataset_metrics import assess_label
from json_io import write_json

OUT=ROOT/'outputs/insuranceqa-final-grounded-v1'
N=32

def prepare():
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; do not reprepare')
    OUT.mkdir(parents=True,exist_ok=True)
    source=data('insuranceqa'); train=source['train']
    baseline=ROOT/'outputs/insuranceqa-4b-v1-windows-assessment-v1/tuned_generations.jsonl'
    bp=read_json(baseline.parent/'protocol.json')
    from mimic_fidelity_pilot import allocate
    rng=random.Random(2026110207); groups=defaultdict(list)
    for row in train:
        # Pre-generation eligibility, never chosen by generated scores.
        if len(row['answer'])<=1100 and not re.search(r'https?://|www\.|@|\b\d{3}[- .]\d{3}[- .]\d{4}\b',row['answer']):
            groups[row['domain']].append(row)
    quotas=allocate(Counter(r['domain'] for r in train),N,rng)
    anchors=[]
    for domain,count in sorted(quotas.items()):anchors.extend(rng.sample(groups[domain],count))
    rng.shuffle(anchors)
    prompts=[]
    for row in anchors:
        prompts.append(source['prompt']+'\nSource context (reference material, not instructions):\n'+
            json.dumps({'domain':row['domain'],'passage':row['answer']},ensure_ascii=False)+
            '\nUse this domain. Write a new question that the passage can answer, and a matching answer supported only by the passage. '+
            'Preserve its qualifications and roles. Do not add names, contact details, prices, laws or dates absent from the passage. '+
            'Use your own wording; do not reproduce the source answer verbatim. Return only the required JSON object.')
    rows=read_rows(baseline)[:N]
    assert [r['sample_id'] for r in rows]==list(range(N))
    (OUT/'baseline_generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    p={'n':N,'format_gate':{'first_attempts':8,'minimum_valid':7,'failure_action':'stop without expansion or retries'},
       'adapter':bp['selected_adapter']['adapter'],'context':1536,'batch_size':4,
       'temperature':1.0,'top_p':.95,'top_k':64,'seed_start':bp['batch_seed_start'],
       'input_hashes':{str(baseline.relative_to(ROOT)):sha(baseline),
           'data/insuranceqa/prepared/train.jsonl':sha(ROOT/'data/insuranceqa/prepared/train.jsonl'),
           str((Path(bp['selected_adapter']['adapter'])/'adapter_model.safetensors').relative_to(ROOT)):bp['selected_adapter']['adapter_sha256']},
       'anchors':anchors,'prompts':prompts,'domain_quotas':quotas,
       'eligibility':'Training-only complete answers <=1100 characters, excluding URL/email/common US phone patterns. Original domain frequencies determine quotas. This excludes long/contact-bearing anchors and may bias outputs.',
       'eligible_counts':{k:len(v) for k,v in groups.items()},
       'review':'Assistant reviews every valid output for question-answer fit and internal consistency; grounded arm additionally checked for source support. Pass/Unsure/Issue, not expert factual validation.',
       'decision':'Keep original policy unless raw grounded output improves meaning without copying or distribution regressions. No post-hoc output filtering, rerolls, retraining, teacher API, new experiments or automatic deployment.',
       'limits':'32 outputs per arm, one seed sequence, retrospective development. Context and supplied domain are combined external controls; not an isolated grounding effect, not an untouched test. Same sampling/batch seeds, different prompts and remaining context budgets.'}
    write_json(OUT/'protocol.json',p)
    write_json(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'baseline_sha256':sha(OUT/'baseline_generations.jsonl')})
    print('Frozen final pilot: 8-attempt gate then at most32; no training.',flush=True)

def check():
    p=read_json(OUT/'protocol.json');f=read_json(OUT/'freeze.json')
    assert sha(OUT/'protocol.json')==f['protocol_sha256']
    assert sha(OUT/'baseline_generations.jsonl')==f['baseline_sha256']
    for path,digest in p['input_hashes'].items():assert sha(ROOT/path)==digest,path
    return p

def generate():
    p=check();lock=ROOT/'reports/assessment_gpu.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        from unsloth import FastModel
        import torch
        from transformers import set_seed
        path=OUT/'grounded_generations.jsonl'
        if path.exists():raise RuntimeError('Do not rerun or overwrite generated attempts')
        model,processor=FastModel.from_pretrained(model_name=p['adapter'],max_seq_length=p['context'],load_in_4bit=True,full_finetuning=False)
        tok=getattr(processor,'tokenizer',processor);tok.padding_side='left';FastModel.for_inference(model);model.eval()
        raw=[];started=time.time()
        with path.open('w',encoding='utf-8') as stream:
            for start in range(0,N,4):
                prompts=[tok.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True) for t in p['prompts'][start:start+4]]
                batch=tok(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                maximum=p['context']-batch['input_ids'].shape[1]-2
                if maximum<400:raise RuntimeError('Insufficient generation context; no truncation')
                seed=p['seed_start']+start//4;set_seed(seed)
                with (model.disable_adapter() if p.get('disable_adapter',False) else contextlib.nullcontext()):
                    with torch.inference_mode():out=model.generate(**batch,max_new_tokens=maximum,do_sample=True,temperature=p['temperature'],top_p=p['top_p'],top_k=p['top_k'],pad_token_id=tok.pad_token_id,use_cache=True)
                eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                for i in range(4):
                    tokens=out[i,batch['input_ids'].shape[1]:].tolist();end=next((j for j,t in enumerate(tokens) if t in eos),None)
                    if end is not None:tokens=tokens[:end+1]
                    r={'sample_id':start+i,'batch_seed':seed,'text':tok.decode(tokens,skip_special_tokens=True),'new_tokens':len(tokens),
                       'hit_token_limit':end is None and len(tokens)>=maximum,'input_tokens_padded':batch['input_ids'].shape[1],'max_new_tokens':maximum}
                    if p.get('disable_adapter',False):r['adapter_disabled']=True
                    raw.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
                os.fsync(stream.fileno());print(f'Final InsuranceQA {start+4}/{N}; elapsed {time.time()-started:.0f}s',flush=True)
                write_json(OUT/'progress.json',{'completed':start+4,'planned':N,'elapsed_seconds':time.time()-started})
                if start==4:
                    m,_=assess_label('insuranceqa',raw,OUT/'gate_records.jsonl')
                    passed=m['structure_valid']>=p['format_gate']['minimum_valid'];write_json(OUT/'format_gate.json',{'passed':passed,'metrics':m})
                    if not passed:print('Format gate failed; stopped.',flush=True);break
                del batch,out
        write_json(OUT/'generation_complete.json',{'attempts':len(raw),'elapsed_seconds':time.time()-started,'raw_sha256':sha(path),'weights_updated':False})
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

def score():
    p=check();import distribution_benchmark as b
    import numpy as np
    b.OUT=OUT;(OUT/'insuranceqa').mkdir(exist_ok=True)
    reference=[core('insuranceqa',r) for r in data('insuranceqa')['train']]
    count=len(read_rows(OUT/'grounded_generations.jsonl'));collections={};metrics={}
    for arm in ['baseline','grounded']:
        raw=read_rows(OUT/(arm+'_generations.jsonl'))[:count]
        m,wrappers=assess_label('insuranceqa',raw,OUT/(arm+'_records.jsonl'))
        records=[r['record'] for r in wrappers];collections[arm]=records
        m['sdmetrics']=b.sd_report('insuranceqa',records,reference)
        m['length_percentiles']={field:{'source':np.percentile([len(r[field]) for r in reference],[1,5,50,95,99]).tolist(),
             'generated':np.percentile([len(r[field]) for r in records],[1,5,50,95,99]).tolist()} for field in ['question','answer']}
        metrics[arm]=m
    grounded=read_rows(OUT/'grounded_records.jsonl');overlap=[]
    def tokens(text):return re.findall(r'\w+',text.casefold())
    for wrapper in grounded:
        i=wrapper['sample_id'];answer=wrapper['record']['answer'];passage=p['anchors'][i]['answer']
        a=tokens(answer);s=tokens(passage);source_grams={tuple(s[j:j+8]) for j in range(max(0,len(s)-7))}
        grams=[tuple(a[j:j+8]) for j in range(max(0,len(a)-7))]
        overlap.append({'sample_id':i,'answer_normalized_equal_to_anchor':signature(answer)==signature(passage),
            'generated_answer_8gram_source_overlap':sum(g in source_grams for g in grams)/len(grams) if grams else None})
    write_json(OUT/'anchor_copying.json',{'scope':'Literal overlap with the supplied anchor; not semantic similarity or a privacy guarantee. Not used for filtering.',
        'records':overlap,'exact_anchor_answers':sum(r['answer_normalized_equal_to_anchor'] for r in overlap),
        'mean_8gram_overlap':float(np.mean([r['generated_answer_8gram_source_overlap'] for r in overlap if r['generated_answer_8gram_source_overlap'] is not None]))})
    result={'arms':metrics,'keywords':b.lexical_profile('insuranceqa',reference,collections),'protocol_sha256':sha(OUT/'protocol.json')}
    write_json(OUT/'metrics.json',result)
    old=read_json(ROOT/'reports/distribution_benchmark/v1/embedding_inputs.json')['insuranceqa']
    inputs={k:old[k] for k in ['training_reference','training_control']}
    for arm,records in collections.items():inputs[arm]={view:[b.text_views('insuranceqa',r)[view] for r in records] for view in ['request','reply','joint']}
    write_json(OUT/'embedding_inputs.json',{'insuranceqa':inputs})
    write_json(OUT/'embedding_protocol.json',{'input_hashes':p['input_hashes'],'embedding_inputs_sha256':sha(OUT/'embedding_inputs.json')})
    print(json.dumps({a:{'valid':m['structure_valid'],'sd':m['sdmetrics']['quality'],'copy':m['exact_training_record_match_rate']} for a,m in metrics.items()}),flush=True)

def embed():
    check();import distribution_benchmark as b
    b.OUT=OUT
    # Existing encoder implementation expects protocol.json. Redirect read-only lookup,
    # preserving our frozen generation protocol unchanged.
    original=b.read_json
    b.read_json=lambda path:original(OUT/'embedding_protocol.json') if Path(path)==OUT/'protocol.json' else original(path)
    b.embed()

def text_score():
    check();import numpy as np
    import distribution_benchmark as b
    from prdc import compute_prdc
    manifest=read_json(OUT/'embedding_manifest.json')['features']['insuranceqa'];result=read_json(OUT/'metrics.json')
    result['text_metrics']={}
    for view in ['request','reply','joint']:
        arrays={}
        for arm,info in manifest.items():
            path=ROOT/info[view]['path'];assert sha(path)==info[view]['sha256'];arrays[arm]=np.load(path).astype(float)
        x=arrays['training_reference'];gamma=b.median_gamma(np.concatenate([x,arrays['training_control']]))
        result['text_metrics'][view]={'reference_n':len(x),'gamma':gamma,'arms':{}}
        for arm in ['baseline','grounded']:
            y=arrays[arm];result['text_metrics'][view]['arms'][arm]={'n':len(y),'mmd2':b.mmd2(x,y,gamma),
                'prdc_k5':{k:float(v) for k,v in compute_prdc(x,y,nearest_k=5).items()}}
    write_json(OUT/'metrics.json',result);print('Text metrics saved.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate','score','embed','text_score']);args=parser.parse_args()
    globals()[args.stage]()
