"""One bounded consistency-instruction experiment on fixed scenarios."""
import argparse
import json
import os
from pathlib import Path
import time

from dataset_catalog import ROOT,data,read_json,read_rows,sha
from dataset_metrics import assess_label
from generate_comparison import completed_rows
from json_io import write_json

OUT=ROOT/'outputs/reliability-consistency-v1'
PARENT=ROOT/'outputs/prefix-alignment-v1/reliability'
IDS=[0,1,7,14,18,22,29,33,36,42,52,63]
ISSUES=[18,33,42,63]
GUARD=('\nKeep the supplied scenario unchanged. Make the workflow consistent with the customer message, '
    'policy context and conversation history: do not repeat an action the history says is completed; '
    'confirm it instead. If the requested policy conflicts with the context, clarify or verify the '
    'relevant policy before policy-specific conclusions or updates. Address each explicit customer '
    'question through the required actions and response elements. Do not invent policy values.')


def prefix_for(record):
    return json.dumps({k:record[k] for k in ['category','subcategory','difficulty','input']},ensure_ascii=False)[:-1]+', "expected_output": '


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);p=read_json(PARENT/'protocol.json');lookup={w['sample_id']:w['record'] for w in read_rows(PARENT/'controlled_records.jsonl')}
    cases=[{'sample_id':i,'source_sample_id':ident,'record':lookup[ident]} for i,ident in enumerate(IDS)]
    protocol={'dataset':'reliability','n':12,'source_ids':IDS,'previous_issue_ids':ISSUES,'passing_control_ids':[i for i in IDS if i not in ISSUES],
        'adapter':p['adapter'],'adapter_sha256':p['adapter_sha256'],'source_raw_sha256':sha(PARENT/'controlled_generations.jsonl'),
        'source_train_sha256':sha(ROOT/'data/reliability/prepared/train.jsonl'),
        'prompts':{'baseline':data('reliability')['prompt'],'candidate':data('reliability')['prompt']+GUARD},
        'assistant_prefixes':[prefix_for(c['record']) for c in cases],
        'batch_size':4,'seed_start':2026110150,'temperature':1.,'top_p':.95,'top_k':64,'context':1024,
        'scope':'Fixed-input workflow completion, not unconditional synthetic-record generation. Inputs/category labels supplied externally; request similarity is not a model result. Four previously flagged cases plus8 fixed passing controls; unblinded AI review, no accuracy/population claim.',
        'gate':'All12 schema-valid; no source-constraint or whole-record-copy regression against the newly generated baseline; no token-limit/placeholder failures; all8 passing controls remain Pass; at least2 prior Issues become Pass; candidate Issue count lower; action-count Wasserstein and TV and workflow-character Wasserstein do not exceed baseline. Descriptive small sample, no distribution-equivalence claim.',
        'policy':'Exactly one prompt variant,24 attempts maximum; no repairs, rerolls, further prompt revisions, broad retraining, test scoring or default deployment. Stop and package even if it fails.'}
    if (OUT/'protocol.json').exists():assert read_json(OUT/'protocol.json')==protocol
    else:
        write_json(OUT/'protocol.json',protocol)
        (OUT/'cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
        write_json(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'cases_sha256':sha(OUT/'cases.jsonl')})
        for arm in ['baseline','candidate']:
            (OUT/arm).mkdir(exist_ok=True);write_json(OUT/arm/'progress.json',{'phase':'prepared','completed':0,'total':12})
    return protocol


def run():
    p=read_json(OUT/'protocol.json');f=read_json(OUT/'freeze.json')
    assert sha(OUT/'protocol.json')==f['protocol_sha256'] and sha(OUT/'cases.jsonl')==f['cases_sha256']
    assert sha(PARENT/'controlled_generations.jsonl')==p['source_raw_sha256']
    assert sha(Path(p['adapter'])/'adapter_model.safetensors')==p['adapter_sha256']
    assert sha(ROOT/'data/reliability/prepared/train.jsonl')==p['source_train_sha256']
    lock=ROOT/'reports/assessment_gpu.lock'
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():raise RuntimeError('Training owns GPU')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    status={'phase':'running','datasets':{'baseline':'pending','candidate':'pending'},'counts':{'baseline':12,'candidate':12},'current':None}
    started=time.time()
    def save():status['elapsed_seconds']=round(time.time()-started,2);write_json(ROOT/'reports/reliability_consistency_status.json',status,best_effort=True)
    def progress(arm,phase,n):write_json(OUT/arm/'progress.json',{'phase':phase,'completed':n,'total':12,'elapsed_seconds':round(time.time()-started,2)},best_effort=True)
    try:
        save()
        from unsloth import FastModel
        import torch
        from transformers import set_seed
        model,processor=FastModel.from_pretrained(model_name=p['adapter'],max_seq_length=1024,load_in_4bit=True,full_finetuning=False)
        t=getattr(processor,'tokenizer',processor);t.padding_side='left';FastModel.for_inference(model);model.eval()
        metrics={}
        for arm in ['baseline','candidate']:
            status['current']=arm;status['datasets'][arm]='running';save()
            path=OUT/arm/'generations.jsonl';count=len(completed_rows(path,12));progress(arm,'generation',count)
            with path.open('a',encoding='utf-8') as stream:
                for start in range(count//4*4,12,4):
                    prompts=[t.apply_chat_template([{'role':'user','content':p['prompts'][arm]}],tokenize=False,add_generation_prompt=True)+prefix for prefix in p['assistant_prefixes'][start:start+4]]
                    inputs=t(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');maximum=1024-inputs['input_ids'].shape[1]-2
                    if maximum<200:raise RuntimeError('Insufficient completion context')
                    seed=p['seed_start']+start//4;set_seed(seed);began=time.time()
                    with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=maximum,do_sample=True,temperature=1.,top_p=.95,top_k=64,pad_token_id=t.pad_token_id,use_cache=True)
                    duration=time.time()-began;eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                    for offset in range(4):
                        index=start+offset
                        if index<count:continue
                        tokens=generated[offset,inputs['input_ids'].shape[1]:].tolist();end=next((i for i,x in enumerate(tokens) if x in eos),None)
                        if end is not None:tokens=tokens[:end+1]
                        text=t.decode(tokens,skip_special_tokens=True)
                        row={'sample_id':index,'source_sample_id':IDS[index],'batch_seed':seed,'batch_size':4,'generated_text':text,'assistant_prefix':p['assistant_prefixes'][index],
                            'text':p['assistant_prefixes'][index]+text,'new_tokens':len(tokens),'hit_token_limit':end is None and len(tokens)>=maximum,'batch_seconds':duration}
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    os.fsync(stream.fileno());progress(arm,'generation',start+4);print(arm,f'{start+4}/12',f'{duration:.1f}s',flush=True);del inputs,generated
            metrics[arm],_=assess_label('reliability',read_rows(path),OUT/arm/'records.jsonl')
            metrics[arm]['raw_sha256']=sha(path);progress(arm,'complete',12);status['datasets'][arm]='complete';save()
        write_json(OUT/'metrics.json',metrics);write_json(OUT/'completion.json',{'protocol_sha256':sha(OUT/'protocol.json'),'metrics_sha256':sha(OUT/'metrics.json'),'weights_updated':False,'new_test_scoring':False})
        status.update(phase='complete',current=None,review='Pending assistant comparison; no automatic adoption');save()
    except BaseException as exc:status.update(phase='failed',error=str(exc));save();raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':
    if '--prepare' in __import__('sys').argv:prepare()
    else:run()
