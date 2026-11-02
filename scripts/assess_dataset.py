"""Frozen base/adapter generation comparison, then a no-update held-out test."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

from dataset_catalog import CATALOG, ROOT, core, data, read_json, read_rows, sha
from dataset_metrics import assess_label, fidelity
from generate_comparison import completed_rows
from json_io import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['insuranceqa','reliability','mimic_demo'])
    parser.add_argument('--samples',type=int,default=128)
    args=parser.parse_args()
    source=data(args.dataset); spec=source['spec']; prepared=ROOT/spec['prepared']; out=ROOT/spec['assessment']
    out.mkdir(parents=True,exist_ok=True)
    selected=read_json(ROOT/spec['selection'])
    assert sha(Path(selected['adapter'])/'adapter_model.safetensors')==selected['adapter_sha256']
    for name, expected in selected['input_hashes'].items():
        if sha(prepared/name)!=expected: raise RuntimeError('Training source artifact changed: '+name)
    protocol={'dataset':args.dataset,'selected_adapter':selected,'samples_per_label':args.samples,'batch_size':4,
              'batch_seed_start':202611010,'temperature':1.,'top_p':.95,'top_k':64,'context':spec['context'],
              'prompt_sha256':sha(prepared/'generation_prompt.txt'),'schema_sha256':sha(prepared/'schema.json'),
              'train_sha256':sha(prepared/'train.jsonl'),'validation_sha256':sha(prepared/'validation.jsonl'),'test_sha256':sha(prepared/'test.jsonl'),
              'generation_limit':'Remaining trained context minus 2 tokens; no repaired/replaced failed attempts',
              'test_policy':'Selection and generation settings frozen before held-out scoring; no tuning or weight updates',
              'scope':spec['scope']}
    if (out/'protocol.json').exists():
        if read_json(out/'protocol.json')!=protocol: raise RuntimeError('Assessment protocol changed; preserve this run')
    else: write_json(out/'protocol.json',protocol)
    if (out/'assessment.json').exists():
        print('Assessment already complete:',args.dataset,flush=True); return
    started=time.time()
    def progress(phase,done,total,**extra):
        write_json(out/'progress.json',{'dataset':args.dataset,'phase':phase,'completed':done,'total':total,
                   'elapsed_seconds':round(time.time()-started,2),**extra},best_effort=True)
    progress('loading',0,args.samples)
    from unsloth import FastModel
    import torch
    from transformers import set_seed
    model,processor=FastModel.from_pretrained(model_name=selected['adapter'],max_seq_length=spec['context'],load_in_4bit=True,full_finetuning=False)
    tokenizer=getattr(processor,'tokenizer',processor)
    FastModel.for_inference(model); model.eval()
    prompt=source['prompt']
    prefix=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
    prefix_ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    max_new=spec['context']-len(prefix_ids)-2
    if max_new<300: raise RuntimeError('Insufficient generation context')
    for label in ['base','tuned']:
        path=out/(label+'_generations.jsonl'); count=len(completed_rows(path,args.samples))
        with path.open('a',encoding='utf-8') as stream:
            for start in range(count//4*4,args.samples,4):
                size=min(4,args.samples-start); seed=protocol['batch_seed_start']+start//4
                set_seed(seed)
                inputs=tokenizer([prefix]*size,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                began=time.time()
                with (model.disable_adapter() if label=='base' else contextlib.nullcontext()),torch.inference_mode():
                    generated=model.generate(**inputs,max_new_tokens=max_new,do_sample=True,temperature=1.,top_p=.95,top_k=64,pad_token_id=tokenizer.pad_token_id,use_cache=True)
                elapsed=time.time()-began
                eos=model.generation_config.eos_token_id
                eos=[eos] if isinstance(eos,int) else eos or []
                for offset in range(size):
                    if start+offset<count: continue
                    tokens=generated[offset,inputs['input_ids'].shape[1]:].tolist()
                    ended=next((i for i,t in enumerate(tokens) if t in eos),None)
                    if ended is not None: tokens=tokens[:ended+1]
                    row={'sample_id':start+offset,'batch_seed':seed,'text':tokenizer.decode(tokens,skip_special_tokens=True),
                         'new_tokens':len(tokens),'hit_token_limit':ended is None and len(tokens)>=max_new,'batch_seconds':elapsed,'batch_size':size}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                os.fsync(stream.fileno())
                progress('generation',start+size,args.samples,label=label)
                print(f'{args.dataset} {label}: {start+size}/{args.samples}',flush=True)
                del inputs,generated
    result={'protocol':protocol,'labels':{},'limitations':[
        '128 attempts per model: exploratory sample sizes, not proof of distribution equivalence.',
        'Field distributions and copying use structurally valid records; all failed attempts are retained and counted.',
        'Source-vocabulary constraints are separate from structural validity and semantic correctness.',
        'Loss is prediction of source records; it does not certify answer or workflow correctness.',
        'KS p-values assume independent samples and are unadjusted across fields; do not interpret non-significance as equivalence.',
        spec['scope']]}
    valid={}
    for label in ['base','tuned']:
        result['labels'][label],valid[label]=assess_label(args.dataset,read_rows(out/(label+'_generations.jsonl')),out/(label+'_records.jsonl'))
    result['validation_reference_control']=fidelity(args.dataset,read_rows(prepared/'validation.jsonl'),source['train'])
    write_json(out/'generation_metrics.json',result)
    # Freeze the exact raw outputs before accessing held-out values for scoring.
    freeze={'protocol_sha256':sha(out/'protocol.json'),**{label+'_generation_sha256':sha(out/(label+'_generations.jsonl')) for label in ['base','tuned']}}
    if (out/'test_freeze.json').exists() and read_json(out/'test_freeze.json')!=freeze: raise RuntimeError('Frozen outputs changed')
    write_json(out/'test_freeze.json',freeze)
    test=read_rows(prepared/'test.jsonl'); loss_path=out/'test_record_losses.jsonl'
    losses=read_rows(loss_path) if loss_path.exists() else []
    if [r['source_id'] for r in losses]!=[r['source_id'] for r in test[:len(losses)]]: raise RuntimeError('Test resume IDs changed')
    with loss_path.open('a',encoding='utf-8') as stream:
        for i in range(len(losses),len(test)):
            row=test[i]
            answer=json.dumps(core(args.dataset,row),ensure_ascii=False)
            text=tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':answer}],tokenize=False,add_generation_prompt=False)
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            if len(ids)>spec['context'] or ids[:len(prefix_ids)]!=prefix_ids: raise RuntimeError(f'Unsupported test context/mask at {row["source_id"]}; no silent truncation')
            inputs=torch.tensor([ids],device='cuda'); labels=inputs.clone(); labels[:,:len(prefix_ids)]=-100
            with torch.inference_mode(): output=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels,use_cache=False)
            loss=float(output.loss.float().cpu())
            if not math.isfinite(loss): raise RuntimeError('Non-finite held-out loss')
            item={'source_id':row['source_id'],'target_tokens':len(ids)-len(prefix_ids),'mean_token_nll':loss}
            losses.append(item); stream.write(json.dumps(item)+'\n'); stream.flush()
            if (i+1)%10==0 or i+1==len(test):
                progress('held_out_test',i+1,len(test));print(f'{args.dataset} test: {i+1}/{len(test)}',flush=True)
            del output,inputs,labels
    nll=sum(r['mean_token_nll']*r['target_tokens'] for r in losses)/sum(r['target_tokens'] for r in losses)
    result['test']={'records':len(losses),'mean_record_loss':sum(r['mean_token_nll'] for r in losses)/len(losses),
                    'token_weighted_nll':nll,'perplexity':math.exp(nll),'weights_updated':False,
                    'tuned_fidelity_vs_test':fidelity(args.dataset,[r['record'] for r in valid['tuned']],test),'freeze':freeze}
    write_json(out/'assessment.json',result)
    progress('complete',len(test),len(test))
    lines=[f'# {spec["label"]}: frozen synthetic-data assessment','',spec['scope'],'',
           '| Measure | Original | Fine-tuned |','|---|---:|---:|']
    for title,key in [('Structural records','structure_valid'),('Source constraints satisfied','source_constraint_valid'),('Full-record copying','exact_training_record_match_rate'),('Request copying','training_request_match_rate'),('Reply copying','training_reply_match_rate')]:
        cells=[]
        for label in ['base','tuned']:
            value=result['labels'][label][key]
            cells.append('n/a' if value is None else str(value)+'/128' if type(value) is int else f'{value:.3f}')
        lines.append('| '+title+' | '+' | '.join(cells)+' |')
    lines += ['',f'Held-out records: {len(losses)}. Mean record completion loss: {result["test"]["mean_record_loss"]:.4f}; token-weighted perplexity: {math.exp(nll):.3f}.',
              '', 'Per-field category distances, ranges, missingness, KS/Wasserstein distances and failures are in the machine-readable assessment.',
              '',*['- '+note for note in result['limitations']],'']
    (ROOT/'reports'/(args.dataset+'_assessment.md')).write_text('\n'.join(lines),encoding='utf-8')
    print('Assessment saved:',out,flush=True)


if __name__=='__main__': main()
