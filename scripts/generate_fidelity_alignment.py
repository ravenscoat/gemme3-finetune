"""Generate each frozen, source-profile-controlled batch without output selection."""
import argparse
import json
import os
import time
from pathlib import Path

from dataset_catalog import ROOT,CATALOG,read_json,sha
from fidelity_alignment import OUT,score
from generate_comparison import completed_rows
from json_io import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=CATALOG)
    parser.add_argument('--experiment-root',type=Path,default=OUT);args=parser.parse_args()
    import fidelity_alignment as alignment
    experiment=args.experiment_root.resolve()
    if not experiment.is_relative_to((ROOT/'outputs').resolve()):raise RuntimeError('Experiment must stay within project outputs')
    alignment.OUT=experiment
    key=args.dataset;directory=experiment/key;p=read_json(directory/'protocol.json');freeze=read_json(directory/'baseline_freeze.json')
    if sha(directory/'protocol.json')!=freeze['protocol_sha256']:raise RuntimeError('Frozen protocol changed')
    if sha(directory/'baseline_generations.jsonl')!=freeze['baseline_copy_sha256']:raise RuntimeError('Frozen baseline changed')
    if sha(ROOT/p['baseline_path'])!=p['baseline_sha256']:raise RuntimeError('Original baseline changed')
    if sha(ROOT/CATALOG[key]['prepared']/'train.jsonl')!=p['source_train_sha256']:raise RuntimeError('Training source changed')
    if sha(Path(p['adapter'])/'adapter_model.safetensors')!=p['adapter_sha256']:raise RuntimeError('Frozen adapter changed')
    if (directory/'completion.json').exists():print(key,'already complete.');return
    started=time.time()
    def progress(phase,done,**values):write_json(directory/'progress.json',{'phase':phase,'completed':done,'total':p['n'],'elapsed_seconds':round(time.time()-started,2),**values},best_effort=True)
    path=directory/'controlled_generations.jsonl';count=len(completed_rows(path,p['n']))
    progress('loading',count)
    from unsloth import FastModel
    import torch
    from transformers import set_seed
    model,processor=FastModel.from_pretrained(model_name=p['adapter'],max_seq_length=p['context'],load_in_4bit=True,full_finetuning=False)
    tokenizer=getattr(processor,'tokenizer',processor);tokenizer.padding_side='left';FastModel.for_inference(model);model.eval()
    progress('generation',count)
    with path.open('a',encoding='utf-8') as stream:
        for start in range(count//4*4,p['n'],4):
            size=min(4,p['n']-start)
            prefixes=p.get('assistant_prefixes',['']*p['n'])[start:start+size]
            prompts=[tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)+prefix for prompt,prefix in zip(p['prompts'][start:start+size],prefixes)]
            inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');maximum=p['context']-inputs['input_ids'].shape[1]-2
            if maximum<200:raise RuntimeError('Insufficient trained generation context; no truncation')
            seed=p['baseline_batch_seed_start']+start//4;set_seed(seed);began=time.time()
            with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=maximum,do_sample=True,temperature=p['temperature'],top_p=p['top_p'],top_k=p['top_k'],pad_token_id=tokenizer.pad_token_id,use_cache=True)
            elapsed=time.time()-began;eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
            for offset in range(size):
                if start+offset<count:continue
                tokens=generated[offset,inputs['input_ids'].shape[1]:].tolist();end=next((i for i,t in enumerate(tokens) if t in eos),None)
                if end is not None:tokens=tokens[:end+1]
                decoded=tokenizer.decode(tokens,skip_special_tokens=True)
                row={'sample_id':start+offset,'batch_seed':seed,'text':prefixes[offset]+decoded,'new_tokens':len(tokens),
                    'hit_token_limit':end is None and len(tokens)>=maximum,'batch_size':size,'batch_seconds':elapsed,
                    'input_tokens_padded':inputs['input_ids'].shape[1],'max_new_tokens':maximum}
                if prefixes[offset]:row.update(assistant_prefix=prefixes[offset],generated_text=decoded)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            os.fsync(stream.fileno());progress('generation',start+size);print(key,f'{start+size}/{p["n"]}',f'batch {elapsed:.1f}s',flush=True);del inputs,generated
    progress('scoring',p['n']);score(key)
    write_json(directory/'completion.json',{'status':'complete','protocol_sha256':sha(directory/'protocol.json'),'raw_sha256':sha(path),
        'metrics_sha256':sha(directory/'metrics.json'),'elapsed_seconds':time.time()-started,'weights_updated':False,'new_test_scoring':False})
    progress('complete',p['n']);print(key,'alignment batch scored.',flush=True)


if __name__=='__main__':main()
