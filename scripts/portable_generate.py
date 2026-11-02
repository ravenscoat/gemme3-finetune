"""Load a bundled adapter against an explicitly supplied local base model."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--dataset',required=True,choices=['bitext','insuranceqa','reliability','mimic_demo'])
    parser.add_argument('--base-model',type=Path,required=True)
    parser.add_argument('--batch-start',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('generated.jsonl'))
    parser.add_argument('--max-new-tokens',type=int,help='Optional smoke-test cap; changes the study budget.')
    args=parser.parse_args();bundle=args.bundle.resolve();base=args.base_model.resolve()
    if not (base/'config.json').exists():raise ValueError('Supply the downloaded, pinned local base-model folder')
    p=json.loads((bundle/'generation_profiles.json').read_text(encoding='utf-8'))[args.dataset]
    start=args.batch_start
    if start<0 or start%4 or start+4>len(p['prompts']):raise ValueError('Choose a valid multiple-of-four batch start')
    cache=bundle/'.runtime-cache';cache.mkdir(exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['UNSLOTH_DISABLE_STATISTICS']='1'
    with tempfile.TemporaryDirectory(dir=cache,prefix='adapter-') as folder:
        target=Path(folder).resolve()
        if not target.is_relative_to(cache.resolve()):raise RuntimeError('Temporary adapter escaped bundle cache')
        for source in (bundle/'adapters'/args.dataset).iterdir():
            if source.is_file():shutil.copy2(source,target/source.name)
        config=json.loads((target/'adapter_config.json').read_text(encoding='utf-8'))
        config['base_model_name_or_path']=str(base)
        (target/'adapter_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        from unsloth import FastModel
        import torch
        from transformers import set_seed
        model,processor=FastModel.from_pretrained(model_name=str(target),max_seq_length=p['context'],load_in_4bit=True,full_finetuning=False)
        tokenizer=getattr(processor,'tokenizer',processor);tokenizer.padding_side='left';FastModel.for_inference(model);model.eval()
        prefixes=p['assistant_prefixes'][start:start+4]
        prompts=[tokenizer.apply_chat_template([{'role':'user','content':s}],tokenize=False,add_generation_prompt=True)+prefix for s,prefix in zip(p['prompts'][start:start+4],prefixes)]
        inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        maximum=p['context']-inputs['input_ids'].shape[1]-2
        if args.max_new_tokens is not None:maximum=min(maximum,args.max_new_tokens)
        if maximum<=0:raise ValueError('No remaining context')
        set_seed(p['seed_start']+start//4)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=maximum,do_sample=True,temperature=p['temperature'],top_p=p['top_p'],top_k=p['top_k'],pad_token_id=tokenizer.pad_token_id,use_cache=True)
        rows=[]
        for i,prefix in enumerate(prefixes):
            generated=tokenizer.decode(out[i,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            rows.append({'sample_id':start+i,'assistant_prefix':prefix,'generated_text':generated,'text':prefix+generated,'smoke_cap':args.max_new_tokens})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        del model,processor,inputs,out;gc.collect();torch.cuda.empty_cache()
    print('Wrote',args.output,'; exact text can vary across GPU/library versions.')


if __name__=='__main__':main()
