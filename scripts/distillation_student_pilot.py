"""Matched-budget continuation control and distilled Reliability candidate.

This is a development pilot. It never replaces the existing adapter or test results.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from dataset_catalog import ROOT,core,data,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'outputs/reliability-distillation-v1'
SEED=2026110121


def prepare():
    p=read_json(OUT/'protocol.json');review=read_rows(OUT/'teacher_review.jsonl')
    if not p.get('preserve_source_context'):raise RuntimeError('The context-drifting first teacher pilot is not eligible for training')
    good={r['sample_id'] for r in review if r['verdict']=='Pass'}
    teachers=[r for r in read_rows(OUT/'teacher_valid_records.jsonl') if r['sample_id'] in good]
    review_by_id={r['sample_id']:r for r in review}
    for row in teachers:
        digest=hashlib.sha256(json.dumps(row['record'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert review_by_id[row['sample_id']]['record_sha256']==digest,'Reviewed teacher record changed'
    if len(teachers)<8:raise RuntimeError('Fewer than eight reviewed teacher examples; no training')
    for family in ['history_consistency','policy_context','request_completeness','ordinary_control']:
        if not any(r['family']==family for r in teachers):raise RuntimeError('Missing reviewed family: '+family)
    source=data('reliability');rows=source['train'];lookup={r['source_id']:r for r in rows}
    requests={r['sample_id']:r for r in read_rows(OUT/'requests.jsonl')}
    rng=random.Random(SEED);replay=rng.sample(rows,96)
    common=[{'prompt':source['prompt'],'completion':core('reliability',r),'source':'original_train','source_id':r['source_id']} for r in replay]
    views={arm:list(common) for arm in ['replay','distilled']}
    for _ in range(4):
        for teacher in teachers:
            anchor=lookup[requests[teacher['sample_id']]['training_source_id']]
            views['replay'].append({'prompt':source['prompt'],'completion':core('reliability',anchor),'source':'matched_anchor','source_id':anchor['source_id']})
            views['distilled'].append({'prompt':source['prompt'],'completion':teacher['record'],'source':'teacher','teacher_sample_id':teacher['sample_id']})
    # The label mix, exposure count and row positions match across training arms.
    assert len(views['replay'])==len(views['distilled'])
    for a,b in zip(views['replay'],views['distilled']):
        assert all(a['completion'][f]==b['completion'][f] for f in ['category','subcategory','difficulty'])
        assert a['completion']['input']['context']==b['completion']['input']['context'],'Source-context mix changed between training arms'
    validation=read_rows(ROOT/'data/reliability/prepared/validation.jsonl')
    views['validation']=[{'prompt':source['prompt'],'completion':core('reliability',r),'source':'original_validation','source_id':r['source_id']} for r in validation]
    for name,items in views.items():
        path=OUT/(name+'_train.jsonl' if name!='validation' else 'validation.jsonl')
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items)
        if path.exists():assert path.read_text(encoding='utf-8')==content
        else:path.write_text(content,encoding='utf-8')
    parent=read_json(ROOT/'outputs/prefix-alignment-v1/reliability/protocol.json')
    trial=read_json(ROOT/'outputs/reliability-consistency-v1/protocol.json')
    protocol={'teacher_protocol_sha256':sha(OUT/'protocol.json'),'review_sha256':sha(OUT/'teacher_review.jsonl'),
        'teacher_records_sha256':sha(OUT/'teacher_valid_records.jsonl'),
        'baseline_adapter':p['baseline_adapter'],'baseline_adapter_sha256':p['baseline_adapter_sha256'],
        'teacher_examples':len(teachers),'teacher_views_in_dataset':4*len(teachers),'original_replay':96,'rows_per_arm':len(views['replay']),
        'seed':SEED,'max_steps':18,'learning_rate':3e-5,'micro_batch':1,'gradient_accumulation':8,'context':1024,
        'training_policy':'Both arms continue the same original adapter for exactly18 updates. The replay control replaces each teacher row with its source anchor at the same row position, retaining the same leading label mix. Validation loss is monitored, not used to select a checkpoint. Save at9/18; evaluate fixed final checkpoint only.',
        'view_hashes':{name:sha(OUT/(name+'_train.jsonl' if name!='validation' else 'validation.jsonl')) for name in views},
        'evaluation':{'generic_n':16,'workflow_n':12,'generic_parent_protocol_sha256':sha(ROOT/'outputs/prefix-alignment-v1/reliability/protocol.json'),
            'generic_prompts':parent['prompts'][:16],'generic_prefixes':parent['assistant_prefixes'][:16],
            'generic_seed_start':parent['baseline_batch_seed_start'],'workflow_prompts':[source['prompt']]*12,
            'workflow_prefixes':trial['assistant_prefixes'],'workflow_seed_start':trial['seed_start'],
            'baseline_generic_path':'outputs/prefix-alignment-v1/reliability/controlled_generations.jsonl',
            'baseline_workflow_path':'outputs/reliability-consistency-v1/baseline/generations.jsonl',
            'baseline_generic_sha256':sha(ROOT/'outputs/prefix-alignment-v1/reliability/controlled_generations.jsonl'),
            'baseline_workflow_sha256':sha(ROOT/'outputs/reliability-consistency-v1/baseline/generations.jsonl')},
        'scope':'Small exploratory development comparison after earlier test use. Generic cohorts are label-prefixed and baseline outputs are reused. Workflow cases are12 already-inspected diagnostic cases, not fresh independent test data. No teacher is used to judge the student.',
        'review_gate':'Require at least2 of4 prior Issue cases become assistant-reviewed Pass; all8 prior passing controls remain Pass; candidate improves over the matched replay control. Require16/16 generic and12/12 workflow schema-valid, no decrease in source-constraint counts or increase in whole-record copies. All generic numeric Wasserstein distances must be no more than1.25x the reused baseline. No automatic deployment even if these exploratory checks pass.',
        'limits':'One seed block, small selected cohorts, one uncalibrated assistant review, and a teacher-data acceptance filter. No population accuracy, causal proof of general distillation benefit, or distribution-equivalence claim. No original test records read or rescored.'}
    if (OUT/'student_protocol.json').exists():assert read_json(OUT/'student_protocol.json')==protocol
    else:write_json(OUT/'student_protocol.json',protocol)
    print('Student pilot prepared:',protocol['rows_per_arm'],'views per arm;18 updates each.',flush=True)
    return protocol


def train(arm):
    p=read_json(OUT/'student_protocol.json');run=OUT/arm
    if (run/'training_summary.json').exists():print(arm,'already trained');return
    if run.exists() and any(run.iterdir()):raise RuntimeError('Partial run exists; inspect before explicit recovery')
    run.mkdir(parents=True);started=time.time()
    from unsloth import FastModel
    import torch
    from datasets import Dataset
    from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed
    set_seed(p['seed']);assert sha(Path(p['baseline_adapter'])/'adapter_model.safetensors')==p['baseline_adapter_sha256']
    model,processor=FastModel.from_pretrained(model_name=p['baseline_adapter'],max_seq_length=1024,load_in_4bit=True,full_finetuning=False)
    tokenizer=getattr(processor,'tokenizer',processor)
    def encode(path):
        encoded=[]
        for row in read_rows(path):
            prefix=tokenizer(tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True),add_special_tokens=False)['input_ids']
            text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']},{'role':'assistant','content':json.dumps(row['completion'],ensure_ascii=False)}],tokenize=False,add_generation_prompt=False)
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            if len(ids)>1024 or ids[:len(prefix)]!=prefix:raise RuntimeError('Context/masking boundary failed; no truncation')
            encoded.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prefix)+ids[len(prefix):]})
        return Dataset.from_list(encoded)
    path=OUT/(arm+'_train.jsonl');assert sha(path)==p['view_hashes'][arm]
    assert sha(OUT/'validation.jsonl')==p['view_hashes']['validation']
    training=encode(path);validation=encode(OUT/'validation.jsonl')
    FastModel.for_training(model);model.config.use_cache=False
    trainable=sum(v.numel() for v in model.parameters() if v.requires_grad)
    if not trainable:raise RuntimeError('No trainable adapter parameters')
    def collate(batch):
        size=((max(len(r['input_ids']) for r in batch)+7)//8)*8
        return {k:torch.tensor([r[k]+[pad]*(size-len(r[k])) for r in batch]) for k,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}
    class Progress(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            write_json(OUT/'student_progress.json',{'phase':'training','arm':arm,'completed':state.global_step,'total':state.max_steps,'elapsed_seconds':round(time.time()-started,1)},best_effort=True)
    args=TrainingArguments(output_dir=str(run/'checkpoints'),max_steps=p['max_steps'],per_device_train_batch_size=1,gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,learning_rate=p['learning_rate'],warmup_ratio=.05,weight_decay=.01,lr_scheduler_type='linear',bf16=True,fp16=False,
        optim='adamw_8bit',logging_steps=3,eval_strategy='no',save_strategy='steps',save_steps=9,save_total_limit=2,load_best_model_at_end=False,
        report_to='none',seed=p['seed'],data_seed=p['seed'],gradient_checkpointing=True,remove_unused_columns=False,dataloader_num_workers=0)
    trainer=Trainer(model=model,args=args,train_dataset=training,eval_dataset=validation,data_collator=collate,callbacks=[Progress()])
    before=trainer.evaluate();result=trainer.train();after=trainer.evaluate()
    model.save_pretrained(run/'adapter');processor.save_pretrained(run/'adapter')
    write_json(run/'training_summary.json',{'arm':arm,'steps':trainer.state.global_step,'training_rows':len(training),'trainable_parameters':trainable,
        'validation_before':before,'validation_after':after,'train_metrics':result.metrics,'adapter_sha256':sha(run/'adapter/adapter_model.safetensors'),
        'initial_adapter_sha256':p['baseline_adapter_sha256'],'elapsed_seconds':time.time()-started,'test_read':False,'selected_by':'Fixed final18 updates, not validation-loss selection'})
    print(arm,'adapter saved.',flush=True)


def evaluate(arm):
    p=read_json(OUT/'student_protocol.json');e=p['evaluation'];run=OUT/arm
    from dataset_metrics import assess_label
    if arm=='baseline':
        for kind in ['generic','workflow']:
            assert sha(ROOT/e['baseline_'+kind+'_path'])==e['baseline_'+kind+'_sha256']
            rows=read_rows(ROOT/e['baseline_'+kind+'_path'])[:e[kind+'_n']]
            dest=run/kind;dest.mkdir(parents=True,exist_ok=True)
            (dest/'generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            m,_=assess_label('reliability',rows,dest/'records.jsonl');write_json(dest/'metrics.json',m)
        return
    from unsloth import FastModel
    import torch
    from transformers import set_seed
    model,processor=FastModel.from_pretrained(model_name=str(run/'adapter'),max_seq_length=1024,load_in_4bit=True,full_finetuning=False)
    t=getattr(processor,'tokenizer',processor);t.padding_side='left';FastModel.for_inference(model);model.eval()
    for kind in ['generic','workflow']:
        dest=run/kind;dest.mkdir(exist_ok=True);path=dest/'generations.jsonl'
        if path.exists() and len(read_rows(path))!=e[kind+'_n']:raise RuntimeError('Partial evaluation needs explicit recovery')
        if not path.exists():
            with path.open('w',encoding='utf-8') as stream:
                for start in range(0,e[kind+'_n'],4):
                    prefixes=e[kind+'_prefixes'][start:start+4]
                    prompts=[t.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)+prefix for prompt,prefix in zip(e[kind+'_prompts'][start:start+4],prefixes)]
                    inputs=t(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');maximum=1024-inputs['input_ids'].shape[1]-2
                    if maximum<200:raise RuntimeError('Insufficient generation context; no truncation')
                    seed=e[kind+'_seed_start']+start//4;set_seed(seed);began=time.time()
                    with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=maximum,do_sample=True,temperature=1.,top_p=.95,top_k=64,pad_token_id=t.pad_token_id,use_cache=True)
                    seconds=time.time()-began;eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                    for offset,prefix in enumerate(prefixes):
                        tokens=generated[offset,inputs['input_ids'].shape[1]:].tolist();end=next((i for i,v in enumerate(tokens) if v in eos),None)
                        if end is not None:tokens=tokens[:end+1]
                        text=t.decode(tokens,skip_special_tokens=True)
                        row={'sample_id':start+offset,'text':prefix+text,'generated_text':text,'assistant_prefix':prefix,'batch_seed':seed,
                             'new_tokens':len(tokens),'hit_token_limit':end is None and len(tokens)>=maximum,'batch_seconds':seconds}
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    os.fsync(stream.fileno());write_json(OUT/'student_progress.json',{'phase':'evaluation','arm':arm,'kind':kind,'completed':start+4,'total':e[kind+'_n']},best_effort=True)
                    print(arm,kind,start+4,'/',e[kind+'_n'],flush=True);del inputs,generated
        metrics,_=assess_label('reliability',read_rows(path),dest/'records.jsonl');write_json(dest/'metrics.json',metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate']);parser.add_argument('--arm',choices=['baseline','replay','distilled'])
    parser.add_argument('--experiment-root',type=Path,default=OUT);args=parser.parse_args();OUT=args.experiment_root.resolve()
    if not OUT.is_relative_to((ROOT/'outputs').resolve()):raise RuntimeError('Experiment must stay within project outputs')
    if args.stage!='prepare' and not args.arm:parser.error('--arm is required for training/evaluation')
    if args.stage=='train' and args.arm=='baseline':parser.error('The baseline is frozen and cannot be trained')
    if args.stage=='prepare':prepare()
    elif args.stage=='train':train(args.arm)
    else:evaluate(args.arm)
