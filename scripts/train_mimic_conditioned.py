"""Independent QLoRA candidate; two prompt views, no generation during training."""
import json
import os
import time

from dataset_catalog import ROOT,read_json,read_rows,sha
from json_io import write_json
from mimic_conditioned_data import PREPARED,RUN,prepare

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    manifest=prepare();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'training_summary.json').exists():print('Candidate training already complete.');return
    if (RUN/'protocol.json').exists():raise RuntimeError('Incomplete candidate exists; inspect before an explicit resume')
    protocol={'dataset':'mimic_demo','base':str(ROOT/'.cache/models/gemma3-4b-training'),
        'initialization':'New independent adapter from the original base; old adapter untouched',
        'data_manifest_sha256':sha(PREPARED/'manifest.json'),'view_hashes':manifest['view_hashes'],
        'seed':20261031,'epochs':2,'learning_rate':1e-4,'rank':8,'alpha':16,'context':1024,
        'gradient_accumulation':8,'batch_size':1,'checkpoint_every':29,
        'selection_metric':'eval_generic_loss on21 validation admissions',
        'budget_note':'460 prompt views x2 epochs; more target exposures than old230 x3, not a compute-matched ablation',
        'adoption_policy':'No automatic replacement. Require post-training generation fidelity/copying comparison; validation loss alone does not establish success.',
        'test_policy':'No test data read or scored; development experiment after earlier tests were used'}
    write_json(RUN/'protocol.json',protocol);started=time.time()
    def progress(**values):write_json(RUN/'progress.json',{'elapsed_seconds':round(time.time()-started,2),**values},best_effort=True)
    progress(phase='loading_training',completed=0,total=116)
    from unsloth import FastModel
    import torch
    from datasets import Dataset
    from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed
    set_seed(protocol['seed'])
    model,processor=FastModel.from_pretrained(model_name=protocol['base'],max_seq_length=1024,load_in_4bit=True,full_finetuning=False)
    tokenizer=getattr(processor,'tokenizer',processor)
    def encode(name):
        encoded=[];lengths=[];prefix_lengths=[]
        for row in read_rows(PREPARED/(name+'.jsonl')):
            prefix_text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True)
            prefix=tokenizer(prefix_text,add_special_tokens=False)['input_ids']
            answer=json.dumps(row['completion'],ensure_ascii=False)
            text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']},{'role':'assistant','content':answer}],tokenize=False,add_generation_prompt=False)
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            if len(ids)>1024 or ids[:len(prefix)]!=prefix:raise RuntimeError('Unsupported context or prompt boundary: '+row['source_id'])
            encoded.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prefix)+ids[len(prefix):]})
            lengths.append(len(ids));prefix_lengths.append(len(prefix))
        return Dataset.from_list(encoded),{'n':len(encoded),'max_tokens':max(lengths),'min_prompt_tokens':min(prefix_lengths),'max_prompt_tokens':max(prefix_lengths),'overlength':0}
    train,a=encode('train');generic,b=encode('validation_generic');conditioned,c=encode('validation_conditioned')
    write_json(RUN/'tokenization.json',{'train':a,'validation_generic':b,'validation_conditioned':c})
    model=FastModel.get_peft_model(model,finetune_vision_layers=False,finetune_language_layers=True,
        finetune_attention_modules=True,finetune_mlp_modules=True,r=8,lora_alpha=16,lora_dropout=0,bias='none',random_state=20261031,use_gradient_checkpointing='unsloth')
    FastModel.for_training(model);model.config.use_cache=False
    def collate(batch):
        size=((max(len(r['input_ids']) for r in batch)+7)//8)*8
        return {key:torch.tensor([r[key]+[pad]*(size-len(r[key])) for r in batch]) for key,pad in [('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100)]}
    class Progress(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            progress(phase='training',completed=state.global_step,total=state.max_steps,epoch=state.epoch,best_validation_loss=state.best_metric)
            if state.global_step>=state.max_steps:control.should_evaluate=True;control.should_save=True
            return control
    args=TrainingArguments(output_dir=str(RUN/'checkpoints'),num_train_epochs=2,per_device_train_batch_size=1,
        gradient_accumulation_steps=8,per_device_eval_batch_size=1,learning_rate=1e-4,warmup_ratio=.05,weight_decay=.01,
        lr_scheduler_type='linear',bf16=True,fp16=False,optim='adamw_8bit',logging_steps=5,
        eval_strategy='steps',eval_steps=29,save_strategy='steps',save_steps=29,save_total_limit=3,
        load_best_model_at_end=True,metric_for_best_model='eval_generic_loss',greater_is_better=False,
        report_to='none',seed=20261031,data_seed=20261031,gradient_checkpointing=True,
        remove_unused_columns=False,dataloader_num_workers=0)
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset={'generic':generic,'conditioned':conditioned},data_collator=collate,callbacks=[Progress()])
    before=trainer.evaluate();write_json(RUN/'validation_before.json',before);torch.cuda.reset_peak_memory_stats()
    trained=trainer.train();after=trainer.evaluate()
    model.save_pretrained(RUN/'adapter');processor.save_pretrained(RUN/'adapter')
    selection={'dataset':'mimic_demo','adapter':str(RUN/'adapter'),'adapter_sha256':sha(RUN/'adapter/adapter_model.safetensors'),
        'best_checkpoint':trainer.state.best_model_checkpoint,'selection_metric':protocol['selection_metric'],
        'base_validation_loss':before['eval_generic_loss'],'selected_validation_loss':after['eval_generic_loss'],'test_used_for_selection':False}
    write_json(RUN/'selection.json',selection)
    write_json(RUN/'training_summary.json',{'protocol':protocol,'validation_before':before,'validation_after':after,
        'train_metrics':trained.metrics,'global_steps':trainer.state.global_step,'elapsed_seconds':time.time()-started,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'selection':selection})
    progress(phase='trained_waiting_for_comparison',completed=trainer.state.global_step,total=trainer.state.max_steps)
    print('Candidate adapter saved. Post-training fidelity comparison follows.',flush=True)


if __name__=='__main__':main()
