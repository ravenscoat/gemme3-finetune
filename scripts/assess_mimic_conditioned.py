"""Matched development generation for the independent mixed-prompt candidate."""
import json
import os
import shutil
import time

from dataset_catalog import ROOT,data,read_json,read_rows,sha
from dataset_metrics import assess_label
from generate_comparison import completed_rows
from json_io import write_json
from mimic_conditioned_data import RUN,conditioned_prompt
from mimic_fidelity_pilot import LABS,missing_pattern
from advanced_exploratory_audits import icd_prefix_issue

N=64


def prepare_plan():
    parent=ROOT/'outputs/mimic-demo-fidelity-conditions-only-v1';p=read_json(parent/'protocol.json')
    paths={'old_generic':parent/'baseline_generations.jsonl','old_conditioned':parent/'condition_only_generations.jsonl'}
    plan={'n':N,'batch_size':4,'seed_start':p['batch_seed_start'],'temperature':p['temperature'],'top_p':p['top_p'],'top_k':p['top_k'],
        'context':p['context'],'generic_prompt':p['arms']['baseline'],'conditions':p['conditions'],
        'baselines':{k:{'path':str(v.relative_to(ROOT)),'sha256':sha(v)} for k,v in paths.items()},
        'parent_protocol_sha256':sha(parent/'protocol.json'),
        'policy':'Matched64-attempt development comparisons, not new final-test evidence. Same prompts, source conditions and sampling settings for each method pair. No repairs or replacement.',
        'adoption':'No automatic deployment; assess field-level gains, regressions, consistency and copying, not training loss alone.'}
    RUN.mkdir(parents=True,exist_ok=True);path=RUN/'comparison_plan.json'
    if path.exists() and read_json(path)!=plan:raise RuntimeError('Comparison plan changed')
    if not path.exists():write_json(path,plan)
    for label,item in plan['baselines'].items():
        target=RUN/(label+'_generations.jsonl')
        if target.exists() and sha(target)!=item['sha256']:raise RuntimeError('Copied baseline changed')
        if not target.exists():shutil.copyfile(ROOT/item['path'],target)
    return plan


def score(plan):
    result={'comparison_plan_sha256':sha(RUN/'comparison_plan.json'),'arms':{},'scope':plan['policy']}
    train=data('mimic_demo')['train'];known={(r['primary_diagnosis_code'],r['primary_diagnosis_version']) for r in train}
    for label in ['old_generic','old_conditioned','candidate_generic','candidate_conditioned']:
        path=RUN/(label+'_generations.jsonl')
        if not path.exists() or len(completed_rows(path,N))!=N:continue
        measure,rows=assess_label('mimic_demo',read_rows(path),RUN/(label+'_records.jsonl'))
        measure.update(prefix_version_flags=sum(icd_prefix_issue(r['record']) is not None for r in rows),
            known_training_diagnosis_pair_count=sum((r['record']['primary_diagnosis_code'],r['record']['primary_diagnosis_version']) in known for r in rows),raw_sha256=sha(path))
        if 'conditioned' in label:
            matched=0
            for row in rows:
                c=plan['conditions'][row['sample_id']];r=row['record']
                matched+=r['primary_diagnosis_code']==c['primary_diagnosis_code'] and r['primary_diagnosis_version']==c['primary_diagnosis_version'] and missing_pattern(r)==c['lab_missing_pattern']
            measure['condition_compliance']={'all':matched,'attempts':N}
        result['arms'][label]=measure
    write_json(RUN/'pilot_metrics.json',result)
    lines=['# Mixed-prompt MIMIC candidate: development comparison','',
        '| Method | Valid /64 | Source constraints /64 | ICD prefix flags | Full copies |','|---|---:|---:|---:|---:|']
    for label,m in result['arms'].items():
        copy='n/a' if m['exact_training_record_match_rate'] is None else f"{m['exact_training_record_match_rate']:.3f}"
        lines.append(f"| {label} | {m['structure_valid']} | {m['source_constraint_valid']} | {m['prefix_version_flags']} | {copy} |")
    lines+=['','| Method / field | Wasserstein distance | Median | Source median | Missing | Source missing |','|---|---:|---:|---:|---:|---:|']
    for label,m in result['arms'].items():
        for field,v in m['fidelity_vs_training']['numeric'].items():
            if 'generated' in v:lines.append(f"| {label} / {field} | {v['wasserstein']:.3f} | {v['generated']['median']:.3f} | {v['reference']['median']:.3f} | {100*v['generated_missing_rate']:.1f}% | {100*v['reference_missing_rate']:.1f}% |")
    lines+=['',plan['policy'],'','Conditional diagnosis pairs and null patterns are externally selected; conditional scores do not establish learned frequencies. No clinical certification. The original test sets were not rescored.','']
    (RUN/'comparison_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    plan=prepare_plan();selected=read_json(RUN/'selection.json')
    if sha(RUN/'adapter/adapter_model.safetensors')!=selected['adapter_sha256']:raise RuntimeError('Candidate adapter changed')
    protocol={'comparison_plan_sha256':sha(RUN/'comparison_plan.json'),'candidate_selection_sha256':sha(RUN/'selection.json'),'candidate_adapter_sha256':selected['adapter_sha256']}
    path=RUN/'generation_protocol.json'
    if path.exists() and read_json(path)!=protocol:raise RuntimeError('Generation protocol changed')
    write_json(path,protocol);score(plan);started=time.time()
    from unsloth import FastModel
    import torch
    from transformers import set_seed
    write_json(RUN/'progress.json',{'phase':'loading_generation','completed':0,'total':2*N},best_effort=True)
    model,processor=FastModel.from_pretrained(model_name=selected['adapter'],max_seq_length=1024,load_in_4bit=True,full_finetuning=False)
    tokenizer=getattr(processor,'tokenizer',processor);tokenizer.padding_side='left';FastModel.for_inference(model);model.eval()
    for arm_index,label in enumerate(['candidate_generic','candidate_conditioned']):
        path=RUN/(label+'_generations.jsonl');count=len(completed_rows(path,N))
        with path.open('a',encoding='utf-8') as stream:
            for start in range(count//4*4,N,4):
                prompts=[]
                for ident in range(start,start+4):
                    prompt=plan['generic_prompt'] if label=='candidate_generic' else conditioned_prompt(plan['generic_prompt'],plan['conditions'][ident])
                    prompts.append(tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True))
                inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                maximum=1024-inputs['input_ids'].shape[1]-2
                if maximum<250:raise RuntimeError('Insufficient trained generation context')
                seed=plan['seed_start']+start//4;set_seed(seed);began=time.time()
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=maximum,do_sample=True,temperature=plan['temperature'],top_p=plan['top_p'],top_k=plan['top_k'],pad_token_id=tokenizer.pad_token_id,use_cache=True)
                elapsed=time.time()-began;eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                for offset in range(4):
                    if start+offset<count:continue
                    tokens=generated[offset,inputs['input_ids'].shape[1]:].tolist();end=next((i for i,t in enumerate(tokens) if t in eos),None)
                    if end is not None:tokens=tokens[:end+1]
                    row={'sample_id':start+offset,'batch_seed':seed,'text':tokenizer.decode(tokens,skip_special_tokens=True),'new_tokens':len(tokens),
                        'hit_token_limit':end is None and len(tokens)>=maximum,'batch_seconds':elapsed,'batch_size':4,'input_tokens_padded':inputs['input_ids'].shape[1],'max_new_tokens':maximum}
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                os.fsync(stream.fileno());write_json(RUN/'progress.json',{'phase':'generation','arm':label,'completed':arm_index*N+start+4,'total':2*N,'elapsed_seconds':round(time.time()-started,2)},best_effort=True)
                print(label,f'{start+4}/{N}',f'batch {elapsed:.1f}s',flush=True);del inputs,generated
        score(plan)
    score(plan);print('Candidate fidelity comparison complete.',flush=True)


if __name__=='__main__':main()
