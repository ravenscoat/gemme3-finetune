"""Development-only prompt/conditional pilots; preserve baseline and every attempt."""
from collections import Counter,defaultdict
import argparse
import json
import os
from pathlib import Path
import random
import time

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha
from dataset_metrics import assess_label
from generate_comparison import completed_rows
from json_io import write_json
from advanced_exploratory_audits import icd_prefix_issue

OUT=ROOT/'outputs/mimic-demo-fidelity-pilot-v1'
N=64
LABS=['creatinine_mg_dl','glucose_mg_dl','hemoglobin_g_dl']
RULE=('Consistency rule: primary_diagnosis_version and primary_diagnosis_code must agree. '
      'ICD-9-CM diagnosis codes begin with a digit, E or V. ICD-10-CM diagnosis codes begin '
      'with a letter, including U. This is a diagnosis-code rule, not a procedure-code rule.')


def missing_pattern(row):return ''.join('1' if row[f] is None else '0' for f in LABS)


def allocate(counts,n,rng):
    """Largest-remainder allocation; randomize ties before sorting."""
    total=sum(counts.values());keys=sorted(counts);rng.shuffle(keys)
    result={key:n*counts[key]//total for key in keys}
    ranked=sorted(keys,key=lambda key:(n*counts[key])%total,reverse=True)
    for key in ranked[:n-sum(result.values())]:result[key]+=1
    return result


def prepare(conditions_only=False):
    source=data('mimic_demo');selected=read_json(ROOT/source['spec']['selection'])
    if sha(Path(selected['adapter'])/'adapter_model.safetensors')!=selected['adapter_sha256']:raise RuntimeError('Adapter changed')
    for name,digest in selected['input_hashes'].items():
        if sha(ROOT/source['spec']['prepared']/name)!=digest:raise RuntimeError('Source changed: '+name)
    original=ROOT/source['spec']['assessment'];old=read_json(original/'protocol.json')
    baseline_path=original/'tuned_generations.jsonl'
    if sha(baseline_path)!=read_json(original/'test_freeze.json')['tuned_generation_sha256']:raise RuntimeError('Original raw outputs changed')
    baseline=completed_rows(baseline_path,128)[:N]
    if len(baseline)!=N or any(r['batch_seed']!=old['batch_seed_start']+r['sample_id']//4 for r in baseline):raise RuntimeError('Baseline IDs/seeds differ')
    rng=random.Random(2026110117);groups=defaultdict(list)
    for row in source['train']:groups[(row['primary_diagnosis_version'],missing_pattern(row))].append(row)
    allocation=allocate({key:len(rows) for key,rows in groups.items()},N,rng)
    conditions=[]
    for group,quota in allocation.items():
        for _ in range(quota):
            row=rng.choice(groups[group])
            conditions.append({'primary_diagnosis_code':row['primary_diagnosis_code'],
                'primary_diagnosis_version':row['primary_diagnosis_version'],'lab_missing_pattern':group[1]})
    rng.shuffle(conditions)
    protocol={'dataset':'mimic_demo','purpose':'Exploratory development; no training or new test scoring',
        'n_per_arm':N,'adapter':selected['adapter'],'adapter_sha256':selected['adapter_sha256'],
        'source_hashes':{str((ROOT/source['spec']['prepared']/name).relative_to(ROOT)):digest for name,digest in selected['input_hashes'].items()},
        'baseline_source':str(baseline_path.relative_to(ROOT)),'baseline_source_sha256':sha(baseline_path),
        'baseline_source_protocol_sha256':sha(original/'protocol.json'),'baseline_sample_ids':list(range(N)),
        'batch_size':4,'batch_seed_start':old['batch_seed_start'],'temperature':old['temperature'],
        'top_p':old['top_p'],'top_k':old['top_k'],'context':old['context'],'padding':'left',
        'max_new_tokens':'Remaining trained context minus2; varies with prompt length; limit hits reported',
        'arms':{'baseline':source['prompt'],'prompt_rule':source['prompt']+'\n'+RULE,
                'conditional_rule':source['prompt']+'\n'+RULE},
        'condition_seed':2026110117,'lab_order':LABS,'conditions':conditions,
        'source_joint_counts':{str(k):len(v) for k,v in groups.items()},'planned_joint_counts':{str(k):v for k,v in allocation.items()},
        'conditioning_policy':'Quotas for joint diagnosis-version/lab-null patterns; code sampled from matching training rows. No patient identifiers or complete source row supplied.',
        'comparison_policy':'All attempts retained. Controlled arm must be reported separately; externally selected fields are not evidence of learned frequencies.',
        'decision_policy':'Assess structure, prefix consistency, source support, missingness, marginals, relationships and copies together. Pilot is not a new final test; confirm useful results on additional seed blocks.'}
    if conditions_only:
        parent=ROOT/'outputs/mimic-demo-fidelity-pilot-v1'
        if not (parent/'completion.json').exists():raise RuntimeError('Complete the parent pilot first')
        protocol['arms']={'baseline':source['prompt'],'condition_only':source['prompt']}
        protocol['followup']={'hypothesis':'Remove the extra ICD rule while retaining exactly the same conditions and seeds; assess numeric drift',
            'parent_protocol_sha256':sha(parent/'protocol.json'),'parent_metrics_sha256':sha(parent/'pilot_metrics.json')}
        if protocol['conditions']!=read_json(parent/'protocol.json')['conditions']:raise RuntimeError('Follow-up conditions differ')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read_json(OUT/'protocol.json')!=protocol:raise RuntimeError('Pilot protocol changed')
    else:
        write_json(OUT/'protocol.json',protocol)
        (OUT/'baseline_generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in baseline),encoding='utf-8')
        write_json(OUT/'baseline_freeze.json',{'sha256':sha(OUT/'baseline_generations.jsonl'),'protocol_sha256':sha(OUT/'protocol.json')})
        write_json(OUT/'progress.json',{'phase':'prepared','completed':0,'total':(len(protocol['arms'])-1)*N})
    if sha(OUT/'baseline_generations.jsonl')!=read_json(OUT/'baseline_freeze.json')['sha256']:raise RuntimeError('Pilot baseline changed')
    return protocol


def score(protocol):
    result={'protocol_sha256':sha(OUT/'protocol.json'),'arms':{},'limits':[
        '64 attempts per arm; exploratory, reused baseline seeds, not independent final-test evidence.',
        'Conditional arm is a source-derived pipeline; diagnosis/code and missingness conditions are externally chosen.',
        'Every raw output is preserved, with no repairs, retries or selective replacements.',
        'Prefix consistency and source pair membership do not establish clinical validity.',
        'Prompt lengths/generation budgets differ; token-limit hits are reported. No weights or test scores changed.']}
    train=data('mimic_demo')['train'];known={(r['primary_diagnosis_code'],r['primary_diagnosis_version']) for r in train}
    for label in protocol['arms']:
        path=OUT/(label+'_generations.jsonl')
        if not path.exists() or len(completed_rows(path,N))!=N:continue
        raw=read_rows(path);measure,rows=assess_label('mimic_demo',raw,OUT/(label+'_records.jsonl'))
        issues=[{'sample_id':r['sample_id'],'issue':icd_prefix_issue(r['record'])} for r in rows if icd_prefix_issue(r['record'])]
        measure.update(prefix_version_flags=len(issues),prefix_issues=issues,
            known_training_diagnosis_pair_count=sum((r['record']['primary_diagnosis_code'],r['record']['primary_diagnosis_version']) in known for r in rows),
            lab_missing_patterns=dict(Counter(missing_pattern(r['record']) for r in rows)),raw_sha256=sha(path))
        if label in ['conditional_rule','condition_only']:
            conditions=protocol['conditions'];matched=[];pair=[];mask=[]
            for row in rows:
                expected=conditions[row['sample_id']];record=row['record']
                a=all(record[f]==expected[f] for f in ['primary_diagnosis_code','primary_diagnosis_version'])
                b=missing_pattern(record)==expected['lab_missing_pattern']
                pair.append(a);mask.append(b);matched.append(a and b)
            measure['condition_compliance']={'all':sum(matched),'diagnosis_pair':sum(pair),'lab_pattern':sum(mask),'attempts':N}
        result['arms'][label]=measure
    write_json(OUT/'pilot_metrics.json',result)
    lines=['# MIMIC fidelity development pilot','',
           '| Arm | Valid /64 | Prefix/version flags among valid | Known source pairs | Full-record copying |','|---|---:|---:|---:|---:|']
    for label,m in result['arms'].items():
        copying='n/a' if m['exact_training_record_match_rate'] is None else f"{m['exact_training_record_match_rate']:.3f}"
        lines.append(f"| {label} | {m['structure_valid']} | {m['prefix_version_flags']} | {m['known_training_diagnosis_pair_count']} | {copying} |")
    lines+=['','## Numerical distributions and missingness','',
            '| Arm / field | Generated median | Source median | Generated missing | Source missing | Wasserstein |','|---|---:|---:|---:|---:|---:|']
    for label,m in result['arms'].items():
        for field,v in m['fidelity_vs_training']['numeric'].items():
            if 'generated' in v:lines.append(f"| {label} / {field} | {v['generated']['median']:.3f} | {v['reference']['median']:.3f} | {v['generated_missing_rate']:.3f} | {v['reference_missing_rate']:.3f} | {v['wasserstein']:.3f} |")
    for label in ['conditional_rule','condition_only']:
        if label in result['arms']:
            c=result['arms'][label]['condition_compliance'];lines+=['',f"{label} condition compliance: {c['all']}/64 overall; diagnosis pair {c['diagnosis_pair']}/64; lab pattern {c['lab_pattern']}/64."]
    lines+=['',*['- '+v for v in result['limits']],'']
    (OUT/'pilot_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--conditions-only',action='store_true');args=parser.parse_args()
    if args.conditions_only:OUT=ROOT/'outputs/mimic-demo-fidelity-conditions-only-v1'
    protocol=prepare(args.conditions_only)
    if args.prepare_only:score(protocol);print('Frozen pilot protocol and baseline; no GPU job started.');return
    if (OUT/'completion.json').exists():print('Pilot already complete.');return
    lock=ROOT/'reports/assessment_gpu.lock'
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():raise RuntimeError('Training owns GPU')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    arms=[label for label in protocol['arms'] if label!='baseline'];total=len(arms)*N
    started=time.time();state={'phase':'loading','completed':0,'total':total}
    def progress(**values):
        state.update(values,elapsed_seconds=round(time.time()-started,2));write_json(OUT/'progress.json',state,best_effort=True)
    try:
        progress()
        from unsloth import FastModel
        import torch
        from transformers import set_seed
        model,processor=FastModel.from_pretrained(model_name=protocol['adapter'],max_seq_length=protocol['context'],load_in_4bit=True,full_finetuning=False)
        tokenizer=getattr(processor,'tokenizer',processor);tokenizer.padding_side='left'
        FastModel.for_inference(model);model.eval()
        for arm_index,label in enumerate(arms):
            path=OUT/(label+'_generations.jsonl');count=len(completed_rows(path,N))
            with path.open('a',encoding='utf-8') as stream:
                for start in range(count//4*4,N,4):
                    prompts=[]
                    for ident in range(start,start+4):
                        prompt=protocol['arms'][label]
                        if label in ['conditional_rule','condition_only']:
                            condition=protocol['conditions'][ident]
                            prompt+='\nUse these required conditions for this fictional record: diagnosis code '+json.dumps(condition['primary_diagnosis_code'])+', diagnosis version '+str(condition['primary_diagnosis_version'])+'. '
                            prompt+=' '.join(f'{field} must be '+('null.' if bit=='1' else 'a non-null numeric value.') for field,bit in zip(LABS,condition['lab_missing_pattern']))
                        prompts.append(tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True))
                    inputs=tokenizer(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                    maximum=protocol['context']-inputs['input_ids'].shape[1]-2
                    if maximum<250:raise RuntimeError('Insufficient trained generation context')
                    seed=protocol['batch_seed_start']+start//4;set_seed(seed);began=time.time()
                    with torch.inference_mode():
                        generated=model.generate(**inputs,max_new_tokens=maximum,do_sample=True,temperature=protocol['temperature'],top_p=protocol['top_p'],top_k=protocol['top_k'],pad_token_id=tokenizer.pad_token_id,use_cache=True)
                    elapsed=time.time()-began;eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
                    for offset in range(4):
                        if start+offset<count:continue
                        tokens=generated[offset,inputs['input_ids'].shape[1]:].tolist();end=next((i for i,t in enumerate(tokens) if t in eos),None)
                        if end is not None:tokens=tokens[:end+1]
                        row={'sample_id':start+offset,'batch_seed':seed,'text':tokenizer.decode(tokens,skip_special_tokens=True),
                            'new_tokens':len(tokens),'hit_token_limit':end is None and len(tokens)>=maximum,'batch_seconds':elapsed,'batch_size':4,
                            'input_tokens_padded':inputs['input_ids'].shape[1],'max_new_tokens':maximum}
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    os.fsync(stream.fileno());progress(phase='generation',arm=label,arm_completed=start+4,completed=arm_index*N+start+4)
                    print(label,f'{start+4}/{N}',f'batch {elapsed:.1f}s',flush=True);del inputs,generated
            score(protocol)
        result=score(protocol);write_json(OUT/'completion.json',{'phase':'complete','elapsed_seconds':time.time()-started,'metrics_sha256':sha(OUT/'pilot_metrics.json'),'adapter_sha256':protocol['adapter_sha256'],'weights_updated':False})
        progress(phase='complete',completed=total);print('Pilot complete. All attempts saved.',flush=True)
    except BaseException as exc:progress(phase='failed',error=str(exc));raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
