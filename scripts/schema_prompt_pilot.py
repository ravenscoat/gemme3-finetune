"""A bounded diagnostic of JSON-template presentation; all v1 artifacts stay frozen."""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dataset_catalog import ROOT,CATALOG,read_json,read_rows,sha
from dataset_metrics import assess_label
from fidelity_alignment import OUT as PARENT
from json_io import write_json

OUT=ROOT/'outputs/schema-prompt-pilot-v1'
SIZES={'reliability':16,'bitext':8,'insuranceqa':8}
STATUS=ROOT/'reports/schema_prompt_pilot_status.json'
MARKER='<GENERATE:'


def nested_template(key,condition):
    required=condition['required']
    if key=='bitext':return {**required,'instruction':'<GENERATE: new customer request>', 'response':'<GENERATE: matching support reply>'}
    if key=='insuranceqa':return {**required,'question':'<GENERATE: new insurance question>','answer':'<GENERATE: corresponding professional answer>'}
    history=[{'role':'user' if i%2==0 else 'assistant','content':'<GENERATE: prior conversation turn>'} for i in range(condition['history_turns'])]
    return {**{f:required[f] for f in ['category','subcategory','difficulty']},
        'input':{'customer_message':'<GENERATE: new customer message>','context':required['input.context'],'conversation_history':history},
        'expected_output':{**{f:required['expected_output.'+f] for f in ['intent','routing_decision','priority']},
            'required_actions':['<GENERATE: required action>' for _ in range(condition['required_action_count'])],
            'compliance_flags':['<GENERATE: applicable compliance flag, or use an empty list>'],
            'expected_response_elements':['<GENERATE: appropriate response element>']}}


def improved_prompt(key,prompt,condition):
    separator='\nRequired field values (dotted paths refer to nested JSON fields): '
    prefix,serialized=prompt.split(separator,1)
    required,end=json.JSONDecoder().raw_decode(serialized)
    assert required==condition['required'] and serialized[end]=='.'
    suffix=serialized[end+1:]
    return (prefix+'\nReturn exactly one complete JSON object using this structure. '
        'Keep the fixed field values and replace every <GENERATE: ...> string with new content. '
        'Arrays for flags and response elements may have the appropriate number of items. '
        'Return the complete object, not separate fields.\n'+json.dumps(nested_template(key,condition),ensure_ascii=False)+suffix)


def prepare():
    for key,n in SIZES.items():
        old=PARENT/key;p=copy.deepcopy(read_json(old/'protocol.json'));d=OUT/key;d.mkdir(parents=True,exist_ok=True)
        p.update(n=n,baseline_ids=list(range(n)),conditions=p['conditions'][:n],selected_source_rows=p['selected_source_rows'][:n])
        p['parent_protocol_sha256']=sha(old/'protocol.json')
        p['parent_balance_full_batch']=p.pop('balance')
        p['prompts']=[improved_prompt(key,s,c) for s,c in zip(p['prompts'][:n],p['conditions'])]
        p['method']='Diagnostic presentation change: nested JSON template replaces dotted required-value mapping. Fixed values, ranges/style/history/action requirements and generation settings retained.'
        p['policy']='First frozen profiles of prior randomized plan, chosen without selecting successful outputs. Small development formatting diagnostic, not a distribution-equivalence test. All raw attempts retained; no repairs/rerolls, new training, test scoring or deployment. Do not auto-promote or launch a large batch.'
        if (d/'protocol.json').exists():assert read_json(d/'protocol.json')==p,'Pilot protocol changed'
        else:
            write_json(d/'protocol.json',p)
            for label,source in [('baseline','baseline'),('previous','controlled')]:
                raw=read_rows(old/(source+'_generations.jsonl'))[:n]
                (d/(label+'_generations.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw),encoding='utf-8')
            write_json(d/'baseline_freeze.json',{'protocol_sha256':sha(d/'protocol.json'),'baseline_copy_sha256':sha(d/'baseline_generations.jsonl')})
            write_json(d/'progress.json',{'phase':'prepared','completed':0,'total':n})
    print('All 32 pilot prompts frozen before generation.',flush=True)


def report():
    results={};lines=['# Nested-template prompt diagnostic','',
        'Small fixed development batches. Same source profiles, adapters and sampling settings as the prior controlled run. Only required-field presentation changes to a complete nested JSON template; existing length/style/history/action instructions remain. Prompt token lengths and remaining completion budgets differ. No single-cause causal proof or distribution-equivalence claim.','',
        'Every raw attempt retained, including malformed responses. No parser repair, reroll, training, new test scoring or automatic deployment. These profiles cover only a subset of modes and do not establish broad fidelity.','',
        '| Dataset | Attempts/arm | Generic schema-valid | Previous controlled schema-valid | Nested-template schema-valid | Source-valid: generic / previous / new | Template-marker leaks: new |',
        '|---|---:|---:|---:|---:|---|---:|']
    for key,n in SIZES.items():
        d=OUT/key;p=read_json(d/'protocol.json');f=read_json(d/'baseline_freeze.json');done=read_json(d/'completion.json');m=read_json(d/'metrics.json')
        assert sha(d/'protocol.json')==f['protocol_sha256']==done['protocol_sha256']
        assert sha(d/'metrics.json')==done['metrics_sha256']
        assert sha(d/'controlled_generations.jsonl')==done['raw_sha256']
        assert sha(d/'baseline_generations.jsonl')==f['baseline_copy_sha256']
        assert sha(PARENT/key/'protocol.json')==p['parent_protocol_sha256']
        assert sha(Path(p['adapter'])/'adapter_model.safetensors')==p['adapter_sha256']
        previous,rows=assess_label(key,read_rows(d/'previous_generations.jsonl'),d/'previous_records.jsonl')
        arms={'generic':m['arms']['baseline'],'previous':previous,'nested':m['arms']['controlled']}
        for label in ['baseline','previous','controlled']:
            raw=read_rows(d/(label+'_generations.jsonl'))
            assert len(raw)==n and [r['sample_id'] for r in raw]==list(range(n))
            assert all(r['batch_seed']==p['baseline_batch_seed_start']+r['sample_id']//4 and r['batch_size']==4 for r in raw)
            if label!='controlled':
                oldlabel='baseline' if label=='baseline' else 'controlled'
                assert raw==read_rows(PARENT/key/(oldlabel+'_generations.jsonl'))[:n]
        leaks=sum(MARKER in r['text'] for r in read_rows(d/'controlled_generations.jsonl'))
        counts=[arms[k]['structure_valid'] for k in arms];strict=[arms[k]['source_constraint_valid'] for k in arms]
        lines.append(f'| {key} | {n} | {counts[0]} | {counts[1]} | {counts[2]} | {strict[0]} / {strict[1]} / {strict[2]} | {leaks} |')
        restored=counts[2]==n and strict[2]>=strict[0] and leaks==0 and arms['nested']['token_limit_count']==0
        results[key]={'arms':arms,'template_marker_leaks':leaks,'formatting_gate_passed':restored,'adapter_unchanged':True,'raw_ids_seeds_baseline_copies_verified':True}
    for key,r in results.items():
        lines += ['',f'## {key}','',f'Formatting gate passed: {r["formatting_gate_passed"]}. Gate requires all attempts schema-valid, source-valid count at least the generic baseline, zero marker leaks and zero token-limit hits. Passing this small diagnostic does not establish final-batch reliability.','',
            '| Field | Wasserstein: generic / previous / nested |','|---|---|']
        for field in r['arms']['generic']['fidelity_vs_training']['numeric']:
            vals=[r['arms'][a]['fidelity_vs_training']['numeric'][field].get('wasserstein') for a in r['arms']]
            lines.append('| '+field+' | '+' / '.join('not estimable' if v is None else f'{v:.4g}' for v in vals)+' |')
        a=r['arms']['nested'];lines+=['',f'New exact-field/range compliance: {a["condition_compliance"]}.',
            'New source warnings: '+json.dumps(a['source_warning_counts'])+'.',
            'New failures: '+json.dumps(a['failures'])+'.',
            f'Normalized whole-record copy fraction, valid rows: {a["exact_training_record_match_rate"]}. No privacy guarantee.']
    write_json(OUT/'results.json',results);(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return results


def run():
    lock=ROOT/'reports/assessment_gpu.lock'
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():raise RuntimeError('Training owns GPU')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    started=time.time();status={'phase':'running','datasets':{k:'pending' for k in SIZES},'counts':SIZES,'current':None}
    def save():status['elapsed_seconds']=round(time.time()-started,2);write_json(STATUS,status,best_effort=True)
    try:
        for key in SIZES:
            status['current']=key;status['datasets'][key]='running';save()
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/generate_fidelity_alignment.py'),'--dataset',key,'--experiment-root',str(OUT)],cwd=ROOT,check=True)
            status['datasets'][key]='complete';save()
        status['phase']='reporting';save();results=report()
        status.update(phase='complete',current=None,formatting_gates={k:r['formatting_gate_passed'] for k,r in results.items()});save()
        print('Pilot complete; report:',OUT/'report.md',flush=True)
    except BaseException as exc:status.update(phase='failed',error=str(exc));save();raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:run()
