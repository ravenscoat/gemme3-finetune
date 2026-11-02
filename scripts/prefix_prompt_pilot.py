"""Fixed label prefixes plus the original training prompt; no template placeholders."""
import copy
import json
from pathlib import Path
import sys

from dataset_catalog import ROOT,CATALOG,data,read_json,read_rows,sha
from json_io import write_json
import schema_prompt_pilot as queue

OUT=ROOT/'outputs/prefix-prompt-pilot-v1'
SIZES={'reliability':8,'bitext':8,'insuranceqa':8}
PARENT=queue.OUT


def prefix_for(key,required):
    fixed=json.dumps(required,ensure_ascii=False)[:-1]
    return fixed+(', "input": {"customer_message": "' if key=='reliability' else ', "instruction": "' if key=='bitext' else ', "question": "')


def prepare():
    assert all((PARENT/k/'completion.json').exists() for k in SIZES),'Finish template diagnostic first'
    for key,n in SIZES.items():
        old=PARENT/key;original=read_json(old/'protocol.json');p=copy.deepcopy(original);d=OUT/key;d.mkdir(parents=True,exist_ok=True)
        fields={'reliability':['category','subcategory','difficulty'],'bitext':['intent','category','tags'],'insuranceqa':['domain']}[key]
        conditions=[{'required':{f:c['required'][f] for f in fields},'ranges':{}} for c in original['conditions'][:n]]
        p.update(n=n,conditions=conditions,baseline_ids=list(range(n)),selected_source_rows=p['selected_source_rows'][:n],
            prompts=[data(key)['prompt']]*n,assistant_prefixes=[prefix_for(key,c['required']) for c in conditions],
            parent_protocol_sha256=sha(old/'protocol.json'),
            method='External label prefix plus exact original user prompt. Gemma generates the continuation; supplied labels are not model predictions.',
            policy='Fixed first eight parent profiles. No free-text exemplars, template markers, forced length/style/numeric/history/action ranges, repair, reroll, training, test scoring or deployment. Raw prefix and generated continuation stored separately. Diagnostic only, not distribution proof.')
        p.pop('parent_balance_full_batch',None)
        if (d/'protocol.json').exists():assert read_json(d/'protocol.json')==p
        else:
            write_json(d/'protocol.json',p)
            for label,source in [('baseline','baseline'),('previous','controlled')]:
                raw=read_rows(old/(source+'_generations.jsonl'))[:n]
                (d/(label+'_generations.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw),encoding='utf-8')
            write_json(d/'baseline_freeze.json',{'protocol_sha256':sha(d/'protocol.json'),'baseline_copy_sha256':sha(d/'baseline_generations.jsonl')})
            write_json(d/'progress.json',{'phase':'prepared','completed':0,'total':n})
    print('24 prefix-conditioned attempts frozen.',flush=True)


def report():
    from dataset_metrics import assess_label
    results={};lines=['# Label-prefix diagnostic','',
        'The exact original training prompt is used. An externally supplied assistant prefix fixes the opening category fields; Gemma generates all remaining tokens. The complete record is the predeclared prefix plus untouched generated continuation, both stored separately. This is hybrid controlled generation, not learned category frequencies.','',
        'No free-text placeholders or length/style/range constraints are supplied. This is a practical change of generation policy, not a single-factor ablation. All attempts retained. No repair/reroll, new weights, test scoring or deployment. These fixed development batches do not establish source-distribution equivalence. The previous arm comes from the parent experiment recorded in each protocol.','',
        '| Dataset | Generic valid | Previous valid | Prefix valid | Marker leaks: previous → prefix | Source constraints: generic / previous / prefix |',
        '|---|---:|---:|---:|---:|---|']
    for key,n in SIZES.items():
        d=OUT/key;p=read_json(d/'protocol.json');freeze=read_json(d/'baseline_freeze.json');done=read_json(d/'completion.json');m=read_json(d/'metrics.json')
        assert sha(d/'protocol.json')==freeze['protocol_sha256']==done['protocol_sha256']
        assert sha(d/'metrics.json')==done['metrics_sha256'] and sha(d/'controlled_generations.jsonl')==done['raw_sha256']
        assert sha(d/'baseline_generations.jsonl')==freeze['baseline_copy_sha256']
        assert sha(PARENT/key/'protocol.json')==p['parent_protocol_sha256']
        parent_done=read_json(PARENT/key/'completion.json');assert sha(PARENT/key/'controlled_generations.jsonl')==parent_done['raw_sha256']
        assert sha(Path(p['adapter'])/'adapter_model.safetensors')==p['adapter_sha256']
        assert sha(ROOT/CATALOG[key]['prepared']/'train.jsonl')==p['source_train_sha256']
        previous,_=assess_label(key,read_rows(d/'previous_generations.jsonl'),d/'previous_records.jsonl')
        arms={'generic':m['arms']['baseline'],'previous':previous,'prefix':m['arms']['controlled']}
        leaks={}
        for label in ['baseline','previous','controlled']:
            raw=read_rows(d/(label+'_generations.jsonl'));assert len(raw)==n and [r['sample_id'] for r in raw]==list(range(n))
            assert all(r['batch_seed']==p['baseline_batch_seed_start']+r['sample_id']//4 and r['batch_size']==4 for r in raw)
            leaks[label]=sum('<GENERATE:' in r['text'] for r in raw)
            if label=='controlled':
                for r in raw:assert r['text']==p['assistant_prefixes'][r['sample_id']]+r['generated_text'] and r['assistant_prefix']==p['assistant_prefixes'][r['sample_id']]
            else:
                parent_label='baseline' if label=='baseline' else 'controlled'
                assert raw==read_rows(PARENT/key/(parent_label+'_generations.jsonl'))[:n]
        a,b,c=[arms[k] for k in arms]
        gate=c['structure_valid']==n and c['source_constraint_valid']>=a['source_constraint_valid'] and leaks['controlled']==0 and c['token_limit_count']==0
        results[key]={'arms':arms,'marker_leaks':leaks,'formatting_gate_passed':gate,'verified':True}
        lines.append(f'| {key} | {a["structure_valid"]}/{n} | {b["structure_valid"]}/{n} | {c["structure_valid"]}/{n} | {leaks["previous"]} → {leaks["controlled"]} | {a["source_constraint_valid"]} / {b["source_constraint_valid"]} / {c["source_constraint_valid"]} |')
    for key,r in results.items():
        lines+=['',f'## {key}','',f'Formatting gate passed: {r["formatting_gate_passed"]}. Supplied-label compliance is guaranteed by the prefix when the record parses correctly, and is not an independent model-quality score.','',
            '| Field | Wasserstein: generic / previous / prefix |','|---|---|']
        for field in r['arms']['generic']['fidelity_vs_training']['numeric']:
            vals=[r['arms'][a]['fidelity_vs_training']['numeric'][field].get('wasserstein') for a in r['arms']]
            lines.append('| '+field+' | '+' / '.join('not estimable' if v is None else f'{v:.4g}' for v in vals)+' |')
        a=r['arms']['prefix'];lines+=['',f'Whole-record normalized copy fraction among valid records: {a["exact_training_record_match_rate"]}. No privacy guarantee.',
            f'Token-limit hits: {a["token_limit_count"]}. Failures: '+json.dumps(a['failures'])+'. Source warnings: '+json.dumps(a['source_warning_counts'])+'.']
        for field in ['training_request_match_rate','training_reply_match_rate']:
            lines.append(field+' (generic / previous / prefix; fractions of valid rows): '+' / '.join(str(r['arms'][arm][field]) for arm in r['arms'])+'.')
        if key=='reliability':lines.append('Reply reuse here compares the structured expected_output workflow, not natural-language answer text. Repeated finite route/action label combinations are expected; this metric alone is not evidence of copying customer messages.')
        b=r['arms']['generic'];c=r['arms']['prefix']
        lines+=['','| Source characteristic | TV generic | TV prefix |','|---|---:|---:|']
        for field,x in b['feature_fidelity'].items():
            before,after=x['TV'],c['feature_fidelity'][field]['TV']
            lines.append(f'| {field} | {before if before is None else round(before,4)} | {after if after is None else round(after,4)} |')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write_json(OUT/'results.json',results);return results


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:
        queue.OUT=OUT;queue.PARENT=PARENT;queue.SIZES=SIZES;queue.STATUS=ROOT/'reports/prefix_prompt_pilot_status.json';queue.report=report;queue.run()
