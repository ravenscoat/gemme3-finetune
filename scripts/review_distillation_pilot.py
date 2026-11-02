"""Hide arm labels for assistant review, then report all fixed pilot gates."""
import argparse
from collections import Counter
import hashlib
import json
import random
import secrets
from pathlib import Path
from dataset_catalog import ROOT,read_json,read_rows,sha,parse_output,validate_record
from json_io import write_json

OUT=ROOT/'outputs/reliability-distillation-v2'


def prepare_review():
    if (OUT/'review_pack.jsonl').exists():print('Frozen review pack already exists.');return
    source=read_rows(ROOT/'outputs/reliability-consistency-v1/cases.jsonl')
    pack=[];mapping={}
    for arm in ['baseline','replay','distilled']:
        raw=read_rows(OUT/arm/'workflow/generations.jsonl')
        assert len(raw)==12
        for r in raw:
            index=r['sample_id'];ident=secrets.token_hex(5)
            record=None;format_error=None
            try:
                record=parse_output(r['text']);errors=validate_record('reliability',record)
                if errors:raise ValueError('; '.join(errors))
            except (ValueError,TypeError,KeyError) as exc:format_error=str(exc);record=None
            digest=hashlib.sha256(r['text'].encode()).hexdigest()
            mapping[ident]={'arm':arm,'sample_id':index,'text_sha256':digest}
            pack.append({'review_id':ident,'case_index':index,'input':source[index]['record']['input'],
                'expected_workflow':record['expected_output'] if record else None,'format_error':format_error,
                'raw_workflow_if_invalid':r.get('generated_text') if format_error else None})
    random.SystemRandom().shuffle(pack)
    (OUT/'review_pack.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pack),encoding='utf-8')
    write_json(OUT/'review_mapping.json',mapping)
    write_json(OUT/'review_protocol.json',{'n':36,'pack_sha256':sha(OUT/'review_pack.jsonl'),'mapping_sha256':sha(OUT/'review_mapping.json'),
        'rubric':'Pass: complete valid workflow addresses the supplied request, does not repeat a completed transaction, and verifies mismatched policies before policy-specific actions. Issue: clear violation or malformed output. Unsure: insufficiently explicit or materially ambiguous.',
        'blinding':'Arm labels withheld from the JSON review pack and revealed only when joining saved judgments. Same previously-inspected12 cases across arms; this is not a fresh independent test or expert calibration.',
        'scope':'Workflow logic only, not insurance factual/legal accuracy or actual tool execution. User previously authorized assistant judgments.'})
    print('Prepared36 arm-hidden records for review.',flush=True)


def report():
    protocol=read_json(OUT/'review_protocol.json')
    assert sha(OUT/'review_pack.jsonl')==protocol['pack_sha256']
    assert sha(OUT/'review_mapping.json')==protocol['mapping_sha256']
    mapping=read_json(OUT/'review_mapping.json');judgments=read_rows(OUT/'workflow_judgments.jsonl')
    raw_outputs={arm:{r['sample_id']:r for r in read_rows(OUT/arm/'workflow/generations.jsonl')} for arm in ['baseline','replay','distilled']}
    assert all(hashlib.sha256(raw_outputs[r['arm']][r['sample_id']]['text'].encode()).hexdigest()==r['text_sha256'] for r in mapping.values())
    assert len(judgments)==36 and {r['review_id'] for r in judgments}==set(mapping)
    assert all(r['verdict'] in ['Pass','Unsure','Issue'] and r.get('reason') for r in judgments)
    joined=[{**mapping[r['review_id']],**r} for r in judgments]
    by={(r['arm'],r['sample_id']):r for r in joined}
    counts={arm:dict(Counter(r['verdict'] for r in joined if r['arm']==arm)) for arm in ['baseline','replay','distilled']}
    metrics={arm:{kind:read_json(OUT/arm/kind/'metrics.json') for kind in ['generic','workflow']} for arm in counts}
    issues=[4,7,9,11];controls=[i for i in range(12) if i not in issues]
    improved=[i for i in issues if by['distilled',i]['verdict']=='Pass']
    p=read_json(OUT/'student_protocol.json')
    limits=p['limits'].replace('No original test records read or rescored.','No fresh final test was conducted; cached, previously inspected diagnostic inputs were reused.')
    gates={
        'generic16_and_workflow12_schema_valid':all(metrics['distilled'][kind]['structure_valid']==n for kind,n in [('generic',16),('workflow',12)]),
        'no_source_constraint_count_regression':all(metrics['distilled'][kind]['source_constraint_valid']>=metrics['baseline'][kind]['source_constraint_valid'] for kind in ['generic','workflow']),
        'no_whole_record_copy_increase':all(metrics['distilled'][kind]['exact_training_record_match_rate']<=metrics['baseline'][kind]['exact_training_record_match_rate'] for kind in ['generic','workflow']),
        'at_least_two_known_issues_pass':len(improved)>=2,
        'all_eight_prior_passing_controls_preserved':all(by['distilled',i]['verdict']=='Pass' for i in controls),
        'more_passes_than_matched_replay_control':counts['distilled'].get('Pass',0)>counts['replay'].get('Pass',0),
    }
    base=metrics['baseline']['generic']['fidelity_vs_training']['numeric'];new=metrics['distilled']['generic']['fidelity_vs_training']['numeric']
    distances={f:{'baseline':x.get('wasserstein'),'replay':metrics['replay']['generic']['fidelity_vs_training']['numeric'][f].get('wasserstein'),'distilled':new[f].get('wasserstein')} for f,x in base.items()}
    gates['generic_numeric_distances_within_1_25x_baseline']=all(v['baseline'] is not None and v['distilled'] is not None and v['distilled']<=1.25*v['baseline'] for v in distances.values())
    decision={'decision':'retain_as_exploratory_candidate' if all(gates.values()) else 'reject_keep_existing_policy',
        'gates':gates,'workflow_review_counts':counts,'previous_issue_case_indices_now_pass':improved,'generic_numeric_wasserstein':distances,
        'automatic_adoption':False,'baseline_adapter_unchanged':sha(Path(p['baseline_adapter'])/'adapter_model.safetensors')==p['baseline_adapter_sha256'],
        'scope':p['scope'],'limits':limits,'review_protocol_sha256':sha(OUT/'review_protocol.json'),'judgments_sha256':sha(OUT/'workflow_judgments.jsonl')}
    write_json(OUT/'decision.json',decision)
    lines=['# Black-box distillation: bounded Reliability pilot','',
        'Gemini3.8 Flash supplied text targets; no teacher weights, logits or gradients were used. The user confirmed a separate permission basis for this account. Teacher outputs were reviewed as candidates, not insurance-expert ground truth.','',
        'Version1:9 returned examples/12 attempts;3 transport timeouts. Rejected before training because all returned contexts changed policy_type to personal_auto.',
        'Version2:10 returned examples/12 attempts;2 transport timeouts. Contexts were preserved exactly. Assistant review accepted8;2 were withheld for workflow/source-characteristic concerns. Rejected/timeout attempts remain in the raw audit.','',
        'Both student arms continue the same original adapter for 18 updates at learning rate 3e-5. Each has 96 original replay rows plus 32 matched topic/context views. One arm uses the original anchors; the other substitutes the eight accepted teacher examples repeated four times. The raw prepared test split was not opened, but cached, previously inspected diagnostic cases were reused; this is development evaluation, not a fresh final test.','',
        '## Results','',f'Decision: **{decision["decision"]}**. Existing live defaults and frozen assessment cohorts remain unchanged.','',
        '| Arm | Generic valid /16 | Workflow valid /12 | Workflow AI Pass / Unsure / Issue |','|---|---:|---:|---|']
    for arm in counts:
        c=counts[arm];lines.append(f'| {arm} | {metrics[arm]["generic"]["structure_valid"]}/16 | {metrics[arm]["workflow"]["structure_valid"]}/12 | {c.get("Pass",0)} / {c.get("Unsure",0)} / {c.get("Issue",0)} |')
    lines+=['','None of the four known issue cases became Pass in the distilled arm. One previously passing driver case gained an incorrect vehicle intent. The replay control also failed to fix the four issues and introduced an ambiguous loss-run workflow. These are assistant rubric judgments, not expert-validated insurance accuracy.','',
        '| Source checks / attempted | Original | Replay control | Distilled |','|---|---:|---:|---:|']
    for kind,n in [('generic',16),('workflow',12)]:
        lines.append('| '+kind+' | '+' | '.join(f'{metrics[arm][kind]["source_constraint_valid"]}/{n}' for arm in counts)+' |')
    summaries={arm:read_json(OUT/arm/'training_summary.json') for arm in ['replay','distilled']}
    lines+=['','| Source-validation loss | Replay control | Distilled |','|---|---:|---:|']
    for point in ['before','after']:
        lines.append('| '+point+' | '+' | '.join(f'{summaries[arm]["validation_"+point]["eval_loss"]:.6f}' for arm in summaries)+' |')
    lines+=['','The fixed final step-18 adapters were compared; no validation-based checkpoint selection was performed. Source-validation loss is a text-prediction measure, not a workflow correctness score.','',
        'Lower Wasserstein distance and total variation (TV) mean smaller gaps to training data. Request/workflow length distances use characters; action-count distance uses actions. Workflow length measures serialized expected-output JSON, not a customer-facing reply. TV ranges from 0 to 1. Distances and reuse rates exclude malformed records; validity counts keep all attempts visible.']
    lines+=['','| Numeric distance, generic cohort | Original | Replay control | Distilled |','|---|---:|---:|---:|']
    for f,v in distances.items():lines.append('| '+f+' | '+' | '.join('Not estimable' if v[k] is None else f'{v[k]:.4f}' for k in ['baseline','replay','distilled'])+' |')
    lines+=['','| Generated-field TV, generic cohort | Original | Replay control | Distilled |','|---|---:|---:|---:|']
    for field in ['expected_output.routing_decision','expected_output.priority','input.context.policy_type','input.context.state']:
        values=[metrics[arm]['generic']['fidelity_vs_training']['categorical'][field]['total_variation'] for arm in counts]
        lines.append('| '+field+' | '+' | '.join('Not estimable' if v is None else f'{v:.4f}' for v in values)+' |')
    lines+=['','These generated-field categorical gaps are reported even when not part of the narrower predeclared numeric gate. The leading category/subcategory/difficulty labels were supplied externally and are not counted as learned category-frequency improvements.','',
        '| Reuse among valid generic records | Original | Replay control | Distilled |','|---|---:|---:|---:|']
    for field in ['training_request_match_rate','training_reply_match_rate','exact_training_record_match_rate']:
        values=[metrics[arm]['generic'][field] for arm in counts]
        lines.append('| '+field+' | '+' | '.join('Not applicable' if v is None else f'{v:.2%}' for v in values)+' |')
    lines+=['','## Fixed gates','',*['- '+k+': '+str(v) for k,v in gates.items()], '',
        '## Interpretation limits','',p['scope'],limits,
        'The frozen protocol wording about not reading/rescoring original test records refers to the raw prepared split file, not to independence: its previously inspected diagnostic inputs were reused and new student outputs on them were scored. No claim of untouched held-out testing is made.',
        'Arm labels were withheld during this assistant review. The reviewer remains one uncalibrated assistant, and these12 cases were already known before training. Generic records use externally supplied leading labels; matching label proportions is not a learned model result. No population accuracy or distribution-equivalence claim is made.','',
        'All protocols, raw API attempts, accepted/rejected examples, review hashes, training views, checkpoints and unfiltered generated outputs are saved under this experiment directory. Neither original source files nor original adapters were overwritten.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status_path=ROOT/'reports/distillation_status.json'
    if status_path.exists():
        status=read_json(status_path)
        if status.get('experiment_root')==str(OUT):
            status.update(phase='complete',semantic_review_completed=True,review_outcome=decision['decision'],decision_path=str(OUT/'decision.json'))
            write_json(status_path,status)
    print(json.dumps(decision,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare_review()
    else:report()
