"""Record the predeclared rejection; no additional prompt search."""
from collections import Counter
import json
from pathlib import Path

from dataset_catalog import ROOT,read_json,read_rows,sha
from dataset_metrics import tv
from json_io import write_json
from reliability_consistency_trial import OUT,IDS,ISSUES


def main():
    p=read_json(OUT/'protocol.json');f=read_json(OUT/'freeze.json');done=read_json(OUT/'completion.json');m=read_json(OUT/'metrics.json')
    assert sha(OUT/'protocol.json')==f['protocol_sha256']==done['protocol_sha256']
    assert sha(OUT/'cases.jsonl')==f['cases_sha256'] and sha(OUT/'metrics.json')==done['metrics_sha256']
    cases=read_rows(OUT/'cases.jsonl');judgments=[]
    for arm in ['baseline','candidate']:
        raw=read_rows(OUT/arm/'generations.jsonl');assert len(raw)==12
        assert sha(OUT/arm/'generations.jsonl')==m[arm]['raw_sha256']
        parsed={r['sample_id']:r['record'] for r in read_rows(OUT/arm/'records.jsonl')}
        for i,row in enumerate(raw):
            assert row['sample_id']==i and row['source_sample_id']==IDS[i]
            assert row['text']==p['assistant_prefixes'][i]+row['generated_text']
            assert row['batch_seed']==p['seed_start']+i//4
            if i in parsed:assert parsed[i]['input']==cases[i]['record']['input']
            ident=IDS[i]
            if ident not in ISSUES:
                verdict,reason='Pass','The workflow still addresses the supplied request with no explicit input/history contradiction observed under the limited rubric.'
                if arm=='candidate' and ident==1:verdict,reason='Unsure','The loss-run workflow drops document delivery from required_actions; delivery is mentioned only in response elements.'
            elif ident==18:
                verdict,reason=('Issue','The workflow still emphasizes incident response without an explicit deductible answer.') if arm=='baseline' else ('Issue','The generated workflow is malformed JSON; it cannot be accepted as a complete scenario.')
            elif ident==33:verdict,reason='Issue','The homeowners request and commercial-liability context remain unreconciled before an endorsement update.'
            elif ident==42:verdict,reason='Issue','process_payment remains in actions although history states the payment is already completed.'
            elif ident==63:
                verdict,reason=('Issue','The health-policy context remains unreconciled before a flood-exclusion explanation.') if arm=='baseline' else ('Unsure','lookup_flood_policy could identify a separate policy, but reconciliation with the supplied health-policy context is not explicit before exclusion advice.')
            judgments.append({'arm':arm,'sample_id':i,'source_sample_id':ident,'verdict':verdict,'reason':reason,'reviewer':'one_unblinded_assistant'})
    counts={arm:dict(Counter(j['verdict'] for j in judgments if j['arm']==arm)) for arm in m}
    controls=all(j['verdict']=='Pass' for j in judgments if j['arm']=='candidate' and j['source_sample_id'] not in ISSUES)
    fixes=sum(j['verdict']=='Pass' for j in judgments if j['arm']=='candidate' and j['source_sample_id'] in ISSUES)
    numeric={field:{arm:m[arm]['fidelity_vs_training']['numeric'][field].get('wasserstein') for arm in m} for field in ['reply_characters','required_action_count']}
    gates={'all_schema_valid':m['candidate']['structure_valid']==12,
        'source_constraints_no_worse':m['candidate']['source_constraint_valid']>=m['baseline']['source_constraint_valid'],
        'whole_record_copy_no_worse':m['candidate']['exact_training_record_match_rate']<=m['baseline']['exact_training_record_match_rate'],
        'passing_controls_preserved':controls,'at_least_two_issues_to_pass':fixes>=2,
        'issue_count_lower':counts['candidate'].get('Issue',0)<counts['baseline'].get('Issue',0),
        'numeric_distances_no_worse':all(v['candidate']<=v['baseline']+1e-12 for v in numeric.values()),
        'no_token_or_marker_failure':m['candidate']['token_limit_count']==0 and all('<GENERATE:' not in r['text'] for r in read_rows(OUT/'candidate/generations.jsonl'))}
    from dataset_catalog import data
    train=data('reliability')['train'];ref=[len(r['expected_output']['required_actions']) for r in train]
    action_tv={arm:tv([len(w['record']['expected_output']['required_actions']) for w in read_rows(OUT/arm/'records.jsonl')],ref) for arm in m}
    gates['action_count_TV_no_worse']=action_tv['candidate']<=action_tv['baseline']+1e-12
    accepted=all(gates.values())
    result={'decision':'accept_as_development_candidate' if accepted else 'reject_keep_existing_policy','gates':gates,'review_counts':counts,'prior_issues_to_pass':fixes,
        'numeric_wasserstein':numeric,'action_count_TV':action_tv,'metrics':m,
        'scope':p['scope'],'limit':'One unblinded AI review; selected failure/control cases, not a population accuracy estimate. Fixed inputs were externally supplied; validity-conditioned numeric distances do not excuse malformed attempts.',
        'other_datasets_unchanged':True,'additional_experiments_planned':False}
    (OUT/'judgments.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in judgments),encoding='utf-8')
    write_json(OUT/'decision.json',result)
    lines=['# Final bounded Reliability consistency attempt','',f'**Decision: {result["decision"]}.**','',
        'Exactly24 attempts:12 fixed scenarios under each prompt. The four previously flagged scenarios and eight passing controls were selected before generation. No repairs, retries, training or final-test scoring. Original inputs and labels were supplied externally; Gemma completed only the expected workflow. This differs from unconditional synthetic generation.','',
        'The guard was rejected: one malformed record, the already-completed payment is still processed again, and action-count distance worsens. One previously passing document workflow also drops its explicit delivery action. No additional prompt revision will be tried before submission.','',
        '| Check | Baseline | Guard |','|---|---:|---:|',
        f'| Schema valid | {m["baseline"]["structure_valid"]}/12 | {m["candidate"]["structure_valid"]}/12 |',
        f'| AI review Pass / Unsure / Issue | {counts["baseline"]} | {counts["candidate"]} |',
        f'| Action-count Wasserstein | {numeric["required_action_count"]["baseline"]:.4f} | {numeric["required_action_count"]["candidate"]:.4f} |',
        f'| Workflow-character Wasserstein | {numeric["reply_characters"]["baseline"]:.4f} | {numeric["reply_characters"]["candidate"]:.4f} |','',
        result['limit'],'','## Predeclared gates','',*['- '+k+': '+str(v) for k,v in gates.items()],
        '', '## Every judgment','', '| Arm | Original case | Verdict | Reason |','|---|---:|---|---|']
    lines += [f'| {j["arm"]} | {j["source_sample_id"]} | {j["verdict"]} | {j["reason"]} |' for j in judgments]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(result['decision'],counts,gates)


if __name__=='__main__':main()
