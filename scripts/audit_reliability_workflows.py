"""Source-relative workflow diagnostics; support mismatch is not semantic failure."""
from collections import Counter,defaultdict
import json
from pathlib import Path

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha,signature
from dataset_metrics import tv
from json_io import write_json

OUT=ROOT/'reports/reliability_workflow_audit/v1'
INPUT=ROOT/'outputs/prefix-alignment-v1/reliability'


def reference(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['category'],row['subcategory'],row['difficulty'])].append(row)
    return groups


def analyze(records,train):
    groups=reference(train);vocabulary={a for row in train for a in row['expected_output']['required_actions']}
    expected=Counter();findings=[];actual=Counter()
    for wrapper in records:
        row=wrapper['record'];key=tuple(row[f] for f in ['category','subcategory','difficulty']);refs=groups.get(key,[])
        actions=row['expected_output']['required_actions'];actual[len(actions)]+=1
        counts=Counter(len(r['expected_output']['required_actions']) for r in refs)
        if refs:
            for count,n in counts.items():expected[count]+=n/len(refs)
        findings.append({'sample_id':wrapper['sample_id'],'source_group':list(key),'source_group_size':len(refs),
            'action_count':len(actions),'source_action_count_histogram':dict(counts),
            'count_outside_observed_group_support':bool(refs) and len(actions) not in counts,
            'actions_unseen_in_training':sorted(set(actions)-vocabulary),
            'duplicate_action_tokens':len(actions)-len(set(actions)),
            'source_action_sets':[json.loads(s) for s in sorted({json.dumps(r['expected_output']['required_actions']) for r in refs})],
            'route_outside_observed_group_support':bool(refs) and row['expected_output']['routing_decision'] not in {r['expected_output']['routing_decision'] for r in refs},
            'intent_outside_observed_group_support':bool(refs) and row['expected_output']['intent'] not in {r['expected_output']['intent'] for r in refs}})
    return {'attempts':len(records),'actual_action_counts':dict(actual),'expected_counts_given_generated_groups':dict(expected),
        'unknown_group_records':sum(f['source_group_size']==0 for f in findings),
        'count_outside_group_support':sum(f['count_outside_observed_group_support'] for f in findings),
        'records_with_unseen_actions':sum(bool(f['actions_unseen_in_training']) for f in findings),
        'route_outside_group_support':sum(f['route_outside_observed_group_support'] for f in findings),
        'intent_outside_group_support':sum(f['intent_outside_observed_group_support'] for f in findings),
        'findings':findings}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'source_train_sha256':sha(ROOT/'data/reliability/prepared/train.jsonl'),
        'controlled_raw_sha256':sha(INPUT/'controlled_generations.jsonl'),
        'controlled_parsed_sha256':sha(INPUT/'controlled_records.jsonl'),
        'generic_parsed_sha256':sha(INPUT/'baseline_records.jsonl'),'ids':list(range(64)),
        'scope':'All64 fixed prefix-cohort records. Generic cohort gets objective diagnostics only; no comparative semantic accuracy claim.',
        'judge':'One unblinded assistant review, authorized by user; not human/expert calibrated. Pass means request/workflow relevance and internal coherence only. No insurance/legal correctness certification.',
        'policy':'No raw changes, action padding, canonical action replacement, rejection or rerolls. Source support diagnostics are flags, not correctness labels. No additional training or test scoring.'}
    if (OUT/'protocol.json').exists():assert read_json(OUT/'protocol.json')==protocol
    else:write_json(OUT/'protocol.json',protocol)
    source=[core('reliability',r) for r in data('reliability')['train']]
    results={'source_action_counts':dict(Counter(len(r['expected_output']['required_actions']) for r in source)),
        'source_action_vocabulary_size':len({a for r in source for a in r['expected_output']['required_actions']}),
        'generic':analyze(read_rows(INPUT/'baseline_records.jsonl'),source),'prefix':analyze(read_rows(INPUT/'controlled_records.jsonl'),source)}
    write_json(OUT/'diagnostics.json',results)
    return results


if __name__=='__main__':prepare()
