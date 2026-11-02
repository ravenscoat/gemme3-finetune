"""Bounded, train-only black-box teacher pilot. Credentials stay in the environment.

Teacher outputs are candidates, never ground truth or automatically selected models.
"""
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha,signature,validate_record,source_warnings
from json_io import write_json

OUT=ROOT/'outputs/reliability-distillation-v1'
MODEL='gemini-3.8-flash'
FIXED_CONTEXT=False
FAMILIES={
 'history_consistency':['payment_then_change','claim_then_status'],
 'policy_context':['flood_question','add_endorsement'],
 'request_completeness':['deductible_question','cyber_breach'],
 'ordinary_control':['add_driver','claim_status'],
}
SYSTEM='''You create fictional insurance workflow training records. Return JSON only.
Use only the fixed category, subcategory and difficulty supplied in the request.
Create a NEW customer message and context, not a copy of the source anchor.
The source anchor is style/input context, not authoritative policy advice.
Use these exact top-level fields: category, subcategory, difficulty, input, expected_output.
input has customer_message (string), context (object), conversation_history (list of role/content objects).
expected_output has intent (string), routing_decision (string), required_actions (list of strings), priority (string), compliance_flags (list of strings), expected_response_elements (list of strings).
Use short snake_case workflow labels and the source-like record length. Do not pad action lists.
For history_consistency, history must clearly state an action was already completed. The workflow confirms it, handles any remaining request, and never repeats that completed transaction.
For policy_context, create a deliberate mismatch between the requested policy and the policy context. Clarify or verify the relevant policy before policy-specific advice or changes.
For request_completeness, include an explicit deductible or coverage-detail question. The workflow retrieves the relevant policy detail and answers that question; never invent a deductible or policy value.
For ordinary_control, create a coherent straightforward example without introducing a policy mismatch or completed transaction.
Preserve plausible conversation roles. Context fields should use supplied source-supported values.
Do not add commentary, explanations, source IDs, teacher labels, placeholders or reasoning to the record.
'''


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=data('reliability');train=source['train'];rng=random.Random(2026110119)
    selection=read_json(ROOT/source['spec']['selection'])
    context_values={}
    for row in train:
        for name,value in row['input']['context'].items():
            context_values.setdefault(name,set()).add(str(value))
    allowed={name:sorted(values) for name,values in context_values.items()}
    routing=sorted({r['expected_output']['routing_decision'] for r in train})
    priorities=sorted({r['expected_output']['priority'] for r in train})
    cases=[]
    for family,subcategories in FAMILIES.items():
        pool=[r for r in train if r['subcategory'] in subcategories]
        if FIXED_CONTEXT and family=='ordinary_control':
            # Claim-status scenarios are coherent for every source policy type.
            pool=[r for r in train if r['category']=='claim_status']
        if len(pool)<3:raise RuntimeError(f'Insufficient training anchors for {family}: {len(pool)}')
        for row in rng.sample(pool,3):
            anchor={k:row[k] for k in ['category','subcategory','difficulty','input']}
            cases.append({'sample_id':len(cases),'family':family,'training_source_id':row['source_id'],
                'request':{'failure_family':family,'fixed_labels':{k:row[k] for k in ['category','subcategory','difficulty']},
                    'source_anchor':anchor,'allowed_context_values':allowed,'allowed_routing_decisions':routing,'allowed_priorities':priorities}})
    protocol={'dataset':'reliability','teacher_model':MODEL,'teacher_endpoint':'https://aiplatform.googleapis.com/v1/publishers/google/models/'+MODEL+':generateContent',
        'stage':'12-output teacher pilot; manual assistant review required before any training',
        'user_permission_basis':'User answered yes when asked whether this Google Cloud account has separate permission for using Gemini outputs to train Gemma. This records their assertion, not a legal determination.',
        'system_prompt':SYSTEM+('''
Keep input.context EXACTLY equal to source_anchor.input.context: every key and value.
Do not replace policy_type, state, customer tier, status, time or channel with defaults.
Create new wording and histories appropriate to that fixed policy context, even when
the source anchor message itself is inconsistent. For policy_context cases, make the
new customer request explicitly refer to a DIFFERENT policy, and clarify before any
policy-specific conclusion. For other families, make the request fit the fixed policy.
''' if FIXED_CONTEXT else ''),'preserve_source_context':FIXED_CONTEXT,
        'temperature':.3,'max_output_tokens':1536,'n':12,'seed':2026110119,
        'source_train_sha256':sha(ROOT/source['spec']['prepared']/'train.jsonl'),
        'schema_sha256':sha(ROOT/source['spec']['prepared']/'schema.json'),
        'baseline_adapter':selection['adapter'],'baseline_adapter_sha256':selection['adapter_sha256'],
        'data_policy':'Only original training records are used as anchors. No validation/test content or prior reviewed synthetic cases are sent to the teacher. Validation and test file bytes are not read by this preparation.',
        'retention':'One request per case, no semantic rerolls or JSON repair. All raw attempts, API failures, model versions and usage retained. Stop on account/quota errors.',
        'scope':'Task-specific output/sequence distillation; no teacher weights, logits, gradients or private reasoning are used. Teacher candidates are not insurance-expert ground truth.',
        'next_stage':'Only after reviewing this pilot, freeze a separate training mix and comparison protocol. No automatic training, adoption, current-cohort overwrite or rescore of original tests.'}
    if (OUT/'protocol.json').exists():
        assert read_json(OUT/'protocol.json')==protocol,'Protocol changed; use a new version'
    else:
        write_json(OUT/'protocol.json',protocol)
        (OUT/'requests.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
        write_json(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'requests_sha256':sha(OUT/'requests.jsonl')})
    return protocol


def score():
    protocol=read_json(OUT/'protocol.json');cases=read_rows(OUT/'requests.jsonl')
    train=data('reliability')['train'];seen={signature(core('reliability',r)) for r in train}
    rows=read_rows(OUT/'teacher_attempts.jsonl');valid=[];audit=[]
    for attempt in rows:
        case=cases[attempt['sample_id']];errors=[];warnings=[];record=None
        try:
            if attempt.get('api_error'):raise ValueError('API error: '+attempt['api_error'])
            record=json.loads(attempt['text']);errors=validate_record('reliability',record)
            if errors:raise ValueError('; '.join(errors))
            if any(record[k]!=v for k,v in case['request']['fixed_labels'].items()):raise ValueError('Teacher changed fixed source labels')
            warnings=source_warnings('reliability',record)
            if warnings:raise ValueError('; '.join(warnings))
            if protocol.get('preserve_source_context') and record['input']['context']!=case['request']['source_anchor']['input']['context']:
                raise ValueError('Teacher changed the fixed source context')
            if signature(record) in seen:raise ValueError('Whole record is a normalized training match')
            if signature(record['input']['customer_message'])==signature(case['request']['source_anchor']['input']['customer_message']):raise ValueError('Customer message copied from the anchor')
            valid.append({'sample_id':attempt['sample_id'],'family':case['family'],'record':record})
        except (ValueError,KeyError,TypeError) as exc:errors=[str(exc)]
        audit.append({'sample_id':attempt['sample_id'],'family':case['family'],'preflight_pass':not errors,'errors':errors,'source_warnings':warnings})
    (OUT/'teacher_valid_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in valid),encoding='utf-8')
    write_json(OUT/'teacher_preflight.json',{'attempted':len(rows),'planned':protocol['n'],'preflight_pass':len(valid),'audit':audit,
        'attempts_sha256':sha(OUT/'teacher_attempts.jsonl'),'scope':'Format, selected source constraints and exact reuse only; semantic review is still required. No model was trained.'})
    print('Teacher preflight:',len(valid),'/',len(rows),'passed; semantic review pending.',flush=True)


def run():
    protocol=prepare();freeze=read_json(OUT/'freeze.json')
    assert sha(OUT/'protocol.json')==freeze['protocol_sha256']
    assert sha(OUT/'requests.jsonl')==freeze['requests_sha256']
    key=os.environ.get('GEMINI_API_KEY')
    if not key:raise RuntimeError('Set GEMINI_API_KEY in the process environment; do not put it in files')
    path=OUT/'teacher_attempts.jsonl';previous=read_rows(path) if path.exists() else []
    assert [r['sample_id'] for r in previous]==list(range(len(previous))),'Noncontiguous or duplicate attempt IDs'
    cases=read_rows(OUT/'requests.jsonl');started=time.time()
    with path.open('a',encoding='utf-8') as stream:
        for case in cases[len(previous):]:
            payload={'systemInstruction':{'parts':[{'text':protocol['system_prompt']}]},
                'contents':[{'role':'user','parts':[{'text':json.dumps(case['request'],ensure_ascii=False)}]}],
                'generationConfig':{'temperature':protocol['temperature'],'maxOutputTokens':protocol['max_output_tokens'],'responseMimeType':'application/json'}}
            request=urllib.request.Request(protocol['teacher_endpoint'],data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','x-goog-api-key':key},method='POST')
            row={'sample_id':case['sample_id'],'family':case['family'],'requested_model':MODEL}
            began=time.time()
            try:
                with urllib.request.urlopen(request,timeout=90) as response:
                    result=json.load(response)
                row.update(model_version=result.get('modelVersion'),usage=result.get('usageMetadata'),
                    finish_reasons=[r.get('finishReason') for r in result.get('candidates',[])],
                    text=''.join(p.get('text','') for r in result.get('candidates',[]) for p in r.get('content',{}).get('parts',[]) if not p.get('thought')))
            except urllib.error.HTTPError as exc:
                detail=json.loads(exc.read().decode('utf-8',errors='replace')).get('error',{})
                row.update(http_status=exc.code,api_error=str(detail.get('message','')).replace(key,'[REDACTED]'))
            except Exception as exc:
                row.update(api_error=str(exc).replace(key,'[REDACTED]'))
            row['seconds']=round(time.time()-began,2)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
            write_json(OUT/'progress.json',{'phase':'teacher_generation','completed':case['sample_id']+1,'total':len(cases),'elapsed_seconds':round(time.time()-started,1)},best_effort=True)
            print('Teacher',case['sample_id']+1,'/',len(cases),'API error' if row.get('api_error') else row.get('finish_reasons'),flush=True)
            if row.get('http_status') in [401,402,403,429]:break
    score()
    write_json(OUT/'progress.json',{'phase':'teacher_pilot_complete' if len(read_rows(path))==len(cases) else 'stopped_on_account_error',
        'attempted':len(read_rows(path)),'total':len(cases),'training_started':False},best_effort=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--score',action='store_true')
    parser.add_argument('--experiment-root',type=Path,default=OUT);parser.add_argument('--fixed-context',action='store_true');args=parser.parse_args()
    OUT=args.experiment_root.resolve();FIXED_CONTEXT=args.fixed_context
    if not OUT.is_relative_to((ROOT/'outputs').resolve()):raise RuntimeError('Experiment must stay within project outputs')
    if args.prepare:prepare();print('Prepared twelve train-only teacher requests.')
    elif args.score:score()
    else:run()
