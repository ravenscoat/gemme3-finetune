"""Bounded Gemini workflow-completion pilot. No training or automatic adoption.

API keys are read from the environment or an echo-suppressed terminal prompt.
No automatic retry, repair, provider fallback, tools or search.
"""
import argparse
import copy
import getpass
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dataset_catalog import ROOT, core, data, read_json, read_rows, sha, validate_record
from json_io import write_json

OUT=ROOT/'outputs/reliability-distillation-v3'
MODEL='gemini-3.8-flash'
ENDPOINT=f'https://aiplatform.googleapis.com/v1/publishers/google/models/{MODEL}:generateContent'
SYSTEM='''Complete the expected insurance workflow for the supplied fictional scenario.
Return only a JSON object with exactly intent, routing_decision, required_actions,
priority, compliance_flags, expected_response_elements. Do not generate or change
the customer message, policy context or history. Do not provide hidden reasoning.
Read history before choosing actions. Never process a payment already completed;
acknowledge it and handle the remaining request. If the requested policy differs
from the context, verify or clarify the relevant policy before changing it or
making a policy-specific coverage statement. Address each explicit question:
retrieve policy-specific deductible/coverage details before explaining them.
Never invent policy values or coverage decisions. Use the supplied source label
vocabularies and short, source-style action lists. Choose only necessary actions.
The source context is not permission to ignore a mismatch in the request.
All scenarios are hypothetical; no payment or policy action is actually executed.
'''


def reservation_usd(payload, protocol):
    # Deliberately conservative byte-based input allowance, plus metadata margin.
    input_allowance=len(json.dumps(payload,ensure_ascii=False).encode('utf-8'))+4096
    if input_allowance>protocol['max_input_allowance']:
        raise ValueError('Input exceeds the frozen input budget')
    rates=protocol['reserved_rates_per_million']
    return (input_allowance*rates['input']+protocol['max_output_tokens']*rates['output'])/1_000_000


def may_reserve(existing,next_reservation,protocol):
    return (len(existing)<protocol['max_generation_requests'] and
            sum(r['reserved_usd'] for r in existing)+next_reservation<=protocol['budget_usd'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=data('reliability');rows=source['train']
    pools={
        'history_consistency':[r for r in rows if r['subcategory']=='payment_then_change'],
        'policy_context':[r for r in rows if r['subcategory']=='flood_question' and r['input']['context']['policy_type']!='homeowners'],
        'request_completeness':[r for r in rows if r['source_id']=='policy_inquiry_052']+[r for r in rows if r['subcategory']=='cyber_breach'],
        'ordinary_control':[r for r in rows if r['category']=='claim_status'],
    }
    vocab={f:sorted({r['expected_output'][f] for r in rows}) for f in ['intent','routing_decision','priority']}
    vocab.update({f:sorted({v for r in rows for v in r['expected_output'][f]}) for f in ['required_actions','compliance_flags','expected_response_elements']})
    cases=[]
    for family,pool in pools.items():
        if len(pool)<3:raise RuntimeError('Insufficient training anchors: '+family)
        for j,row in enumerate(pool[:3]):
            record=core('reliability',copy.deepcopy(row));changes=[]
            if family=='history_consistency':
                record['input']['conversation_history']=[
                    {'role':'customer','content':'Please pay the outstanding bill.'},
                    {'role':'agent','content':'Your payment has already been processed successfully.'},
                    {'role':'customer','content':'Thank you. I also need to add a new car.'}]
                record['input']['customer_message']='The payment is complete. Please help me add my new car to my auto policy.'
                changes=['Controlled completed-payment history and remaining request; source context preserved.']
            elif family=='policy_context':
                record['input']['customer_message']='I am asking about my homeowners policy. Am I covered for flooding?'
                changes=['Explicit homeowners request against a different source policy context.']
            elif family=='request_completeness' and row['subcategory']=='cyber_breach':
                record['input']['customer_message']+=' What is my deductible?'
                changes=['Added an explicit deductible question to a source cyber scenario.']
            fixed={k:record[k] for k in ['category','subcategory','difficulty','input']}
            payload={'systemInstruction':{'parts':[{'text':SYSTEM}]},
                'contents':[{'role':'user','parts':[{'text':json.dumps({'scenario':fixed,'allowed_workflow_labels':vocab},ensure_ascii=False)}]}],
                'generationConfig':{'temperature':0.2,'maxOutputTokens':1536,
                    'responseMimeType':'application/json','thinkingConfig':{'thinkingLevel':'LOW'}}}
            cases.append({'sample_id':len(cases),'family':family,'future_partition':'teacher_training_candidate' if j<2 else 'development_only',
                'training_source_id':row['source_id'],'changes':changes,'scenario':fixed,'original_source_workflow':row['expected_output'],'payload':payload})
    selected=read_json(ROOT/source['spec']['selection'])
    protocol={'dataset':'reliability','teacher_model':MODEL,'endpoint':ENDPOINT,'method':'SeqKD workflow-completion pilot, not the GAD paper reproduction',
        'system_prompt':SYSTEM,'budget_usd':1.0,'max_generation_requests':12,'max_output_tokens':1536,'max_input_allowance':24000,
        'reserved_rates_per_million':{'input':1.50,'output':7.50},
        'published_rates_per_million':{'input':0.75,'output':3.75},
        'pricing_checked':'2026-11-02; https://cloud.google.com/model-garden/pricing',
        'billing_scope':'Local conservative reservation, not a cloud-account spending cap or credit-balance verification. Unknown timeout usage remains reserved. No tools/caching/retries.',
        'previous_experiment':'v2 used8 teacher records and18 updates; failed its checks. v3 tests fixed-input workflow targets before training.',
        'source_train_sha256':sha(ROOT/source['spec']['prepared']/'train.jsonl'),
        'original_adapter':selected['adapter'],'original_adapter_sha256':selected['adapter_sha256'],
        'vocabularies':vocab,'n':len(cases),
        'scope':'Training-source anchors with declared controlled input edits. Already-trained source contexts, not a fresh independent test. No raw validation/test rows read. Cases fixed before teacher calls.',
        'data_use_permission':'User previously confirmed separate permission to use Gemini outputs to train Gemma; assertion recorded, not independently verified.',
        'next_step':'Review every teacher result. No training, source changes, model selection, report changes or automatic deployment in this script.'}
    protocol['planned_maximum_reservation_usd']=sum(reservation_usd(c['payload'],protocol) for c in cases)
    if protocol['planned_maximum_reservation_usd']>protocol['budget_usd']:raise RuntimeError('Planned requests exceed budget')
    requests=''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases)
    if (OUT/'protocol.json').exists():
        assert read_json(OUT/'protocol.json')==protocol,'Use a new experiment version'
        assert (OUT/'requests.jsonl').read_text(encoding='utf-8')==requests,'Requests changed'
    else:
        write_json(OUT/'protocol.json',protocol)
        (OUT/'requests.jsonl').write_text(requests,encoding='utf-8')
        write_json(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'requests_sha256':sha(OUT/'requests.jsonl')})
    print(f"Prepared {len(cases)} cases. Conservative reservation for all requests: ${protocol['planned_maximum_reservation_usd']:.4f}; local limit $1. Preparation itself makes no API call or training update.",flush=True)
    return protocol


def score():
    p=read_json(OUT/'protocol.json');cases=read_rows(OUT/'requests.jsonl');results=[];used=0.0
    for path in sorted((OUT/'attempts').glob('*.json')):
        a=read_json(path);case=cases[a['sample_id']];errors=[];workflow=None
        try:
            if a.get('api_error') or a['state']!='returned':raise ValueError(a.get('api_error','Unresolved reserved request; do not retry'))
            if a.get('finish_reason')!='STOP':raise ValueError('Incomplete output: '+str(a.get('finish_reason')))
            workflow=json.loads(a['text']);record={**case['scenario'],'expected_output':workflow}
            errors=validate_record('reliability',record)
            if errors:raise ValueError('; '.join(errors))
            for field,value in workflow.items():
                values=value if isinstance(value,list) else [value]
                if any(v not in p['vocabularies'][field] for v in values):raise ValueError('Unknown source workflow label: '+field)
        except (ValueError,KeyError,TypeError) as exc:errors=[str(exc)]
        usage=a.get('usage',{});input_tokens=usage.get('promptTokenCount',0)
        output_tokens=usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0)
        used+=(input_tokens*p['published_rates_per_million']['input']+output_tokens*p['published_rates_per_million']['output'])/1e6
        results.append({'sample_id':a['sample_id'],'family':case['family'],'partition':case['future_partition'],'workflow':workflow,'preflight_pass':not errors,'errors':errors,'usage':usage})
    write_json(OUT/'preflight.json',{'attempted':len(results),'planned':p['n'],'preflight_pass':sum(r['preflight_pass'] for r in results),'items':results,'scope':'Schema/vocabulary checks only; all outputs still require semantic review.'})
    ledger=read_json(OUT/'budget.json') if (OUT/'budget.json').exists() else {'reservations':[]}
    write_json(OUT/'usage.json',{'known_returned_usage_estimate_usd':used,'reserved_upper_estimate_usd':sum(a['reserved_usd'] for a in ledger['reservations']),
        'budget_usd':p['budget_usd'],'calls_reserved':len(ledger['reservations']),'note':'Estimate, not a Google billing balance. Timeouts can have incurred usage and remain fully reserved. Reasoning tokens included. No actual free-credit eligibility claimed.'})
    print(f"Preflight {sum(r['preflight_pass'] for r in results)}/{len(results)}; returned-usage estimate ${used:.6f}. Semantic review required.",flush=True)


def run(limit,read_key):
    protocol=prepare();freeze=read_json(OUT/'freeze.json')
    assert sha(OUT/'protocol.json')==freeze['protocol_sha256'] and sha(OUT/'requests.jsonl')==freeze['requests_sha256']
    if not 1<=limit<=protocol['n']:raise ValueError('Limit must be1..12')
    key=os.environ.get('GEMINI_API_KEY')
    if not key and read_key:key=getpass.getpass('Gemini API key (hidden): ').strip()
    if not key:raise RuntimeError('GEMINI_API_KEY is missing; use --read-key for hidden terminal input')
    attempts=OUT/'attempts';attempts.mkdir(exist_ok=True)
    lock=OUT/'teacher.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        ledger=read_json(OUT/'budget.json') if (OUT/'budget.json').exists() else {'reservations':[]}
        if any(read_json(f).get('api_error') or read_json(f).get('state')!='returned' for f in attempts.glob('*.json')):
            raise RuntimeError('Prior API failure/unresolved request; inspect before another run. No automatic retry.')
        for case in read_rows(OUT/'requests.jsonl')[:limit]:
            path=attempts/f"{case['sample_id']:03}.json"
            if path.exists():continue
            reserve=reservation_usd(case['payload'],protocol)
            if not may_reserve(ledger['reservations'],reserve,protocol):raise RuntimeError('Request or dollar limit reached')
            item={'sample_id':case['sample_id'],'reserved_usd':reserve};ledger['reservations'].append(item)
            write_json(OUT/'budget.json',ledger)  # Reserve before any billable network action.
            result={**item,'state':'reserved','requested_model':MODEL,'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            write_json(path,result);started=time.monotonic()
            request=urllib.request.Request(ENDPOINT,data=json.dumps(case['payload']).encode('utf-8'),headers={'Content-Type':'application/json','x-goog-api-key':key},method='POST')
            try:
                with urllib.request.urlopen(request,timeout=60) as response:raw=json.load(response)
                candidate=raw.get('candidates',[{}])[0]
                result.update(state='returned',model_version=raw.get('modelVersion'),usage=raw.get('usageMetadata',{}),finish_reason=candidate.get('finishReason'),
                    text=''.join(v.get('text','') for v in candidate.get('content',{}).get('parts',[]) if not v.get('thought')))
            except urllib.error.HTTPError as exc:
                message=exc.read().decode('utf-8',errors='replace').replace(key,'[REDACTED]')
                result.update(state='api_error',http_status=exc.code,api_error=message[:2000])
            except Exception as exc:result.update(state='api_error',api_error=str(exc).replace(key,'[REDACTED]'))
            result['seconds']=round(time.monotonic()-started,2);write_json(path,result)
            print(f"Teacher {case['sample_id']+1}/{protocol['n']}: {result['state']}; {result['seconds']}s",flush=True)
            if result['state']!='returned':break
            if result.get('finish_reason')!='STOP':break
    finally:
        key=None
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()
        score()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','score']);parser.add_argument('--limit',type=int,default=12);parser.add_argument('--read-key',action='store_true');args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='score':score()
    else:run(args.limit,args.read_key)
