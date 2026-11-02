"""Read-only evidence views; never select or load a different live adapter."""
from pathlib import Path

from dataset_catalog import ROOT,read_json,read_rows,sha

SNAPSHOT=ROOT/'reports/assessment_snapshot/snapshot.json'
AUDIT=ROOT/'reports/reliability_workflow_audit/v1'


def development_metrics(key):
    if not SNAPSHOT.exists():return None
    snapshot=read_json(SNAPSHOT);item=snapshot['datasets'].get(key)
    if item is None:return None
    result={**item,'scope':snapshot['scope']}
    if key=='reliability':
        result['final_trial']=snapshot.get('final_reliability_attempt')
        diagnostics=read_json(AUDIT/'diagnostics.json')['prefix']
        result['review']={**read_json(AUDIT/'review_summary.json'),
            'judgments':[{k:w[k] for k in ['sample_id','verdict','reason']} for w in read_rows(AUDIT/'ai_judgments.jsonl')]}
        result['workflow_diagnostics']={k:v for k,v in diagnostics.items() if k!='findings'}
    return result


def development_saved(key,index):
    item=development_metrics(key)
    if item is None:raise ValueError('Development evidence is unavailable')
    path=(ROOT/item['records_path']).resolve()
    if not path.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Invalid evidence path')
    if sha(path)!=item['records_sha256']:raise ValueError('Frozen evidence cohort changed; rebuild and verify its snapshot')
    rows=read_rows(path)
    if not rows:raise ValueError('No valid records in this cohort')
    index=index%len(rows);wrapper=rows[index]
    judgment=next((w for w in item.get('review',{}).get('judgments',[]) if w['sample_id']==wrapper['sample_id']),None)
    return {'dataset':key,'index':index,'total':len(rows),'sample_id':wrapper['sample_id'],'record':wrapper['record'],
        'source_warnings':wrapper.get('source_warnings',[]),'whole_record_match':wrapper.get('exact_training_record_match'),
        'request_match':wrapper.get('request_seen'),'reply_match':wrapper.get('response_seen'),
        'review':judgment,'policy':item['policy'],'live':False,'development':True}
