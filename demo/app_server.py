"""Local multi-dataset demo backed by the four saved QLoRA adapters."""
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import gc
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs,urlparse
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dataset_catalog import CATALOG,core,data,parse_output,read_json,read_rows,sha,signature,source_warnings,text_at,validate_record,value_at
from bitext_results import metrics as bitext_metrics
from development_evidence import development_metrics,development_saved

STATIC=Path(__file__).resolve().parent/'static'
MODEL=TOKENIZER=MODEL_KEY=None
ENGINE={'state':'idle','error':None}
GPU_LOCK=threading.Lock();JOB_LOCK=threading.Lock();SEARCH_LOCK=threading.Lock()
JOBS={};SEARCH={}


def reserved():
    if (ROOT/'reports/assessment_gpu.lock').exists():return 'evaluating'
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():return 'training'
    return None


@lru_cache(maxsize=4)
def known_records(key):
    source=data(key);rows=[core(key,r) for r in source['train']]
    return {field:{signature(value_at(r,path)) for r in rows} if path else set()
            for field,path in [('record',None),('request',source['spec']['request']),('response',source['spec']['reply'])]}|{'record':{signature(r) for r in rows}}


def seen(key,record):
    source=data(key);known=known_records(key)
    return {'record_seen':signature(record) in known['record'],
            'request_seen':signature(value_at(record,source['spec']['request'])) in known['request'] if known['request'] else None,
            'response_seen':signature(value_at(record,source['spec']['reply'])) in known['response'] if known['response'] else None}


def related(key,record):
    source=data(key);rows=source['train'];path=source['spec']['request']
    if path:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        with SEARCH_LOCK:
            if key not in SEARCH:
                vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=20000,dtype=np.float32)
                SEARCH[key]=(vectorizer,vectorizer.fit_transform([text_at(r,path) for r in rows]))
        vectorizer,matrix=SEARCH[key]
        scores=(matrix@vectorizer.transform([text_at(record,path)]).T).toarray().ravel()
        indices=scores.argsort()[-3:][::-1];method='lexical similarity'
    else:
        numeric=source['schema']['numeric_fields'];ranges={f:max(r[f] for r in rows if r[f] is not None)-min(r[f] for r in rows if r[f] is not None) for f in numeric}
        def similarity(other):
            distances=[]
            for f in source['schema']['fields']:
                a,b=record[f],other[f]
                if a is None or b is None:distances.append(0 if a is b else 1)
                elif f in numeric:distances.append(min(1,abs(a-b)/(ranges[f] or 1)))
                else:distances.append(0 if a==b else 1)
            return 1-sum(distances)/len(distances)
        scores=[similarity(r) for r in rows];indices=sorted(range(len(rows)),key=lambda i:scores[i],reverse=True)[:3];method='field similarity'
    return [{'record':core(key,rows[int(i)]),'source_line':rows[int(i)].get('source_line',rows[int(i)].get('source_id')),
             'similarity':float(scores[int(i)]),'similarity_method':method} for i in indices]


def metadata(key):
    source=data(key);spec=source['spec'];rows=source['train'];group=spec['group']
    schema=source['schema'];groups=sorted({str(value_at(r,group)) for r in rows})
    selection=read_json(ROOT/spec['selection'])
    checkpoint=Path(selection.get('best_checkpoint',selection.get('adapter',''))).name
    if key=='bitext':checkpoint='checkpoint-3900'
    return {'dataset':key,'label':spec['label'],'kind':spec['kind'],'scope':spec['scope'],
            'training_rows':len(rows),'group_field':group,'groups':groups,'fields':schema['fields'],
            'intents':schema.get('intent_category',{}),'categories':dict(Counter(str(value_at(r,group)) for r in rows)),
            'checkpoint':checkpoint,'epochs':3 if key in ['reliability','mimic_demo'] else 1,
            'saved_count':len(saved_rows(key)), 'model':'Gemma 3 4B'}


def saved_rows(key):
    if key=='bitext':return read_rows(ROOT/'outputs/bitext-4b-full-pass-evaluation-500-windows/tuned_unfenced_manual_review.jsonl')
    path=ROOT/CATALOG[key]['assessment']/'tuned_records.jsonl'
    try:return read_rows(path) if path.exists() else []
    except (OSError,json.JSONDecodeError):return []


def saved(key,index):
    rows=saved_rows(key)
    if not rows:
        complete=(ROOT/CATALOG[key]['assessment']/'assessment.json').exists()
        return {'pending':not complete,'empty':complete,'dataset':key,'record':None,'references':[],'live':False}
    index=index%len(rows);row=rows[index];record=core(key,row) if key=='bitext' else row['record']
    return {'dataset':key,'index':index,'sample_id':row['sample_id'],'record':record,'live':False,
            'references':related(key,record),'source_warnings':source_warnings(key,record),**seen(key,record)}


def primary_metrics(key):
    if key=='bitext':return {'dataset':key,'kind':'support','complete':True,**bitext_metrics()}
    out=ROOT/CATALOG[key]['assessment'];path=out/'assessment.json';partial=out/'generation_metrics.json'
    if path.exists():return {'dataset':key,'kind':CATALOG[key]['kind'],'complete':True,**read_json(path)}
    return {'dataset':key,'kind':CATALOG[key]['kind'],'complete':False,
            'progress':read_json(out/'progress.json') if (out/'progress.json').exists() else {},
            **(read_json(partial) if partial.exists() else {})}


def advanced_metrics(key):
    directory=ROOT/'reports/advanced_analysis/v1';result={}
    path=directory/'ai_review_summary.json'
    if path.exists():
        review=read_json(path)
        if key in review['counts']:
            counts=review['counts'][key]
            result['review']={'counts':counts,'n':sum(counts.values()),'limits':review['limits'],
                'judgments':[r for r in read_rows(directory/'ai_judgments.jsonl') if r['dataset']==key]}
    path=directory/'prdc_results.json'
    if path.exists():
        values=read_json(path)
        if key in values['datasets']:result['prdc']={'fields':values['datasets'][key],'limits':values['limits']}
    path=directory/'sdmetrics_results.json'
    if key=='mimic_demo' and path.exists():
        values=read_json(path)
        result['sdmetrics']={'collections':{label:{k:r[k] for k in ['n','overall_quality','properties','per_column']} for label,r in values['collections'].items()},'limits':values['limits']}
    path=directory/'exploratory_results.json'
    if path.exists():
        values=read_json(path)
        if key=='mimic_demo':result['icd']={label:{k:r[k] for k in ['n','flagged']} for label,r in values['icd_prefix_version'].items() if isinstance(r,dict)}
        if key=='insuranceqa':result['factual_spot_checks']=values['factual_spot_checks']
    return result


def metrics(key):
    return {**primary_metrics(key),'advanced':advanced_metrics(key),'development':development_metrics(key)}


def load_model(key):
    global MODEL,TOKENIZER,MODEL_KEY
    if MODEL is not None and MODEL_KEY==key:return
    ENGINE.update(state='loading',error=None)
    from unsloth import FastModel
    import torch
    if MODEL is not None:
        MODEL=None;TOKENIZER=None;MODEL_KEY=None;gc.collect();torch.cuda.empty_cache()
    spec=CATALOG[key];selected=read_json(ROOT/spec['selection'])
    if sha(Path(selected['adapter'])/'adapter_model.safetensors')!=selected['adapter_sha256']:raise RuntimeError('Frozen adapter hash changed')
    model,processor=FastModel.from_pretrained(model_name=selected['adapter'],max_seq_length=spec['context'],load_in_4bit=True,full_finetuning=False)
    FastModel.for_inference(model);model.eval()
    MODEL,TOKENIZER,MODEL_KEY=model,getattr(processor,'tokenizer',processor),key
    ENGINE.update(state='ready',error=None)


def patch_job(ident,**values):
    with JOB_LOCK:JOBS[ident].update(values)


def run_job(ident,request):
    started=time.time();thread=None;stop=threading.Event();key=request['dataset']
    try:
        patch_job(ident,status='loading' if MODEL_KEY!=key else 'preparing');load_model(key)
        if request.get('action')=='load':
            patch_job(ident,status='complete',seconds=round(time.time()-started,2));return
        import torch
        from transformers import TextIteratorStreamer,set_seed,StoppingCriteria,StoppingCriteriaList
        class StopRequested(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):return stop.is_set()
        source=data(key);prompt=source['prompt'];custom=request['prompt'].strip();group=request.get('intent','')
        if custom.casefold().rstrip('.')!='generate synthetic data':prompt+='\nAdditional generation request: '+custom
        if group:prompt+='\nGenerate the record with '+source['spec']['group']+' equal to '+group+'.'
        text=TOKENIZER.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        inputs=TOKENIZER(text,add_special_tokens=False,return_tensors='pt').to('cuda')
        available=source['spec']['context']-inputs['input_ids'].shape[1]-2
        if available<300:raise ValueError('Please shorten the prompt to leave room for a complete record.')
        maximum=min(600,available) if key=='bitext' else available
        streamer=TextIteratorStreamer(TOKENIZER,skip_prompt=True,skip_special_tokens=True,timeout=180.)
        results=[];errors=[];set_seed(request['seed'])
        def generate():
            try:
                with torch.inference_mode():results.append(MODEL.generate(**inputs,streamer=streamer,max_new_tokens=maximum,
                    do_sample=True,temperature=request['temperature'],top_p=.95,top_k=64,pad_token_id=TOKENIZER.pad_token_id,
                    use_cache=True,stopping_criteria=StoppingCriteriaList([StopRequested()])))
            except Exception as exc:errors.append(exc);streamer.end()
        ENGINE['state']='generating';patch_job(ident,status='generating')
        thread=threading.Thread(target=generate,daemon=True);thread.start();partial=''
        for chunk in streamer:partial+=chunk;patch_job(ident,text=partial)
        thread.join()
        if errors:raise errors[0]
        record=None;validation_error=None;warnings=[]
        try:
            candidate=parse_output(partial);errors=validate_record(key,candidate)
            if errors:raise ValueError('; '.join(errors))
            record=candidate;warnings=source_warnings(key,record)
        except (ValueError,TypeError,KeyError) as exc:validation_error=str(exc)
        tokens=int(results[0].shape[1]-inputs['input_ids'].shape[1])
        result={'dataset':key,'status':'complete','live':True,'text':partial,'record':record,'validation_error':validation_error,
                'source_warnings':warnings,'seconds':round(time.time()-started,2),'new_tokens':tokens,'hit_token_limit':tokens>=maximum,
                'requested_intent':group,'seed':request['seed'],'temperature':request['temperature']}
        if record:result.update(references=related(key,record),**seen(key,record))
        patch_job(ident,**result);ENGINE['state']='ready'
    except Exception as exc:
        import traceback
        traceback.print_exc();patch_job(ident,status='error',error=str(exc));ENGINE.update(state='ready' if MODEL is not None else 'error',error=str(exc))
    finally:
        stop.set()
        if thread is not None:thread.join()
        GPU_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    def send_json(self,value,status=200):
        payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)))
        self.end_headers();self.wfile.write(payload)

    def do_GET(self):
        parsed=urlparse(self.path);query=parse_qs(parsed.query);key=query.get('dataset',['bitext'])[0]
        try:
            if key not in CATALOG:raise ValueError('Unknown dataset')
            if parsed.path=='/api/catalog':return self.send_json({k:{f:s[f] for f in ['label','kind','scope']} for k,s in CATALOG.items()})
            if parsed.path=='/api/status':
                state=reserved() or ('idle' if MODEL_KEY!=key and ENGINE['state']=='ready' else ENGINE['state'])
                return self.send_json({'engine':{'state':state,'error':ENGINE['error']},'model':'Gemma 3 4B','checkpoint':3900 if key=='bitext' else metadata(key)['checkpoint'],'dataset':key,'loaded_dataset':MODEL_KEY})
            if parsed.path=='/api/meta':return self.send_json(metadata(key))
            if parsed.path=='/api/metrics':return self.send_json(metrics(key))
            if parsed.path=='/api/development':return self.send_json(development_metrics(key))
            if parsed.path=='/api/development/saved':return self.send_json(development_saved(key,int(query.get('index',['0'])[0])))
            if parsed.path=='/api/saved':return self.send_json(saved(key,int(query.get('index',['0'])[0])))
            if parsed.path=='/api/dataset':
                source=data(key);term=query.get('q',[''])[0].casefold();group=query.get('intent',[''])[0]
                rows=[r for r in source['train'] if (not group or str(value_at(r,source['spec']['group']))==group) and (not term or term in json.dumps(core(key,r),ensure_ascii=False).casefold())]
                offset=max(0,int(query.get('offset',['0'])[0]));limit=min(20,max(1,int(query.get('limit',['8'])[0])))
                return self.send_json({'total':len(rows),'offset':offset,'records':[{'record':core(key,r),'source_line':r.get('source_line',r.get('source_id'))} for r in rows[offset:offset+limit]]})
            if parsed.path.startswith('/api/jobs/'):
                with JOB_LOCK:job=dict(JOBS[parsed.path.rsplit('/',1)[-1]]) if parsed.path.rsplit('/',1)[-1] in JOBS else None
                return self.send_json(job or {'error':'Job not found'},200 if job else 404)
            evidence={'/evidence/criteria':(ROOT/'reports/criteria_evidence/v1/index.html','text/html'),
                '/evidence/simple-report':(ROOT/'output/pdf/gemma_report.pdf','application/pdf'),
                '/evidence/report-text':(ROOT/'reports/gemma_report.md','text/plain'),
                '/evidence/distributions':(ROOT/'outputs/prefix-alignment-v1/distributions.html','text/html'),
                '/evidence/final-reliability-trial':(ROOT/'outputs/reliability-consistency-v1/report.md','text/plain'),
                '/evidence/assessment-report':(ROOT/'reports/assessment_snapshot/report.md','text/plain'),
                '/evidence/workflow-review':(ROOT/'reports/reliability_workflow_audit/v1/report.md','text/plain')}
            if parsed.path in evidence:
                file,kind=evidence[parsed.path];payload=file.read_bytes();self.send_response(200)
                self.send_header('Content-Type',kind+('; charset=utf-8' if kind.startswith('text/') else ''));self.send_header('Content-Length',str(len(payload)))
                self.end_headers();return self.wfile.write(payload)
            assets={'/':'simple.html','/app.js':'simple.js','/style.css':'simple.css',
                '/details':'detailed.html','/detailed.js':'detailed.js','/detailed.css':'detailed.css'}
            if parsed.path in assets:
                file=STATIC/assets[parsed.path];payload=file.read_bytes();self.send_response(200)
                self.send_header('Content-Type',{'html':'text/html','js':'text/javascript','css':'text/css'}[file.suffix[1:]]+'; charset=utf-8')
                self.send_header('Content-Length',str(len(payload)));self.end_headers();return self.wfile.write(payload)
            self.send_json({'error':'Not found'},404)
        except (ValueError,KeyError,TypeError) as exc:self.send_json({'error':str(exc)},400)
        except Exception as exc:self.send_json({'error':str(exc)},500)

    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send_json({'error':'Cross-origin requests are not accepted'},403)
        if self.path!='/api/generate':return self.send_json({'error':'Not found'},404)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=8192:raise ValueError('Invalid request size')
            request=json.loads(self.rfile.read(size))
            if not isinstance(request,dict):raise ValueError('Request must be a JSON object')
            key=request.get('dataset','bitext')
            if key not in CATALOG:raise ValueError('Unknown dataset')
            request['dataset']=key
            if request.get('action')!='load':
                if not isinstance(request.get('prompt'),str) or not 0<len(request['prompt'].strip())<=1000:raise ValueError('Enter a prompt of 1–1,000 characters')
                group=request.get('intent','')
                if group and group not in metadata(key)['groups']:raise ValueError('Unknown focus value')
                request['temperature']=float(request.get('temperature',1.));request['seed']=int(request.get('seed',202610300))
                if not .1<=request['temperature']<=1.5 or not 0<=request['seed']<2**32:raise ValueError('Temperature or seed out of range')
            if reserved():return self.send_json({'error':'The GPU is reserved for assessment or training. Saved outputs and source browsing are available.'},409)
            if not GPU_LOCK.acquire(blocking=False):return self.send_json({'error':'The model is busy; wait for this request to finish.'},409)
            ident=uuid.uuid4().hex
            with JOB_LOCK:
                if len(JOBS)>=100:JOBS.pop(next(iter(JOBS)))
                JOBS[ident]={'id':ident,'dataset':key,'status':'queued','text':''}
            threading.Thread(target=run_job,args=(ident,request),daemon=True).start();self.send_json({'id':ident},202)
        except (ValueError,TypeError,KeyError) as exc:self.send_json({'error':str(exc)},400)


def main():
    server=ThreadingHTTPServer(('127.0.0.1',8765),Handler)
    print('Simile: http://127.0.0.1:8765',flush=True);server.serve_forever()


if __name__=='__main__':main()
