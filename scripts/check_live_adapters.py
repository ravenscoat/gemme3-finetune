"""One extra live API example per adapter, separate from frozen assessments."""
import json
from pathlib import Path
import sys
import time
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dataset_catalog import validate_record
from json_io import write_json


def api(path,payload=None):
    request=Request('http://127.0.0.1:8765'+path,data=json.dumps(payload).encode() if payload is not None else None,
                    headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=120) as response:return json.load(response)


def main():
    results={'scope':'Extra live frontend API checks; not part of the frozen 128-attempt assessment.','models':{}}
    for key in ['insuranceqa','mimic_demo','reliability','bitext']:
        job=api('/api/generate',{'dataset':key,'prompt':'Generate synthetic data.','temperature':1.,'seed':202611011})
        started=time.monotonic()
        while True:
            current=api('/api/jobs/'+job['id'])
            if current['status']=='error':raise RuntimeError(key+': '+current['error'])
            if current['status']=='complete':break
            if time.monotonic()-started>600:raise RuntimeError(key+' timed out; job may still be running')
            time.sleep(2)
        status=api('/api/status?dataset='+key)
        assert status['loaded_dataset']==key
        assert current['dataset']==key and current['live'] is True
        errors=validate_record(key,current['record']) if current.get('record') else [current.get('validation_error','No record')]
        results['models'][key]={'loaded_dataset':status['loaded_dataset'],'structure_valid':not errors,'errors':errors,'output':current}
        write_json(ROOT/'reports/frontend_live_check.json',results)
        print(key+': live adapter returned output; structure valid='+str(not errors)+'; '+str(current['seconds'])+' seconds',flush=True)
    print('All four live adapter routes checked.',flush=True)


if __name__=='__main__':main()
