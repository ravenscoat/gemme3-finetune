"""Own the GPU for two small continuation arms and their fixed development checks."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataset_catalog import ROOT,read_json,sha
from json_io import write_json

OUT=ROOT/'outputs/reliability-distillation-v1'


def main():
    script=ROOT/'scripts/distillation_student_pilot.py'
    subprocess.run([sys.executable,str(script),'prepare','--experiment-root',str(OUT)],check=True)
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=3) as response:demo=json.load(response)
    except (OSError,ValueError):demo=None
    if demo and (demo.get('loaded_dataset') or demo['engine']['state'] in ['loading','generating']):
        raise RuntimeError('The demo has a live model loaded; release it before this separate GPU pilot')
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():raise RuntimeError('Another training queue owns the GPU')
    lock=ROOT/'reports/assessment_gpu.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time();status={'phase':'running','experiment_root':str(OUT),'current':None,'completed_jobs':[],'automatic_adoption':False}
    def save():
        status['elapsed_seconds']=round(time.time()-started,1);write_json(ROOT/'reports/distillation_status.json',status,best_effort=True)
    try:
        for stage,arm in [('evaluate','baseline'),('train','replay'),('train','distilled'),('evaluate','replay'),('evaluate','distilled')]:
            status['current']=stage+'_'+arm;save();print('Starting',status['current'],flush=True)
            with (OUT/(stage+'_'+arm+'.log')).open('a',encoding='utf-8') as stream:
                subprocess.run([sys.executable,str(script),stage,'--arm',arm,'--experiment-root',str(OUT)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            status['completed_jobs'].append(status['current']);save();print('Finished',status['current'],flush=True)
        p=read_json(OUT/'student_protocol.json')
        assert sha(__import__('pathlib').Path(p['baseline_adapter'])/'adapter_model.safetensors')==p['baseline_adapter_sha256']
        status.update(phase='complete_pending_semantic_review',current=None);save()
    except BaseException as exc:
        status.update(phase='failed',error=str(exc));save();raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment-root',type=__import__('pathlib').Path,required=True);args=parser.parse_args();OUT=args.experiment_root.resolve()
    if not OUT.is_relative_to((ROOT/'outputs').resolve()):raise RuntimeError('Experiment must stay within project outputs')
    main()
