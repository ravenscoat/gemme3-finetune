"""Serialize the three new frozen-model assessments on the local GPU."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from dataset_catalog import CATALOG, ROOT
from json_io import write_json


def main():
    if (ROOT/'reports/multi_dataset_gpu.lock').exists(): raise RuntimeError('Training still owns the GPU')
    lock=ROOT/'reports/assessment_gpu.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    jobs=[{'dataset':key,'assessment':CATALOG[key]['assessment'],'phase':'pending'} for key in ['reliability','mimic_demo','insuranceqa']]
    status={'phase':'starting','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs}
    status_path=ROOT/'reports/remaining_evaluation_status.json'
    child=None
    try:
        for job in jobs:
            if (ROOT/job['assessment']/'assessment.json').exists():
                job['phase']='complete'; continue
            log=ROOT/'reports'/(job['dataset']+'_assessment.log')
            with log.open('ab') as stream:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/assess_dataset.py'),'--dataset',job['dataset'],'--samples','128'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                job.update(phase='running',pid=child.pid,log=str(log));status.update(phase='running',current_dataset=job['dataset'])
                write_json(status_path,status,best_effort=True)
                while child.poll() is None: time.sleep(5)
                if child.returncode:
                    job.update(phase='failed',exit_code=child.returncode)
                    raise RuntimeError(job['dataset']+' assessment failed; inspect '+str(log))
            job['phase']='complete';write_json(status_path,status,best_effort=True)
        status.update(phase='complete',current_dataset=None,finished_utc=datetime.now(timezone.utc).isoformat())
        print('All new assessments completed.',flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            if os.name=='nt': subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
            else: child.terminate()
            child.wait()
        status.update(phase='failed',error=str(exc));raise
    finally:
        try: write_json(status_path,status,best_effort=True)
        finally:
            if lock.exists() and lock.read_text().strip()==str(os.getpid()): lock.unlink()


if __name__=='__main__':main()
