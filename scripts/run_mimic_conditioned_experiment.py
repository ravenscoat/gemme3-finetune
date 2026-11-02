"""Reserve GPU across independent training/comparison subprocesses; release in finally."""
import os
import subprocess
import sys
import time

from dataset_catalog import ROOT,read_json,sha
from json_io import write_json
from mimic_conditioned_data import RUN,prepare
from assess_mimic_conditioned import prepare_plan


def main():
    prepare();prepare_plan()
    if (RUN/'completion.json').exists():print('Experiment already complete.');return
    lock=ROOT/'reports/assessment_gpu.lock'
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():raise RuntimeError('Another training job owns GPU')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    started=time.time()
    try:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_mimic_conditioned.py')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/assess_mimic_conditioned.py')],cwd=ROOT,check=True)
        write_json(RUN/'completion.json',{'status':'complete','elapsed_seconds':time.time()-started,
            'candidate_adapter_sha256':read_json(RUN/'selection.json')['adapter_sha256'],'comparison_sha256':sha(RUN/'pilot_metrics.json'),
            'deployed':False,'new_test_scoring':False})
        write_json(RUN/'progress.json',{'phase':'complete','completed':128,'total':128,'elapsed_seconds':round(time.time()-started,2)},best_effort=True)
    except BaseException as exc:
        write_json(RUN/'progress.json',{'phase':'failed','completed':0,'total':116,'elapsed_seconds':round(time.time()-started,2),'error':str(exc)},best_effort=True);raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
