"""Sequential GPU queue across all four trained dataset adapters."""
import os
import subprocess
import sys
import time

from dataset_catalog import ROOT,read_json
from fidelity_alignment import OUT,ORDER,SIZES
from json_io import write_json


def main():
    lock=ROOT/'reports/assessment_gpu.lock'
    if (ROOT/'reports/multi_dataset_gpu.lock').exists():raise RuntimeError('Training owns GPU')
    for key in ORDER:
        if not (OUT/key/'protocol.json').exists():raise RuntimeError('Freeze all protocols before starting')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    started=time.time();status={'phase':'running','datasets':{key:'pending' for key in ORDER},'counts':SIZES,'current':None}
    def save():status['elapsed_seconds']=round(time.time()-started,2);write_json(ROOT/'reports/fidelity_alignment_status.json',status,best_effort=True)
    try:
        for key in ORDER:
            if (OUT/key/'completion.json').exists():status['datasets'][key]='complete';save();continue
            status.update(current=key);status['datasets'][key]='running';save()
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/generate_fidelity_alignment.py'),'--dataset',key],cwd=ROOT,check=True)
            status['datasets'][key]='complete';save()
        status.update(phase='complete',current=None);save();print('All four fidelity batches complete.',flush=True)
    except BaseException as exc:
        if status['current']:status['datasets'][status['current']]='failed'
        status.update(phase='failed',error=str(exc));save();raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
