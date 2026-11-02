"""Sequential local-GPU jobs. Preserves completed Bitext work and all new checkpoints."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from json_io import write_json


def dump(path, value):
    return write_json(path, value, best_effort=True)


def main():
    lock = ROOT / 'reports/multi_dataset_gpu.lock'
    status_file = ROOT / 'reports/remaining_training_status.json'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    jobs = [
        {'dataset': 'reliability', 'run': 'reliability-4b-v1-windows', 'epochs': 3, 'max_length': 1024, 'checkpoint_every': 25},
        {'dataset': 'mimic_demo', 'run': 'mimic-demo-4b-v2-windows', 'epochs': 3, 'max_length': 1024, 'checkpoint_every': 25},
        {'dataset': 'insuranceqa', 'run': 'insuranceqa-4b-v1-windows', 'epochs': 1, 'max_length': 1536, 'checkpoint_every': 100},
    ]
    status = {'phase': 'starting', 'queue_pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs}
    child = None
    try:
        for job in jobs:
            out = ROOT / 'outputs' / job['run']
            job['phase'] = 'pending'
            if (out / 'training_summary.json').exists():
                job['phase'] = 'already_complete'
                continue
            command = [sys.executable, '-u', str(ROOT / 'scripts/train_dataset.py'), '--dataset', job['dataset'],
                       '--run', job['run'], '--epochs', str(job['epochs']), '--max-length', str(job['max_length']),
                       '--checkpoint-every', str(job['checkpoint_every']), '--eval-limit', '128']
            checkpoints = sorted((out / 'checkpoints').glob('checkpoint-*'), key=lambda p: int(p.name.rsplit('-', 1)[1]))
            checkpoints = [p for p in checkpoints if (p / 'trainer_state.json').exists() and (p / 'optimizer.pt').exists()]
            if checkpoints:
                command += ['--resume', str(checkpoints[-1])]
            log = ROOT / 'reports' / (job['run'] + '.log')
            print('Starting:', job['dataset'], 'Log:', log, flush=True)
            with log.open('ab') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                job.update(phase='training', pid=child.pid, log=str(log))
                status.update(phase='training', current_dataset=job['dataset'])
                dump(status_file, status)
                while child.poll() is None:
                    time.sleep(5)
                if child.returncode:
                    job.update(phase='failed', exit_code=child.returncode)
                    raise RuntimeError(f"{job['dataset']} failed; inspect {log}")
            job['phase'] = 'complete'
            dump(status_file, status)
            print('Completed:', job['dataset'], flush=True)
        status.update(phase='complete', current_dataset=None, finished_utc=datetime.now(timezone.utc).isoformat())
        print('All three adapters saved. Generation evaluation remains a separate step.', flush=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            else:
                child.terminate()
            child.wait()
        status.update(phase='failed', error=str(exc))
        raise
    finally:
        try:
            dump(status_file, status)
        finally:
            if lock.exists() and lock.read_text().strip() == str(os.getpid()):
                lock.unlink()


if __name__ == '__main__':
    main()
