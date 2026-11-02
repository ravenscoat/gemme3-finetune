"""Read-only terminal view of the existing three-dataset training queue."""
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
PKT = timezone(timedelta(hours=5))
history = deque(maxlen=60)
previous = None


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def render():
    global previous
    status = read(ROOT / 'reports/remaining_training_status.json')
    print('\033[2J\033[H', end='')
    print('GEMMA - THREE DATASET TRAINING QUEUE')
    print(datetime.now(PKT).strftime('%a %d %b %Y | %I:%M:%S %p Pakistan time'))
    print('Watching existing jobs. Ctrl+C closes this monitor; training continues.\n')
    jobs = status.get('jobs', [])
    for i, job in enumerate(jobs, 1):
        phase = job.get('phase', 'pending')
        print(f"  {i}. {job['dataset']:14} {phase:17} {job['epochs']} epoch(s)")
    active = status.get('current_dataset')
    job = next((j for j in jobs if j['dataset'] == active), None)
    active_key = job.get('run') if job else None
    if active_key != previous:
        history.clear()
        previous = active_key
    if job:
        progress = read(ROOT / 'outputs' / job['run'] / 'progress.json')
        step, total = progress.get('global_step', 0), progress.get('target_steps', 0)
        print(f"\nCURRENT: {active} | {job['run']}")
        if total:
            fraction = step / total
            bar = '#' * int(fraction * 32) + '-' * (32 - int(fraction * 32))
            print(f'[{bar}] {fraction:.1%} | Step {step:,} / {total:,} | Remaining {max(0,total-step):,}')
            elapsed = progress.get('elapsed_seconds')
            if elapsed is not None and (not history or history[-1][0] != step):
                history.append((step, elapsed))
            print(f"Epoch: {progress.get('epoch', 0):.2f} | Best validation loss: {progress.get('best_validation_loss')}")
            if status.get('phase') != 'training' or job.get('phase') != 'training':
                print('No active training ETA; the progress above is the last saved update.')
            elif len(history) > 1 and history[-1][0] - history[0][0] >= 5:
                seconds_per_step = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
                remaining = max(0, total - step) * seconds_per_step
                eta = datetime.now(PKT) + timedelta(seconds=remaining)
                print(f'Estimated CURRENT dataset finish: {eta:%I:%M %p PKT} (~{remaining/60:.0f} min)')
                print('Estimate changes with validation/checkpoint saving. Queued datasets run afterward.')
            else:
                print('Current-dataset ETA: measuring training speed; available after 5 observed steps.')
        else:
            print('Loading model / tokenizing / measuring baseline validation loss...')
        print('\nLATEST LOG:')
        log = ROOT / 'reports' / (job['run'] + '.log')
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 16000))
                text = stream.read().decode('utf-8', errors='replace')
            lines = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text).splitlines()
            print('\n'.join([line for line in lines if line.strip()][-6:]))
    phase = status.get('phase', 'waiting')
    if phase == 'complete':
        print('\nALL THREE TRAINING RUNS COMPLETE. Adapters and summaries are saved.')
        print('Synthetic generation and final assessment are the next stage.')
    elif phase == 'failed':
        print('\nQUEUE STOPPED:', status.get('error', 'Check the job log.'))
        print('Existing checkpoints are preserved. This monitor will not restart training.')
    print('\nRefresh: 5 seconds | Full status: reports/remaining_training_status.json', flush=True)
    return phase


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    try:
        while True:
            phase = render()
            if '--once' in sys.argv or phase == 'complete':
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nMonitor closed. Training continues independently.')
