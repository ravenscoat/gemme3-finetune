"""Read-only terminal monitor for the existing training process."""
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bitext-4b-full-pass-windows'
history = deque(maxlen=100)
pakistan = timezone(timedelta(hours=5))


def tail(name, count):
    path = ROOT / 'reports' / name
    if path.exists():
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        print(re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', '\n'.join(lines[-count:])))


try:
    while True:
        print('\033[2J\033[H', end='')
        print('GEMMA TRAINING - LIVE MONITOR\n')
        print('Watching the existing GPU job. Ctrl+C closes this monitor only.\n')
        progress = None
        try:
            progress = json.loads((OUT / 'progress.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if progress:
            now = time.monotonic()
            step, target = progress['global_step'], progress['target_steps']
            if not history or history[-1][0] != step:
                history.append((step, now))
            print(f'Step: {step:,} / {target:,}    Remaining: {target-step:,}')
            print(f'Total pass completed: {step/target:.1%}')
            if history and step - history[0][0] >= 10:
                seconds_per_step = (now-history[0][1]) / (step-history[0][0])
                eta = datetime.now(pakistan) + timedelta(seconds=seconds_per_step*(target-step))
                print('Estimated finish:', eta.strftime('%a %d %b, %I:%M %p'), 'Pakistan time')
                print('Estimate changes with speed; validation and saving can extend it.')
            else:
                print('Estimated finish: measuring speed; available after 10 observed steps.')
            print('Best validation loss:', progress.get('best_validation_loss'))
        else:
            print('Preparing model / checking validation baseline before training updates.')
        print('\nLATEST TRAINING LOG:')
        tail('full_pass_windows.log', 5)
        print('\nLATEST PROGRESS / DIAGNOSTICS:')
        tail('full_pass_windows_errors.log', 3)
        if (OUT / 'training_summary.json').exists():
            summary = json.loads((OUT / 'training_summary.json').read_text(encoding='utf-8'))
            print('\nTRAINING COMPLETE. Selected checkpoint:', summary['best_checkpoint'])
            break
        print('\nRefreshes every 5 seconds.', flush=True)
        time.sleep(5)
except KeyboardInterrupt:
    print('\nMonitor closed. Training continues independently.')
