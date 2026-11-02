"""Read-only live display of the existing full-pass evaluation job."""
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bitext-4b-full-pass-evaluation-500-windows'
history = deque(maxlen=60)


def tail(path, count):
    if path.exists():
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        print(re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', '\n'.join(lines[-count:])))


try:
    while True:
        print('\033[2J\033[H', end='')
        print('GEMMA FULL-PASS EVALUATION - LIVE MONITOR\n')
        print('Watching the existing job. Ctrl+C closes this monitor only.\n')
        target = OUT / 'tuned_generations.jsonl'
        count = len(target.read_text(encoding='utf-8').splitlines()) if target.exists() else 0
        print('Original model: 500/500 reused from saved comparison')
        print(f'Full-pass model: {count}/500 generated ({count/500:.1%})')
        now = time.monotonic()
        if not history or history[-1][0] != count:
            history.append((count, now))
        if count < 500 and count - history[0][0] >= 12:
            seconds = (now-history[0][1]) / (count-history[0][0]) * (500-count)
            eta = datetime.now(timezone(timedelta(hours=5))) + timedelta(seconds=seconds)
            print('Approximate generation finish:', eta.strftime('%I:%M %p'), 'Pakistan time')
            print('Automatic scoring follows; estimate changes with speed and interruptions.')
        elif count < 500:
            print('Loading model / measuring speed. Estimate appears after 12 observed examples.')
        print('\nStages:')
        for title, path in [('Generation', target), ('Strict format checks', OUT/'evaluation.json'),
                            ('Distribution checks', OUT/'evaluation_unfenced.json'),
                            ('Lexical/style analysis', OUT/'extended_evaluation.json'),
                            ('MAUVE', ROOT/'reports/bitext-4b-full-pass-evaluation-500-windows_mauve.md'),
                            ('Final comparison report', ROOT/'reports/training_stage_comparison.md')]:
            done = count == 500 if title == 'Generation' else path.exists()
            print(f'  {"DONE" if done else "PENDING"}: {title}')
        print('\nLATEST LOG:')
        tail(ROOT/'reports/full_pass_evaluation.log', 5)
        print('\nLATEST DIAGNOSTICS:')
        tail(ROOT/'reports/full_pass_evaluation_errors.log', 3)
        if (ROOT/'reports/training_stage_comparison.md').exists():
            print('\nAUTOMATIC COMPARISON COMPLETE. Human semantic review remains pending.')
            break
        print('\nRefreshes every 5 seconds.', flush=True)
        time.sleep(5)
except KeyboardInterrupt:
    print('\nMonitor closed. Evaluation continues independently.')
