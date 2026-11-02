"""Read-only VS Code monitor for the frozen-model test assessment."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/final-assessment'


def tail(name, count):
    path = ROOT/'reports'/name
    if path.exists():
        text = '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-count:])
        print(re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text))


try:
    while True:
        print('\033[2J\033[H', end='')
        print('GEMMA FINAL TEST - LIVE MONITOR\n')
        print('Selected full-pass model. Evaluation only; no training.\n')
        try:
            progress = json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            progress = None
        if progress:
            completed, total = progress['completed'], progress['total']
            print(f'Test records assessed: {completed:,}/{total:,} ({completed/total:.1%})')
            print(f'Remaining: {total-completed:,}')
            if completed >= 100:
                seconds = progress['elapsed_this_invocation_seconds'] / completed * (total-completed)
                eta = datetime.now(timezone(timedelta(hours=5))) + timedelta(seconds=seconds)
                print('Approximate finish:', eta.strftime('%I:%M %p'), 'Pakistan time')
                print('Estimate includes observed loading time and changes with speed.')
        else:
            print('Loading the saved model. Progress appears after the first 50 records.')
        print('\nLATEST LOG:')
        tail('final_test.log', 5)
        print('\nLATEST DIAGNOSTICS:')
        tail('final_test_errors.log', 4)
        if (ROOT/'reports/final_test_assessment.md').exists():
            print('\nFINAL TEST ASSESSMENT COMPLETE.')
            print('Report: D:/gemma-finetune/reports/final_test_assessment.md')
            break
        print('\nRefreshes every 5 seconds. Ctrl+C closes this monitor only.', flush=True)
        time.sleep(5)
except KeyboardInterrupt:
    print('\nMonitor closed. Assessment continues independently.')
