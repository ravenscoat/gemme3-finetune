"""Visible, read-only monitor; closing it never stops assessment workers."""
from collections import deque
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
history=deque(maxlen=40);previous=None
PKT=timezone(timedelta(hours=5))


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):return {}


def render():
    global previous
    status=read(ROOT/'reports/remaining_evaluation_status.json')
    print('\033[2J\033[H',end='')
    print('GEMMA - SYNTHETIC DATA ASSESSMENT\n'+datetime.now(PKT).strftime('%a %d %b | %I:%M:%S %p Pakistan time'))
    print('No training or weight changes. Ctrl+C closes this monitor only.\n')
    for job in status.get('jobs',[]):print(f"  {job['dataset']:15} {job['phase']}")
    job=next((j for j in status.get('jobs',[]) if j['dataset']==status.get('current_dataset')),None)
    if job:
        p=read(ROOT/job['assessment']/'progress.json');done=p.get('completed',0);total=p.get('total',128)
        phase=p.get('phase','loading');label=p.get('label','')
        key=(job['dataset'],phase,label)
        if key!=previous:history.clear();previous=key
        print(f"\nCURRENT: {job['dataset']} | {phase} {label}")
        print(f'{done}/{total} ({done/total:.1%})')
        if status.get('phase')=='running':
            if not history or history[-1][0]!=done:history.append((done,time.monotonic()))
            if len(history)>1 and history[-1][0]>history[0][0]:
                remaining=(total-done)*(history[-1][1]-history[0][1])/(history[-1][0]-history[0][0])
                eta=datetime.now(PKT)+timedelta(seconds=remaining)
                print(f'Estimated CURRENT stage finish: {eta:%I:%M %p PKT} (~{remaining/60:.0f} min)')
            else:print('Stage ETA: measuring speed...')
        log=Path(job['log'])
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-8000));lines=f.read().decode('utf-8',errors='replace').splitlines()
            print('\nLATEST LOG:\n'+'\n'.join([l for l in lines if l.strip()][-5:]))
    if status.get('phase')=='failed':print('\nSTOPPED: '+status.get('error','Check log'))
    if status.get('phase')=='complete':print('\nALL THREE ASSESSMENTS COMPLETE. Results saved on D:.')
    print('\nRefresh: 5 seconds | reports/remaining_evaluation_status.json',flush=True)
    return status.get('phase')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    try:
        while True:
            phase=render()
            if '--once' in sys.argv or phase=='complete':break
            time.sleep(5)
    except KeyboardInterrupt:print('Monitor closed; background assessments continue.')
