"""Read-only live monitor; closing this terminal does not stop the GPU worker."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--conditions-only',action='store_true');args=parser.parse_args()
OUT=ROOT/('outputs/mimic-demo-fidelity-conditions-only-v1' if args.conditions_only else 'outputs/mimic-demo-fidelity-pilot-v1')
while True:
    state=json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
    print('\033[2J\033[H',end='')
    print('GEMMA: MIMIC FIDELITY IMPROVEMENT PILOT\n')
    print('Saved model; new generation only. No retraining or test scoring.')
    print('Source conditions without extra ICD wording.' if args.conditions_only else 'Arms: clearer code/version rule, then source-conditioned generation.')
    print('Matched baseline:64 previous attempts. New attempts:',state['total'],'\n')
    print('Phase:',state['phase'],'| Arm:',state.get('arm','loading'))
    print('New outputs:',state['completed'],'/',state['total'],'| Elapsed:',round(state.get('elapsed_seconds',0)/60,1),'min')
    if state.get('error'):print('Error:',state['error'])
    log=ROOT/('reports/mimic_fidelity_conditions_only_stdout.log' if args.conditions_only else 'reports/mimic_fidelity_pilot_stdout.log')
    if log.exists():print('\nLatest log:\n'+'\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-4:]))
    print('\nCtrl+C closes this monitor; the background worker continues.')
    if state['phase'] in ['complete','failed']:break
    time.sleep(5)
