"""Visible CMD monitor of saved experiment status/results; does not own GPU jobs."""
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
rates={}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


while True:
    try:
        paths=sorted((ROOT/'outputs').glob('mimic-demo-fidelity-*/progress.json'),key=lambda p:p.stat().st_mtime)
        if os.name=='nt':os.system('cls')
        else:print('\033[2J\033[H',end='',flush=True)
        print('GEMMA - FIDELITY EXPERIMENTS\n')
        print('Refresh:5 seconds. Ctrl+C closes this monitor; GPU workers are separate.\n')
        for path in paths:
            state=read(path)
            print(path.parent.name)
            print('  Phase:',state['phase'],'| '+('Steps:' if state['phase']=='training' else 'Outputs:'),str(state['completed'])+'/'+str(state['total']),
                  '| Elapsed:',str(round(state.get('elapsed_seconds',0)/60,1))+'min')
            if state['phase']=='training':
                key=str(path);history=rates.setdefault(key,[]);history.append((time.monotonic(),state['completed']))
                history[:]=[v for v in history if time.monotonic()-v[0]<90]
                elapsed=history[-1][0]-history[0][0];steps=history[-1][1]-history[0][1]
                if elapsed>20 and steps>0:print('  Training estimate:',round((state['total']-state['completed'])*elapsed/steps/60,1),'min remaining; generation comparison follows.')
                print('  Epoch:',state.get('epoch','-'),'| Best generic validation loss:',state.get('best_validation_loss','not measured yet'))
            if state.get('error'):print('  Error:',state['error'])
        print('\nCOMPLETED COMPARISONS - 64 attempts per arm')
        print(f'{"Approach":<22} {"Valid":>7} {"ICD flags":>10} {"Lab missing C/G/H":>22} {"Age distance":>13}')
        shown=set()
        for path in paths:
            metrics=path.parent/'pilot_metrics.json'
            if not metrics.exists():continue
            for label,m in read(metrics)['arms'].items():
                if label in shown:continue
                shown.add(label);numeric=m['fidelity_vs_training']['numeric']
                missing='/'.join('n/a' if numeric[f]['generated_missing_rate'] is None else f"{100*numeric[f]['generated_missing_rate']:.1f}%" for f in ['creatinine_mg_dl','glucose_mg_dl','hemoglobin_g_dl'])
                distance=numeric['age'].get('wasserstein');distance='n/a' if distance is None else f'{distance:.3f}'
                print(f"{label:<22} {m['structure_valid']:>4}/64 {m['prefix_version_flags']:>10} {missing:>22} {distance:>13}")
        print('\nSource missing C/G/H:9.1% /10.4% /10.4%. Lower age distance is closer.')
        print('Controlled approaches receive source-derived conditions; these are not proof of learned probabilities.')
        active=[read(p)['phase'] for p in paths if read(p)['phase'] not in ['complete','failed']]
        print('\nSTATUS:', 'An experiment is preparing/running.' if active else 'All started experiments have finished. No new training is running.')
        if active:
            log=ROOT/'reports/mimic_conditioned_experiment_stdout.log'
            if log.exists():
                lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
                print('\nLatest training/comparison log:\n'+'\n'.join(lines[-3:]))
        print('Reports and raw outputs: D:\\gemma-finetune\\outputs\\mimic-demo-fidelity-*')
    except (OSError,ValueError,KeyError) as exc:print('Waiting for a complete status file:',exc,flush=True)
    time.sleep(5)
