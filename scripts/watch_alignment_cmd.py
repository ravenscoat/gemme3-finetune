"""Read-only CMD display for the dataset-fidelity generation queue."""
import json
import os
from pathlib import Path
import math
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
while True:
    try:
        runs={'fidelity_alignment_status.json':('fidelity-alignment-v1','fidelity_alignment_stdout.log'),
            'schema_prompt_pilot_status.json':('schema-prompt-pilot-v1','schema_prompt_pilot_stdout.log'),
            'prefix_prompt_pilot_status.json':('prefix-prompt-pilot-v1','prefix_prompt_pilot_stdout.log'),
            'prefix_alignment_status.json':('prefix-alignment-v1','prefix_alignment_stdout.log'),
            'reliability_consistency_status.json':('reliability-consistency-v1','reliability_consistency_stdout.log')}
        available=[ROOT/'reports'/name for name in runs if (ROOT/'reports'/name).exists()]
        status_path=max(available,key=lambda p:p.stat().st_mtime)
        status_file=status_path.name
        prefix_pilot=status_file=='prefix_prompt_pilot_status.json'
        pilot=status_file!='fidelity_alignment_status.json'
        experiment,log_name=runs[status_file]
        status=json.loads((ROOT/'reports'/status_file).read_text(encoding='utf-8'))
        os.system('cls')
        print('GEMMA - TRAINING-DATA FIDELITY\n')
        print('Experiment:',experiment,'\n')
        print('Goal:match training ranges, frequencies and characteristics.')
        print('Generation only. No new training or final-test scoring.\n')
        for key,state in status['datasets'].items():
            path=ROOT/'outputs'/experiment/key/'progress.json'
            progress=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            done=progress.get('completed',0)
            print(f"{key:<14} {state:<10} {done:>3}/{status['counts'][key]} {progress.get('phase','')}")
            if state=='running':
                print('  Current dataset elapsed:',round(progress.get('elapsed_seconds',0)/60,1),'min')
                raw_path=path.parent/('generations.jsonl' if experiment=='reliability-consistency-v1' else 'controlled_generations.jsonl')
                if progress.get('phase')=='generation' and done and raw_path.exists():
                    raw=[json.loads(s) for s in raw_path.read_text(encoding='utf-8').splitlines() if s.strip()]
                    batches=[r['batch_seconds'] for r in raw if r['sample_id']%4==0][-3:]
                    if batches:print('  Estimated remaining generation for this dataset:',round(math.ceil((status['counts'][key]-done)/4)*statistics.median(batches)/60,1),'min; queued datasets follow')
        print('\nCurrent:',status['current'] or 'none','| Queue:',status['phase'])
        print('Elapsed:',round(status.get('elapsed_seconds',0)/60,1),'min (status refreshed at stage boundaries).')
        report_status=ROOT/'reports/fidelity_alignment_report_status.json'
        if report_status.exists() and not pilot:
            report=json.loads(report_status.read_text(encoding='utf-8'))
            print('Verification/report:',report.get('phase','unknown'))
            if report.get('error'):print('REPORT ERROR:',report['error'])
            if report.get('phase')=='complete':print('Results: outputs/fidelity-alignment-v1/report.md and distributions.html')
        print('\nLatest generation log:')
        log=ROOT/'reports'/log_name
        if log.exists():print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-4:]))
        if status.get('error'):print('\nERROR:',status['error'])
        if pilot and status['phase']=='complete' and (ROOT/'outputs'/experiment/'report.md').exists():print('Results: outputs/'+experiment+'/report.md')
        print('\nConditions are externally selected from training statistics.')
        print('All failures and raw outputs are retained. Ctrl+C closes only this monitor.')
    except (OSError,ValueError,KeyError) as exc:print('Waiting for status:',exc,flush=True)
    time.sleep(5)
