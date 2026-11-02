"""CPU-only completion supervisor: verify/report even if the chat is closed."""
from datetime import datetime,timezone
import subprocess
import sys
import time

from dataset_catalog import ROOT,read_json
from fidelity_alignment import OUT,ORDER
from json_io import write_json


def main():
    status_path=ROOT/'reports/fidelity_alignment_status.json'
    report_status=ROOT/'reports/fidelity_alignment_report_status.json'
    try:
        write_json(report_status,{'phase':'waiting_for_generation','pid':__import__('os').getpid()})
        while True:
            status=read_json(status_path)
            if status['phase']=='failed':raise RuntimeError('Generation queue failed; raw files preserved: '+str(status.get('error')))
            if status['phase']=='complete':break
            time.sleep(5)
        write_json(report_status,{'phase':'verifying_and_reporting'})
        for script in ['report_fidelity_alignment.py','plot_fidelity_alignment.py']:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
        summary=read_json(OUT/'summary.json')
        bullets=[]
        for key in ORDER:
            arms=summary['datasets'][key]['arms'];b,c=arms['baseline'],arms['controlled']
            bullets.append(f'- {key}: {c["attempted"]} controlled attempts; schema {b["structure_valid"]} → {c["structure_valid"]}; source constraints {b["source_constraint_valid"]} → {c["source_constraint_valid"]}; exact-field/range compliance {c["condition_compliance"]["both"]}/{c["attempted"]}.')
        note=('\n## COMPLETE: source-profile fidelity comparison\n\nAll four fixed batches and automatic CPU verification/reporting completed.\n'
              '348 new raw attempts retained. See `outputs/fidelity-alignment-v1/report.md`,\n'
              '`distributions.html`, `summary.json` and `verification.json`.\n\n'+'\n'.join(bullets)+'\n\n'
              'These are small development comparisons of a strongly controlled pipeline.\n'
              'Do not claim learned source probabilities, full clinical/factual validity or\n'
              'automatic assessment success. Read every regression and warning in the report.\n'
              'No new weights, test scores or deployment. Original adapters verified unchanged.\n')
        for name in ['NEXT_SESSION.md','decision.md']:
            path=ROOT/name;text=path.read_text(encoding='utf-8')
            if note not in text:
                head,sep,tail=text.partition('\n');path.write_text(head+'\n'+note+'\n'+tail,encoding='utf-8')
        write_json(report_status,{'phase':'complete','finished_utc':datetime.now(timezone.utc).isoformat(),
            'report':str(OUT/'report.md'),'plots':str(OUT/'distributions.html')})
        print('All fidelity batches verified, reported, plotted and logged.',flush=True)
    except BaseException as exc:
        write_json(report_status,{'phase':'failed','error':str(exc)});raise


if __name__=='__main__':main()
