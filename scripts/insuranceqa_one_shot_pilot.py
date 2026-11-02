"""One added demonstration versus frozen source-grounded prompts; no model updates."""
import argparse
import json
from pathlib import Path
import shutil

from dataset_catalog import ROOT,read_json,sha
from json_io import write_json
import insuranceqa_final_pilot as engine

OUT=ROOT/'outputs/insuranceqa-one-shot-v1'
PRIOR=ROOT/'outputs/insuranceqa-final-grounded-v1'
DEMONSTRATION='''Worked example (fictional policy, demonstrating the transformation only):
Example source:
{"domain":"auto-insurance","passage":"This example policy covers towing after an accident, but excludes towing after a mechanical breakdown."}
Example output:
{"domain":"auto-insurance","question":"Does this policy pay for towing if my engine fails?","answer":"No. Its towing benefit applies after an accident; mechanical breakdowns are excluded."}
End of worked example. The actual task and source follow.

'''

def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen; do not prepare again')
    old=read_json(PRIOR/'protocol.json');freeze=read_json(PRIOR/'freeze.json')
    assert sha(PRIOR/'protocol.json')==freeze['protocol_sha256']
    assert sha(PRIOR/'grounded_generations.jsonl')==read_json(PRIOR/'generation_complete.json')['raw_sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(PRIOR/'grounded_generations.jsonl',OUT/'baseline_generations.jsonl')
    p={**old,'experiment':'one-shot-v1','comparison':'Same32 source anchors; zero-shot grounding versus identical prompt prefixed with one fictional worked example.',
       'input_hashes':{**old['input_hashes'],str((PRIOR/'protocol.json').relative_to(ROOT)):sha(PRIOR/'protocol.json'),
           str((PRIOR/'grounded_generations.jsonl').relative_to(ROOT)):sha(PRIOR/'grounded_generations.jsonl'),
           str((PRIOR/'judgments.jsonl').relative_to(ROOT)):sha(PRIOR/'judgments.jsonl')},
       'prompts':[DEMONSTRATION+prompt for prompt in old['prompts']],
       'demonstration':DEMONSTRATION,'demonstration_source':'Explicitly fictional illustration; not extracted from validation/test or an assertion about real insurance policies.',
       'decision':'One newly authorized bounded experiment. Keep every attempt, no retries/filtering/teacher/training. Reject a replacement if copying or support/format regresses, and report mixed distribution effects. No automatic deployment or follow-up experiment.',
       'review':'Reuse prior frozen judgments for the identical baseline outputs. Assistant reviews all valid new outputs for alignment/coherence and passage support, unblinded and not expert calibrated.',
       'limits':'One seed sequence and32 paired source anchors, already inspected development material. Different generated questions and remaining generation budgets. One worked example may bias wording/topics; cannot establish factual accuracy or broad generalization.'}
    write_json(OUT/'protocol.json',p)
    write_json(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'baseline_sha256':sha(OUT/'baseline_generations.jsonl')})
    print('Frozen32-example one-shot comparison; identical source anchors/sampling/adapter.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate','score','embed','text_score']);a=parser.parse_args()
    engine.OUT=OUT
    if a.stage=='prepare':prepare()
    else:getattr(engine,a.stage)()

if __name__=='__main__':main()
