"""Matched base-versus-adapter source-task diagnostic and stratified source audit."""
import argparse
from collections import defaultdict
import random
import shutil
from dataset_catalog import ROOT,read_json,read_rows,sha,data
from json_io import write_json
import insuranceqa_final_pilot as engine

OUT=ROOT/'outputs/insuranceqa-base-diagnostic-v1'
PRIOR=ROOT/'outputs/insuranceqa-final-grounded-v1'

def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    p=read_json(PRIOR/'protocol.json');assert sha(PRIOR/'protocol.json')==read_json(PRIOR/'freeze.json')['protocol_sha256']
    OUT.mkdir(parents=True,exist_ok=True);shutil.copy2(PRIOR/'grounded_generations.jsonl',OUT/'baseline_generations.jsonl')
    p={**p,'disable_adapter':True,'arm_names':{'baseline':'InsuranceQA adapter enabled, reused frozen source-only outputs','grounded':'Original quantized base with adapter disabled'},
       'format_gate':{'first_attempts':8,'minimum_valid':0,'failure_action':'Record failures and complete all32 for fair comparison'},
       'input_hashes':{**p['input_hashes'],str((PRIOR/'protocol.json').relative_to(ROOT)):sha(PRIOR/'protocol.json'),str((PRIOR/'grounded_generations.jsonl').relative_to(ROOT)):sha(PRIOR/'grounded_generations.jsonl')},
       'decision':'Diagnostic only:32 identical source prompts, seeds, sampling and context. No teacher, training or default replacement. No retries or output selection. Complete all32 even if the base format differs.',
       'limits':'Previously inspected development anchors; unblinded assistant judgments. Can reveal a tradeoff on this particular source-generation task, not prove general capability loss or factual accuracy.'}
    write_json(OUT/'protocol.json',p);write_json(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'baseline_sha256':sha(OUT/'baseline_generations.jsonl')})
    groups=defaultdict(list)
    for r in data('insuranceqa')['train']:groups[r['domain']].append(r)
    rng=random.Random(2026110219);selected=[]
    for domain,rows in sorted(groups.items()):
        ordered=sorted(rows,key=lambda r:len(r['answer']));mid=max(1,len(rows)//2)
        for band,pool in [('shorter_half',ordered[:mid]),('longer_half',ordered[mid:])]:
            if pool:selected.append({'audit_id':len(selected),'length_stratum':band,'record':rng.choice(pool)})
    write_json(OUT/'source_audit_sample.json',{'seed':2026110219,'sampling':'One random training row from each domain x shorter/longer answer half. Equal domain allocation, not representative of corpus frequencies. No held-out data used.','rows':selected})
    print('Prepared32matched base generations and',len(selected),'source audit rows.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate','score','embed','text_score']);args=parser.parse_args();engine.OUT=OUT
    if args.stage=='prepare':prepare()
    else:getattr(engine,args.stage)()
