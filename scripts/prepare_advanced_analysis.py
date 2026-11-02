"""Freeze descriptive follow-up evaluations of existing outputs; no tuning/test reuse."""
from collections import defaultdict
import json
from pathlib import Path
import random

from dataset_catalog import CATALOG,ROOT,core,data,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'
SEED=2026110101


def generated(key,label):
    if key=='bitext':
        directory=ROOT/'outputs/bitext-4b-full-pass-evaluation-500-windows'
        name=label+'_unfenced_manual_review.jsonl'
        path=directory/name
        return path,[{'sample_id':r['sample_id'],'record':core(key,r)} for r in read_rows(path)]
    path=ROOT/CATALOG[key]['assessment']/(label+'_records.jsonl')
    return path,read_rows(path)


def main():
    if (OUT/'protocol.json').exists():
        print('Advanced protocol already frozen; not resampling.');return
    OUT.mkdir(parents=True,exist_ok=True)
    rng=random.Random(SEED)
    reviews=[];origins={}
    for key,quota,prefix in [('bitext',40,'B'),('insuranceqa',30,'Q'),('reliability',30,'W')]:
        path,rows=generated(key,'tuned');origins[str(path.relative_to(ROOT))]=sha(path)
        grouped=defaultdict(list)
        for row in rows:grouped[row['record'][CATALOG[key]['group']]].append(row)
        for values in grouped.values():rng.shuffle(values)
        groups=sorted(grouped);rng.shuffle(groups);chosen=[]
        while len(chosen)<quota:
            for group in groups:
                if grouped[group] and len(chosen)<quota:chosen.append(grouped[group].pop())
        for i,row in enumerate(chosen,1):
            reviews.append({'review_id':f'{prefix}{i:02d}','dataset':key,'sample_id':row['sample_id'],
                            'stratum':row['record'][CATALOG[key]['group']],'record':row['record']})
    (OUT/'review_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviews),encoding='utf-8')
    fields={'bitext':{'request':'instruction','reply':'response'},'insuranceqa':{'request':'question','reply':'answer'}}
    feature_inputs={};selections={}
    for key in fields:
        bp,base=generated(key,'base');tp,tuned=generated(key,'tuned')
        origins[str(bp.relative_to(ROOT))]=sha(bp);origins[str(tp.relative_to(ROOT))]=sha(tp)
        n=min(128,len(base),len(tuned))
        source=data(key);directory=ROOT/CATALOG[key]['prepared']
        validation=read_rows(directory/'validation.jsonl')
        collections={'base':rng.sample(base,n),'tuned':rng.sample(tuned,n),
            'training_reference':rng.sample(source['train'],n),'validation_control':rng.sample(validation,n)}
        selections[key]={'n':n,'selected_ids':{},'field_names':fields[key],
                         'original_valid':len(base),'tuned_valid':len(tuned)}
        feature_inputs[key]={}
        for label,rows in collections.items():
            selections[key]['selected_ids'][label]=[r.get('sample_id',r.get('source_id',r.get('source_line'))) for r in rows]
            feature_inputs[key][label]={field:[(r.get('record',r))[path] for r in rows] for field,path in fields[key].items()}
    write_json(OUT/'embedding_inputs.json',feature_inputs)
    legacy=read_json(ROOT/'outputs/bitext-4b-evaluation-500-windows/mauve_protocol.json')
    protocol={'seed':SEED,'purpose':'Descriptive follow-up on frozen outputs; no retraining, re-selection or final-test reuse',
              'review':{'counts':{'bitext':40,'insuranceqa':30,'reliability':30},'sampling':'Round-robin stratified by source label; not population-representative',
                        'rubric':'rubric_v1.md','input_sha256':sha(OUT/'review_records.jsonl'),'human_calibration':'Not available; no expert/accuracy claim'},
              'embedding':{'feature_model':legacy['feature_model'],'feature_model_revision':legacy['feature_model_revision'],
                           'representation':'Last non-padding token, float32, L2 normalization','max_tokens':512,'batch_size':4,
                           'prdc_nearest_k':[5,10],'selections':selections,'input_sha256':sha(OUT/'embedding_inputs.json'),
                           'scope':'Experimental text-embedding adaptation of image-generation PRDC; small-sample and representation-sensitive'},
              'sdmetrics':{'reference':'mimic_demo training admissions','outputs':'All structurally valid original and tuned frozen records',
                           'missingness':'Kept as null/NaN; no imputation','checks':['QualityReport','MissingValueSimilarity','RangeCoverage','CategoryCoverage','BoundaryAdherence','CategoryAdherence']},
              'source_hashes':origins}
    for key in ['bitext','insuranceqa','mimic_demo']:
        for split in ['train','validation']:
            path=ROOT/CATALOG[key]['prepared']/(split+'.jsonl');protocol['source_hashes'][str(path.relative_to(ROOT))]=sha(path)
    write_json(OUT/'protocol.json',protocol)
    print('Frozen100-review sample and embedding selections:',{k:v['n'] for k,v in selections.items()})


if __name__=='__main__':main()
