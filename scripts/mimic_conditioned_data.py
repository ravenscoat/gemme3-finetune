"""Versioned generic/condition-specific views of existing training patients only."""
import json

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha
from json_io import write_json
from mimic_fidelity_pilot import LABS,missing_pattern

PREPARED=ROOT/'data/mimic_demo/conditioned_v1'
RUN=ROOT/'outputs/mimic-demo-fidelity-conditioned-v1-windows'


def condition_from_record(record):
    return {'primary_diagnosis_code':record['primary_diagnosis_code'],
            'primary_diagnosis_version':record['primary_diagnosis_version'],'lab_missing_pattern':missing_pattern(record)}


def conditioned_prompt(prompt,condition):
    text=prompt+'\nUse these required conditions for this fictional record: diagnosis code '+json.dumps(condition['primary_diagnosis_code'])+', diagnosis version '+str(condition['primary_diagnosis_version'])+'. '
    return text+' '.join(f'{field} must be '+('null.' if bit=='1' else 'a non-null numeric value.') for field,bit in zip(LABS,condition['lab_missing_pattern']))


def prepare():
    source=data('mimic_demo');directory=ROOT/source['spec']['prepared']
    validation=read_rows(directory/'validation.jsonl');train=source['train']
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in validation})
    views={'train':[],'validation_generic':[],'validation_conditioned':[]}
    for split,rows in [('train',train),('validation',validation)]:
        for row in rows:
            for view in ['generic','conditioned']:
                item={'source_id':row['source_id'],'group_id':row['group_id'],'view':view,
                    'prompt':source['prompt'] if view=='generic' else conditioned_prompt(source['prompt'],condition_from_record(row)),
                    'completion':core('mimic_demo',row)}
                views['train' if split=='train' else 'validation_'+view].append(item)
    contents={name:''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows) for name,rows in views.items()}
    PREPARED.mkdir(parents=True,exist_ok=True)
    for name,content in contents.items():
        path=PREPARED/(name+'.jsonl')
        if path.exists() and path.read_text(encoding='utf-8')!=content:raise RuntimeError('Candidate views changed: '+name)
        if not path.exists():path.write_text(content,encoding='utf-8')
    manifest={'source_hashes':{str((directory/name).relative_to(ROOT)):sha(directory/name) for name in ['train.jsonl','validation.jsonl','schema.json','generation_prompt.txt']},
              'view_hashes':{name:sha(PREPARED/(name+'.jsonl')) for name in views},'view_counts':{name:len(rows) for name,rows in views.items()},
              'source_train_admissions':len(train),'source_train_patients':len({r['group_id'] for r in train}),
              'source_validation_admissions':len(validation),'source_validation_patients':len({r['group_id'] for r in validation}),
              'scope':'Same real records; two prompt views are not additional patients. No test records read. Unchanged target fields/order and source distributions.',
              'mask_policy':'Completion-only loss; per-example prompt boundary verified during tokenization'}
    if (PREPARED/'manifest.json').exists() and read_json(PREPARED/'manifest.json')!=manifest:raise RuntimeError('Candidate data manifest changed')
    write_json(PREPARED/'manifest.json',manifest)
    return manifest


if __name__=='__main__':print(json.dumps(prepare(),indent=2))
