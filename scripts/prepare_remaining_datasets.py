"""Prepare source-faithful synthetic-record tasks; keep split metadata out of targets."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261031


def norm(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(name, splits, fields, prompt, audit, categorical, numeric=()):
    out = ROOT / 'data' / name / 'prepared'
    if out.exists():
        raise RuntimeError(f'Prepared data already exist: {out}. Preserve their split identity.')
    ids = [{r['group_id'] for r in splits[s]} for s in ['train', 'validation', 'test']]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    out.mkdir(parents=True)
    hashes = {}
    for split, rows in splits.items():
        random.Random(SEED).shuffle(rows)
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (out / (split + '.jsonl')).write_text(text, encoding='utf-8')
        hashes[split] = hashlib.sha256(text.encode()).hexdigest()
    schema = {'dataset': name, 'fields': fields, 'categorical_fields': list(categorical), 'numeric_fields': list(numeric)}
    (out / 'schema.json').write_text(json.dumps(schema, indent=2), encoding='utf-8')
    (out / 'generation_prompt.txt').write_text(prompt, encoding='utf-8')
    audit.update(counts={s: len(r) for s, r in splits.items()}, split_sha256=hashes, seed=SEED,
                 field_order=fields, training_metadata_excluded=['source_id', 'group_id'])
    (out / 'preparation_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(name, audit['counts'], flush=True)


def insuranceqa():
    source = ROOT / 'data/insuranceqa/raw'
    vocab = dict(line.split('\t', 1) for line in (source / 'vocabulary').read_text(encoding='utf-8').splitlines())
    def decode(text):
        return ' '.join(vocab[t] if t.startswith('idx_') else t for t in text.split())
    with gzip.open(source / 'InsuranceQA.label2answer.raw.encoded.gz', 'rt', encoding='utf-8') as stream:
        answers = {label: decode(answer) for label, answer in (line.rstrip('\n').split('\t', 1) for line in stream)}
    splits, original_counts, purged = {}, {}, {}
    seen_questions, seen_answers = set(), set()
    for split, filename in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
        rows = []
        with gzip.open(source / f'InsuranceQA.question.anslabel.raw.100.pool.solr.{filename}.encoded.gz', 'rt', encoding='utf-8') as stream:
            lines = list(stream)
        original_counts[split] = len(lines)
        for index, line in enumerate(lines):
            domain, question, positives, pool = line.rstrip('\n').split('\t')
            question = decode(question)
            labels = positives.split()
            assert labels and all(label in answers for label in labels)
            # One source-positive answer per question; never use retrieved distractors.
            selected = labels[int(digest([domain, question]), 16) % len(labels)]
            answer = answers[selected]
            q, a = norm(question), norm(answer)
            if q in seen_questions or a in seen_answers:
                continue
            rows.append({'domain': domain, 'question': question, 'answer': answer,
                         'source_id': f'{filename}:{index+1}:answer:{selected}', 'group_id': digest(q)[:20]})
        # Purge lower-priority split overlaps; shared answers within a split are retained.
        seen_questions.update(norm(r['question']) for r in rows)
        seen_answers.update(norm(r['answer']) for r in rows)
        purged[split] = len(lines) - len(rows)
        splits[split] = rows
    domains = sorted({r['domain'] for r in splits['train']})
    prompt = ('Generate synthetic data. Return exactly one JSON object with fields domain, question, answer. '
              'Create a new insurance question and its corresponding professional-style answer, matching the '
              'wording, length and topic distribution of InsuranceQA V2. No commentary or markdown. '
              'The domain must be one of: ' + ', '.join(domains) + '.')
    save('insuranceqa', splits, ['domain', 'question', 'answer'], prompt,
         {'source': 'InsuranceQA V2 raw encoded, pinned official repository', 'official_counts': original_counts,
          'overlap_rows_removed': purged, 'answer_selection': 'SHA256(question,domain) modulo source positive labels; one answer per question',
          'split_policy': 'Official splits; normalized question/selected-answer overlaps removed from lower-priority validation/test',
          'limits': 'Original de-tokenized spacing is preserved; long records are audited separately at tokenization.'}, ['domain'])


def reliability():
    source = ROOT / 'data/reliability/raw'
    fields = ['category', 'subcategory', 'difficulty', 'input', 'expected_output']
    splits, seen, dropped = {}, set(), {}
    for split in ['train', 'validation', 'test']:
        rows = []
        originals = [json.loads(line) for line in (source / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        for row in originals:
            key = digest(row['input'])
            if key in seen:
                continue
            rows.append({**{k: row[k] for k in fields}, 'source_id': row['id'], 'group_id': key})
        seen.update(r['group_id'] for r in rows)
        splits[split] = rows
        dropped[split] = len(originals) - len(rows)
    categories = sorted({r['category'] for r in splits['train']})
    prompt = ('Generate synthetic data. Return one insurance workflow scenario as a JSON object with exactly '
              'category, subcategory, difficulty, input, expected_output. input contains customer_message, context, '
              'conversation_history. expected_output contains intent, routing_decision, required_actions, priority, '
              'compliance_flags, expected_response_elements. Match the source scenario distribution and structure. '
              'Create a new example, without commentary or markdown. Categories: ' + ', '.join(categories) + '.')
    save('reliability', splits, fields, prompt,
         {'source': 'pashas/insurance-ai-reliability-benchmark pinned official splits',
          'duplicate_input_overlap_removed': dropped, 'split_policy': 'Official splits with shared exact input removed from lower-priority splits',
          'limits': 'This adapter generates benchmark-style scenarios and labels. A split-level assessment is not an independent benchmark of agent reliability. Original templates and label inconsistencies may remain.',
          'excluded_source_fields': ['id', 'evaluation_criteria', 'metadata']}, ['category', 'subcategory', 'difficulty'])


def mimic():
    source = ROOT / 'data/mimic-iv-demo-2.2/mimic-iv-clinical-database-demo-2.2/hosp'
    def table(name):
        with gzip.open(source / (name + '.csv.gz'), 'rt', newline='', encoding='utf-8') as stream:
            yield from csv.DictReader(stream)
    patients = {r['subject_id']: r for r in table('patients')}
    admissions = {r['hadm_id']: r for r in table('admissions')}
    diagnoses = {}
    for row in table('diagnoses_icd'):
        if row['seq_num'] == '1':
            assert row['hadm_id'] not in diagnoses
            diagnoses[row['hadm_id']] = row
    lab_fields = {'50912': ('creatinine_mg_dl', 'mg/dl'), '50931': ('glucose_mg_dl', 'mg/dl'), '51222': ('hemoglobin_g_dl', 'g/dl')}
    labs, rejected_units = {}, Counter()
    for row in table('labevents'):
        if row['itemid'] not in lab_fields or row['hadm_id'] not in admissions or not row['valuenum']:
            continue
        admission = admissions[row['hadm_id']]
        if not admission['admittime'] <= row['charttime'] <= admission['dischtime']:
            continue
        field, unit = lab_fields[row['itemid']]
        if row['valueuom'].casefold() != unit:
            rejected_units[row['valueuom']] += 1
            continue
        value = float(row['valuenum'])
        if not math.isfinite(value):
            continue
        key = (row['hadm_id'], field)
        if key not in labs or row['charttime'] < labs[key][0]:
            labs[key] = (row['charttime'], value)
    subjects = sorted(patients)
    random.Random(SEED).shuffle(subjects)
    groups = {subject: 'train' if i < 80 else 'validation' if i < 90 else 'test' for i, subject in enumerate(subjects)}
    splits = {s: [] for s in ['train', 'validation', 'test']}
    for hadm_id, row in admissions.items():
        patient = patients[row['subject_id']]
        age = int(patient['anchor_age']) + datetime.fromisoformat(row['admittime']).year - int(patient['anchor_year'])
        diagnosis = diagnoses.get(hadm_id)
        record = {'age': min(90, age), 'age_topcoded': age >= 90, 'gender': patient['gender'],
                  'admission_type': row['admission_type'], 'insurance': row['insurance'], 'race': row['race'],
                  'length_of_stay_days': round((datetime.fromisoformat(row['dischtime']) - datetime.fromisoformat(row['admittime'])).total_seconds()/86400, 3),
                  'hospital_expire_flag': int(row['hospital_expire_flag']),
                  'primary_diagnosis_code': diagnosis['icd_code'] if diagnosis else None,
                  'primary_diagnosis_version': int(diagnosis['icd_version']) if diagnosis else None,
                  **{field: labs.get((hadm_id, field), (None, None))[1] for field, unit in lab_fields.values()}}
        assert record['age'] >= 0 and record['length_of_stay_days'] >= 0
        splits[groups[row['subject_id']]].append({**record, 'source_id': hadm_id, 'group_id': row['subject_id']})
    fields = list(record)
    prompt = ('Generate synthetic data. Return exactly one fictional hospital-admission record as JSON, matching '
              'the ranges, frequencies, missingness and relationships in the MIMIC-IV demo training records. '
              'Fields: ' + ', '.join(fields) + '. age is an integer, with 90 representing age 90 or older; '
              'age_topcoded is boolean. length_of_stay_days is numeric. hospital_expire_flag is 0 or 1. '
              'primary_diagnosis_version is 9 or 10. Diagnosis fields and first inpatient lab values may be null. '
              'Do not include identifiers, dates, commentary or markdown.')
    save('mimic_demo', splits, fields, prompt,
         {'source': 'Official MIMIC-IV demo v2.2; 100 patients, 275 admissions',
          'patient_counts': {'train': 80, 'validation': 10, 'test': 10}, 'split_policy': 'Patient-disjoint 80/10/10 split',
          'unit_rejections': dict(rejected_units), 'lab_selection': 'Earliest finite numeric value between admission and discharge, with verified unit; missing stays null',
          'diagnosis_selection': 'Source seq_num=1 only; no generated clinical labels',
          'age_derivation': 'anchor_age + admission_year - anchor_year; ages >=90 topcoded to90',
          'limits': 'Small clinical demo, not the full MIMIC dataset or representative U.S. population. Admission-level records, not free-text notes.'},
         ['age_topcoded', 'gender', 'admission_type', 'insurance', 'race', 'hospital_expire_flag', 'primary_diagnosis_version'],
         ['age', 'length_of_stay_days', 'creatinine_mg_dl', 'glucose_mg_dl', 'hemoglobin_g_dl'])


if __name__ == '__main__':
    insuranceqa()
    reliability()
    mimic()
