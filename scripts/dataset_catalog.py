"""Dataset definitions and structural checks shared by assessment and local UI."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CATALOG = {
 'bitext': {'label':'Bitext Insurance', 'kind':'support', 'run':'bitext-4b-full-pass-windows', 'selection':'outputs/final-assessment/selection.json', 'context':1024, 'group':'intent', 'request':'instruction', 'reply':'response', 'scope':'Synthetic insurance-support source; completed 500-output assessment.'},
 'insuranceqa': {'label':'InsuranceQA V2', 'kind':'qa', 'run':'insuranceqa-4b-v1-windows', 'context':1536, 'group':'domain', 'request':'question', 'reply':'answer', 'scope':'Source insurance questions and professional answers; one positive answer per question.'},
 'reliability': {'label':'Insurance Reliability', 'kind':'workflow', 'run':'reliability-4b-v1-windows', 'context':1024, 'group':'category', 'request':'input.customer_message', 'reply':'expected_output', 'scope':'Benchmark-style scenario generation; held-out split is an in-domain check after training.'},
 'mimic_demo': {'label':'MIMIC-IV Demo', 'kind':'clinical', 'run':'mimic-demo-4b-v2-windows', 'context':1024, 'group':'admission_type', 'request':None, 'reply':None, 'scope':'100-patient official demo. Structured admissions, not the full MIMIC database or U.S. population.'},
}
for key, spec in CATALOG.items():
    spec.setdefault('selection', f"outputs/{spec['run']}/selection.json")
    spec['prepared'] = f'data/{key}/prepared'
    spec['assessment'] = f"outputs/{spec['run']}-assessment-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def data(key):
    spec = CATALOG[key]
    directory = ROOT / spec['prepared']
    schema = read_json(directory / 'schema.json')
    return {'spec':spec, 'schema':schema, 'train':read_rows(directory / 'train.jsonl'),
            'prompt':(directory / 'generation_prompt.txt').read_text(encoding='utf-8').strip()}


def value_at(record, path):
    if not path:
        return record
    value = record
    for field in path.split('.'):
        value = value.get(field) if isinstance(value, dict) else None
    return value


def text_at(record, path):
    value = value_at(record, path)
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def core(key, record):
    return {field:record[field] for field in data(key)['schema']['fields']}


def signature(value):
    def normalized(item):
        if isinstance(item, str):
            return ' '.join(re.findall(r'\w+', item.casefold()))
        if isinstance(item, dict):
            return {k:normalized(v) for k,v in item.items()}
        if isinstance(item, list):
            return [normalized(v) for v in item]
        if type(item) in [int, float]:
            return float(item)
        return item
    return json.dumps(normalized(value), sort_keys=True, ensure_ascii=False)


def parse_output(text):
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, flags=re.DOTALL | re.IGNORECASE)
    return json.loads(fenced.group(1).strip() if fenced else text)


def validate_record(key, record):
    """Check shape/types separately from source-vocabulary fidelity or correctness."""
    fields = data(key)['schema']['fields']
    errors = []
    if not isinstance(record, dict) or set(record) != set(fields):
        return ['Expected exactly the dataset fields: ' + ', '.join(fields)]
    if key in ['insuranceqa', 'bitext']:
        if any(not isinstance(record[f], str) or not record[f].strip() for f in fields):
            errors.append('All fields must be non-empty strings')
    elif key == 'reliability':
        for field in ['category', 'subcategory', 'difficulty']:
            if not isinstance(record[field], str) or not record[field].strip():
                errors.append(field + ' must be a non-empty string')
        inp, expected = record['input'], record['expected_output']
        if not isinstance(inp, dict) or set(inp) != {'customer_message','context','conversation_history'}:
            errors.append('input must contain customer_message, context, conversation_history')
        else:
            if not isinstance(inp['customer_message'], str) or not inp['customer_message'].strip(): errors.append('Invalid customer_message')
            if not isinstance(inp['context'], dict): errors.append('context must be an object')
            if not isinstance(inp['conversation_history'], list): errors.append('conversation_history must be a list')
            elif any(not isinstance(turn, dict) or set(turn) != {'role','content'} or any(not isinstance(v,str) for v in turn.values()) for turn in inp['conversation_history']): errors.append('Invalid conversation turn')
        expected_fields = {'intent','routing_decision','required_actions','priority','compliance_flags','expected_response_elements'}
        if not isinstance(expected, dict) or set(expected) != expected_fields:
            errors.append('Unexpected expected_output fields')
        else:
            for field in ['intent','routing_decision','priority']:
                if not isinstance(expected[field], str) or not expected[field].strip(): errors.append('Invalid expected_output.'+field)
            for field in ['required_actions','compliance_flags','expected_response_elements']:
                if not isinstance(expected[field], list) or any(not isinstance(v,str) for v in expected[field]): errors.append('Invalid expected_output.'+field)
    else:
        for field in ['age', 'hospital_expire_flag']:
            if type(record[field]) is not int: errors.append(field + ' must be integer')
        if type(record['age_topcoded']) is not bool: errors.append('age_topcoded must be boolean')
        for field in ['gender','admission_type','insurance','race']:
            if not isinstance(record[field], str) or not record[field].strip(): errors.append('Invalid '+field)
        for field in data(key)['schema']['numeric_fields']:
            value = record[field]
            if value is None and field in ['creatinine_mg_dl','glucose_mg_dl','hemoglobin_g_dl']: continue
            if type(value) not in [int,float] or not math.isfinite(value): errors.append('Invalid numeric '+field)
        if record['primary_diagnosis_code'] is not None and not isinstance(record['primary_diagnosis_code'],str): errors.append('Invalid diagnosis code')
        version = record['primary_diagnosis_version']
        if version is not None and (type(version) is not int or version not in [9,10]): errors.append('Invalid ICD version')
    return errors


def source_warnings(key, record):
    warnings = []
    source = data(key)
    paths = source['schema'].get('categorical_fields', [])
    for path in paths:
        values = {signature(value_at(r,path)) for r in source['train']}
        if signature(value_at(record,path)) not in values: warnings.append('Unseen training category: '+path)
    if key == 'mimic_demo':
        if not 0 <= record['age'] <= 90 or record['age_topcoded'] != (record['age']==90): warnings.append('Age/topcoding constraint violated')
        if record['length_of_stay_days'] < 0: warnings.append('Negative length of stay')
        if record['hospital_expire_flag'] not in [0,1]: warnings.append('Mortality flag must be 0 or 1')
        if (record['primary_diagnosis_code'] is None) != (record['primary_diagnosis_version'] is None): warnings.append('Diagnosis code/version missingness differs')
    if key == 'bitext':
        if source['schema']['intent_category'].get(record['intent']) != record['category']: warnings.append('Intent/category mismatch')
        if any(t not in source['schema']['tag_characters'] for t in record['tags']): warnings.append('Unknown language tag')
    return warnings
