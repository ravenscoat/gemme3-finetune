"""Explicitly exploratory diagnostics prompted by already-inspected outputs."""
from collections import Counter,defaultdict
import hashlib
import json
import re

from dataset_catalog import ROOT,core,data,read_json,read_rows,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'
EMAIL=re.compile(r'[A-Za-z0-9_.+%\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
PHONE=re.compile(r'(?<!\d)(?:\+?1[\s.\-]*)?(?:\(?\d{3}\)?[\s.\-]*)\d{3}[\s.\-]*\d{4}(?!\d)')


def icd_prefix_issue(record):
    code,version=record['primary_diagnosis_code'],record['primary_diagnosis_version']
    if code is None or version is None:return 'missing_pair' if (code is None)!=(version is None) else None
    leading=code.strip().upper()[0] if code.strip() else ''
    if not leading:return 'empty_code'
    if version==10 and leading not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':return 'ICD10_nonletter_prefix'
    if version==9 and leading not in '0123456789EV':return 'ICD9_disallowed_letter_prefix'
    return None


def contacts(text):
    result=set()
    for value in EMAIL.findall(text):result.add(('email',value.casefold()))
    for value in PHONE.findall(text):
        digits=re.sub(r'\D','',value)
        if len(digits)==11 and digits.startswith('1'):digits=digits[1:]
        if len(digits)==10:result.add(('candidate_US_phone',digits))
    return result


def main():
    addendum=read_json(OUT/'exploratory_addendum.json')
    result={'status':'Exploratory, post-inspection follow-up; not preregistered primary results',
            'addendum_sha256':sha(OUT/'exploratory_addendum.json'),'icd_prefix_version':{},'contact_reuse':{}}
    source=data('mimic_demo');collections={'training_reference':[core('mimic_demo',r) for r in source['train']],
        'validation_control':[core('mimic_demo',r) for r in read_rows(ROOT/source['spec']['prepared']/'validation.jsonl')]}
    for label in ['base','tuned']:
        collections[label]=[r['record'] for r in read_rows(ROOT/source['spec']['assessment']/(label+'_records.jsonl'))]
    for label,rows in collections.items():
        issues=[{'row_index':i,'code':r['primary_diagnosis_code'],'version':r['primary_diagnosis_version'],'issue':icd_prefix_issue(r)} for i,r in enumerate(rows) if icd_prefix_issue(r)]
        result['icd_prefix_version'][label]={'n':len(rows),'attempts':128 if label in ['base','tuned'] else len(rows),
            'non_null_code_version_pairs':sum(r['primary_diagnosis_code'] is not None and r['primary_diagnosis_version'] is not None for r in rows),
            'flagged':len(issues),'issues':issues}
    source=data('insuranceqa');known=defaultdict(list)
    for r in source['train']:
        for token in contacts(r['answer']):known[token].append(r['source_id'])
    for label in ['base','tuned']:
        rows=read_rows(ROOT/source['spec']['assessment']/(label+'_records.jsonl'));items=[]
        for row in rows:
            tokens=contacts(row['record']['answer'])
            if not tokens:continue
            items.append({'sample_id':row['sample_id'],'candidate_count':len(tokens),
                'tokens':[{'kind':kind,'normalized_sha256':hashlib.sha256(value.encode()).hexdigest(),
                           'seen_in_training':(kind,value) in known,'training_source_ids':known[(kind,value)][:10]}
                          for kind,value in sorted(tokens)]})
        result['contact_reuse'][label]={'structurally_valid':len(rows),'attempts':128,'with_candidates':len(items),
            'with_training_contact_reuse':sum(any(t['seen_in_training'] for t in item['tokens']) for item in items),'records':items}
    result['contact_reuse']['limits']=['Regex candidates only; no verification that a phone/email is real, current or belongs to the named speaker.',
        'Professional contacts may be publicly published or illustrative. Matches are partial source reuse, not proof of private-data leakage.',
        'Zero full-answer copies does not rule out exact contact-substring reuse; nonmatches can also be invented identities.',
        'Only answers and exact normalized email/US-phone patterns are scanned; this is not a comprehensive privacy or memorization audit.']
    result['factual_spot_checks']=[
        {'review_id':'Q07','sample_id':25,'claim_summary':'Medicare is the first/only payer and there are no secondary health insurers.',
         'finding':'Contradicted','evidence_summary':'CMS gives common circumstances in which an employer group health plan pays first and Medicare pays second.',
         'source':'https://www.cms.gov/medicare/coordination-benefits-recovery/overview/secondary-payer','accessed':'2026-11-01'},
        {'review_id':'Q09','sample_id':100,'claim_summary':'401(k) plans were established in1974.',
         'finding':'Contradicted','evidence_summary':'The U.S. Bureau of Labor Statistics identifies the Revenue Act of1978 as including the provision that became section401(k).',
         'source':'https://www.bls.gov/opub/btn/volume-5/defined-contribution-retirement-plans-who-has-them-and-what-do-they-cost.htm','accessed':'2026-11-01'}]
    result['icd_prefix_version']['limits']=['Leading-character compatibility only, not a full code-format or dictionary check.',
        'A passing prefix does not establish that a diagnosis code exists, is billable, has the right version/date or fits the patient.',
        'Codes are diagnoses, not ICD-10-PCS procedures. U prefixes are allowed; the old2010 U exception is deliberately not enforced.']
    write_json(OUT/'exploratory_results.json',result)
    lines=['# Exploratory failure analysis','',
           'Added after inspecting frozen outputs. These are targeted diagnostics, not preregistered benchmark results or population error estimates.','',
           '## MIMIC code/version leading-character screen','',
           '| Collection | Structurally valid/source rows | Flagged pairs |','|---|---:|---:|']
    for label,value in result['icd_prefix_version'].items():
        if isinstance(value,dict):lines.append(f"| {label} | {value['n']} | {value['flagged']} |")
    lines+=['','ICD-10-CM diagnoses start with a letter; ICD-9-CM diagnoses start with a digit or E/V. [CMS structural comparison](https://www.cms.gov/medicare/coding/icd10/downloads/032310_ICD10_Transcript.pdf). This deliberately narrow screen allows U in modern ICD-10 and does not validate code existence.','',
            '## InsuranceQA contact candidates','',
            '| Collection | Valid outputs /128 | Contains email/phone candidate | Reuses a training contact |','|---|---:|---:|---:|']
    for label in ['base','tuned']:
        value=result['contact_reuse'][label];lines.append(f"| {label} | {value['structurally_valid']} | {value['with_candidates']} | {value['with_training_contact_reuse']} |")
    lines+=['',*['- '+v for v in result['contact_reuse']['limits']],'','## Targeted official-source factual spot-checks','',
            'Two examples selected because the qualitative review raised concerns. These do not calibrate the judge or estimate a factual error rate. A relevance/coherence Pass can still be factually wrong.','']
    for item in result['factual_spot_checks']:lines.append(f"- **{item['review_id']}, sample{item['sample_id']}: {item['finding']}.** Generated claim: {item['claim_summary']} Evidence: {item['evidence_summary']} [Official source]({item['source']}).")
    lines+=['',*['- '+v for v in result['icd_prefix_version']['limits']],'']
    (OUT/'exploratory_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('ICD:',{k:v['flagged'] for k,v in result['icd_prefix_version'].items() if isinstance(v,dict)})
    print('Contact reuse:',{k:result['contact_reuse'][k]['with_training_contact_reuse'] for k in ['base','tuned']})


if __name__=='__main__':main()
