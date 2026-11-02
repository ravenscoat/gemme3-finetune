"""Assistant source-quality triage, not professional factual adjudication."""
from collections import Counter
import json,re
from dataset_catalog import data,sha,ROOT,read_json
from insuranceqa_base_diagnostic import OUT
from json_io import write_json

NOTES=[
 ('Retain','Directly discusses combining multiple annuities with qualified tax language.'),
 ('Review','Blanket safe/no-market-risk, creditor-protection and everyone-should-buy claims need product/jurisdiction qualifications.'),
 ('Retain','Names rating services and provides a comparison route; minor grammar errors.'),
 ('Review','Detailed state-dependent coverage descriptions and uninsured/underinsured terminology need expert verification before reuse as facts.'),
 ('Retain','Explains supplemental cancer coverage in relation to out-of-pocket medical costs.'),
 ('Retain','Explains pricing factors without inventing a universal quote.'),
 ('Review','Tax deductibility/benefit treatment stated broadly without policy or taxpayer conditions.'),
 ('Review','Specific60percent/80percent replacement rules are presented as general; applicable policy assumptions need verification.'),
 ('Retain','Explicitly clarifies a likely question abbreviation before answering.'),
 ('Review','Dated ACA framing and broad claim that other life/disability insurance requires an exam need qualification.'),
 ('Clarify','Opening says covered unless specifically included, which conflicts with the apparent intended exception. Do not silently correct to excluded.'),
 ('Retain','Explains several reasons premiums can change with source qualifiers.'),
 ('Review','Fixed3-5day clearing estimate and policy-as-loan-collateral advice lack context.'),
 ('Retain','Qualifies underwriting discussion by severity, treatment and insurer.'),
 ('Review','Specific monthly premium estimate lacks a date, benefit specification and jurisdiction.'),
 ('Review','Promotional never-needed-is-no-good and cannot-lose framing obscures policy conditions.'),
 ('Retain','Conditional medical-necessity explanation; typo does not erase the meaning.'),
 ('Review','Broad Medicare Advantage/gap and supplement description needs product-specific checking.'),
 ('Review','Joke and malformed sentence reduce professional-style consistency, though warning signs address the question.'),
 ('Retain','Answers the subjective question with an explanation of the insurer perspective; financial assertions remain unverified.'),
 ('Retain','Provides a qualified existing-agent/carrier route to renters coverage.'),
 ('Review','Broad landlord-frequency, all-state legal and replacement-in-days claims need dated evidence.'),
 ('Retain','Directly explains partial/full vesting and portability in general terms.'),
 ('Review','Halifax/Registered Retirement Plan references indicate non-US context; self-promotion and tax jurisdiction are unresolved.')]

def main():
    sample=read_json(OUT/'source_audit_sample.json');assert len(sample['rows'])==len(NOTES)
    rows=[{**r,'triage':n[0],'reason':n[1],'scope':'Alignment, clarity, qualifications and source-scope triage. Not expert factual verification.'} for r,n in zip(sample['rows'],NOTES)]
    train=data('insuranceqa')['train'];patterns={'contact_or_web':r'https?://|www\.|@|\b[a-z0-9-]+\.(?:com|gov|org)\b','explicit_year':r'\b(?:19|20)\d{2}\b','dollar_amount':r'\$\s*\d','absolute_phrase':r'\beveryone should\b|\bcan.t lose\b|\bcan.t loose\b|\bnever\b|\balways\b'}
    flags={name:sum(bool(re.search(pattern,r['answer'],re.I)) for r in train) for name,pattern in patterns.items()}
    report={'sample_sha256':sha(OUT/'source_audit_sample.json'),'sample_n':len(rows),'sampling':sample['sampling'],'triage_counts':dict(Counter(r['triage'] for r in rows)),
       'judgments':rows,'corpus_rows':len(train),'corpus_lexical_flags':flags,'flag_scope':'Heuristic markers, not confirmed errors, privacy breaches or grounds for automatic exclusion.',
       'decision':'Create a reviewed source version before additional training; preserve original reference. Clarify row10; quarantine broad/dated/contact/jurisdiction cases for review rather than automatic deletion. No cleaned dataset or new adapter created.'}
    write_json(OUT/'source_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['judgments','sampling']},indent=2))

if __name__=='__main__':main()
