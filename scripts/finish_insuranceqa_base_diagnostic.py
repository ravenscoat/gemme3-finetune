"""Independent diagnostic audit and readable decision report."""
import json
from pathlib import Path
from dataset_catalog import ROOT,read_json,read_rows,sha
from insuranceqa_base_diagnostic import OUT,PRIOR
import insuranceqa_final_pilot as engine
import verify_insuranceqa_final_pilot as verifier
from json_io import write_json

def main():
    engine.OUT=OUT;verifier.OUT=OUT;verifier.check=engine.check;verifier.main()
    p=read_json(OUT/'protocol.json');old=read_json(PRIOR/'protocol.json')
    for field in ['anchors','prompts','adapter','context','batch_size','seed_start','temperature','top_p','top_k']:assert p[field]==old[field]
    assert p['disable_adapter'] is True
    raw=read_rows(OUT/'grounded_generations.jsonl');assert len(raw)==32 and all(r['adapter_disabled'] for r in raw)
    assert sha(OUT/'baseline_generations.jsonl')==sha(PRIOR/'grounded_generations.jsonl')
    source=read_json(OUT/'source_audit.json');assert sha(OUT/'source_audit_sample.json')==source['sample_sha256']
    m=read_json(OUT/'metrics.json');r=read_json(OUT/'review_summary.json')
    lines=['# InsuranceQA: base-versus-adapter diagnostic','',
       '## Decision','',
       'The original quantized Gemma handles these source-conditioned generation prompts substantially better in this assistant review. This is evidence of a task-specific adapter tradeoff, not proof of general catastrophic forgetting. Stop prompt-only sweeps; review the source data and explicitly train any future source-conditioned candidate on that task. No training or automatic replacement was performed.','',
       '## Same task, two model states','',
       '32 identical source passages and prompts, same seeds/batch size/sampling/context. Adapter-enabled outputs are reused without changes from the frozen source-only experiment. New outputs use the same quantized base weights with the PEFT adapter disabled inside generation. Different generated questions, one seed schedule, inspected development material and unblinded, uncalibrated assistant review limit conclusions. These are support/coherence judgments, not expert factual accuracy.','',
       '| Measure | InsuranceQA adapter | Original Gemma |','|---|---:|---:|',
       '| Coherence Pass / Unsure / Issue | 14 / 8 / 10 | 28 / 0 / 4 |',
       '| Supported by supplied source | 8/32 | 24/32 |',
       '| Schema-valid after existing fenced-JSON parser | 32/32 | 32/32 |',
       '| Markdown fences, contrary to prompt | 0/32 | 32/32 |']
    for label,values in [('SDMetrics domain/length quality (higher closer)',[m['arms'][a]['sdmetrics']['quality'] for a in ['baseline','grounded']]),('Joint MMD-squared (lower closer)',[m['text_metrics']['joint']['arms'][a]['mmd2'] for a in ['baseline','grounded']]),('Keyword JS distance (lower closer)',[m['keywords']['collections'][a]['js_distance_base2'] for a in ['baseline','grounded']]),('Exact training-answer reuse',[m['arms'][a]['training_reply_match_rate'] for a in ['baseline','grounded']])]:
        lines.append('| '+label+' | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines+=['','Schema validity does not mean literal instruction compliance: original Gemma wrapped every JSON object in Markdown fences. That is an interface issue to handle explicitly; it does not erase answer-quality differences. Distribution scores and factual/source quality measure different things. These results do not justify replacing the required fine-tuned submission with an untrained baseline.','',
       '## Source audit','',
       'Reviewed24 training records: one random record from each of12domains, separately in its shorter and longer answer halves.11Retain,12Review,1Clarify. Equal-domain sampling is not a corpus error-rate estimate. Retain means no obvious issue under this triage, not expert factual approval.','',
       'Examples: audit10 says coverage applies unless specifically included, an unclear exception that must be clarified; audit7 presents60%/80% disability benefit amounts broadly; audit14 gives a dated/unspecified premium quote; audit23 mixes Halifax/Registered Retirement Plan language and self-promotion, requiring market-scope review. Source audit and generated-answer review use different rubrics.','',
       'Corpus heuristic counts, not confirmed errors: '+json.dumps(source['corpus_lexical_flags'])+'. Dates, prices and URLs can be valid source characteristics; do not automatically delete these rows.','',
       '## Best next steps','',
       '1. Build a versioned review table for flagged source records. Resolve the coverage exception and domain/jurisdiction mismatches. Preserve both original source and any corrected version; document evidence for corrections.','2. Define the desired input/output task. Generic joint record generation remains valid for the assignment. If we want passage-grounded generation, explicitly include source passages and reviewed supported paraphrases in training. Do not use the exact source answer as both input and target.','3. Train a separate small context-conditioned candidate only after usable targets exist. Keep rank and most settings steady; compare it with both the original adapter and base on frozen development cases. No such training has begun.','4. Recheck source support, question-answer fit, copying and original distribution matching with equal-sized cohorts and additional seeds. Preserve raw attempts; test failures and known passing cases. Do not claim a new untouched test from previously inspected material.','',
       '## Evidence and reproduction','',
       'Full evidence: protocol.json, all raw generations, parsed records, judgments.jsonl, review_summary.json, source_audit_sample.json, source_audit.json, metrics.json, embedding_manifest.json and verification.json in this folder.','',
       'Scripts: insuranceqa_base_diagnostic.py prepare/generate/score/embed/text_score; audit_insuranceqa_source.py; review_insuranceqa_base_diagnostic.py; finish_insuranceqa_base_diagnostic.py. Use existing analysis environment for CPU stages and Windows GPU environment with scripts/env_windows.ps1 for generation/embedding. Generation refuses to overwrite attempts.','',
       'Original four adapters, selected cohorts, website defaults and previous reports remain preserved. No teacher calls, retraining, filtering or deployment in this diagnostic.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'decision.json',{'decision':'audit_and_task_matched_training_recommended_no_training_started','base_support':24,'adapter_support':8,'base_coherence_pass':28,'adapter_coherence_pass':14,'scope':'Unblinded32-case development diagnostic, not general forgetting or factual accuracy','report_sha256':sha(OUT/'report.md')})
    v=read_json(OUT/'verification.json');v.update(identical_prompts_anchors_sampling_verified=True,base_adapter_disabled_recorded=True,source_audit_sample_verified=True,
        artifacts={str(path.relative_to(OUT)):sha(path) for path in OUT.iterdir() if path.is_file() and path.name!='verification.json'})
    write_json(OUT/'verification.json',v);print('Diagnostic and source audit complete:',OUT/'report.md')

if __name__=='__main__':main()
