"""Freeze the final pilot decision and easy-to-read evidence summary."""
from dataset_catalog import read_json,sha
from insuranceqa_final_pilot import OUT,check
from json_io import write_json

def main():
    check();m=read_json(OUT/'metrics.json');review=read_json(OUT/'review_summary.json');copy=read_json(OUT/'anchor_copying.json')
    a=m['arms']['baseline'];b=m['arms']['grounded'];text=m['text_metrics']['joint']['arms'];kw=m['keywords']['collections']
    findings=[
      f"All 32 attempts in each arm were structurally valid; neither arm hit its generation token limit.",
      f"Domain/length SD quality improved {a['sdmetrics']['quality']:.4f} -> {b['sdmetrics']['quality']:.4f}; keyword JS distance improved {kw['baseline']['js_distance_base2']:.4f} -> {kw['grounded']['js_distance_base2']:.4f}. These are distribution scores, not accuracy.",
      f"Joint MMD-squared changed {text['baseline']['mmd2']:.5f} -> {text['grounded']['mmd2']:.5f} (lower is closer). Use only this matched 32-output comparison; do not compare its magnitude directly with the main 128-output score.",
      'Assistant coherence review: original 10 Pass / 8 Unsure / 14 Issue; grounded 14 Pass / 8 Unsure / 10 Issue. The reviewer was unblinded, these are different generated questions, and this is not a factual-accuracy estimate or a test of repairing the same questions.',
      'Source support review: only 8/32 grounded outputs passed; 24/32 introduced material the supplied passage did not establish. This is an assistant support judgment, not proof that all added claims are false.',
      f"Exact training-answer reuse increased 0/32 -> 3/32. All three equal the supplied anchor answer. Mean literal 8-word-phrase overlap with the anchor was {copy['mean_8gram_overlap']:.1%}. Whole-record matches were zero in both arms; changing the question can hide answer copying if only full-record matches are checked.",
      'The source also contains dated claims, questionable broad rules and a web-address typo. Source support alone cannot establish current insurance truth. The eligibility regex excludes HTTP/www URLs and common contact patterns, but it did not remove bare-domain web addresses.',
    ]
    decision={'decision':'reject_keep_existing_policy','plain_english':'The final prompt improved some resemblance and coherence measures, but it did not reliably follow its source and increased exact answer copying. Keep the original saved InsuranceQA policy. Document the mixed result and stop experiments.',
      'findings':findings,'review':review,'no_more_experiments':True,'original_adapter_and_selected_cohorts_unchanged':True,
      'protocol_sha256':sha(OUT/'protocol.json'),'metrics_sha256':sha(OUT/'metrics.json'),'judgments_sha256':sha(OUT/'judgments.jsonl')}
    write_json(OUT/'decision.json',decision)
    lines=['# Final InsuranceQA source-context experiment','',decision['plain_english'],'',
       '## Fixed design','', '32 attempts per arm; 8/8 passed the initial format gate. Same adapter, batch size4, temperature1.0, top-p.95, top-k64 and seed schedule. Context1536; different input and remaining-output budgets. Source passages and domains supplied externally from training only. No training, teacher calls, rerolls or post-generation filtering. Full protocol and every attempt are saved.','',
       '## Results','',*['- '+v for v in findings],'',
       '## Limits and final decision','',
       'One small development cohort, unblinded assistant review, different questions, no expert calibration and no untouched final test. Source-anchor eligibility excludes long/contact-bearing examples and changes the source mix within domains. Do not interpret the coherence counts as a controlled factual improvement. No original model/default or selected cohort is replaced. This is the final experiment.','',
       'Evidence: protocol.json, freeze.json, baseline_generations.jsonl, grounded_generations.jsonl, metrics.json, anchor_copying.json, judgments.jsonl, review_summary.json, embedding_manifest.json, decision.json and verification.json.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(decision['plain_english'])

if __name__=='__main__':main()
