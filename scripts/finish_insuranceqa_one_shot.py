"""Write the evidence-based decision for the newly authorized one-example test."""
from dataset_catalog import read_json,sha
from insuranceqa_one_shot_pilot import OUT
from json_io import write_json

def main():
    m=read_json(OUT/'metrics.json');review=read_json(OUT/'review_summary.json');copy=read_json(OUT/'anchor_copying.json')
    a=m['arms']['baseline'];b=m['arms']['grounded'];t=m['text_metrics']['joint']['arms'];kw=m['keywords']['collections']
    findings=[
      'Assistant coherence review changed from 14 Pass / 8 Unsure / 10 Issue to 17 Pass / 8 Unsure / 7 Issue. This is a small unblinded review of different generated questions, not an accuracy gain or statistical significance claim.',
      'Source-support passes changed 8/32 -> 7/32; unsupported/contradictory outputs 24/32 -> 25/32. Two formerly unsupported source cases became supported, while three formerly supported cases became unsupported. Unsupported additions may be true, false or unverifiable; support review does not decide that.',
      f"Exact training-answer reuse fell 3/32 -> 2/32. The two equal the supplied source answers. Mean generated 8-word phrase overlap with its source is {copy['mean_8gram_overlap']:.1%}, versus 27.9% before. Full-record matches remain 0/32 in both arms; partial copying still occurs.",
      f"Distribution matching regressed in field/length SD quality ({a['sdmetrics']['quality']:.4f} -> {b['sdmetrics']['quality']:.4f}) and joint MMD2 ({t['baseline']['mmd2']:.5f} -> {t['grounded']['mmd2']:.5f}, lower is closer). Keyword JS improved ({kw['baseline']['js_distance_base2']:.4f} -> {kw['grounded']['js_distance_base2']:.4f}). Metrics do not agree on a universal improvement.",
      f"Median answer length fell {a['length_percentiles']['answer']['generated'][2]:g} -> {b['length_percentiles']['answer']['generated'][2]:g} characters; the 95th percentile fell {a['length_percentiles']['answer']['generated'][3]:.1f} -> {b['length_percentiles']['answer']['generated'][3]:.1f}. Shorter answers are not automatically better for the source-distribution requirement.",
      'Both arms are 32/32 valid, preserve all 32 supplied domain labels and hit no generation limits. Simple lexical checks found no towing/engine-failure demonstration-topic leakage absent from the source. This narrow check does not prove no semantic influence or copying.',
      'A supplied renters-labelled anchor discusses homeowners insurance. The new output faithfully follows that inconsistent source. Better prompting cannot by itself make contradictory source labels and prose jointly correct.',
    ]
    d={'decision':'reject_keep_original_generic_policy','plain_english':'One worked example made some outputs more coherent and reduced exact answer copying slightly, but did not improve source support and worsened two distribution measures. Do not adopt this prompt or the earlier source-only prompt. Keep the original saved generic InsuranceQA policy and document the limits.',
       'findings':findings,'examples':[
         ['Output 2','A source-based definition of Social Security benefits stays on topic; much phrasing still resembles the source.'],
         ['Output 10','Source allows ordinary insurance for a bus driver\'s private car; output says drivers do not qualify, then discusses commercial buses.'],
         ['Output 11','The unemployment/home-insurance answer is copied in full despite the example showing paraphrasing.'],
         ['Output 27','The answer paraphrases the passage well, but the supplied renters label conflicts with the homeowners subject.']],
       'interpretation':'This specific one-example prompt is not a reliable fix. It changes generation style more successfully than it enforces source support. Both inference-time source prompts differ from the generic task used during training; that mismatch remains a plausible explanation, not a proven cause. Source auditing and context-conditioned training would be separate future work, not actions performed here.',
       'original_selected_policies_unchanged':True,'automatic_deployment':False,'further_experiments_planned':False,
       'protocol_sha256':sha(OUT/'protocol.json'),'metrics_sha256':sha(OUT/'metrics.json'),'judgments_sha256':sha(OUT/'judgments.jsonl')}
    write_json(OUT/'decision.json',d)
    (OUT/'report.md').write_text('\n'.join(['# InsuranceQA one-example prompt comparison','',d['plain_english'],'',
       'The user explicitly reopened experiments after the earlier freeze. One fictional passage-to-QA example was prepended to the exact earlier32prompts. Same source anchors, adapter, seeds and sampling; smaller remaining output budgets. Format gate8/8; all32attempts completed. No training/teacher calls/rerolls/output filtering. Baseline is the earlier source-only pipeline, which was already rejected, not the retained generic policy.','',
       '## Results','',*['- '+f for f in findings],'','## Interpretation','',d['interpretation'],'',
       'One seed schedule, small already-inspected development cohort and unblinded uncalibrated assistant review. No untouched test, expert factual validation or equivalence claim. PRDC, three embedding views, length percentiles, all keyword frequencies and copying indicators are saved in metrics.json and associated CSVs. Full protocol, raw outputs, judgments, decision and independent verification are retained. No additional experiment or default replacement is planned.'])+'\n',encoding='utf-8')
    print(d['plain_english'])

if __name__=='__main__':main()
