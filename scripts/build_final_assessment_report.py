"""Evidence-first final assessment report, with a plain-English reading path."""
from pathlib import Path
import json
import html
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'outputs/insuranceqa-final-grounded-v1'

def main(include_one_shot=False):
    m=json.loads((PILOT/'metrics.json').read_text());decision=json.loads((PILOT/'decision.json').read_text())
    bench=json.loads((ROOT/'reports/distribution_benchmark/v1/results.json').read_text())['datasets']
    stem='gemma_final_assessment_updated' if include_one_shot else 'gemma_final_assessment'
    output=ROOT/('output/pdf/'+stem+'.pdf');md=ROOT/('reports/'+stem+'.md')
    ink=colors.HexColor('#203448');teal=colors.HexColor('#126c75');light=colors.HexColor('#edf5f6')
    styles={'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=ink,spaceAfter=16),
      'head':ParagraphStyle('head',fontName='Helvetica-Bold',fontSize=19,leading=24,textColor=ink,spaceAfter=14),
      'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=teal,spaceBefore=10,spaceAfter=7),
      'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10.5,leading=14.7,textColor=ink,spaceAfter=9),
      'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9,leading=12,textColor=ink)}
    story=[];markdown=[]
    def p(text,style='body'):
        story.append(Paragraph(html.escape(text),styles[style]));markdown.extend([text,''])
    def page(title,first=False):
        if not first:story.append(PageBreak())
        p(title,'title' if first else 'head');markdown[-2]='# '+title
    def table(headers,rows,widths):
        cells=[[Paragraph(html.escape(str(c)),styles['cell']) for c in row] for row in [headers,*rows]]
        t=Table(cells,colWidths=[w*mm for w in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),light),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.8,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe4'))]))
        story.extend([t,Spacer(1,9)]);markdown.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows],''])
    page('Gemma synthetic data: final assessment',True)
    p('Evidence and decisions | 2 November 2026')
    p('The assignment: fine-tune Gemma to generate synthetic examples that match the ranges, distribution and characteristics of its training data.')
    p('We trained four separate QLoRA adapters, generated new records, compared them with their sources, tested improvements and retained the useful policies. Bitext provides the strongest demonstrated case. InsuranceQA and Reliability still have answer/workflow weaknesses. The clinical result is limited to a small MIMIC demo.')
    p('Three questions keep the evidence clear','sub')
    table(['Question','What we measure'],[
      ['Does it resemble the source?','Topic and keyword frequencies; field/length distributions; embedding similarity and coverage.'],
      ['Does it make sense?','Question-answer alignment, internal consistency, workflow handling and limited source-support review.'],
      ['Is it simply copying?','Normalized full-record, request and reply matches; diversity and source overlap.']], [54,120])
    p('There is no single overall accuracy score. A distribution score can be high while an answer is wrong. The reported numbers describe saved outputs; they do not establish production readiness or professor approval.')
    p('Reading order: data and training -> evaluation rules -> retained results -> experiments -> InsuranceQA prompt comparisons -> conclusion and evidence.')
    if include_one_shot:p('Updated edition: includes the subsequently requested one-example prompt comparison. Earlier selected results remain unchanged.')

    page('1. Data and training, in plain words')
    p('An adapter is a small learned addition to the existing model. QLoRA keeps the large model weights fixed in 4-bit form and trains adapter weights. It makes a 4B model feasible on the local 8 GB GPU. Rank 8 and alpha 16 were used for the original adapters.')
    table(['Dataset','Train / validation / test','What one example contains'],[
      ['Bitext','31,200 / 3,900 / 3,900','Intent, category, language tags, customer request and support response.'],
      ['InsuranceQA','12,889 / 1,961 / 1,963','Insurance domain, question and one annotated positive answer. Distractor answers are excluded.'],
      ['Reliability','357 / 76 / 77','Customer message, policy context/history and expected workflow actions.'],
      ['MIMIC-IV demo','230 / 21 / 24 admissions','Structured hospital-admission fields; 80 / 10 / 10 distinct patients.']], [31,47,96])
    p('Training teaches patterns. Validation checks progress and chooses checkpoints. The test set assesses a selected model, but our original test sets have already been inspected. Later experiments are development comparisons, not a fresh untouched final test.')
    p('Training performed','sub')
    p('Bitext: 200 pilot updates followed by 3,700 more, totaling one epoch. InsuranceQA: one full epoch, 1,612 updates; validation selected checkpoint 1,600. Reliability: three epochs, 135 updates. Original MIMIC: three epochs. A separate conditioned clinical candidate used 460 prompt views over two epochs; its budget differed from the original model.')
    p('InsuranceQA did not lose its long training examples: all 12,889 were retained; the longest was 1,402 tokens within a 1,536-token training context. None of its original 128 generated outputs reached the generation limit. More tokens or more epochs were therefore not an established fix.')
    p('MIMIC is the official 100-patient demo. It is not the complete MIMIC database and does not represent the U.S. population. Its diagnosis codes and clinical relationships were not independently validated.')

    page('2. How we measure the requirement')
    table(['Measure','Easy explanation and interpretation'],[
      ['SDMetrics QualityReport','Do selected fields and relationships resemble the source? Higher is closer. InsuranceQA uses only domain and question/answer lengths; this is not an answer-quality score.'],
      ['MMD-squared','How different are two collections in a fixed text-feature space? Lower is closer. We use an RBF kernel and report the biased nonnegative statistic.'],
      ['PRDC','Precision: resembles source regions. Recall: covers source variety. Density: nearby source concentration (can exceed 1). Coverage: source neighborhoods reached. None measures factual accuracy.'],
      ['Keyword JS distance','Do the source top 100 content words appear in similar proportions? Lower is closer; OTHER retains the remaining vocabulary mass.'],
      ['Ranges and frequencies','Character-length percentiles, Wasserstein distance, category/diagnosis counts and missing values show which characteristics differ.'],
      ['Validity, copying, review','Count every attempted output. Check exact normalized reuse separately. Assistant Pass/Unsure/Issue reviews examine limited meaning/coherence, not expert factual accuracy.']], [43,131])
    p('Comparability rules','sub')
    p('Main text embeddings use pinned GPT2-large terminal-token features, float32, L2 normalization and a 512-token cap. Request, reply and joint views are scored separately. MMD and PRDC share this encoder, so they are complementary views rather than independent semantic confirmation.')
    p('The main benchmark uses 128 embedding records per text collection, except Reliability (64). Source reference/control subsets, IDs, hashes and sampling settings are saved. Descriptive real-real comparisons provide context, not pass thresholds or confidence intervals. Small-sample tails and PRDC are uncertain.')
    p('SDMetrics uses all valid records. Invalid outputs remain in validity counts. Clinical text metrics and keyword scores are not applicable to structured records; diagnosis frequencies are saved instead. SDMetrics fields differ by dataset, so their overall scores must not rank datasets against each other.')

    page('3. Retained four-dataset results')
    rows=[]
    for key,arm,label in [('bitext','selected_prefix','Bitext'),('insuranceqa','original_generic','InsuranceQA'),('reliability','selected_prefix','Reliability'),('mimic_demo','selected_candidate','MIMIC demo')]:
        q=bench[key];c=q['cohorts'][arm];text=q.get('text_metrics');kw=q.get('keywords')
        rows.append([label,str(c['core']['structure_valid'])+'/'+str(c['core']['attempted']),f"{c['sdmetrics']['quality']:.4f}",f"{text['joint']['collections'][arm]['mmd2']:.5f}" if text else 'N/A',f"{kw['collections'][arm]['js_distance_base2']:.4f}" if key!='mimic_demo' else 'N/A'])
    table(['Retained policy','Valid','SD quality','Joint MMD2','Keyword JS'],rows,[37,24,35,39,39])
    p('Bitext','sub')
    p('On matched 156-output arms, externally supplied intent/category prefixes improved SD quality from 0.7893 to 0.9465, joint MMD2 from 0.01344 to 0.00714 and keyword JS from 0.0683 to 0.0329. This demonstrates a more faithful controlled pipeline, not independently learned category frequencies. One full record and 26.92% of replies matched training; copying remains a limitation.')
    p('InsuranceQA','sub')
    p('The retained 128 outputs are structurally valid, with joint PRDC recall 0.75. Domain/length resemblance is reasonably strong, but semantic coverage and answer quality remain weaker. A prior stratified review found 15 Pass, 4 Unsure and 11 Issue in 30 records. No full-record or answer matches were found in 128 outputs; 13 questions matched training. This means no exact complete-answer reuse was detected, not that every answer is good or privacy is guaranteed.')
    p('Reliability','sub')
    p('On matched 64-output arms, topic prefixes improved SD quality 0.6879 -> 0.8228, MMD2 0.01943 -> 0.01478 and keyword JS 0.2559 -> 0.2075. The selected 64 passed source checks; assistant review was 45 Pass, 15 Unsure, 4 Issue. Deductible omissions, policy-context errors and payment/history handling remain.')
    p('MIMIC demo','sub')
    p('With shared diagnosis/null conditions, the separate candidate improved SD quality 0.6428 -> 0.6705 and several numeric distances. Age distance improved 7.249 -> 2.535 years, but generated ages 41-86 still miss younger source ages. Some category/relationship metrics worsened; 63/64 passed source constraints. Diagnosis/null inputs are external controls; clinical correctness is unverified.')

    page('4. The experiment trajectory')
    table(['What we tried','What happened and why we kept/rejected it'],[
      ['Train and select adapters','Completed four adapters; selected using validation loss. Lower loss showed better prediction of source-style completions, not guaranteed factual quality.'],
      ['Larger, detailed profile prompts','Some topic/length scores improved, but format reliability worsened. InsuranceQA fell from 64/64 to 54/64 valid. Rejected as a default.'],
      ['Short prefix controls','Small pilot followed by larger matched comparisons. Kept for Bitext and Reliability because distribution resemblance improved. Prior InsuranceQA 7/8 valid versus 8/8 generic did not pass its gate.'],
      ['Conditioned clinical candidate','Retained as a separate assessment candidate for modest structured improvement. Original clinical model remains saved; unequal training budgets disclosed.'],
      ['Reliability consistency prompt','On 12 diagnostic cases, original 8 Pass/4 Issue versus guarded 7 Pass/2 Unsure/3 Issue; only 11/12 valid and no known failure became Pass. Rejected.'],
      ['Gemini black-box distillation','First teacher batch collapsed policy types and was rejected. Revised batch yielded 8 accepted examples. Matched 18-update distilled arm had 7 Pass/5 Issue versus original 8 Pass/4 Issue; no known issue fixed. Rejected. This was not full adversarial GAD.'],
      ['Unified distribution benchmark','Computed and saved SDMetrics, MMD2, PRDC, keyword and range/copying evidence. Independent verification recalculated 39 MMD and 7 keyword profiles; all 47 pre-pilot tests passed.']], [45,129])
    p('A failed experiment is evidence when the comparison is saved and the worse policy is rejected. We did not treat a better-looking single metric as sufficient to replace a model.')

    page('5. InsuranceQA source-context experiment')
    p('Hypothesis: providing a short source answer may reduce invented details and mismatched question-answer pairs. We used the existing adapter without retraining, with an eight-attempt format gate and a maximum of 32 attempts. All raw outputs are saved; there are no retries or post-hoc exclusions.')
    p('Inputs came only from training data. Full answers up to 1,100 characters, excluding HTTP/www links, email and common phone patterns, were eligible. Bare-domain web addresses could remain. Domain quotas came from the original training frequencies. This pre-generation eligibility excludes long/contact-bearing anchors and may bias the result. The new prompt asks for a new question and supported answer in its own wording.')
    p('The control is the first matching number of original generic outputs, with the same batch size, sampling settings and seed schedule. Prompt/context budgets differ. Source context and domain control change together, so this tests a combined pipeline. This small comparison cannot establish broad improvement or repair the 11 earlier failures.')
    pilotrows=[]
    for arm in ['baseline','grounded']:
        a=m['arms'][arm];t=m.get('text_metrics',{}).get('joint',{}).get('arms',{}).get(arm,{})
        pilotrows.append([arm,str(a['structure_valid'])+'/'+str(a['attempted']),f"{a['sdmetrics']['quality']:.4f}",f"{t['mmd2']:.5f}" if t else 'N/A',f"{m['keywords']['collections'][arm]['js_distance_base2']:.4f}"])
    table(['Arm','Valid','SD quality','Joint MMD2','Keyword JS'],pilotrows,[37,24,35,39,39])
    p('Decision','sub');p(decision['plain_english'])
    for text in decision['findings'][:4]:p(text)
    page('5b. Why source context alone was rejected')
    for text in decision['findings'][4:]:p(text)
    p('Concrete examples from the saved outputs','sub')
    table(['Observed example','What it teaches us'],[
      ['Grounded output 11','The question is about unemployment and home insurance. The answer repeats the complete supplied passage. It fits the question, but does not create a new answer.'],
      ['Grounded output 15','The passage defines bonus annuities. The generated question asks about nonforfeitable annuities while the answer repeats the bonus definition. Copying source words does not ensure question-answer fit.'],
      ['Grounded output 13','A passage about dated PPACA conditions leads to many invented zip-code prices and internally conflicting rating statements. Supplying context does not guarantee the adapter follows it.']], [46,128])
    p('Review limitations','sub')
    p('Assistant judgments are unblinded and not expert calibrated. Source support is only checked in the grounded arm, because the baseline received no source. A source-supported answer can repeat an old or incorrect insurance claim. Reuse and answer-substring overlap are reported separately; no cheap lexical filter is treated as proof of truth.')

    if include_one_shot:
        directory=ROOT/'outputs/insuranceqa-one-shot-v1'
        latest=json.loads((directory/'metrics.json').read_text());d=json.loads((directory/'decision.json').read_text())
        review=json.loads((directory/'review_summary.json').read_text())
        page('6. One worked example: follow-up test')
        p('After the earlier report, the owner explicitly requested one further prompt experiment. The aim was to show the model the desired transformation, rather than add more verbal rules. No new fine-tuning was performed.')
        p('Exactly what changed','sub')
        p('A single short fictional demonstration was placed before each existing source-grounded prompt. It shows a policy covering accident towing but excluding mechanical breakdown, followed by a new engine-failure question and a matching paraphrased answer. It is a teaching example, not a statement about real insurance coverage.')
        p('We reused the same 32 source passages, domain labels, adapter, batch size, seed schedule, temperature 1.0, top-p 0.95 and top-k 64. The control is the previous source-grounded output, not the original generic generator. Longer inputs reduce the remaining output budget. All raw attempts are saved without retries or filtering.')
        table(['Arm','Valid','SD quality','Joint MMD2','Keyword JS'],[
           ['Source only',str(latest['arms']['baseline']['structure_valid'])+'/'+str(latest['arms']['baseline']['attempted']),f"{latest['arms']['baseline']['sdmetrics']['quality']:.4f}",f"{latest['text_metrics']['joint']['arms']['baseline']['mmd2']:.5f}",f"{latest['keywords']['collections']['baseline']['js_distance_base2']:.4f}"],
           ['Source + example',str(latest['arms']['grounded']['structure_valid'])+'/'+str(latest['arms']['grounded']['attempted']),f"{latest['arms']['grounded']['sdmetrics']['quality']:.4f}",f"{latest['text_metrics']['joint']['arms']['grounded']['mmd2']:.5f}",f"{latest['keywords']['collections']['grounded']['js_distance_base2']:.4f}"]],[37,24,35,39,39])
        p('What happened','sub')
        for finding in d['findings'][:3]:p(finding)
        p('The baseline judgments are reused unchanged from the prior experiment. New outputs are reviewed by the same conversation assistant, unblinded and without expert calibration. Passage support does not establish current factual truth. These are paired source inputs, but the generated questions differ.')
        page('6b. Follow-up decision and examples')
        p(d['plain_english'])
        for finding in d['findings'][3:]:p(finding)
        p('Examples from the saved follow-up','sub')
        table(['Output','Observation'],d['examples'],[31,143])
        p('What this changes in our conclusion','sub')
        p(d['interpretation'])
        p('Limits','sub')
        p('One demonstration and one seed sequence on already inspected development anchors. No untouched-test or statistical significance claim. Demonstration content can itself change the topic/wording distribution. Small samples do not establish tail coverage. No result here validates the current truth of historical InsuranceQA answers.')
    page(('7.' if include_one_shot else '6.')+' Conclusion, evidence and reproduction')
    p('We completed the engineering task and measured the requested characteristics with a fixed framework. The evidence supports useful distribution improvements in controlled generation. It also identifies remaining answer/workflow problems and clinical scope limits. It does not prove exact distribution equivalence, factual accuracy, privacy or general U.S.-market fidelity.')
    p('Final boundary','sub')
    if include_one_shot:p('The one-example follow-up is complete and documented above. Original adapters and previously retained cohorts remain saved. No further training, prompt sweeps or teacher calls are planned; no website default was automatically replaced.')
    else:p('This was the last experiment. Preserve the original adapters and previously retained cohorts. No further training, prompt sweeps or teacher calls are planned. The final pilot is documented as accepted or rejected on its evidence; original outputs remain available either way.')
    table(['Evidence','Project-relative location'],[
      ['Four-dataset metrics and charts','reports/distribution_benchmark/v1/report.md and index.html'],
      ['Full scores, inputs and audit','reports/distribution_benchmark/v1/results.json, protocol.json, provenance.json, verification.json'],
      ['Final pilot and every attempt','outputs/insuranceqa-final-grounded-v1/ (protocol, raw outputs, metrics, judgments, decision, verification)'],
      ['Prior experiments','outputs/prefix-alignment-v1/; outputs/fidelity-alignment-v1/; outputs/reliability-consistency-v1/; outputs/reliability-distillation-v2/'],
      ['Source research and decisions','reports/insuranceqa_improvement_research.md; decision.md'],
      ['Owner learning guide','reports/gemma_explained.md; output/pdf/gemma_explained.pdf']], [48,126])
    p('Commands','sub')
    if include_one_shot:
        p('Follow-up evidence: outputs/insuranceqa-one-shot-v1/ contains the report, protocol, every raw output, metrics, judgments, decision and verification. Reproduction stages: scripts/insuranceqa_one_shot_pilot.py score / embed / text_score. Independent check: scripts/verify_insuranceqa_one_shot.py. See protocol.json for frozen prompts and settings; generation refuses to overwrite attempts.')
    else:
        p('Analysis environment: .venv-analysis/Scripts/python.exe scripts/insuranceqa_final_pilot.py score; then text_score after embeddings. GPU embeddings: load scripts/env_windows.ps1, set HF_HUB_OFFLINE=1, then .venv-windows/Scripts/python.exe scripts/insuranceqa_final_pilot.py embed. Generation is intentionally guarded against overwriting attempts; its exact frozen prompts are in protocol.json.')
    p('Full benchmark verification: .venv-analysis/Scripts/python.exe scripts/verify_distribution_benchmark.py. Pilot verification separately checks protocol/raw/model/source hashes and calculations. Build with the bundled document Python: scripts/build_final_assessment_report.py'+(' --include-one-shot.' if include_one_shot else '.'))
    p('Method references','sub')
    p('MMD: jmlr.org/papers/v13/gretton12a.html. PRDC: github.com/clovaai/generative-evaluation-prdc. SDMetrics: docs.sdv.dev/sdmetrics/data-metrics/quality. Source-grounding inspiration: aclanthology.org/P19-1620/ and aclanthology.org/2020.emnlp-main.468/. Those QA studies are not direct InsuranceQA/Gemma evidence.')
    def footer(canvas,doc):
        canvas.setFillColor(ink);canvas.setFont('Helvetica',8);canvas.drawString(18*mm,12*mm,'Gemma assessment | 2 November 2026 | Development evidence');canvas.drawRightString(192*mm,12*mm,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=20*mm,title='Gemma synthetic data - final assessment').build(story,onFirstPage=footer,onLaterPages=footer)
    md.write_text('\n'.join(markdown),encoding='utf-8');reader=PdfReader(output)
    print('Final report:',len(reader.pages),'pages;',output)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--include-one-shot',action='store_true');args=parser.parse_args()
    main(args.include_one_shot)
