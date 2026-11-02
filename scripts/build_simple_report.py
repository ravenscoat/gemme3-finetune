"""Plain-English assessment report with a separate methods/evidence appendix."""
import html
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from pypdf import PdfReader
from dataset_catalog import ROOT,read_json,sha

PDF=ROOT/'output/pdf/gemma_report.pdf'
MD=ROOT/'reports/gemma_report.md'
INK=colors.HexColor('#203045');TEAL=colors.HexColor('#176d77');LIGHT=colors.HexColor('#edf4f6')
SOURCES=[
('1','Training comparison','reports/training_stage_comparison.md'),
('2','Original four-dataset assessment','reports/all_datasets_assessment.md'),
('3','Detailed prompts failed','outputs/fidelity-alignment-v1/report.md'),
('4','Template diagnostic','outputs/schema-prompt-pilot-v1/report.md'),
('5','Simpler topic controls','outputs/prefix-alignment-v1/report.md'),
('6','Clinical candidate comparison','outputs/mimic-demo-fidelity-conditioned-v1-windows/recommendation.md'),
('7','Rejected workflow fix','outputs/reliability-consistency-v1/report.md'),
('8','Final selections and limitations','reports/assessment_snapshot/report.md'),
('9','Real-sample comparison context','reports/criteria_evidence/v1/report.md'),
]

def main():
    snapshot=read_json(ROOT/'reports/assessment_snapshot/snapshot.json')
    freeze=read_json(ROOT/'reports/final_experiment_freeze.json')
    for key,item in snapshot['datasets'].items():
        assert sha(ROOT/item['records_path'])==freeze['selected_cohorts'][key]['records_sha256']
    style={
      'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=INK,spaceAfter=14),
      'heading':ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=19,leading=24,textColor=INK,spaceAfter=12),
      'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=12,spaceAfter=6),
      'body':ParagraphStyle('body',fontName='Helvetica',fontSize=11,leading=16,textColor=INK,spaceAfter=10),
      'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=12,textColor=INK,spaceAfter=6),
      'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9.4,leading=13,textColor=INK),
    }
    story=[];md=[]
    def p(text,kind='body'):
        story.append(Paragraph(html.escape(text),style[kind]));md.extend([text,''])
    def heading(text,first=False):
        story.append(Paragraph(html.escape(text),style['title' if first else 'heading']));md.extend([('# ' if first else '## ')+text,''])
    def sub(text):
        story.append(Paragraph(html.escape(text),style['sub']));md.extend(['### '+text,''])
    def table(headers,rows,widths):
        t=Table([[Paragraph(html.escape(str(v)),style['cell']) for v in row] for row in [headers,*rows]],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dbe3eb'))]))
        story.extend([t,Spacer(1,9)])
        md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(str(v) for v in row)+' |' for row in rows],''])

    heading('Fine-tuning Gemma',True)
    p('Synthetic data assessment | Evidence frozen 1 November 2026','small')
    sub('The task')
    p('Fine-tune Gemma to create new examples that resemble its training data. The assignment asks for matching ranges, distribution and characteristics.')
    p('In easy words: values should stay within source-like ranges, common topics should appear often, and examples should have the source format and style. Matching these statistics does not establish factual or clinical correctness.')
    sub('The four datasets')
    table(['Dataset','Train','Validation','Test'],[
        ['Bitext support requests/replies','31,200','3,900','3,900'],
        ['InsuranceQA question/answer pairs','12,889','1,961','1,963'],
        ['Reliability workflow scenarios','357','76','77'],
        ['MIMIC-IV demo admissions','230','21','24'],
    ],[72*mm,34*mm,34*mm,34*mm])
    sub('What we did')
    for title,text in [
      ('1. Start small.','We tested a short Bitext training run before spending time on a full pass.'),
      ('2. Train all four.','We saved four separate adapters: small learned changes to the same Gemma 3 4B base model. QLoRA made training practical on the 8 GB GPU.'),
      ('3. Compare the outputs.','We checked format, values, category proportions, copying and examples that did not make sense.'),
      ('4. Try targeted improvements.','We tested simpler topic controls for insurance data and a separate clinical training candidate.'),
      ('5. Keep the evidence.','We kept improvements, reported regressions, rejected a failed final fix and froze the experiments.'),
    ]:p(title+' '+text)
    p('MIMIC uses the official 100-patient demo, not the full database or a representative U.S. population. Its patient-disjoint split is 80/10/10 patients. Training taught the models; validation guided checkpoint choice; the original selected models were assessed on the reserved tests without weight updates. Later candidate/prompt work is development evidence, not a new independent test. [2, 8]','small')

    story.append(PageBreak());heading('What improved - and what failed')
    p('Each change answered a specific problem. We did not assume that more training or a longer prompt would improve everything.')
    table(['What we tried','Evidence','What we did next'],[
      ['200-step Bitext pilot, continued to 3,900 total steps','Original -> pilot -> full pass: valid records 483 -> 499 -> 500 out of 500; topics covered 2 -> 37 -> 37 of 39. [1]','Keep the selected full-pass model. Category distance was worse than the pilot, and reply reuse increased.'],
      ['Longer, detailed prompts','Valid records dropped to 121/156 for Bitext and 2/64 for Reliability. [3]','Reject the approach and investigate the format.'],
      ['Complete JSON templates','Reliability reached 15/16 valid, but 13/16 contained template markers. [4]','Reject. A valid format was not enough.'],
      ['Simpler topic prefixes','Bitext: 156/156 valid, all 39 topics covered; reply-length distance 22.35 -> 7.24. Reliability: 64/64 valid; category distance 0.114 -> 0.029. [5]','Keep these evaluated policies, with clear disclosure that we supplied the labels.'],
      ['Clinical training candidate','With the same conditions, age distance fell 7.25 -> 2.54; stay-length distance 2.00 -> 1.04 days. All five measured numeric distances improved. [6]','Retain the candidate for presentation, alongside its remaining category and relationship gaps.'],
      ['Final workflow-consistency prompt','Validity fell 12/12 -> 11/12. No flagged case became a Pass; a completed-payment error persisted. [7]','Reject the change. Stop experimenting and prepare the submission.'],
    ],[42*mm,86*mm,46*mm])
    sub('What those improvements mean')
    p('A smaller distribution distance means closer on that measurement. It is not an accuracy percentage. For example, a reply-length distance of 7.24 describes a difference in lengths, not whether the answer is correct.')
    p('The topic prefixes provide labels before Gemma writes the remaining text. Matching those proportions is a control we applied, not proof of learned topic probabilities. Clinical diagnosis and missing-lab conditions were identical in its two arms; training budgets differed, so this is not a single-factor causal comparison. [5, 6]','small')
    p('The clinical and insurance work followed separate branches; this table groups the decisions so they are easy to follow. The longer project journey keeps more detail.','small')

    story.append(PageBreak());heading('Where we stand now')
    table(['Dataset','What is working','What still needs work'],[
      ['Bitext (156 examples)','Strongest source resemblance; all 39 topics represented.','One whole-record training match; 26.9% of replies match source text.'],
      ['Reliability (64)','Required format and selected label checks pass for all 64.','AI review: 45 Pass, 15 Unsure, 4 Issue. Workflow mistakes remain.'],
      ['InsuranceQA (128)','All 128 have the required format; no whole-record or answer matches.','A prior 30-example AI review found 11 Issue and 4 Unsure. We kept the original method after new prompts failed.'],
      ['MIMIC demo (64)','All five measured numeric distributions improved under shared conditions.','One unsupported race label; some proportions worsened. Clinical correctness is unverified.'],
    ],[34*mm,65*mm,75*mm])
    sub('How we show closeness to the criterion')
    p('Ranges: compare minimum, middle and maximum values, plus missing values. Distribution: compare category bars and numeric charts. Characteristics: inspect format, writing style, relationships and meaningful errors. Copying is shown separately. [8]')
    p('Range matching remains incomplete: source clinical ages span 21-90, while the selected 64 generated records span 41-86. No observed out-of-range values does not mean the tails are covered. Some clinical category proportions and correlations worsened. [8]','small')
    p('We also sampled real training rows 200 times at the same size as each synthetic cohort. This puts distances in context. These bands are not pass thresholds or a new final test: samples overlap the source, and clinical sampling is by admission rather than patient. [9]','small')
    sub('Our honest conclusion')
    p('Fine-tuning improved structure and source resemblance. Additional controls helped some distributions. Results are mixed across datasets, and we have not proved that every part of the criterion is solved. AI reviews are one unblinded assistant review, not expert accuracy.')
    p('Stronger future claims would require independent generation runs, insurance-expert review and clinical validation. The current evidence supports measurable, dataset-specific improvements; it does not establish full distribution equivalence or readiness for clinical use.')
    p('The methods and exact evidence paths are in the appendix. The report is a separate deliverable from the demonstration website.','small')

    story.append(PageBreak());heading('Appendix: methods and evidence')
    sub('Training setup')
    p('Gemma 3 4B with 4-bit QLoRA adapters (rank 8, alpha 16). Bitext and InsuranceQA completed one epoch; original Reliability and clinical training used three. The clinical candidate used two epochs over 460 prompt views of 230 admissions. Exact learning rates, seeds, checkpoints, pinned base revision and generation settings are retained in the bundled protocols and generation_profiles.json. [2, 6]','small')
    sub('How to interpret the comparisons')
    p('Original Bitext training comparison: 500 attempts per model. Later prefix comparison: 156 per arm for Bitext and 64 per arm for Reliability, with eight pilot examples reused in each selected cohort. InsuranceQA retains its original 128-output assessment. The clinical comparison uses 64 per arm under shared conditions. These cohorts are not pooled into one score. [1, 5, 6, 8]','small')
    p('Validity counts every attempt; malformed outputs remain failures. Distribution distances and copying rates use schema-valid outputs. Only complete outer JSON fences may be removed; malformed outputs are not repaired or rerolled. Copy matching ignores case, punctuation and spacing. Zero full-record matches is not a privacy guarantee. [2, 5]','small')
    p('TV is the gap between category proportions (0 is exact agreement). Wasserstein compares numeric distributions in their original units; lower is closer. Neither proves equivalent joint distributions. No acceptance threshold was supplied, so no universal pass score is invented. The 200-repeat real-sample bands are descriptive context, not confidence intervals or pass thresholds. [8, 9]','small')
    p('AI judgments use one unblinded assistant reviewer without expert calibration. The final Reliability trial fixed four known problems and eight passing controls before generation; its 24 outputs assess workflow completion, not unconditional source fidelity. The trial was rejected because validity, passing controls and action-count matching regressed. [7, 8]','small')
    sub('Reproducibility')
    p('The submission includes exact generation profiles/settings, raw outputs, training protocols, source/model provenance and SHA256 manifests. Verify the bundle with: python verify_submission.py. Derived reports can be rebuilt in the original prepared workspace with scripts/build_simple_report.py and scripts/build_criteria_evidence.py, using their document/analysis dependencies. Base weights and raw training data are excluded; see the bundle README.','small')
    sub('Evidence files in the submission bundle')
    p('IDs match the citations in this report. Paths start at the bundle root; the editable Markdown includes direct workspace links.','small')
    for num,title,path in SOURCES:
        assert (ROOT/path).exists()
        p('['+num+'] '+title+': '+path,'small')
    md.extend(['## Clickable evidence',''])
    for num,title,path in SOURCES:md.append(f'- [{num}: {title}]({(ROOT/path).as_posix()})')
    md.extend(['','[Longer project journey]('+(ROOT/'output/pdf/gemma_project_journey.pdf').as_posix()+')',
               '[Interactive charts](http://127.0.0.1:8765/evidence/criteria)'])
    MD.write_text('\n'.join(md)+'\n',encoding='utf-8')
    def footer(c,doc):
        c.saveState();c.setStrokeColor(TEAL);c.line(18*mm,17*mm,192*mm,17*mm);c.setFont('Helvetica',8);c.setFillColor(INK);c.drawString(18*mm,12*mm,'GEMMA / ASSESSMENT REPORT / 1 NOVEMBER 2026');c.drawRightString(192*mm,12*mm,str(doc.page));c.restoreState()
    SimpleDocTemplate(str(PDF),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=23*mm,title='Fine-tuning Gemma for synthetic data - assessment report').build(story,onFirstPage=footer,onLaterPages=footer)
    pages=PdfReader(PDF).pages
    text='\n'.join(p.extract_text() or '' for p in pages)
    for phrase in ['121/156','26.9%','200 times','Where we stand now']:assert phrase in text,phrase
    (ROOT/'output/pdf/gemma_report_verification.json').write_text(json.dumps({'pages':len(pages),'pdf_sha256':sha(PDF),'markdown_sha256':sha(MD),'cohorts_unchanged':True,'sources':{path:sha(ROOT/path) for _,_,path in SOURCES},'visual_review':'Pending rendered-page inspection'},indent=2)+'\n',encoding='utf-8')
    print('Short report:',PDF,'pages:',len(pages))

if __name__=='__main__':main()
