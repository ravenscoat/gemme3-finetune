"""Create an accessible experiment narrative from frozen project evidence.

Run with the bundled document Python (reportlab + pypdf). No ML jobs are run.
"""
import html
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from pypdf import PdfReader

from dataset_catalog import ROOT, read_json, sha

OUT=ROOT/'output/pdf'
MD=ROOT/'reports/gemma_project_journey.md'
TEAL=colors.HexColor('#167d8d')
INK=colors.HexColor('#203045')
ORANGE=colors.HexColor('#d98136')
LIGHT=colors.HexColor('#edf4f6')

EVIDENCE=[
 ('E1','Patient-disjoint clinical preparation and split audit','data/mimic_demo/prepared/preparation_audit.json'),
 ('E2','Original / 200-step pilot / full-pass comparison','reports/training_stage_comparison.md'),
 ('E3','Original four-dataset assessment and test results','reports/all_datasets_assessment.md'),
 ('E4','First detailed-prompt experiment: every attempt','outputs/fidelity-alignment-v1/report.md'),
 ('E5','Nested-template diagnostic and marker leaks','outputs/schema-prompt-pilot-v1/report.md'),
 ('E6','Small label-prefix diagnostic','outputs/prefix-prompt-pilot-v1/report.md'),
 ('E7','Larger Bitext / Reliability prefix comparison','outputs/prefix-alignment-v1/report.md'),
 ('E8','Clinical candidate: comparison and limitations','outputs/mimic-demo-fidelity-conditioned-v1-windows/recommendation.md'),
 ('E9','All 64 Reliability workflow judgments','reports/reliability_workflow_audit/v1/report.md'),
 ('E10','Rejected final workflow-prompt attempt','outputs/reliability-consistency-v1/report.md'),
 ('E11','Current selections, ranges and tradeoffs','reports/assessment_snapshot/report.md'),
 ('E12','New descriptive real-sample reference bands','reports/criteria_evidence/v1/report.md'),
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=read_json(ROOT/'reports/assessment_snapshot/snapshot.json')
    freeze=read_json(ROOT/'reports/final_experiment_freeze.json')
    for key,item in snapshot['datasets'].items():
        assert sha(ROOT/item['records_path'])==freeze['selected_cohorts'][key]['records_sha256']
    clinical=read_json(ROOT/'outputs/mimic-demo-fidelity-conditioned-v1-windows/pilot_metrics.json')['arms']
    prefix={key:read_json(ROOT/f'outputs/prefix-alignment-v1/{key}/metrics.json')['arms'] for key in ['bitext','reliability']}
    bands=read_json(ROOT/'reports/criteria_evidence/v1/reference_bands.json')['datasets']
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleLocal',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=INK,spaceAfter=13))
    styles.add(ParagraphStyle(name='SectionLocal',fontName='Helvetica-Bold',fontSize=18,leading=23,textColor=INK,spaceAfter=12))
    styles.add(ParagraphStyle(name='SubLocal',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=5))
    styles.add(ParagraphStyle(name='BodyLocal',fontName='Helvetica',fontSize=10.2,leading=14.5,textColor=INK,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallLocal',fontName='Helvetica',fontSize=8,leading=11,textColor=INK,spaceAfter=6))
    styles.add(ParagraphStyle(name='CellLocal',fontName='Helvetica',fontSize=8.5,leading=11,textColor=INK))
    story=[];md=[]
    def p(text,small=False):
        text=text.replace('\u2192',' -> ').replace('\u2013','-').replace('\u2014','-')
        safe=html.escape(text)
        safe=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',safe)
        story.append(Paragraph(safe,styles['SmallLocal' if small else 'BodyLocal']))
        md.extend([text,''])
    def h(text,title=False):
        story.append(Paragraph(html.escape(text),styles['TitleLocal' if title else 'SectionLocal']))
        md.extend([('# ' if title else '## ')+text,''])
    def sub(text):
        story.append(Paragraph(html.escape(text),styles['SubLocal']))
        md.extend(['### '+text,''])
    def page():
        story.append(PageBreak())
    def table(headers,rows,widths):
        cells=[[Paragraph(html.escape(str(v)),styles['CellLocal']) for v in row] for row in [headers,*rows]]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
            ('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#dbe3eb'))]))
        story.extend([t,Spacer(1,7)])
        md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                   *['| '+' | '.join(str(v) for v in row)+' |' for row in rows],''])
    def callout(text):
        box=Table([[Paragraph(html.escape(text),styles['BodyLocal'])]],colWidths=[174*mm])
        box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,TEAL),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10)]))
        story.extend([box,Spacer(1,9)]);md.extend(['> '+text,''])
    def bar_chart(title,labels,values,max_value=100,suffix='%',height=150):
        drawing=Drawing(495,height)
        drawing.add(String(0,height-14,title,fontName='Helvetica-Bold',fontSize=10,fillColor=INK))
        left=155;plot_width=280
        for i,(label,value) in enumerate(zip(labels,values)):
            y=height-48-i*29
            drawing.add(String(0,y+4,label,fontName='Helvetica',fontSize=9,fillColor=INK))
            drawing.add(Rect(left,y,plot_width,16,fillColor=LIGHT,strokeColor=None))
            drawing.add(Rect(left,y,plot_width*value/max_value,16,fillColor=TEAL,strokeColor=None))
            drawing.add(String(left+plot_width+8,y+4,f'{value:.1f}{suffix}',fontName='Helvetica-Bold',fontSize=9,fillColor=INK))
        story.extend([drawing,Spacer(1,5)])

    # Page 1
    h('Gemma: the project journey',True)
    p('An easy-to-read account of what we tried, what improved, what failed, and why we changed direction. Evidence frozen as of 1 November 2026.',True)
    callout('The task: fine-tune Gemma to generate synthetic data that matches the ranges, distribution and characteristics of its training data.')
    p('We built four separate fine-tuned adapters on the same Gemma 3 4B base model. An adapter is a small set of learned changes saved alongside the original model. We used QLoRA to make training practical on the local 8 GB GPU. This was not four complete model copies trained from scratch.')
    sub('The data we actually used')
    table(['Dataset','What one record looks like','Train / validation / test'],[
        ['Bitext','Support request + response + intent/category/tags','31,200 / 3,900 / 3,900'],
        ['InsuranceQA V2','Insurance question + one professional-style answer + domain','12,889 / 1,961 / 1,963'],
        ['Reliability','Customer message + context/history + expected workflow','357 / 76 / 77'],
        ['MIMIC-IV demo','Structured admission: age, diagnosis, stay and lab values','230 / 21 / 24 admissions'],
    ],[30*mm,91*mm,53*mm])
    p('Bitext contains 39,000 source examples; 31,200 were used for training. For MIMIC we used the official 100-patient demo, split by patient: 80 training patients, 10 validation and 10 test. It is not the full MIMIC database or a representative sample of the U.S. population. [E1, E3]')
    sub('The route through the project')
    p('Start small -> train the useful adapters -> compare original and fine-tuned outputs -> find distribution gaps -> test prompt changes -> keep the useful changes and reject regressions -> freeze the experiments and present the evidence.')
    p('We did not treat fluent text as proof of success. We checked structure, source resemblance, copying and semantic problems separately. The final result is mixed: stronger source resemblance, with clear remaining limits.')
    p('Reading order: the clinical candidate in Section 4 was trained before the shared prompt diagnostics in Section 2. We group the clinical branch separately so its own decisions and results are easy to follow.',True)

    # Page 2
    page();h('1. We proved the training approach first')
    sub('Why start with a short Bitext run?')
    p('Bitext gave us many consistent examples and clear labels. We first used a 200-step pilot to check that the pipeline worked. Once it showed a useful improvement, we continued from that checkpoint for 3,700 more updates: 3,900 optimizer steps in total, or one full pass. A step is a training update; it is not one example.')
    table(['500 attempts per model','Original Gemma','200-step pilot','Full pass'],[
        ['Structure-valid records','483 / 500','499 / 500','500 / 500'],
        ['Intents covered, out of 39','2','37','37'],
        ['Intent distance (TV; lower is closer)','0.949','0.173','0.150'],
        ['Category distance (TV)','0.744','0.092','0.130'],
        ['Normalized reply reuse','0.0%','0.0%','27.0%'],
        ['Whole-record training matches','0.0%','0.0%','0.6%'],
    ],[77*mm,32*mm,32*mm,33*mm])
    bar_chart('Bitext: intent coverage grew after fine-tuning', ['Original','200-step pilot','Full pass'],[100*2/39,100*37/39,100*37/39])
    p('What improved: the model covered far more of the source topics and produced the required record format more consistently. What did not improve everywhere: category proportions were closer in the pilot than after the full pass, and source reply reuse increased. More training was not a win on every measure. [E2]')
    sub('We then completed the other adapters')
    table(['Original assessment','Original valid','Fine-tuned valid'],[
        ['InsuranceQA','106 / 128','128 / 128'],['Reliability','0 / 128','128 / 128'],['MIMIC demo','127 / 128','128 / 128'],
    ],[86*mm,44*mm,44*mm])
    p('The original fine-tuned runs were one epoch for Bitext and InsuranceQA, and three epochs for Reliability and MIMIC. An epoch is one pass through the training examples. The big gain in Reliability was producing the complete nested scenario format. Valid structure still did not establish correct workflows. [E3]')

    # Page 3
    page();h('2. Longer prompts were not an automatic fix')
    p('The first assessments exposed gaps in topic proportions, text lengths, missing values and supported labels. We tried supplying more explicit instructions to guide generation toward the source.')
    table(['Attempt','What happened','Our next decision'],[
        ['Detailed source-profile prompts','Schema-valid: Bitext 121/156; Reliability 2/64; InsuranceQA 54/64. All were below their matched generic baselines.','Reject this general approach. Investigate output structure. [E4]'],
        ['Complete nested JSON templates','Reliability recovered to 15/16 valid, but 13/16 leaked template markers. Bitext was 8/8 valid, yet only 2/8 passed source checks.','Reject. Correct-looking JSON can still contain placeholders or wrong labels. [E5]'],
        ['Minimal label prefixes','Small pilot: Bitext 8/8 and Reliability 8/8 valid, with no marker leaks. InsuranceQA still produced only 7/8 valid.','Expand only the two passing datasets. Keep the original InsuranceQA method. [E6]'],
    ],[38*mm,87*mm,49*mm])
    sub('What is a label prefix?')
    p('We supply the opening fields of the record, such as its intent and category, and let Gemma complete the remaining text. Think of choosing a topic before asking someone to write a new example. We stopped supplying free-text template placeholders and many extra range/style instructions.')
    callout('Important: choosing the topic proportions outside the model is a generation control. It does not prove that the model learned to choose those proportions by itself.')
    sub('The larger follow-up')
    p('We evaluated 156 Bitext examples and 64 Reliability examples under this policy. Eight examples in each dataset were reused from the pilot; the remainder were fresh attempts. The comparisons below use equally sized generic cohorts from the same adapters. We retained failed attempts and did not repair or reroll outputs. [E7]')
    p('The lesson: clearer, smaller output guidance worked better here than simply making the prompt longer. This is an observed result from these experiments, not a claim that longer prompts always fail.')

    # Page 4
    page();h('3. The simpler controls helped, with tradeoffs')
    sub('Bitext: closer lengths and complete intent coverage')
    b=prefix['bitext'];r=prefix['reliability']
    def numeric(arm,field):return arm['fidelity_vs_training']['numeric'][field]['wasserstein']
    table(['Bitext, 156 attempts per arm','Generic prompt','Label prefix'],[
        ['Structure-valid / source-label checks','156/156; 156/156','156/156; 156/156'],
        ['Intents represented','36 / 39','39 / 39'],
        ['Intent distance (TV)','0.250','0.000'],
        ['Request-length distance (characters)',f'{numeric(b["baseline"],"request_characters"):.3f}',f'{numeric(b["controlled"],"request_characters"):.3f}'],
        ['Reply-length distance (characters)',f'{numeric(b["baseline"],"reply_characters"):.3f}',f'{numeric(b["controlled"],"reply_characters"):.3f}'],
        ['Normalized reply reuse','24.36%','26.92%'],
        ['Whole-record matches','1 / 156','1 / 156'],
    ],[94*mm,40*mm,40*mm])
    p('The zero intent distance was created by the supplied label schedule. The text-length improvement is separate evidence: Gemma generated the request and reply text. Repeated replies and one whole-record training match remained. We kept these weaknesses visible. [E7]')
    sub('Reliability: better labels, mixed workflow statistics')
    table(['Reliability, 64 attempts per arm','Generic prompt','Label prefix'],[
        ['Structure-valid records','64 / 64','64 / 64'],
        ['Source-label checks','48 / 64','64 / 64'],
        ['Broad category distance (TV)','0.1139','0.0287'],
        ['Subcategory distance (TV)','0.5256','0.2078'],
        ['Customer-message length distance',f'{numeric(r["baseline"],"request_characters"):.3f}',f'{numeric(r["controlled"],"request_characters"):.3f}'],
        ['Required-action count distance',f'{numeric(r["baseline"],"required_action_count"):.4f}',f'{numeric(r["controlled"],"required_action_count"):.4f}'],
    ],[94*mm,40*mm,40*mm])
    p('Some labels and message lengths moved closer to the source, while action-count matching became worse. Channel and state distributions also worsened. We therefore did not describe the prefix as a complete fix. The next question was whether the workflows actually made sense. [E7, E9]')
    p('Distances here use Wasserstein unless marked TV. Zero means exact agreement on that measured distribution; lower is closer. Numeric distances keep their original units and are not accuracy percentages.',True)

    # Page 5
    page();h('4. MIMIC needed a different approach')
    p('The early clinical generator made valid records, but it did not reproduce missing lab values well and had diagnosis-prefix/version inconsistencies. We trained a separate candidate using both ordinary prompts and condition-specific prompts for the same 230 training admissions.')
    p('The candidate used 460 prompt views over two epochs. For the comparison below, both old and new adapters received the same 64 diagnosis-pair and lab-missingness conditions. This holds those generation controls constant, but the training budgets differ. [E8]')
    old=clinical['old_conditioned'];new=clinical['candidate_conditioned']
    fields=[('age','Age (years)'),('length_of_stay_days','Stay length (days)'),('creatinine_mg_dl','Creatinine (mg/dL)'),('glucose_mg_dl','Glucose (mg/dL)'),('hemoglobin_g_dl','Hemoglobin (g/dL)')]
    table(['Distribution distance','Old adapter','New candidate'],[
        [label,f'{numeric(old,f):.3f}',f'{numeric(new,f):.3f}'] for f,label in fields
    ],[94*mm,40*mm,40*mm])
    ratios=[100*numeric(new,f)/numeric(old,f) for f,label in fields]
    bar_chart('Remaining numeric distance: old adapter = 100 for each field', [label for _,label in fields],ratios,height=205)
    p('All five measured numeric distances improved. A shorter bar means less distribution mismatch, not clinical accuracy. The SDMetrics statistical-quality score also rose from 0.643 to 0.671. Both adapters obeyed 64/64 supplied conditions and had zero observed diagnosis-prefix flags in this conditioned comparison. [E8]')
    p('What remained: the candidate introduced one race label absent from training. Admission-type, insurance and mortality proportions became less similar, and numeric correlation mismatch rose slightly from 0.0838 to 0.0879. Clinical plausibility and complete diagnosis-code validity were not established.')
    p('A later, heavier controller improved numeric matching further but produced two prefix/version flags and three source-label failures. We retained the lighter candidate for presentation: zero such flags and one source-label failure. It remains separate from the original live MIMIC default. [E11]')

    # Page 6
    page();h('5. We inspected meaning, then rejected a failed fix')
    sub('Reliability: good format did not mean a good workflow')
    p('The assistant reviewed all 64 selected Reliability examples: 45 Pass, 15 Unsure and 4 Issue. This was one unblinded AI review, without expert calibration. It is useful error analysis, not a measured population accuracy rate. [E9]')
    table(['Concrete concern','Why it matters'],[
        ['Case 18: deductible question','The workflow focused on breach response without explicitly answering the deductible question.'],
        ['Case 33: conflicting policy','A homeowners request conflicted with commercial-liability context, yet the workflow proceeded with an update.'],
        ['Case 42: payment already completed','History said payment was processed; the new workflow included process_payment again.'],
        ['Case 63: conflicting coverage context','Flood advice was given against health-policy context without clearly reconciling the policy.'],
    ],[56*mm,118*mm])
    sub('One bounded consistency-prompt experiment')
    p('We froze four problematic cases and eight passing controls. Both prompts received the same original inputs and labels; Gemma generated only the expected workflow. The new prompt asked it to respect history, resolve policy conflicts and answer explicit questions. This was workflow completion, not a fresh unconditional generation test. [E10]')
    table(['12 attempts per arm','Baseline prompt','Added consistency instruction'],[
        ['Structure-valid records','12 / 12','11 / 12'],
        ['AI Pass / Unsure / Issue','8 / 0 / 4','7 / 2 / 3'],
        ['Previously flagged cases becoming Pass','-','0'],
        ['Required-action count distance','0.0805','0.1487'],
    ],[88*mm,38*mm,48*mm])
    p('The completed-payment error persisted. One previously passing document workflow became uncertain, and a deductible case became malformed JSON. Workflow-length matching improved, but that did not compensate for the regressions. We rejected the change and froze the experiments. [E10]')
    callout('This was a useful negative result: we tested a specific idea, found that it did not meet the predeclared checks, and kept the previous policy instead of presenting a failed fix as progress.')

    # Page 7
    page();h('6. How close are we to the assignment criterion?')
    table(['Criterion','What we show','What it cannot prove'],[
        ['Ranges','Minimum, median, maximum, missingness and out-of-range rates','A plausible range does not guarantee correct relationships or adequate tail coverage.'],
        ['Distribution','Category proportions, numeric/text-length distances and charts','A small distance on one field does not establish the full joint distribution.'],
        ['Characteristics','Structure, style, relationships and workflow examples','A valid JSON record may still be factually or clinically wrong.'],
        ['Originality check','Normalized record and field reuse','Zero exact copies does not establish privacy or exclude paraphrased memorization.'],
    ],[30*mm,75*mm,69*mm])
    sub('We added a real-sample reference for the distances')
    p('A genuine small sample also differs from its full dataset. We drew equally sized training samples 200 times and compared each with the same full training reference used for synthetic scores. The middle 90% of these distances gives descriptive context. [E12]')
    rows=[]
    for key,field,label in [('bitext','reply_characters','Bitext reply length'),('reliability','required_action_count','Reliability action count'),('mimic_demo','age','MIMIC age')]:
        v=bands[key]['numeric'][field];b=v['real_sample_reference']
        rows.append([label,f'{v["synthetic_distance"]:.3f}',f'{b["p05"]:.3f} - {b["p95"]:.3f}'])
    table(['Example measurement','Synthetic distance','Real-sample 5th-95th percentile'],rows,[70*mm,40*mm,64*mm])
    p('These are not pass thresholds, confidence intervals or independent final tests. Samples overlap the full source; MIMIC sampling is by admission rather than patient. Controlled labels can match better than random sampling by design. We do not turn these scores into an overall accuracy percentage.')
    p('Supporting analyses included exploratory MAUVE and PRDC text-distribution checks, plus SDMetrics for clinical records. These supplement the direct field-level charts. Small text cohorts limit interpretation; these scores are not insurance or clinical accuracy. [E3]',True)
    sub('Where each dataset stands')
    p('**Bitext:** strongest evidence of source-like generation; reply reuse and a whole-record copy remain. **Reliability:** better source-label control, but workflow mistakes remain. **InsuranceQA:** retain the original generator; 128/128 valid and no whole-record copies, but a prior 30-example AI review found 11 Issue and 4 Unsure. **MIMIC demo:** better numeric matching under shared controls, with categorical tradeoffs and unverified clinical correctness. [E9, E11]')
    p('Selected presentation cohorts are Bitext 156, Reliability 64, InsuranceQA 128 and MIMIC 64. They are not pooled into one score. The prefix cohorts include pilot reuse, and the newer work is development evidence rather than a fresh final test.')

    # Page 8
    page();h('7. Evidence, reproducibility and what comes next')
    sub('How the splits were used')
    p('Training examples taught the model. Validation data helped monitor training and choose checkpoints. The original selected adapters were then assessed on the held-out test splits without updating weights. Test loss measures prediction of source records, not synthetic-data correctness. Those test sets have been used; later prompt/candidate work is reported as development work, with no new final-test scoring. [E3]')
    sub('Open the evidence')
    p('The PDF uses source IDs [E1]-[E12]. All paths below are relative to D:/gemma-finetune. The editable Markdown report contains clickable absolute links. InsuranceQA and Reliability have their own preparation_audit.json under data/<dataset>/prepared; Bitext provenance is data/bitext/source_manifest.json.',True)
    table(['ID','Evidence and project path'],[[ident,label+'; '+path] for ident,label,path in EVIDENCE],[12*mm,162*mm])
    sub('The next useful work')
    p('Prepare the presentation and rehearse the demo. A stronger future study would add independent generation seeds, insurance-expert review and clinical/codebook validation. These are not completed experiments. Attribution, gradient alignment, distillation and downstream training utility were considered but were not part of the completed evidence.')
    p('The saved models and original output cohorts remain unchanged. The report and visual evidence explain both improvements and failures. No claim is made that every dataset fully satisfies the criterion or that the clinical data represent the U.S. population.')

    # Clickable evidence index in Markdown, plus exact data sources for chart values.
    md.extend(['## Clickable evidence index',''])
    for ident,label,path in EVIDENCE:
        assert (ROOT/path).exists(),path
        md.append(f'- **[{ident}]** [{label}]({(ROOT/path).as_posix()})')
    md.extend(['','Interactive evidence: [Open the criterion dashboard](http://127.0.0.1:8765/evidence/criteria).',
        'Offline charts: [Open the saved page]('+(ROOT/'reports/criteria_evidence/v1/index.html').as_posix()+').',
        '', 'Metric inputs for the plotted clinical comparisons: `outputs/mimic-demo-fidelity-conditioned-v1-windows/pilot_metrics.json`. Prefix comparisons: `outputs/prefix-alignment-v1/<dataset>/metrics.json`. All values are descriptive of the saved cohorts.'])
    MD.write_text('\n'.join(md)+'\n',encoding='utf-8')
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL);canvas.setLineWidth(.5);canvas.line(18*mm,18*mm,192*mm,18*mm)
        canvas.setFont('Helvetica',8);canvas.setFillColor(INK)
        canvas.drawString(18*mm,13*mm,'GEMMA / PROJECT JOURNEY / 1 NOVEMBER 2026')
        canvas.drawRightString(192*mm,13*mm,str(doc.page));canvas.restoreState()
    pdf=OUT/'gemma_project_journey.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=17*mm,bottomMargin=24*mm,
        title='Gemma: the project journey',author='Gemma fine-tuning project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(pdf)
    extracted='\n'.join(page.extract_text() or '' for page in reader.pages)
    for text in ['Gemma: the project journey','200-step pilot','121/156','process_payment','0.1487','E12']:
        assert text in extracted,text
    write={'pdf_pages':len(reader.pages),'pdf_sha256':sha(pdf),'markdown_sha256':sha(MD),
        'cohorts_verified_unchanged':True,'note':'Text checks passed; rendered pages require visual inspection.',
        'source_files':{path:sha(ROOT/path) for _,_,path in EVIDENCE}}
    (OUT/'gemma_project_journey_verification.json').write_text(json.dumps(write,indent=2)+'\n',encoding='utf-8')
    print('PDF:',pdf,'pages:',len(reader.pages),flush=True)
    print('Markdown:',MD,flush=True)


if __name__=='__main__':main()
