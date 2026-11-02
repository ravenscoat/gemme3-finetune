"""Build the owner's plain-English learning guide from frozen project evidence."""
from pathlib import Path
import html
import json
import re
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/gemma_explained.pdf'
MD=ROOT/'reports/gemma_explained.md'
INK=colors.HexColor('#233548');TEAL=colors.HexColor('#176c75')
LIGHT=colors.HexColor('#edf5f6');LINE=colors.HexColor('#d6e1e7')
SOURCES=[
 ('1','Training stages','reports/training_stage_comparison.md'),
 ('2','Original four-dataset assessment','reports/all_datasets_assessment.md'),
 ('3','Detailed-prompt experiment','outputs/fidelity-alignment-v1/report.md'),
 ('4','Template experiment','outputs/schema-prompt-pilot-v1/report.md'),
 ('5','Small prefix pilot','outputs/prefix-prompt-pilot-v1/report.md'),
 ('6','Larger prefix comparison','outputs/prefix-alignment-v1/report.md'),
 ('7','Clinical candidate comparison','outputs/mimic-demo-fidelity-conditioned-v1-windows/recommendation.md'),
 ('8','Reliability review of 64 records','reports/reliability_workflow_audit/v1/report.md'),
 ('9','Rejected consistency prompt','outputs/reliability-consistency-v1/report.md'),
 ('10','Selected cohorts and limitations','reports/assessment_snapshot/report.md'),
 ('11','Real-sample reference bands','reports/criteria_evidence/v1/report.md'),
 ('12','Gemini distillation pilot','outputs/reliability-distillation-v2/report.md'),
 ('13','Distillation audit','outputs/reliability-distillation-v2/verification.json'),
]

def main():
    # Fail rather than silently documenting an unfinished or different experiment.
    result=json.loads((ROOT/'outputs/reliability-distillation-v2/decision.json').read_text())
    assert result['decision']=='reject_keep_existing_policy'
    assert result['workflow_review_counts']['distilled']=={'Pass':7,'Issue':5}
    for _,_,path in SOURCES:assert (ROOT/path).is_file(),path
    OUT.parent.mkdir(parents=True,exist_ok=True)
    style={
      'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=29,leading=34,textColor=INK,spaceAfter=14),
      'heading':ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=21,leading=26,textColor=INK,spaceAfter=12),
      'sub':ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=10,spaceAfter=5),
      'body':ParagraphStyle('body',fontName='Helvetica',fontSize=11,leading=15.5,textColor=INK,spaceAfter=9),
      'small':ParagraphStyle('small',fontName='Helvetica',fontSize=9,leading=12.5,textColor=INK,spaceAfter=6),
      'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=9.8,leading=13.5,textColor=INK),
    }
    story=[];md=[]
    def p(text,kind='body'):
        story.append(Paragraph(html.escape(text),style[kind]));md.extend([text,''])
    def title(text,first=False):
        if not first:story.append(PageBreak())
        story.append(Paragraph(html.escape(text),style['title' if first else 'heading']));md.extend([('# ' if first else '## ')+text,''])
    def sub(text):
        story.append(Paragraph(html.escape(text),style['sub']));md.extend(['### '+text,''])
    def table(headers,rows,widths=None):
        width=174*mm
        widths=[width/len(headers)]*len(headers) if widths is None else [x*mm for x in widths]
        content=[[Paragraph(html.escape(str(x)),style['cell']) for x in row] for row in [headers,*rows]]
        obj=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
           ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
           ('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,LINE)]))
        story.extend([obj,Spacer(1,9)])
        md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
        md.extend(['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows]);md.append('')
    def callout(text):
        item=Table([[Paragraph(html.escape(text),style['body'])]],colWidths=[174*mm])
        item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('LEFTPADDING',(0,0),(-1,-1),12),
           ('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([item,Spacer(1,10)]);md.extend(['> '+text,''])

    class ProjectFlow(Flowable):
        def __init__(self):
            Flowable.__init__(self);self.width=174*mm;self.height=48
        def draw(self):
            labels=['Four sources','Train adapters','New records','Compare','Keep / reject']
            gap=12;w=(self.width-4*gap)/5;c=self.canv
            for i,label in enumerate(labels):
                x=i*(w+gap);c.setFillColor(LIGHT);c.setStrokeColor(TEAL)
                c.roundRect(x,9,w,31,5,fill=1,stroke=1)
                c.setFillColor(INK);c.setFont('Helvetica-Bold',9);c.drawCentredString(x+w/2,21,label)
                if i<4:
                    c.setStrokeColor(TEAL);c.line(x+w+2,24,x+w+gap-2,24)
                    c.line(x+w+gap-5,27,x+w+gap-2,24);c.line(x+w+gap-5,21,x+w+gap-2,24)

    title('Your Gemma project, explained',True)
    p('A learning guide for you | Updated 1 November 2026','small')
    p('This report explains the assignment, the decisions we made, the experiments we ran, and what the results actually mean. It includes the latest Gemini experiment. You do not need a machine-learning background to read it.')
    sub('What were we asked to do?')
    callout('Fine-tune Gemma to generate synthetic data. The generated data should match the ranges, distribution and characteristics of the data used to train it.')
    p('Synthetic data means newly generated examples. For insurance, that can be a new customer request and support reply. For MIMIC, it is a fictional structured hospital-admission record. We want believable new examples that resemble the source, rather than simply copying it.')
    table(['The requirement','In everyday language'],[
      ['Ranges','Do values cover a similar span? For example, does generated patient age resemble the source age range?'],
      ['Distribution','Do common topics and values appear often, and rare ones less often?'],
      ['Characteristics','Do the records have the right fields, wording, missing values and sensible relationships?'],
    ],[41,133])
    sub('Where are we now?')
    p('We trained four separate Gemma adapters, measured their outputs, tried improvements and kept the useful ones. Bitext is our strongest case. The other datasets have measurable improvements and remaining problems. The latest Gemini distillation attempt was rejected because it made the small Reliability comparison worse.')
    p('That failed attempt did not erase the earlier work. The original models and selected assessment outputs remain saved and unchanged. [10, 12, 13]')
    p('Reading route: data -> training -> evaluation -> experiments -> current position. The last two pages give you a glossary, a short explanation to practise and the evidence locations.','small')
    story.append(ProjectFlow());md.extend(['Four sources -> train adapters -> generate new records -> compare -> keep or reject.',''])

    title('1. What does our data look like?')
    p('We did not combine all four sources into one mixed dataset. Each teaches a different kind of example, so we trained and evaluated a separate adapter for each.')
    sub('Bitext: a customer asks; support answers')
    p('One record has intent, category, tags, instruction and response. Intent means what the customer wants; instruction is the request; response is the matching support reply. Tags describe language variations.')
    callout('Illustration, not an actual source row: intent = cancel_insurance_policy; instruction = "I want to cancel my policy"; response = a support reply explaining the cancellation process.')
    sub('InsuranceQA: an insurance question and answer')
    p('One record has domain, question and answer. Domain is the insurance topic. We prepared one positive answer per question. This task needs insurance knowledge, not just a convincing support tone.')
    p('Illustration: domain = auto-insurance; question = "What does a deductible mean?"; answer = an explanation of the customer-paid part of a covered claim. This is a teaching illustration, not an evaluated model answer.','small')
    sub('Reliability: a scenario and the expected workflow')
    p('A record contains category, subcategory, difficulty, a customer message, policy context, conversation history and expected workflow. The workflow says where to route the case, which actions are needed and what the response should cover.')
    p('For example, if history says a payment was already processed, the next workflow should recognise that rather than process it again. This dataset checks consistency across several parts of a record.')
    sub('MIMIC demo: a structured hospital admission')
    p('Fields include age, gender, admission type, insurance, race, length of stay, mortality flag, diagnosis code/version and three lab values: creatinine, glucose and hemoglobin. Some values can be missing, written as null.')
    p('We used the official MIMIC-IV demo: 100 patients, with 275 prepared admission records. It is not the full MIMIC database and does not represent the whole U.S. market or population. [2, 10]')

    title('2. Why did we make three data splits?')
    p('Think of training as studying, validation as practice exams and testing as the reserved exam. Each serves a different purpose.')
    table(['Split','What it does','Does it update the model?'],[
      ['Training','The examples the model learns from.','Yes, during training.'],
      ['Validation / evaluation','Checks progress during training and helps compare checkpoints.','No weight updates from evaluating these rows. It can influence our choices.'],
      ['Test','Checks the selected model after training.','No weight updates.'],
    ],[32,102,40])
    table(['Dataset','Training','Validation','Test'],[
      ['Bitext','31,200','3,900','3,900'],
      ['InsuranceQA','12,889','1,961','1,963'],
      ['Reliability','357','76','77'],
      ['MIMIC demo admissions','230','21','24'],
    ],[69,35,35,35])
    p('MIMIC was split by patient: 80 training patients, 10 validation patients and 10 test patients. A person can have several admissions. Keeping each person in one split helps prevent their related admissions appearing on both sides of the comparison. [2]')
    sub('Did we evaluate during training?')
    p('Yes. Validation loss was monitored during the training runs. Loss measures how well the model predicts the source text. Lower usually means better text prediction; it does not directly measure whether a generated insurance workflow is correct.')
    sub('Why do we keep calling later results "development"?')
    p('The original test splits were used in the original assessment. After inspecting results and designing fixes, we cannot honestly call later comparisons a completely fresh exam. Later outputs and the reused diagnostic cases are development evidence. The Gemini pilot reused known cases; it did not provide a new untouched final test. [2, 12]')
    callout('Generating 600 evaluation examples would mean asking for 600 outputs to measure. It would not mean training on 600 more examples.')

    title('3. How did we actually train Gemma?')
    p('The active project uses Gemma 3 4B. "4B" describes about four billion model parameters: the numbers that store its learned patterns. The earlier 270M experiment was dropped; it is not our selected result.')
    sub('Fine-tuning, LoRA and QLoRA')
    p('Fine-tuning means continuing learning from an existing model using our task examples. We did not build Gemma from scratch.')
    p('During training, we give Gemma a generation instruction and a source record as the desired output. It learns to complete records like those examples. Later we give the instruction without the answer and ask for a new record. This is generator training, not just chatbot-answer training.')
    p('LoRA learns small extra sets of numbers called adapters, while keeping the main model weights fixed. QLoRA also stores the main model in a compact 4-bit form to reduce GPU memory use. This made training practical on the local 8 GB GPU. We used rank 8 and alpha 16; these control the adapter setup. [2]')
    callout('One shared base model + four separately trained adapters = four specialised configurations. We did use LoRA through QLoRA; we did not update every base-model weight.')
    table(['Term','What it means in our project'],[
      ['Batch','A group of examples processed together. Gradient accumulation lets several small groups contribute to one update.'],
      ['Step','One optimizer update: the training process changes the adapter numbers. A step is not necessarily one example.'],
      ['Epoch','One complete pass through the training examples.'],
      ['Checkpoint','A saved training state. It lets us preserve progress and compare saved versions.'],
    ],[36,138])
    sub('Why start with 200 steps?')
    p('The short Bitext pilot checked that learning worked before committing more time. We then continued for 3,700 more updates, reaching 3,900 total steps: one full pass over 31,200 training examples. That is eight examples per update on average in this run. [1]')
    p('Bitext and InsuranceQA originally used one epoch. Reliability and the original MIMIC runs used three. Smaller datasets need fewer steps per pass, so three epochs on a small dataset can finish much sooner than one epoch on Bitext. More epochs do not automatically mean better results.')
    p('The project used the D: drive and native Windows training. Earlier WSL setup work is not the active training path. Final adapters and checkpoints are saved on disk; a closed terminal does not erase them.','small')

    title('4. How do we measure "close to the data"?')
    p('A model can produce neat JSON and still give a wrong answer. It can also generate sensible individual examples while getting the overall topic mix wrong. We therefore use several checks together.')
    table(['Check','Simple explanation'],[
      ['Format / schema','Are the required fields present, with the correct types? JSON is just a structured way to write a record.'],
      ['Source checks','Are labels and values supported by the prepared source? This is narrower than factual correctness.'],
      ['Ranges / missingness','Compare low, middle and high values, and how often fields are missing.'],
      ['Category proportions / TV','Compare the mix of topics or labels. TV is a category-distribution gap: 0 is an exact match, 1 is the maximum gap.'],
      ['Numeric distance / Wasserstein','Compare two distributions, not just their averages. Lower is closer. Age distance uses years; text-length distance uses characters.'],
      ['Relationships','Check whether fields move together similarly. Matching age and stay separately does not prove their relationship matches.'],
      ['Copying / reuse','Look for whole records and individual fields matching training data after normalising case, punctuation and spacing.'],
      ['Meaning / AI review','Read examples for missed questions, contradictory context or repeated actions. This is not expert-validated accuracy.'],
    ],[47,127])
    sub('A simple example of distribution matching')
    p('Suppose 40% of source requests concern payments. If only 5% of generated requests do, the generated set misses the source mix even if every payment reply looks good. These numbers are an illustration, not our measured dataset proportions.')
    p('We also ran exploratory MAUVE and PRDC text checks. These use a model to map texts into numerical features, then compare how close and varied the collections look in that map. SDMetrics compares statistical patterns in structured clinical records. These add evidence, not insurance or medical certification; small samples limit them. [2, 10]','small')
    p('No universal pass threshold was supplied. A distance is not an accuracy percentage, and matching a few fields does not prove that the full combination of fields matches.')

    title('5. What did the first experiments teach us?')
    sub('First: training made the format much more reliable')
    table(['Bitext: 500 attempts each','Original','200 steps','Full pass'],[
      ['Records with valid format','483','499','500'],
      ['Topics represented, out of 39','2','37','37'],
      ['Topic-distribution gap, TV','0.949','0.173','0.150'],
      ['Category-distribution gap, TV','0.744','0.092','0.130'],
      ['Replies matching training','0%','0%','27%'],
    ],[78,32,32,32])
    p('Training broadened topic coverage and improved formatting. But full-pass category matching was worse than the pilot, and reply reuse increased. The lesson was to inspect several measures rather than assume more training improves everything. [1]')
    p('In the original 128-output assessments, InsuranceQA formatting rose from 106 to 128 valid records, Reliability from 0 to 128 and MIMIC from 127 to 128. Reliability learned its complicated record structure, but valid structure alone did not prove a correct workflow. [2]')
    sub('Then: longer prompts did not automatically help')
    table(['What we tried','What happened','Decision'],[
      ['Detailed source-profile instructions','Valid: Bitext 121/156; Reliability 2/64; InsuranceQA 54/64. All below their matched generic baselines.','Reject and inspect formatting.'],
      ['Complete nested JSON templates','Reliability 15/16 valid, but 13/16 leaked template markers.','Reject; neat format still contained template text.'],
      ['Minimal topic prefixes','Small pilot: Bitext 8/8 and Reliability 8/8 valid. InsuranceQA only 7/8.','Expand the two passing sources; keep original QA method.'],
    ],[47,91,36])
    p('These trials were separate cohorts, not points on one identical test. We retained failed attempts instead of repairing or rerolling them. [3, 4, 5]','small')

    title('6. Bitext and InsuranceQA: what worked?')
    sub('Bitext: choosing the topic before generation helped')
    p('A prefix means supplying the first fields of a record, such as intent and category, then letting Gemma write the rest. Like asking a writer for a payment example rather than letting them choose any topic, it controls what the example will be about.')
    p('The actual instructions also specify the JSON fields. The phrase "Generate synthetic data" is the task, with formatting and source guidance around it. The exact saved prompts and experiment protocols are listed in the evidence section.')
    table(['Bitext: 156 attempts per arm','Generic prompt','Topic prefix'],[
      ['Valid records / source checks','156/156; 156/156','156/156; 156/156'],
      ['Topics represented','36 of 39','39 of 39'],
      ['Topic gap, TV','0.250','0.000'],
      ['Reply-length distance, characters','22.345','7.242'],
      ['Replies matching training','24.36%','26.92%'],
      ['Whole-record training matches','1/156','1/156'],
    ],[86,44,44])
    callout('The perfect topic mix was supplied by our topic schedule. We cannot claim Gemma independently learned perfect topic probabilities. The generated text-length improvement is separate evidence. [6]')
    p('Eight outputs in this selected prefix cohort came from the pilot. It is a development comparison, not a new independent final test. Bitext is our strongest source because it combines consistent structure, broad topic coverage and better length matching; copying remains a weakness.')
    sub('InsuranceQA: format improved; answer quality remains mixed')
    p('We kept its original fine-tuned generation method after the new prompts failed. All 128 selected outputs had valid format, with no observed whole-record or answer matches. A prior 30-example AI review found 15 Pass, 4 Unsure and 11 Issue. [10]')
    p('A professional-sounding answer can still be inaccurate or incomplete. InsuranceQA needs stronger factual review, and our existing AI review is not a substitute for an insurance expert. Its questions require knowledge that format and length checks cannot verify.')

    title('7. Reliability: why did good JSON still fail?')
    p('Reliability generates scenarios and expected workflows. We are not claiming that the model executed real insurance tools or passed a separate live-agent benchmark. The question is whether the generated record is consistent and complete.')
    p('With the prefix policy, all 64 selected records passed format and source-label checks. Broad category TV improved from 0.1139 to 0.0287, subcategory TV from 0.5256 to 0.2078, and message-length distance from 6.876 to 2.938. But action-count distance worsened from 0.0251 to 0.1430. [6]')
    sub('Reading the records found four concrete problems')
    table(['Problem','What the model did','What was needed'],[
      ['Missed question','Customer asked the ransomware deductible; workflow focused on incident response.','Explicitly retrieve and address the deductible.'],
      ['Wrong policy context','Homeowners request appeared alongside a commercial-liability policy.','Clarify which policy the customer means before updating it.'],
      ['Repeated payment','History said payment was complete; workflow included processing it again.','Recognise the completed payment and handle what remains.'],
      ['Wrong coverage context','Flood advice appeared against health-policy context.','Identify the relevant property/flood policy before policy-specific advice.'],
    ],[34,76,64])
    p('One assistant reviewed all 64: 45 Pass, 15 Unsure and 4 Issue. These are judgments on this chosen set, not an expert-calibrated population accuracy score. [8]')
    sub('We tried a targeted consistency instruction')
    p('We fixed four known problem cases and eight previously passing controls. The instruction asked Gemma to respect history, reconcile policy conflicts and answer explicit questions. The baseline had 12/12 valid outputs and 8 Pass / 4 Issue. The new instruction had 11/12 valid and 7 Pass / 2 Unsure / 3 Issue. None of the four known issues became Pass. [9]')
    p('We rejected it. A lower Issue count alone was misleading because uncertainty and malformed output increased. The completed-payment error persisted. The latest Gemini trial was a separate attempt to teach better workflows; page 10 explains it.')

    title('8. MIMIC: matching numbers takes more work')
    p('The original clinical generator produced valid records, but struggled with missing labs and diagnosis conventions. We trained a separate candidate with ordinary and condition-specific prompts, using the same 230 training admissions. It used 460 prompt views over two epochs. [7]')
    p('A condition tells the generator something to follow, such as a diagnosis pair or which labs should be missing. We compared old and new adapters under the same 64 conditions. Their training budgets differed, so the result does not isolate one single cause.')
    table(['Distance to training data','Old adapter','New candidate'],[
      ['Age, years','7.249','2.535'],
      ['Length of stay, days','2.003','1.040'],
      ['Creatinine, mg/dL','0.311','0.242'],
      ['Glucose, mg/dL','19.507','17.058'],
      ['Hemoglobin, g/dL','0.472','0.320'],
    ],[86,44,44])
    p('All five numeric distances improved: the new samples were closer on these individual measurements. SDMetrics statistical quality rose from 0.643 to 0.671. That is a statistical score, not "67.1% medically correct." [7]')
    sub('The remaining gaps matter')
    p('Training ages covered 21-90, where 90 means 90 or older. The selected generated ages covered only 41-86. The generated values stayed inside the source range but missed its younger and older tails. [10]')
    callout('Being inside the range is not the same as covering the range. A generator that always chooses middle-aged adults can avoid out-of-range errors while still missing important parts of the source.')
    p('The selected candidate passed source checks for 63/64 records: one race label was absent from training. Admission-type, insurance and mortality proportions worsened, and correlation mismatch rose slightly from 0.0838 to 0.0879. Clinical plausibility and full diagnosis-code validity remain unverified.')
    p('A heavier controller introduced more diagnosis/source problems, so we kept the lighter candidate for presentation. It remains separate from the original live MIMIC default. Nothing here establishes U.S.-population matching. [10]','small')

    title('9. Gemini distillation: what did we try?')
    p('Black-box distillation means using a teacher model through its API, collecting its example outputs and training our student on suitable examples. We used Gemini 3.8 Flash as teacher and the saved Reliability Gemma adapter as student. We did not obtain Gemini weights, gradients or private reasoning.')
    sub('First, we checked the teacher data')
    p('Version 1 returned 9/12 planned records; 3 timed out. All nine changed policy type to personal_auto, damaging source diversity. None entered training.')
    p('Version 2 required unchanged policy context. It returned 10/12; 2 timed out. Contexts were preserved. Assistant review accepted 8 and withheld 2 for workflow/source concerns. Requests used training anchors, not validation/test cases or clinical records. [12]')
    sub('Then, we made a fairer small comparison')
    table(['Arm','What changed'],[
      ['Original','Reuse the saved original outputs as baseline.'],
      ['Replay control','Continue training on 96 original rows plus 32 source-anchor views.'],
      ['Distilled student','The same 96 rows plus 8 reviewed teacher examples repeated four times.'],
    ],[40,134])
    p('Both trained arms used 128 views, the same topic/context mix, 18 updates and learning rate. Adapters/checkpoints were saved separately. The control helps separate extra training from teacher influence.')
    table(['12 known workflow cases','Pass','Unsure','Issue'],[
      ['Original','8','0','4'],['Replay control','7','1','4'],['Distilled student','7','0','5'],
    ],[78,32,32,32])
    p('Arm labels were hidden during review. None of the four known issues became Pass. The student produced one malformed generic record out of 16 and worsened length/action and routing measures. Policy/state coverage and template reuse improved, but the candidate failed the overall checks.')
    callout('Decision: reject this candidate and keep the original model. This small recipe failed; it does not prove distillation can never work. We did not automatically start a larger run. [12, 13]')

    title('10. What do we have, and what remains?')
    table(['Dataset','Our current position'],[
      ['Bitext','Strongest evidence: topic control and better text lengths. One whole-record copy and repeated source replies remain.'],
      ['InsuranceQA','Valid output structure; keep the original generation method. Answer-quality problems remain.'],
      ['Reliability','Valid structure and better controlled source labels. Workflow errors remain; both targeted fixes were rejected.'],
      ['MIMIC demo','Better measured numeric distributions under shared conditions. Some categories/relationships worsened; no clinical or U.S.-population validation.'],
    ],[39,135])
    sub('The frontend is a demonstration tool')
    p('The simplified website has Generate, Training data and Results. You can enter a prompt, inspect source examples and browse saved results. The detailed evidence view provides charts. The website has no report hyperlinks, as requested.')
    p('Saved cohorts and live generation are different things. Live generation uses the configured original adapters; the clinical candidate is a separate presentation comparison. These scores assess record generation, not arbitrary chatbot answers. Explain which model and settings produced each shown result.')
    sub('How do we show closeness honestly?')
    p('Show the source and generated records side by side; show range and category charts; report validity and copying; and include the examples that fail. We also drew real training samples 200 times to see how small real samples vary against the full source. Those bands are descriptive context, not a pass threshold or confidence interval. [10, 11]')
    sub('What is complete? What is future work?')
    p('Training, saved adapters, evaluations, prompt comparisons, clinical candidate, frontend and reports are complete. The new distillation audit verified 18 original artifact hashes unchanged, and all 43 code tests passed. Code tests check the software; they do not prove generated data quality. [13]')
    p('Future improvements would need independent generation runs, insurance-expert review, clinical/codebook checks and fresh confirmation data. Gradient alignment, data attribution and downstream training utility were discussed but not demonstrated. Distillation was demonstrated as a bounded experiment, with a negative result.')
    p('This learning guide is new and separate. The earlier submission ZIPs and short PDF were not silently updated to include the Gemini trial.','small')

    title('11. How to explain your own work')
    sub('A short explanation you can practise')
    callout('"I fine-tuned Gemma to generate new examples from four different datasets. I used separate QLoRA adapters so training fit my GPU. I evaluated format, ranges, proportions, relationships, copying and meaningful errors. I tested improvements against existing outputs and rejected changes that regressed. Bitext gave the strongest result. Reliability and clinical data still have important limitations, which I documented rather than hiding."')
    sub('Questions you should be ready to answer')
    table(['Question','Your answer in simple words'],[
      ['Why not train everything at once?','A small pilot checked the pipeline before spending time on a longer pass. The four sources also needed different records and checks.'],
      ['Why not just use a longer prompt?','We tested it. More detailed instructions hurt formatting here. Simpler topic prefixes worked better for two datasets.'],
      ['Did you meet the criterion completely?','We showed measurable improvements, but not complete distribution equivalence or verified factual/clinical correctness.'],
      ['Why keep a failed experiment?','It explains a decision with evidence and protects the baseline from a worse change.'],
      ['Why did Gemini not fix it?','This small set and short recipe did not work. We did not isolate the exact cause, so we cannot claim a bigger run would fix it.'],
    ],[57,117])
    sub('The terms worth remembering')
    p('Adapter: a small learned addition to the base model. Schema: the required record fields/types. Intent: what the customer wants. Prefix: supplied opening fields. Cohort: a particular set of generated examples. Baseline: the version we compare against. Control: a comparison that holds other factors steady. Regression: a change that makes a measured result worse.','small')
    p('TV and Wasserstein are distances, not grades. Loss is text-prediction error, not answer correctness. AI Pass is one reviewer\'s judgment on the reviewed cases, not a hiring score. Checkpoint is a saved version, not automatically the best version.','small')
    p('Data attribution estimates which training examples influenced a prediction. Gradient alignment compares whether examples push model learning in similar directions. We discussed these ideas but did not demonstrate them; neither would directly prove distribution matching.','small')
    p('The ownership you can demonstrate is the chain: identify a problem -> choose a measurable experiment -> compare fairly -> inspect tradeoffs -> keep or reject with evidence. The saved results support that explanation; they do not guarantee the hiring decision.')

    title('12. Where is the evidence?')
    p('All paths below start at D:/gemma-finetune. The editable Markdown copy has clickable evidence links. Older reports describe the state at their own experiment date; this guide includes the later rejected Gemini trial.')
    for ident,label,path in SOURCES:
        p('['+ident+'] '+label+' | '+path,'small')
    sub('Exact prompts, models and decisions')
    p('Original prompts: data/<dataset>/prepared/generation_prompt.txt, where <dataset> is bitext, insuranceqa, reliability or mimic_demo. Later experiments also store their actual protocols, prefixes and raw generations in their experiment directories. The prompt explanations in this guide are paraphrases, not claims that we used only a bare two-word instruction.','small')
    p('Original adapters: outputs/bitext-4b-full-pass-windows/adapter; outputs/insuranceqa-4b-v1-windows/adapter; outputs/reliability-4b-v1-windows/adapter; outputs/mimic-demo-4b-v2-windows/adapter. Clinical candidate: outputs/mimic-demo-fidelity-conditioned-v1-windows/adapter.','small')
    p('Decision history: decision.md. Latest handoff: NEXT_SESSION.md. Latest distillation outputs, checkpoints and judgments: outputs/reliability-distillation-v2/. To regenerate this guide: run scripts/build_explained_report.py with the bundled document Python dependencies.','small')

    def footer(canvas,doc):
        canvas.setStrokeColor(LINE);canvas.line(18*mm,17*mm,192*mm,17*mm)
        canvas.setFont('Helvetica',8);canvas.setFillColor(INK)
        canvas.drawString(18*mm,12*mm,'GEMMA / YOUR PROJECT EXPLAINED / 1 NOV 2026')
        canvas.drawRightString(192*mm,12*mm,str(doc.page))
    doc=SimpleDocTemplate(str(OUT),pagesize=(210*mm,297*mm),rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=23*mm,
        title='Your Gemma project, explained',author='Gemma assessment project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    md+=['## Clickable evidence','']
    for ident,label,path in SOURCES:md.append(f'- [{ident}: {label}]({(ROOT/path).as_posix()})')
    MD.write_text('\n'.join(md)+'\n',encoding='utf-8')
    reader=PdfReader(OUT)
    assert len(reader.pages)==13,f'Expected 13 intentional pages, found {len(reader.pages)}'
    for i,page in enumerate(reader.pages,1):
        text=page.extract_text() or ''
        assert len(text)>300,(i,'unexpected empty page')
    print(json.dumps({'pdf':str(OUT),'markdown':str(MD),'pages':len(reader.pages),'words':len(re.findall(r'\S+',MD.read_text(encoding='utf-8')))},indent=2))

if __name__=='__main__':main()
