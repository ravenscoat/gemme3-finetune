import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root='D:/gemma-finetune';
const runtime='C:/Users/quandale dingle UwU/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const skill='C:/Users/quandale dingle UwU/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs'));
const build=root+'/tmp/slides/hiring';await fs.mkdir(build,{recursive:true});await fs.mkdir(root+'/output/slides',{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
deck.theme.defaultFont='Arial';
const C={ink:'#183246',teal:'#087E83',muted:'#526877',bg:'#F7F9FA',light:'#E8F1F2',white:'#FFFFFF'};
const FONT='Arial';
function text(s,t,x,y,w,h,size=27,color=C.ink,bold=false){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={fontFamily:FONT,fontSize:size,color,bold};return a;}
function slide(title,notes){const s=deck.slides.add();s.background.fill=C.bg;text(s,title,64,44,1152,100,44,C.ink,true);text(s,String(deck.slides.items.length).padStart(2,'0'),1160,670,56,28,18,C.muted);s.speakerNotes.text=notes;return s;}
function table(s,headers,rows,y=185,widths=[300,420,432],height=350){const t=s.tables.add({rows:rows.length+1,columns:headers.length,left:64,top:y,width:1152,height,columnWidths:widths,values:[headers,...rows]});t.borders.assign({fill:'#D9E2E6',width:0.5,style:'solid'});for(let r=0;r<=rows.length;r++)for(let c=0;c<headers.length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.light:C.bg;cell.text.style={fontFamily:FONT,fontSize:24,color:C.ink,bold:r===0};}return t;}
function chart(s,categories,values,x=640,y=235,w=560,h=310){const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,series:[{name:'Distance',values,fill:C.teal}],barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:24,fill:C.ink}},xAxis:{textStyle:{fontSize:23,fill:C.ink}},yAxis:{min:0,textStyle:{fontSize:23,fill:C.ink}},chartFill:C.bg,plotAreaFill:C.bg});applyPresentationChartFont(c,{fontFamily:FONT});return c;}
function note(s,t){text(s,t,64,608,1135,48,22,C.muted);}
const source='Evidence: D:/gemma-finetune/';

let s=slide('Gemma synthetic-data generation','Hiring assessment. Explain that synthetic examples should resemble their training source. Four separately trained adapters share a Gemma 3 4B base. The work runs locally on an RTX5060Laptop8GB GPU. '+source+'reports/gemma_hiring_assessment_report.md');
s.background.fill=C.ink;s.shapes.items[0].text.style={fontFamily:FONT,fontSize:64,color:C.white,bold:true};s.shapes.items[0].position={left:72,top:160,width:1100,height:180};
text(s,'Four specialized adapters on an 8 GB GPU',76,365,1100,76,34,C.white);
text(s,'Hiring assessment presentation',76,500,1000,42,26,'#8CD5D3');

s=slide('What the project delivered','Lead with the implementation and measured achievements. The schema results refer to the separately retained cohorts, not every possible future response and not factual accuracy. Bitext156/156, QA128/128, Reliability64/64, clinical64/64. '+source+'reports/distribution_benchmark/v1/scorecard_selected_policies.csv');
text(s,'4 specialized QLoRA adapters',64,185,1152,70,42,C.teal,true);
text(s,'Valid structured records across all retained assessment cohorts',64,272,1130,96,32);
text(s,'Measured gains in topic, text and clinical distributions',64,390,1130,90,32);
text(s,'Reproducible evaluations and an interactive local demo',64,505,1130,90,32);

s=slide('The requirement and our evaluation','Read the exact criterion: Matches the ranges, distribution, and characteristics of the data it was trained on. Explain that a distribution is how often things occur. Similarity and factual correctness are complementary checks. '+source+'reports/distribution_benchmark/v1/criteria.json');
table(s,['Criterion','Plain-language question','Example'],[['Ranges','Do values span similar ranges?','Age and response length'],['Distribution','Do frequencies resemble the source?','Insurance topics and keywords'],['Characteristics','Do records preserve source patterns?','Fields, style and relationships']],180,[240,520,392],330);
note(s,'Content review and copying checks accompany the distribution measurements.');

s=slide('Four datasets with different strengths','Rows and categories refer to prepared data. InsuranceQA retains16813 after39validation and37test overlaps removed. Bitext39intents/17categories. Reliability10categories/80subcategories/3difficulty levels. MIMIC9admissiontypes,3insurance labels,14race labels. The clinical demo has100patients and275admissions. '+source+'data/*/prepared/schema.json and preparation_audit.json');
table(s,['Dataset','Records','What one row contains'],[['Bitext','39,000','Intent/category/tags, request, reply'],['InsuranceQA V2','16,813 retained','Domain, question, answer'],['Reliability','510','Scenario, context/history, workflow'],['MIMIC-IV demo','275 admissions','13 demographic and clinical fields']],175,[280,250,622],350);
note(s,'Bitext: 39 intents   InsuranceQA: 12 domains   Reliability: 10 categories');

s=slide('QLoRA made local training practical','QLoRA combines a mostly4bit frozen base with trainable LoRA adapters. Ordinary16bit4B weights alone need roughly8GB before activations/working memory. Other LoRA offloading setups can exist, so do not say LoRA universally impossible. Rank8alpha16 and gradientcheckpointing. Four separate adapters initialized from base. '+source+'outputs/insuranceqa-4b-v1-windows/training_summary.json');
text(s,'Gemma 3 4B',64,180,550,68,46,C.teal,true);text(s,'RTX 5060 Laptop GPU\n8 GB VRAM',700,180,490,115,36,C.ink,true);
text(s,'Frozen base weights stored mostly at 4-bit precision',64,340,1135,64,30);
text(s,'Small LoRA adapters learn each dataset separately',64,425,1135,64,30);
text(s,'Gradient checkpointing reduces working memory',64,510,1135,64,30);
note(s,'Windows training with Unsloth and PyTorch. Models, data and caches stored on D:.');

s=slide('Training, validation and testing','Training updates weights. Validation monitors and selects checkpoints. Test evaluates selected models after choices. Original tests now used, so laterprompt experiments are development comparisons. Bitextgroupheuristics aim to reduceoverlap. QAofficialsplits retained withoverlappurging. Reliabilityofficial70/15/15. MIMICsplit80/10/10patients, differentadmissioncounts. '+source+'data/*/prepared/ and reports/gemma_hiring_assessment_report.md');
table(s,['Dataset','Train','Validation','Test'],[['Bitext','31,200','3,900','3,900'],['InsuranceQA','12,889','1,961','1,963'],['Reliability','357','76','77'],['MIMIC admissions','230','21','24']],170,[360,264,264,264],320);
text(s,'Training learns. Validation selects. Testing checks.',64,525,1150,60,32,C.teal,true);
note(s,'MIMIC keeps each patient in one split. Bitext supplies 800 training rows per intent.');

s=slide('What the benchmark scores tell us','Explain each score briefly. SDMetrics compares declared column shapes/relationships. QA uses domain and lengths only. MMD uses RBF biased squared statistic on fixed GPT2large512tokenfeatures. PRDC shares encoder, so it is not independentfactualconfirmation. Keywords use source top100+OTHER. MainembeddingN128 exceptReliability64; sourcecohort sizes differ. Do not rank SDscores between datasets. '+source+'reports/distribution_benchmark/v1/report.md');
table(s,['Measure','What it checks','Interpretation'],[['SDMetrics quality','Field distributions and relationships','Higher is closer'],['MMD-squared','Text-collection difference','Lower is closer'],['PRDC','Resemblance and variety in text space','Four complementary views'],['Keyword JS distance','Important-word frequencies','Lower is closer']],175,[325,515,312],340);
note(s,'These measure resemblance. Review checks meaning and reuse separately.');

s=slide('Bitext improved across training stages','Pilot200updates then3700more to3900total, oneepoch31200. ChartintentTV0.949original,.173pilot,.150full500attemptseach. Valid483,499,500. Controlledselected156externallabelpolicySD.94648 jointMMD.0071437 keywords.032936. Main128embedding. Replyreuse26.92%,whole1/156; preserveasmateriallimit. '+source+'reports/training_stage_comparison.md; outputs/prefix-alignment-v1/bitext/metrics.json');
text(s,'2 to 37',64,190,500,90,60,C.teal,true);text(s,'intents represented, out of 39',64,285,520,60,27);
text(s,'500/500',64,385,500,85,54,C.ink,true);text(s,'valid records after the full pass',64,475,520,70,27);
chart(s,['Original','Pilot','Full pass'],[.949,.173,.150]);text(s,'Intent-frequency distance\nLower is closer',640,160,555,70,26);
note(s,'Source-matched labels further improved topic and text resemblance. Reply reuse remains a focus.');

s=slide('InsuranceQA learned the dataset format','Separateoneepoch1612updates selected1600. Validation3.8888to1.7860 notaccuracy. Original106/128valid, selected128. Domain/lengthSD.890765, MMD.0141737,keywords.104830, PRDCk5precision.914recall.75. Noexactanswer/fullrecordcopies,13questionsmatch. 30assistantreview15P4U11I; unblindednotexpert. Source-task32diagnosticbase24supportedvsadapter8, adapterdistributionbetter. Do not claim generalforgetting. '+source+'reports/distribution_benchmark/v1/insuranceqa/results.json; outputs/insuranceqa-base-diagnostic-v1/report.md');
text(s,'106/128 to 128/128',64,185,1130,80,48,C.teal,true);text(s,'Structure-valid outputs before and after fine-tuning',64,275,1130,60,28);
table(s,['Measure','Retained result'],[['Validation prediction loss','3.8888 to 1.7860'],['Domain/length quality','0.8908'],['Joint MMD-squared','0.01417'],['Keyword JS distance','0.1048']],365,[680,472],210);
note(s,'Next focus: question–answer consistency and reviewed source supervision.');

s=slide('Reliability learned the workflow schema','Original0/128schema vsadapter128/128. Selected64controlledlabels passed64sourcechecks. MatchedSD.6879to.8228 MMD.01943to.01478 keywords.2559to.2075. Assistant45P15U4I. Sourceconstraintsnotcorrectworkflows. Consistency/SeqKDpilotfailedrejected. This is in-domain data trainedonbenchmark, notindependentagentbenchmark. '+source+'outputs/prefix-alignment-v1/reliability/metrics.json; reports/reliability_workflow_audit/v1/report.md');
text(s,'0/128 to 128/128',64,190,550,95,48,C.teal,true);text(s,'Records meeting the full nested schema',64,290,535,88,29);
text(s,'64/64 source checks',64,430,535,60,34,C.ink,true);text(s,'in the selected controlled cohort',64,505,535,60,26);
chart(s,['Generic','Controlled'],[.6879,.8228]);text(s,'SDMetrics quality\nHigher is closer',640,160,555,70,26);
note(s,'Next focus: deductible completeness, policy context and payment history.');

s=slide('Clinical generation moved closer to the demo','OfficialMIMICdemo100patients275admissions prepared230/21/24. Original3epochs87updates best75. Separatecandidate460views2epochs andbotharmsdiagnosis/nullconditioned64each. Unequalbudgetsso notisolatedeffect. SD.6428to.6705,ageW7.249to2.535daysstay2.003to1.040,glucose19.507to17.058. Sourceconstraint63/64. Sourceage21-90 candidate41-86,relationshipsnotallimprove. No clinical/codevalidityclaim. '+source+'outputs/mimic-demo-fidelity-conditioned-v1-windows/recommendation.md');
text(s,'7.249 to 2.535 years',64,190,565,90,43,C.teal,true);text(s,'Age-distribution distance',64,285,550,55,29);
text(s,'2.003 to 1.040 days',64,395,565,80,40,C.ink,true);text(s,'Stay-length distribution distance',64,485,550,65,27);
chart(s,['Original','Candidate'],[7.249,2.535]);text(s,'Age distance, years\nLower is closer',640,160,555,70,26);
note(s,'Separate conditioned candidate. 100-patient demo. Next focus: rare ages and clinical validation.');

s=slide('A reproducible result with clear next steps','Close with deliveredengineering andspecificpriorities. Showlocaldemoifavailable: enterprompt,viewgeneratedrecord,comparewithsourcepreviewandresults. No report hyperlinkswereaddedwebsite. Existingselectedoriginalskeepunlessevidenceimproves. Larger model orvalidateddistillationpotentialfuture choicesnotprovenfix. Allsource/control limitations in report. '+source+'reports/gemma_hiring_assessment_report.md; decision.md');
text(s,'Delivered',64,180,500,55,34,C.teal,true);text(s,'Four saved adapters\nPrepared data and reproducible splits\nSynthetic outputs and benchmark results\nInteractive demo with dataset previews',64,255,1120,205,30);
text(s,'Next targeted improvements',64,500,1100,55,32,C.teal,true);text(s,'Answer consistency, response originality and rare-case coverage',64,565,1130,70,27);

const speaking=[
 'I fine-tuned four separate Gemma adapters to generate examples resembling four supplied datasets. The assessment focuses on ranges, distributions and characteristics. I chose an approach that runs locally within an 8 GB GPU budget.',
 'The deliverable includes trained adapters, prepared data, generated examples, saved evaluations and a local demo. Every output in the selected assessment cohorts met its required structure: 156 Bitext, 128 InsuranceQA, 64 Reliability and 64 clinical records. Structure validity checks fields and types. Content quality is assessed separately.',
 'Ranges describe the span of values. Distribution describes how often topics or values occur. Characteristics include fields, wording and relationships. I use several complementary measures, rather than interpreting one number as overall accuracy.',
 'Bitext contains consistent customer-support examples across 39 intents and 17 categories. InsuranceQA has 12 domains and more varied answers. Reliability has 10 workflow categories and nested context. MIMIC is the official 100-patient demo, with 275 admissions. Prepared files use one JSON record per line.',
 'QLoRA combines a mostly 4-bit frozen base model with small trainable LoRA adapters. Ordinary 16-bit weights for a four-billion-parameter model already require roughly 8 GB before training overhead. Quantization and gradient checkpointing made local training practical. Other LoRA configurations may use offloading, so I do not claim LoRA is universally impossible on this GPU.',
 'Training updates the adapter. Validation monitors progress and selects checkpoints. Test records evaluate the selected model. Bitext uses approximately 80/10/10 splits. InsuranceQA preserves official splits after removing overlaps. Reliability keeps official splits. MIMIC keeps all admissions from each patient in one split. Original test splits have now been used, so later comparisons are development evidence.',
 'SDMetrics checks selected field distributions and relationships. MMD measures difference between collections in a fixed text representation. PRDC examines resemblance, variety, concentration and coverage in that same space. Keyword JS compares source-word frequencies. These measure resemblance, not factual truth. Main embedding samples are 128 per collection, except Reliability with 64. SDMetrics columns differ by dataset, so scores should not rank datasets against one another.',
 'Bitext started with a 200-step pilot and continued to 3,900 total updates, one pass through 31,200 records. In 500-output comparisons, represented intents increased from 2 to 37 of 39 and valid records reached 500 of 500. The chart shows intent-frequency distance falling. Supplying source-matched intent and category labels improved the controlled pipeline further. These labels are external controls. In the selected 156 outputs, one full record and 26.92 percent of replies matched training.',
 'InsuranceQA completed one epoch with 1,612 updates, selecting checkpoint 1,600. It improved structure and source prediction, with no exact complete-answer reuse in the retained 128 outputs. SD quality here covers domain and lengths. Answer consistency is the next priority: a prior 30-record assistant review found 15 Pass, 4 Unsure and 11 Issue. A separate source-following diagnostic showed that the adapter matched source distributions better while original Gemma followed supplied passages more closely. These are small, unblinded reviews, not factual-accuracy estimates.',
 'Reliability improved from zero to 128 schema-valid records in the original comparison. The chart compares matched 64-output generic and controlled cohorts, not the pre-training base. Supplied category labels improved distribution quality, and 64 of 64 selected records passed source checks. Assistant review found 45 Pass, 15 Unsure and 4 Issue. Remaining targets are deductible completeness, policy context and payment history. Training on this benchmark makes held-out results in-domain evidence.',
 'The clinical candidate improved numeric resemblance on the official MIMIC demo. Both comparison arms received diagnosis and missing-lab conditions, and their training budgets differed. The values are distribution distances, not individual prediction errors. The candidate passed source constraints in 63 of 64 outputs. Generated ages still under-cover younger and extreme source ages. Clinical and diagnosis-code correctness require separate validation.',
 'The project demonstrates model adaptation under a local hardware constraint and a repeatable way to compare outputs with their sources. I kept improvements with useful evidence and preserved earlier versions when trials did not help. The next priorities are focused supervision for consistency, greater response originality and better rare-case coverage. The local demo lets a reviewer enter a prompt, inspect a generated record and compare it with a source preview.'
];
for(let i=0;i<deck.slides.items.length;i++){const s=deck.slides.items[i];const citation=String(s.speakerNotes.text).split('Evidence: ')[1]??'D:/gemma-finetune/reports/gemma_hiring_assessment_report.md';s.speakerNotes.text=speaking[i]+'\n\nEvidence: '+citation;}
// Preserve a portable speaking outline and all editable native evidence.
await fs.writeFile(build+'/speaker_notes.json',JSON.stringify(deck.slides.items.map((s,i)=>({slide:i+1,notes:s.speakerNotes.text})),null,2));
for(let i=0;i<deck.slides.items.length;i++){const blob=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));}
const candidate=build+'/candidate.pptx';await(await PresentationFile.exportPptx(deck)).save(candidate);
const final=root+'/output/slides/gemma_hiring_presentation.pptx';
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,6,7,9].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:12,materializeLiteralChartWorkbooks:true,requiredNativeTableOwnerSlides:[3,4,6,7,9],requiredNativeChartOwnerSlides:[8,10,11],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:build+'/validation-final.json'});
console.log(JSON.stringify({final,slides:deck.slides.items.length,result}));

