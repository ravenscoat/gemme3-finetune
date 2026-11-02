import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const ROOT='D:/gemma-finetune', R='C:/Users/quandale dingle UwU/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const SK='C:/Users/quandale dingle UwU/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const {Presentation,PresentationFile}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
const build=ROOT+'/tmp/slides/aesthetic';await fs.mkdir(build,{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});p.theme.defaultFont='Arial';
const C={ink:'#102D35',teal:'#087C78',clay:'#A96542',paper:'#F3F1EA',muted:'#566B70',pale:'#A5DAD0',line:'#D3D9D4'};
const notes=JSON.parse(await fs.readFile(ROOT+'/tmp/slides/hiring/speaker_notes.json','utf8'));
function txt(s,t,x,y,w,h,size=28,color=C.ink,font='Arial',bold=false){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:font,fontSize:size,color,bold};return a;}
function title(s,t,dark=false){txt(s,t,70,52,1130,135,57,dark?C.paper:C.ink,'Georgia');}
function slide(dark=false){const s=p.slides.add();s.background.fill=dark?C.ink:C.paper;s.speakerNotes.text=notes[p.slides.items.length-1].notes;txt(s,String(p.slides.items.length).padStart(2,'0'),1170,670,50,26,18,dark?C.pale:C.muted);return s;}
function foot(s,t,dark=false){txt(s,t,72,623,1125,42,21,dark?C.pale:C.muted);}
function tbl(s,heads,rows,x=72,y=205,widths=[300,280,556],height=330){const t=s.tables.add({rows:rows.length+1,columns:heads.length,left:x,top:y,width:widths.reduce((a,b)=>a+b),height,columnWidths:widths,values:[heads,...rows]});t.borders.assign({fill:C.line,width:0.4,style:'solid'});for(let r=0;r<=rows.length;r++)for(let c=0;c<heads.length;c++){const a=t.getCell(r,c);a.fill=r===0?'#DDE6DF':C.paper;a.text.style={typeface:'Arial',fontSize:24,color:r===0?C.teal:C.ink,bold:r===0};}return t;}
function chart(s,cat,values,{x=625,y=245,w=590,h=320,max}={}){const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories:cat,series:[{name:'Value',values,fill:C.teal,points:values.map((v,i)=>({idx:i,fill:i===values.length-1?C.teal:'#A6B7B7'}))}],barOptions:{direction:'column',gapWidth:160},hasLegend:false,xAxis:{textStyle:{fontSize:22,fill:C.ink},line:{fill:C.line,width:1}},yAxis:{min:0,...(max?{max}:{}),textStyle:{fontSize:20,fill:C.muted}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:27,fill:C.ink,bold:true}},chartFill:C.paper,plotAreaFill:C.paper});applyPresentationChartFont(c,{fontFamily:'Arial'});}

let s=slide(true);s.images.add({blob:new Uint8Array(await fs.readFile(ROOT+'/assets/slides/gemma-cover.png')),contentType:'image/png',fit:'cover',position:{left:0,top:0,width:1280,height:720},alt:'Decorative abstract woven light on a dark blue background'});
txt(s,'Gemma',68,105,800,145,116,C.paper,'Georgia');txt(s,'Synthetic data,\nmeasured against\nits source',74,290,700,245,57,C.paper,'Georgia');txt(s,'Four specialized adapters on an 8 GB GPU',76,600,900,44,27,C.pale);
s.speakerNotes.text+='\nCover artwork: AI-generated decorative abstract, not an image of the model or its measurements.';

s=slide(true);title(s,'Delivered on an 8 GB GPU',true);
for(const [x,big,small] of [[72,'4','Specialized adapters'],[485,'8 GB','Local GPU memory'],[900,'4 / 4','Retained cohorts\nfully schema-valid']]){txt(s,big,x,250,360,150,92,C.pale,'Georgia');txt(s,small,x,420,335,100,29,C.paper);}
foot(s,'Saved models, reproducible evaluations and a local interactive demo',true);

s=slide();title(s,'What successful synthetic data should match');
for(const [i,x,h,b,e] of [[1,72,'Ranges','A similar span of values','Age and response length'],[2,490,'Distribution','A similar frequency of topics','Categories and keywords'],[3,905,'Characteristics','The same useful patterns','Fields, style and relationships']]){txt(s,'0'+i,x,240,290,90,65,C.clay,'Georgia');txt(s,h,x,350,310,60,33,C.ink,'Arial',true);txt(s,b,x,435,300,95,28);txt(s,e,x,550,300,70,22,C.muted);}

s=slide();title(s,'Four sources. Four distinct generation tasks.');
tbl(s,['Source','Prepared records','One example contains'],[['Bitext','39,000','Intent/category/tags, request and reply'],['InsuranceQA','16,813 retained','Domain, question and answer'],['Reliability','510','Scenario, context/history and workflow'],['MIMIC-IV demo','275 admissions','13 demographic and clinical fields']],72,205,[300,270,566],335);
foot(s,'39 Bitext intents     12 InsuranceQA domains     10 workflow categories     100 clinical patients');

s=slide(true);title(s,'A full-size base. A small trainable adapter.',true);
txt(s,'Gemma 3 4B',72,240,800,100,72,C.pale,'Georgia');
txt(s,'Frozen base weights stored mostly at 4-bit precision',76,370,1100,65,31,C.paper);
txt(s,'QLoRA',76,485,370,80,53,C.paper,'Georgia');txt(s,'Small adapter learns each dataset\nCheckpointing reduces working memory',490,480,700,110,29,C.paper);
foot(s,'Windows, Unsloth and PyTorch. Independent adapters initialized from the same base.',true);

s=slide();title(s,'A deliberate split for learning and checking');
tbl(s,['Dataset','Training','Validation','Test'],[['Bitext','31,200','3,900','3,900'],['InsuranceQA','12,889','1,961','1,963'],['Reliability','357','76','77'],['MIMIC admissions','230','21','24']],72,195,[380,252,252,252],300);
txt(s,'Learn',78,540,335,60,36,C.teal,'Georgia');txt(s,'Select checkpoints',485,540,390,60,36,C.teal,'Georgia');txt(s,'Check outcomes',900,540,330,60,36,C.teal,'Georgia');
foot(s,'MIMIC patients stay in one split. Bitext includes 800 training records per intent.');

s=slide();title(s,'One criterion, several complementary checks');
const metrics=[['SDMetrics','Fields and relationships','Higher is closer'],['MMD-squared','Text-collection difference','Lower is closer'],['PRDC','Resemblance and variety','Four views of coverage'],['Keyword JS','Important-word frequencies','Lower is closer']];
for(let i=0;i<4;i++){const y=220+i*87;txt(s,metrics[i][0],75,y,315,60,33,C.teal,'Georgia');txt(s,metrics[i][1],420,y+2,470,60,27);txt(s,metrics[i][2],940,y+4,290,60,24,C.muted);}
foot(s,'Structure, meaning and copying checks complete the evaluation. Similarity scores are not accuracy.');

s=slide();title(s,'Bitext learned a much broader intent mix');
txt(s,'2 to 37',72,230,530,130,88,C.teal,'Georgia');txt(s,'Intents represented, out of 39',78,365,510,75,27);
txt(s,'500 / 500 valid',78,480,540,80,38,C.ink,'Arial',true);
txt(s,'Intent-frequency distance',635,193,560,42,25,C.muted);chart(s,['Original','Pilot','Full pass'],[.949,.173,.150],{max:1});
foot(s,'Lower is closer. Full pass: 3,900 updates. Source-matched labels improved resemblance further.');

s=slide();title(s,'InsuranceQA learned its target format');
txt(s,'128 / 128',72,220,600,125,85,C.teal,'Georgia');txt(s,'Valid QA records after fine-tuning\n106 / 128 before fine-tuning',78,365,540,108,28);
for(const [y,k,v] of [[235,'Validation loss','3.8888 to 1.7860'],[355,'Domain / length quality','0.8908'],[475,'Joint MMD-squared','0.01417']]){txt(s,k,735,y,470,44,25,C.muted);txt(s,v,735,y+48,470,65,38,C.ink,'Georgia');}
foot(s,'Keyword JS: 0.1048. Next focus: answer consistency and reviewed source supervision.');

s=slide();title(s,'Reliability gained the complete workflow schema');
txt(s,'0 to 128',72,230,535,120,78,C.teal,'Georgia');txt(s,'Schema-valid records\nfrom 128 attempts per model',78,365,500,100,28);
txt(s,'64 / 64 source checks',78,510,520,65,34,C.ink,'Arial',true);
txt(s,'Matched SDMetrics quality',635,192,560,45,25,C.muted);chart(s,['Generic','Controlled'],[.6879,.8228],{max:1});
foot(s,'Higher is closer. Supplied labels control the topic mix. Next focus: context and history consistency.');

s=slide();title(s,'Clinical values moved closer to the demo');
txt(s,'7.249',72,230,530,110,82,C.muted,'Georgia');txt(s,'2.535 years',72,350,550,110,77,C.teal,'Georgia');
txt(s,'Stay-length distance\n2.003 to 1.040 days',78,495,530,90,27);
txt(s,'Age distance in years',635,192,560,45,25,C.muted);chart(s,['Original','Candidate'],[7.249,2.535],{max:8});
foot(s,'Lower is closer. Separate conditioned candidate, 64 outputs per arm. Official 100-patient demo.');

s=slide(true);title(s,'A working pipeline, with evidence to show',true);
txt(s,'Train locally.\nGenerate new records.\nMeasure source resemblance.',72,240,1130,260,58,C.paper,'Georgia');
txt(s,'Saved adapters and outputs. Editable prompts. Reproducible scores.',78,555,1100,54,27,C.pale);
foot(s,'Next priorities: answer consistency, response originality and rare-case coverage.',true);

for(let i=0;i<p.slides.items.length;i++){const img=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await img.arrayBuffer()));}
const candidate=build+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
const final=ROOT+'/output/slides/gemma_hiring_presentation_redesigned.pptx';
const tables=[4,6],charts=[8,10,11];
await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tables.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:tables,requiredNativeChartOwnerSlides:charts,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial','Georgia']},verifyArtifactToolImport:true,receiptPath:build+'/validation-final.json'});
await fs.copyFile(ROOT+'/output/slides/gemma_hiring_presentation.pptx',build+'/previous-design.pptx');
await fs.copyFile(final,ROOT+'/output/slides/gemma_hiring_presentation.pptx');
console.log(final);
