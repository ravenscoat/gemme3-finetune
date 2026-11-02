"""Plain-language hiring report from saved training and evaluation evidence."""
import html,json,re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer
from dataset_catalog import ROOT,data,read_json,read_rows,sha,value_at
from hiring_assessment_plain_language import write_sections

def main():
    name='gemma_hiring_assessment_report'
    out=ROOT/('output/pdf/'+name+'.pdf');md=ROOT/('reports/'+name+'.md')
    benchmark=read_json(ROOT/'reports/distribution_benchmark/v1/results.json')['datasets']
    allrows={k:sum([read_rows(ROOT/data(k)['spec']['prepared']/(s+'.jsonl')) for s in ['train','validation','test']],[]) for k in benchmark}
    # Fixed training rows, selected for readable, profanity-free illustrations.
    example_lines={'bitext':22468,'insuranceqa':830,'reliability':116,'mimic_demo':10}
    examples={k:data(k)['train'][line-1] for k,line in example_lines.items()}
    profanity=re.compile(r'\b(?:fuck\w*|shit\w*|bullshit\w*|bitch\w*|ass(?:hole)?s?|damn\w*|bastard\w*|crap\w*|hell|wtf|stfu)\b',re.I)
    assert all(not profanity.search(json.dumps(r)) for r in examples.values()),'Review the selected source examples for profanity'
    snapshot=read_json(ROOT/'reports/assessment_snapshot/snapshot.json')['datasets']
    comparison_lines={'bitext':(22468,95),'insuranceqa':(1518,14),'reliability':(116,23),'mimic_demo':(10,1)}
    comparison_fields={
        'bitext':[('Intent','intent'),('Category','category'),('Tags','tags'),('Customer request','instruction'),('Support reply','response')],
        'insuranceqa':[('Domain','domain'),('Question','question'),('Answer','answer')],
        'reliability':[('Category','category'),('Subcategory','subcategory'),('Difficulty','difficulty'),('Customer message','input.customer_message'),('Policy type','input.context.policy_type'),('Required actions','expected_output.required_actions'),('Routing decision','expected_output.routing_decision')],
        'mimic_demo':[('Age (years)','age'),('Age recorded as 90+','age_topcoded'),('Gender','gender'),('Admission type','admission_type'),('Insurance','insurance'),('Race','race'),('Stay length (days)','length_of_stay_days'),('In-hospital death flag','hospital_expire_flag'),('Diagnosis code','primary_diagnosis_code'),('Diagnosis version','primary_diagnosis_version'),('Creatinine (mg/dL)','creatinine_mg_dl'),('Glucose (mg/dL)','glucose_mg_dl'),('Hemoglobin (g/dL)','hemoglobin_g_dl')],
    }
    comparisons={}
    for key,(train_line,generated_line) in comparison_lines.items():
        source=data(key);generated_path=ROOT/snapshot[key]['records_path']
        assert sha(generated_path)==snapshot[key]['records_sha256'],'Frozen generated cohort changed'
        training=source['train'][train_line-1];wrapper=read_rows(generated_path)[generated_line-1]
        generated=wrapper['record']
        assert not profanity.search(json.dumps([training,generated])),'Review comparison example wording'
        comparisons[key]={'training_path':source['spec']['prepared']+'/train.jsonl','training_line_1_based':train_line,'training_sha256':sha(ROOT/source['spec']['prepared']/'train.jsonl'),'generated_path':str(generated_path.relative_to(ROOT)).replace('\\','/'),'generated_line_1_based':generated_line,'generated_sample_id':wrapper['sample_id'],'generated_sha256':sha(generated_path),'policy':snapshot[key]['policy'],'displayed_cells':[{ 'label':label,'field':field,'training':value_at(training,field),'generated':value_at(generated,field)} for label,field in comparison_fields[key]],'selection':'Illustrative source and saved generated records; not matched prompts, random sampling or an overall quality estimate. No new generation.'}
    example_cells={
        'bitext':[(f,examples['bitext'][f]) for f in data('bitext')['schema']['fields']],
        'insuranceqa':[(f,examples['insuranceqa'][f]) for f in data('insuranceqa')['schema']['fields']],
        'reliability':[(f,examples['reliability'][f]) for f in ['category','subcategory','difficulty']],
        'mimic_demo':[(f,examples['mimic_demo'][f]) for f in data('mimic_demo')['schema']['fields']],
    }
    for field in ['input.customer_message','input.context.policy_type','expected_output.intent','expected_output.routing_decision','expected_output.required_actions','expected_output.priority']:
        value=examples['reliability']
        for part in field.split('.'):value=value[part]
        example_cells['reliability'].append((field,value))
    for cell in comparisons['bitext']['displayed_cells']:
        if cell['field']=='response':
            cell['label']='Reply opening (excerpt)';cell['excerpt']=True
            for arm in ['training','generated']:
                cell['full_'+arm]=cell[arm]
                cell[arm]='\n'.join(cell[arm].splitlines()[:4]).rstrip()
    ink=colors.HexColor('#203448');teal=colors.HexColor('#126c75');light=colors.HexColor('#edf5f6')
    styles={k:ParagraphStyle(k,fontName='Helvetica-Bold' if k in ['title','head','sub'] else 'Helvetica',fontSize=size,leading=lead,textColor=teal if k=='sub' else ink,spaceAfter=gap) for k,size,lead,gap in [('title',27,32,16),('head',20,25,14),('sub',12,16,7),('body',10.5,14.2,8),('cell',9,12,0),('small',9,12.5,7)]}
    story=[];markdown=[]
    def p(text,style='body'):
        story.append(Paragraph(html.escape(text),styles[style]));markdown.extend([text,''])
    def page(title,first=False):
        if not first:story.append(PageBreak())
        p(title,'title' if first else 'head');markdown[-2]='# '+title
    def sub(text):p(text,'sub');markdown[-2]='## '+text
    def table(headers,rows,widths,compact=False):
        cells=[[Paragraph(html.escape(str(c)).replace('\n','<br/>'),styles['cell']) for c in r] for r in [headers,*rows]]
        padding=4 if compact else 6
        t=Table(cells,colWidths=[v*mm for v in widths],repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),light),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),('LINEBELOW',(0,0),(-1,0),.7,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe4'))]));story.extend([t,Spacer(1,8)])
        markdown.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(str(v).replace('|',r'\|').replace('\n','<br>') for v in r)+' |' for r in rows],''])
    def example(key,title,caption=''):
        sub(title)
        values=[(field,value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)) for field,value in example_cells[key]]
        table(['Dataset column / field','Actual cell value'],values,[62,112],compact=True)
        p('Source: data/'+key+'/prepared/train.jsonl, line '+f'{example_lines[key]:,}'+'. '+caption,'small')
    def comparison(key,title):
        item=comparisons[key];sub(title)
        cell=lambda value:value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
        table(['Field','Training example','Saved generated example'],[[c['label'],cell(c['training']),cell(c['generated'])] for c in item['displayed_cells']],[44,65,65],compact=True)
        p('Training: '+item['training_path']+', line '+f"{item['training_line_1_based']:,}"+'. Generated: '+item['generated_path']+', sample '+str(item['generated_sample_id'])+'.','small')
    def result_table(key,arm):
        d=benchmark[key];c=d['cohorts'][arm];core=c['core'];rows=[['Correct record format',f"{core['structure_valid']}/{core['attempted']}",'Examples with all required fields and the right value types.'],['SDMetrics quality',f"{c['sdmetrics']['quality']:.4f}",'Similarity of selected column patterns; higher is closer.']]
        if d['text_metrics']:
            text=d['text_metrics']['joint']['collections'][arm];pr=text['prdc']['5'];rows.extend([
                ['Joint MMD-squared',f"{text['mmd2']:.5f}",'Gap in measured text patterns; lower is closer.'],
                ['PRDC precision / recall',f"{pr['precision']:.3f} / {pr['recall']:.3f}",'Resemblance / variety reached. These do not measure factual correctness.'],
                ['PRDC density / coverage',f"{pr['density']:.3f} / {pr['coverage']:.3f}",'Clustering / source examples reached. Density can be above 1.'],
                ['Keyword JS distance',f"{d['keywords']['collections'][arm]['js_distance_base2']:.4f}",'Gap in important-word use; lower is closer.']])
        else:rows.append(['Text and keyword checks','Not applicable','Clinical rows contain fields and numbers rather than generated prose.'])
        table(['Measure','Result','What it means'],rows,[45,35,94])

    selected_arms={'bitext':'selected_prefix','insuranceqa':'original_generic','reliability':'selected_prefix','mimic_demo':'selected_candidate'}
    distribution_export={}
    for key,arm in selected_arms.items():
        core=benchmark[key]['cohorts'][arm]['core']
        fidelity=core['fidelity_vs_training']
        # The saved Bitext core has no categorical entries. Its frozen snapshot
        # separately records the already-computed supplied-label TV distances.
        categories=(snapshot[key]['chosen']['fidelity_vs_training']['categorical']
                    if key=='bitext' else fidelity['categorical'])
        distribution_export[key]={'arm':arm,'generated_valid_records':core['structure_valid'],
            'source_training_records':len(data(key)['train']),'categorical':categories,'numeric':fidelity['numeric'],
            'records_path':snapshot[key]['records_path'],'records_sha256':snapshot[key]['records_sha256'],
            'metric_sources':['reports/distribution_benchmark/v1/results.json']+(['reports/assessment_snapshot/snapshot.json'] if key=='bitext' else []),
            'scope':'Existing saved metrics, exported without recomputation; percentiles shown are P25/P50/P75.'}

    def distribution_tables():
        fmt=lambda value:f'{value:.3f}'.rstrip('0').rstrip('.')
        quartiles=lambda stats:' / '.join(fmt(stats[k]) for k in ['p25','median','p75'])
        page('Distribution details: insurance datasets')
        p('These are saved scores for the chosen generated batches, compared with their training data. Lower TV and Wasserstein distances mean closer distributions. There is no universal passing score.','small')
        specs=[('bitext','Bitext: 156 generated / 31,200 source',
                [('intent','Intent proportions (supplied labels)')],
                [('reply_characters','Reply lengths','characters')],
                [('request_characters','Request length (characters)'),('reply_characters','Reply length (characters)')]),
               ('insuranceqa','InsuranceQA: 128 generated / 12,889 source',
                [('domain','Insurance-domain proportions')],
                [('reply_characters','Answer lengths','characters')],
                [('request_characters','Question length (characters)'),('reply_characters','Answer length (characters)')]),
               ('reliability','Reliability: 64 generated / 357 source',
                [('category','Category proportions (supplied labels)'),('expected_output.routing_decision','Routing proportions (generated)')],
                [('reply_characters','Workflow lengths','characters'),('required_action_count','Number of required actions','actions')],
                [('reply_characters','Workflow length (characters)'),('required_action_count','Required-action count')])]
        for key,title,tvs,distances,numerics in specs:
            item=distribution_export[key];sub(title)
            rows=[['TV: '+label,f"{item['categorical'][field]['total_variation']:.4f}"] for field,label in tvs]
            rows += [['Wasserstein: '+label,fmt(item['numeric'][field]['wasserstein'])+' '+unit] for field,label,unit in distances]
            table(['Check','Distance from source'],rows,[122,52],compact=True)
            table(['Percentiles: P25 / P50 / P75','Training data','Generated data'],
                [[label,quartiles(item['numeric'][field]['reference']),quartiles(item['numeric'][field]['generated'])] for field,label in numerics],
                [72,51,51],compact=True)
        p('P25 is the lower quarter, P50 is the middle (median), and P75 is the upper quarter. These describe the middle of the data, not its rare extremes. Matching these three values alone does not prove a full match.','small')
        p('Bitext intent and Reliability category labels were supplied before generation, so their TV scores describe our controls. Routing decisions were generated. Workflow length counts the expected-output JSON, not a customer-facing answer.','small')

        page('Distribution details: MIMIC demo')
        item=distribution_export['mimic_demo']
        p('Selected clinical candidate: 64 generated admissions compared with 230 training admissions. The earlier clinical results table shows improvement between model versions; the distances below compare the chosen candidate directly with the source.')
        sub('TV distance: category proportions')
        cats=[('admission_type','Admission type'),('insurance','Insurance label'),('race','Race label'),('hospital_expire_flag','In-hospital death flag')]
        table(['Category checked','TV distance'],[[label,f"{item['categorical'][field]['total_variation']:.4f}"] for field,label in cats],[122,52],compact=True)
        p('Zero means identical proportions for the selected categories. A larger value means a greater gap. This is not a clinical accuracy percentage. These are selected category checks; the full saved export includes the others.','small')
        sub('Percentiles and Wasserstein distances')
        fields=[('age','Age (years)'),('length_of_stay_days','Hospital stay (days)'),('creatinine_mg_dl','Creatinine (mg/dL)'),('glucose_mg_dl','Glucose (mg/dL)'),('hemoglobin_g_dl','Hemoglobin (g/dL)')]
        table(['Measurement','Source P25 / P50 / P75','Generated P25 / P50 / P75','Wasserstein'],
            [[label,quartiles(item['numeric'][field]['reference']),quartiles(item['numeric'][field]['generated']),fmt(item['numeric'][field]['wasserstein'])] for field,label in fields],
            [41,51,51,31],compact=True)
        p('P25 / P50 / P75 means lower quarter / middle / upper quarter. Lab percentiles use only nonmissing measurements. Wasserstein has the unit shown in the first column. Values are rounded; the distance is not the difference between the medians.','small')
        sub('How to read one row')
        p('For glucose, the source median is 120 mg/dL and the generated median is 134 mg/dL. The source upper quarter is 171.5, compared with 167 in generation. Wasserstein is 17.058 mg/dL: it compares the whole set of values, rather than one percentile.')
        p('For Reliability on the previous page, action-count quartiles are 3 / 3 / 4 in both sets, yet Wasserstein is 0.143 actions. The proportions between those points can still differ. This is why we show both percentiles and a full-distribution distance.')
        p('MIMIC diagnosis and missing-lab conditions were supplied as generation controls. Small batches can miss rare values; these descriptive scores do not establish clinical correctness or distribution equivalence. Source: reports/distribution_benchmark/v1/results.json; Bitext TV also uses reports/assessment_snapshot/snapshot.json.','small')

    subcategory_count=len({r['subcategory'] for r in allrows['reliability']})
    write_sections(page,p,sub,table,example,result_table,subcategory_count,comparison,distribution_tables)

    def footer(c,d):
        c.setFillColor(ink);c.setFont('Helvetica',8);c.drawString(18*mm,12*mm,'Gemma | Hiring assessment | 2 November 2026');c.drawRightString(192*mm,12*mm,str(d.page))
    SimpleDocTemplate(str(out),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=20*mm,title='Gemma synthetic-data generation - hiring assessment').build(story,onFirstPage=footer,onLaterPages=footer)
    md.write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    inputs=['reports/distribution_benchmark/v1/results.json','reports/training_stage_comparison.md','reports/all_datasets_assessment.md','outputs/insuranceqa-base-diagnostic-v1/report.md']
    inputs+=['outputs/prefix-alignment-v1/report.md','outputs/insuranceqa-one-shot-v1/report.md','outputs/reliability-consistency-v1/protocol.json','outputs/reliability-consistency-v1/report.md','outputs/fidelity-alignment-v1/report.md']
    inputs+=['scripts/prepare_bitext.py','scripts/prepare_remaining_datasets.py','reports/bitext_audit.json',*[f'data/{k}/prepared/preparation_audit.json' for k in ['insuranceqa','reliability','mimic_demo']]]
    (ROOT/'reports/gemma_hiring_assessment_examples.json').write_text(json.dumps(comparisons,indent=2,ensure_ascii=False),encoding='utf-8')
    inputs+=['reports/gemma_hiring_assessment_examples.json']
    (ROOT/'reports/gemma_hiring_distribution_tables.json').write_text(json.dumps(distribution_export,indent=2,ensure_ascii=False),encoding='utf-8')
    inputs+=['reports/gemma_hiring_distribution_tables.json','reports/assessment_snapshot/snapshot.json']
    write={'evidence_hashes':{p:sha(ROOT/p) for p in inputs},'pdf_sha256':sha(out),'bitext_training_records_verified':31200,'dataset_examples':{k:{'source_path':str(data(k)['spec']['prepared'])+'/train.jsonl','source_sha256':sha(ROOT/data(k)['spec']['prepared']/'train.jsonl'),'source_line_1_based':example_lines[k],'displayed_cells':dict(example_cells[k]),'selection':'Readability and profanity-free source wording; illustrations, not performance samples.'} for k in examples},'scope':'Readable synthesis of saved results; no new training/generation/metric calculation.'}
    (ROOT/('reports/'+name+'_provenance.json')).write_text(json.dumps(write,indent=2),encoding='utf-8')
    print('Created',out)

if __name__=='__main__':main()
