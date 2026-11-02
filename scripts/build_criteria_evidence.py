"""CPU-only presentation and finite-training-sample calibration of frozen cohorts.

No model selection, training, generation or held-out test scoring occurs here.
"""
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from dataset_catalog import ROOT, core, data, read_json, read_rows, sha, signature, text_at, value_at
from dataset_metrics import tv
from json_io import write_json

OUT = ROOT / 'reports/criteria_evidence/v1'
SEED = 2026110117
REPEATS = 200


def numeric_values(key, records, field):
    if field.endswith('_characters'):
        role = field.removesuffix('_characters')
        return [len(text_at(r, data(key)['spec'][role])) for r in records]
    if field == 'required_action_count':
        return [len(r['expected_output']['required_actions']) for r in records]
    return [value_at(r, field) for r in records]


def finite(values):
    return [v for v in values if type(v) in (int, float) and math.isfinite(v)]


def distance(a, b):
    a, b = finite(a), finite(b)
    return float(wasserstein_distance(a, b)) if a and b else None


def band(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'estimable_repeats': 0, 'p05': None, 'median': None, 'p95': None}
    lo, med, hi = np.percentile(values, [5, 50, 95])
    return {'estimable_repeats': len(values), 'p05': float(lo), 'median': float(med), 'p95': float(hi)}


def categorical_sample_distances(reference, samples):
    # Normalize once, exactly as tv() does; reuse integer counts across draws.
    labels = [signature(v) for v in reference]
    vocabulary = {v:i for i,v in enumerate(sorted(set(labels)))}
    codes = np.array([vocabulary[v] for v in labels])
    target = np.bincount(codes, minlength=len(vocabulary))/len(codes)
    return [float(.5*np.abs(np.bincount(codes[indices], minlength=len(vocabulary))/len(indices)-target).sum()) for indices in samples]


def number(value):
    return '—' if value is None else f'{value:.4g}'


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def plot(fig):
    fig.update_layout(template='plotly_white', height=360, margin=dict(l=55,r=20,t=80,b=80),
                      font=dict(family='Arial',size=12), colorway=['#167d8d','#d98136'],
                      legend=dict(orientation='h', y=1.07))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive':True,'displaylogo':False})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_path = ROOT/'reports/assessment_snapshot/snapshot.json'
    snapshot = read_json(snapshot_path)
    protocol = {
        'seed': SEED, 'repeats': REPEATS, 'snapshot_sha256': sha(snapshot_path),
        'purpose': 'Descriptive interpretation of frozen development cohorts; no model/prompt selection.',
        'method': 'For each dataset, draw n training rows without replacement 200 times, where n is the schema-valid synthetic cohort size. Compare each sample with the same full training reference used by the synthetic score. Repeat draws may overlap.',
        'interval': '5th–95th percentiles of finite-training-sample distances; not a confidence interval, p-value, equivalence test or pass threshold.',
        'limitations': ['Samples overlap their full training reference; this understates independent-population variation, especially for small MIMIC.',
            'MIMIC resampling is by admission, not patient; within-patient dependence is not modelled.',
            'Unconditional sample context is not a matched conditional control. Supplied labels/diagnoses/null patterns can outperform random-sample variability by construction.',
            'Only schema-valid outputs enter distributions. Invalid attempts remain in the validity denominator.',
            'Training references only; no validation or test examples are used.']}
    # Persist the design before computing any reference bands.
    write_json(OUT/'protocol.json', protocol)
    rng = np.random.default_rng(SEED)
    results = {'protocol': protocol, 'datasets': {}}
    sections = []
    md = ['# How close is our synthetic data to its source?', '',
          'CPU-only analysis of frozen outputs. No new model training, generation, selection or test scoring.', '',
          '## How to read the reference bands', '', protocol['method'], '', protocol['interval'], '',
          *['- '+s for s in protocol['limitations']], '',
          'Distances are descriptive: lower means closer on that measurement, not more factually correct. Numeric Wasserstein retains the field’s units; do not average different fields.', '']
    for key, item in snapshot['datasets'].items():
        source = data(key)
        train = [core(key,r) for r in source['train']]
        path = ROOT/item['records_path']
        assert sha(path) == item['records_sha256'], 'Frozen records changed'
        wrappers = read_rows(path)
        generated = [w['record'] for w in wrappers]
        n = len(generated)
        assert n == item['chosen']['structure_valid'] and n <= len(train)
        samples = [rng.choice(len(train), size=n, replace=False).tolist() for _ in range(REPEATS)]
        fidelity = item['chosen']['fidelity_vs_training']
        entry = {'n':n, 'source_n':len(train), 'cohort_sha256':sha(path),
                 'training_sha256':sha(ROOT/source['spec']['prepared']/'train.jsonl'),
                 'numeric':{}, 'categorical':{}}
        numeric_rows = []
        for field, observed in fidelity['numeric'].items():
            ref = numeric_values(key,train,field)
            gen = numeric_values(key,generated,field)
            observed_distance = distance(gen,ref)
            assert observed_distance is None or math.isclose(observed_distance,observed['wasserstein'],rel_tol=1e-9,abs_tol=1e-9)
            context = band([distance([ref[i] for i in indices],ref) for indices in samples])
            entry['numeric'][field] = {'synthetic_distance': observed_distance, 'real_sample_reference':context}
            numeric_rows.append([field, number(observed_distance), number(context['median']), number(context['p05'])+' – '+number(context['p95'])])
        category_rows = []
        for field, observed in fidelity['categorical'].items():
            ref = [value_at(r,field) for r in train]
            gen = [value_at(r,field) for r in generated]
            observed_distance = tv(gen,ref)
            assert math.isclose(observed_distance, observed['total_variation'],abs_tol=1e-9)
            context = band(categorical_sample_distances(ref,samples))
            entry['categorical'][field] = {'synthetic_distance':observed_distance,'real_sample_reference':context}
            category_rows.append([field, number(observed_distance), number(context['median']), number(context['p05'])+' – '+number(context['p95'])])
        results['datasets'][key] = entry
        chosen = item['chosen']
        body = f'<section id="{key}"><h2>{esc(item["label"])}</h2><p>{esc(item["policy"])}</p>'
        body += '<div class="stats">'+''.join(f'<div><strong>{esc(v)}</strong><span>{esc(label)}</span></div>' for v,label in [
            (f'{n}/{chosen["attempted"]}', 'Structure valid / attempted'),
            (f'{chosen["source_constraint_valid"]}/{chosen["attempted"]}', 'Source-label checks / attempted'),
            (f'{100*chosen["exact_training_record_match_rate"]:.1f}%', 'Normalized whole-record copying')])+'</div>'
        body += '<details><summary>What these results do and do not show</summary><ul>'+''.join('<li>'+esc(note)+'</li>' for note in item['notes'])+'</ul></details>'
        body += '<h3>1. Do the ranges look similar?</h3><p>Observed source ranges are not safety limits. Covering the range and staying inside it are separate questions.</p>'
        range_rows=[]
        for field, v in fidelity['numeric'].items():
            triple=lambda x: ' / '.join(number(x[t]) for t in ['min','median','max']) if x else '—'
            range_rows.append([field,triple(v.get('reference')),triple(v.get('generated')),
                f'{100*v["reference_missing_rate"]:.1f}% / {100*v["generated_missing_rate"]:.1f}%',
                f'{100*v["generated_outside_reference_range_rate"]:.1f}%' if v.get('generated_outside_reference_range_rate') is not None else '—'])
        body += table(['Field','Source min / median / max','Synthetic min / median / max','Missing: source / synthetic','Outside source range'],range_rows)
        body += '<h3>2. Do the distributions overlap?</h3><p>Choose a field. Both charts use proportions, so unequal sample sizes can be compared. Numeric plots exclude missing values; missing rates appear above.</p><div class="grid">'
        numeric_fig=go.Figure()
        for i,field in enumerate(fidelity['numeric']):
            combined = finite(numeric_values(key,train,field)) + finite(numeric_values(key,generated,field))
            low, high = min(combined), max(combined)
            step = (high-low)/25 if high>low else 1
            for name,rows in [('Training',train),('Synthetic',generated)]:
                numeric_fig.add_trace(go.Histogram(x=finite(numeric_values(key,rows,field)),name=name,histnorm='probability',opacity=.6,visible=i==0,legendgroup=name,xbins=dict(start=low,end=high+step,size=step)))
        numeric_fig.update_layout(barmode='overlay',updatemenus=[dict(y=1.32,x=0,buttons=[dict(label=f,method='update',args=[{'visible':[j//2==i for j in range(len(numeric_fig.data))]},{'xaxis.title':f}]) for i,f in enumerate(fidelity['numeric'])])])
        body += plot(numeric_fig)
        category_fig=go.Figure()
        for i,field in enumerate(fidelity['categorical']):
            counts=[Counter(str(value_at(r,field)) for r in rows) for rows in [train,generated]]
            labels=sorted(set(counts[0])|set(counts[1]))
            for name,c,total in zip(['Training','Synthetic'],counts,[len(train),n]):
                category_fig.add_trace(go.Bar(x=labels,y=[c[v]/total for v in labels],name=name,visible=i==0))
        category_fig.update_layout(barmode='group',updatemenus=[dict(y=1.32,x=0,buttons=[dict(label=f,method='update',args=[{'visible':[j//2==i for j in range(len(category_fig.data))]}]) for i,f in enumerate(fidelity['categorical'])])])
        body += plot(category_fig)+'</div>'
        body += '<details><summary>Inspect all distances and real-sample reference bands</summary><p>The band is descriptive sampling context, not a pass threshold. Controlled labels may match perfectly by construction.</p>'
        body += table(['Numeric field (Wasserstein)','Synthetic → full source','Real sample median','Real sample 5th–95th percentile'],numeric_rows)
        body += table(['Category (total variation)','Synthetic → full source','Real sample median','Real sample 5th–95th percentile'],category_rows)+'</details>'
        body += '<h3>3. Do the characteristics make sense?</h3>'
        if key=='mimic_demo':
            correlations=fidelity['spearman_correlations']
            body += '<p>Relationships between numeric fields, measured by Spearman correlation. Small samples and missing values limit interpretation; correlation does not certify clinical plausibility.</p><div class="grid">'
            fields=list(fidelity['numeric'])
            for label,side in [('Training','reference'),('Synthetic','generated')]:
                pairs=correlations[side]
                matrix=[[1 if a==b else next((pairs[p]['rho'] for p in [a+' / '+b,b+' / '+a] if p in pairs),None) for b in fields] for a in fields]
                fig=go.Figure(go.Heatmap(z=matrix,x=fields,y=fields,zmin=-1,zmax=1,colorscale='RdBu',reversescale=True))
                fig.update_layout(title=label+' numeric relationships')
                body+=plot(fig)
            body+='</div><p>Mean absolute correlation gap: '+number(correlations['mean_absolute_difference'])+'. Expert clinical and full diagnosis-code validation remain incomplete.</p>'
        elif key=='reliability':
            body+='<p><strong>AI workflow review: 45 Pass, 15 Unsure, 4 Issue out of 64.</strong> One unblinded AI reviewer; not expert-validated accuracy. Remaining problems include repeated payments, conflicting policy context and missed questions.</p>'
        elif key=='insuranceqa':
            body+='<p>Prior stratified AI review: 15 Pass, 11 Issue, 4 Unsure among 30 examples. This is not a population error-rate estimate. Insurance factual and answer-coherence concerns remain.</p>'
        else:
            body+='<p>Source-like support style and broad intent coverage are strengths. Labels are externally supplied; one whole-record training match remains. Repeated response templates require inspection.</p>'
        body+='<details><summary>Compare source and generated examples</summary><p>Illustrative records, not paired answers or a quality score. Frozen generated examples are shown in original order; no rerolls.</p><div class="grid">'
        body+='<div><h4>Training example</h4><pre>'+esc(json.dumps(train[0],ensure_ascii=False,indent=2))+'</pre></div><div><h4>Generated example · sample '+esc(wrappers[0]['sample_id'])+'</h4><pre>'+esc(json.dumps(generated[0],ensure_ascii=False,indent=2))+'</pre></div></div></details></section>'
        sections.append(body)
        md += ['## '+item['label'], '',item['policy'], '',f'Training rows: {len(train)}. Schema-valid synthetic rows: {n}; attempted: {chosen["attempted"]}.', '',
               '| Numeric field | Synthetic Wasserstein | Real-sample median | Real-sample 5th–95th percentile |','|---|---:|---:|---|',
               *['| '+' | '.join(r)+' |' for r in numeric_rows], '',
               '| Categorical field | Synthetic TV | Real-sample median | Real-sample 5th–95th percentile |','|---|---:|---:|---|',
               *['| '+' | '.join(r)+' |' for r in category_rows], '',*['- '+s for s in item['notes']], '']
        print(key, 'complete', flush=True)
    write_json(OUT/'reference_bands.json',results)
    md += ['## Method references', '',
        '- [SDMetrics: quality, individual distributions and pairwise trends](https://docs.sdv.dev/sdmetrics/data-metrics/quality)',
        '- [SDMetrics: source/synthetic visualization utilities](https://docs.sdv.dev/sdmetrics/getting-started/visualization-utilities)',
        '- [SDMetrics: boundary adherence differs from range coverage](https://docs.sdv.dev/sdmetrics/data-metrics/diagnostic/boundaryadherence)',
        '- [MAUVE authors: small-sample interpretation limits](https://github.com/krishnap25/mauve)', '',
        'The finite-training-sample reference bands above are a project-specific descriptive calibration, not an SDMetrics certification or a MAUVE result.']
    (OUT/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    nav=''.join(f'<a href="#{key}">{esc(item["label"])}</a>' for key,item in snapshot['datasets'].items())
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gemma · Evidence against the criterion</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#203045;font:16px/1.6 Arial,sans-serif}main{max-width:1200px;margin:auto;padding:32px 24px}h1{font-size:38px;line-height:1.15}h2{font-size:29px}h3{margin-top:30px}p{max-width:950px}.intro,section{background:white;border:1px solid #dbe3eb;border-radius:16px;padding:28px;margin:24px 0}nav{display:flex;gap:10px;flex-wrap:wrap}nav a{background:#173b51;color:white;padding:9px 16px;border-radius:8px;text-decoration:none}.stats,.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.stats{grid-template-columns:repeat(3,1fr);margin:22px 0}.stats div{background:#eef5f6;padding:16px;border-radius:10px}.stats strong,.stats span{display:block}.stats strong{font-size:26px}.stats span{font-size:13px}details{background:#f5f7fa;padding:15px;border-radius:9px;margin:20px 0}summary{cursor:pointer;font-weight:bold}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:11px;border-bottom:1px solid #dce3eb;vertical-align:top}th{background:#edf2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#142637;color:#e5eff6;padding:18px;border-radius:8px;font-size:12px;max-height:450px;overflow:auto}.note{border-left:4px solid #d98136;padding-left:16px}@media(max-width:800px){.grid,.stats{grid-template-columns:1fr}main{padding:15px}section{padding:17px}h1{font-size:30px}}@media print{details{display:block}nav{display:none}section{break-before:page}.grid{display:block}}</style><script>'''+get_plotlyjs()+'''</script><main><div class="intro"><p>GEMMA · FROZEN DEVELOPMENT EVIDENCE</p><h1>How close is the synthetic data?</h1><p><strong>Criterion: “Matches the ranges, distribution, and characteristics of the data it was trained on.”</strong></p><p>Read each dataset separately. Inspect ranges, compare distributions, then check examples and remaining semantic problems. No single percentage represents overall correctness.</p><p class="note">Supplied labels and clinical conditions are external controls. Current cohorts are development evidence, not a new independent test. Original models and generated records are unchanged.</p><nav>'''+nav+'''</nav><details><summary>How to interpret the new real-sample reference bands</summary><p>We draw equally sized training samples without replacement 200 times and compare each with the full training reference. We show the middle 90% of these distances, alongside the synthetic distance to that same reference.</p><p>This is finite-dataset sampling context, not a confidence interval or pass threshold. Samples overlap the reference; MIMIC uses admission rows rather than patient clusters. Controlled label frequencies can look better than random sampling by construction. No held-out test data is used.</p></details></div>'''+''.join(sections)+'''<p>Reproduce in the original workspace: .venv-analysis/Scripts/python.exe scripts/build_criteria_evidence.py · Raw metrics: reference_bands.json · Design: protocol.json. All charts work offline.</p></main></html>'''
    (OUT/'index.html').write_text(document,encoding='utf-8')
    print('Evidence page:',OUT/'index.html',flush=True)


if __name__=='__main__':
    main()
