"""Offline interactive scientific plots from the verified batches (no inference)."""
from collections import Counter
import html
import json

import plotly.graph_objects as go

from dataset_catalog import CATALOG,core,data,read_json,read_rows,text_at
from fidelity_alignment import OUT,ORDER


def measured(key,field,r):
    if field=='request_characters':return len(text_at(r,CATALOG[key]['request']))
    if field=='reply_characters':return len(text_at(r,CATALOG[key]['reply']))
    if field=='required_action_count':return len(r['expected_output']['required_actions'])
    return r[field]


def ecdf(values):
    counts=Counter(v for v in values if v is not None);total=sum(counts.values());x=sorted(counts);running=0;y=[]
    for value in x:running+=counts[value];y.append(running/total)
    return x,y


def main():
    assert (OUT/'verification.json').exists(),'Verify batches first'
    chunks=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Training-source fidelity</title>',
        '<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px;color:#182337}h2{margin-top:60px}p{max-width:1000px;line-height:1.6}nav a{margin-right:20px}</style></head><body>',
        '<h1>Training-source fidelity</h1><p>Fixed development batches. Blue: training source; orange: model-only baseline; green: externally controlled generation. Controls supply source profiles and intervals. Improvements are properties of the combined pipeline. All failures and regressions remain in report.md; charts use valid rows only. No new model training or final-test scoring.</p>',
        '<nav>'+''.join(f'<a href="#{k}">{html.escape(CATALOG[k]["label"])}</a>' for k in ORDER)+'</nav>']
    include=True
    for key in ORDER:
        m=read_json(OUT/key/'metrics.json');p=read_json(OUT/key/'protocol.json');primary=CATALOG[key]['group']
        chunks += [f'<section id="{key}"><h2>{html.escape(CATALOG[key]["label"])}</h2>',
            f'<p>{p["n"]} attempts per arm. Schema-valid: {m["arms"]["baseline"]["structure_valid"]} before; {m["arms"]["controlled"]["structure_valid"]} after. Category counts and ECDF curves are descriptive, not evidence of equivalence.</p>']
        b=m['arms']['baseline']['feature_fidelity'][primary];c=m['arms']['controlled']['feature_fidelity'][primary]
        labels=sorted(set(b['source_counts'])|set(b['generated_counts'])|set(c['generated_counts']))
        fig=go.Figure()
        for name,counts,color in [('Training source',b['source_counts'],'#3282bd'),('Model only',b['generated_counts'],'#d87a23'),('Controlled',c['generated_counts'],'#289b78')]:
            total=sum(counts.values());fig.add_bar(name=name,x=[json.loads(s) for s in labels],y=[counts.get(s,0)/total if total else None for s in labels],marker_color=color)
        fig.update_layout(title=f'{primary}: source proportions versus generated proportions',barmode='group',template='plotly_white',height=600,margin=dict(b=220),yaxis_tickformat='.0%',legend_orientation='h')
        chunks.append(fig.to_html(full_html=False,include_plotlyjs=include));include=False
        records={'Training source':[core(key,r) for r in data(key)['train']],
            'Model only':[w['record'] for w in read_rows(OUT/key/'baseline_records.jsonl')],
            'Controlled':[w['record'] for w in read_rows(OUT/key/'controlled_records.jsonl')]}
        for field in m['arms']['baseline']['fidelity_vs_training']['numeric']:
            fig=go.Figure()
            for name,color in [('Training source','#3282bd'),('Model only','#d87a23'),('Controlled','#289b78')]:
                x,y=ecdf([measured(key,field,r) for r in records[name]])
                fig.add_scatter(name=name,x=x,y=y,mode='lines',line=dict(color=color,shape='hv'))
            fig.update_layout(title=field+' — empirical cumulative distribution (missing values excluded)',template='plotly_white',height=400,xaxis_title=field,yaxis_title='Fraction at or below value',yaxis_range=[0,1],legend_orientation='h')
            chunks.append(fig.to_html(full_html=False,include_plotlyjs=False))
        chunks.append('</section>')
    chunks.append('</body></html>');(OUT/'distributions.html').write_text('\n'.join(chunks),encoding='utf-8')
    print('Wrote offline distributions.html',flush=True)


if __name__=='__main__':main()
