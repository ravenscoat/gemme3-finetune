"""Standalone, offline interactive distribution plots for the candidate comparison."""
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset_catalog import core,data,read_json,read_rows
from mimic_conditioned_data import RUN


def main():
    source=[core('mimic_demo',r) for r in data('mimic_demo')['train']]
    fields=['age','length_of_stay_days','creatinine_mg_dl','glucose_mg_dl','hemoglobin_g_dl']
    for mode in ['generic','conditioned']:
        groups={'Training source':source,
                'Saved model':[r['record'] for r in read_rows(RUN/('old_'+mode+'_records.jsonl'))],
                'Mixed-prompt candidate':[r['record'] for r in read_rows(RUN/('candidate_'+mode+'_records.jsonl'))]}
        fig=make_subplots(rows=2,cols=3,subplot_titles=[f.replace('_',' ') for f in fields]+['Missing labs'])
        colors=['#69758b','#c38d64','#6578d9']
        for group_index,(label,rows) in enumerate(groups.items()):
            if not rows:continue
            for index,field in enumerate(fields):
                values=[r[field] for r in rows if r[field] is not None]
                fig.add_trace(go.Histogram(x=values,histnorm='probability',name=label+' (n='+str(len(rows))+')',
                    legendgroup=label,showlegend=index==0,opacity=.48,marker_color=colors[group_index],
                    bingroup=field,nbinsx=16),row=index//3+1,col=index%3+1)
            fig.add_trace(go.Scatter(x=['Creatinine','Glucose','Hemoglobin'],y=[100*sum(r[f] is None for r in rows)/len(rows) for f in fields[2:]],
                name=label,legendgroup=label,showlegend=False,mode='lines+markers',marker_color=colors[group_index]),row=2,col=3)
        fig.update_layout(title='MIMIC demo: '+mode+' generation — saved model vs mixed-prompt candidate',
            template='plotly_white',barmode='overlay',height=820,width=1320,
            legend=dict(orientation='h',y=-.13),margin=dict(t=100,b=130),font=dict(family='Arial',size=12))
        fig.add_annotation(xref='paper',yref='paper',x=0,y=-.24,showarrow=False,align='left',
            text='Development samples:64 attempts/model; histograms use valid non-null values. Click a legend item to inspect a series.<br>'+
            ('Conditions supply diagnosis pairs and lab-null patterns externally. ' if mode=='conditioned' else '')+
            'No clinical validity or independent final-test claim. Read failures, copying and per-field distances alongside these plots.')
        fig.update_yaxes(title_text='Missing %',row=2,col=3)
        fig.write_html(RUN/('distributions_'+mode+'.html'),include_plotlyjs=True)
    print('Saved two offline interactive distribution plots.')


if __name__=='__main__':main()
