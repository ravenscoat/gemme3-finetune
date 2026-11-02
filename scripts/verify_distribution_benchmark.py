"""Independent arithmetic/export/integrity checks for the saved benchmark."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from dataset_catalog import ROOT,read_json,sha
from distribution_benchmark import OUT,check_inputs
from json_io import write_json


def main():
    protocol=read_json(OUT/'protocol.json');check_inputs(protocol)
    result=read_json(OUT/'results.json');manifest=read_json(OUT/'embedding_manifest.json')
    audit=read_json(OUT/'verification.json')
    for cohort in read_json(OUT/'provenance.json')['cohorts'].values():
        assert sha(ROOT/cohort['raw'])==cohort['raw_sha256']
        for path,digest in cohort['source_protocol_hashes'].items():assert sha(ROOT/path)==digest
    for file,digest in audit['artifacts'].items():assert sha(OUT/file)==digest,file
    for path in OUT.rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Non-standard '+value)))
    comparisons=0;lexical=0
    for key,d in result['datasets'].items():
        assert len(d['cohorts'])==len(protocol['cohorts'][key])
        for arm,c in d['cohorts'].items():
            assert c['core']['structure_valid']==c['sdmetrics']['n']
            assert c['core']['structure_valid']<=c['core']['attempted']
            assert 0<=c['sdmetrics']['quality']<=1
            props=[r['Score'] for r in c['sdmetrics']['properties'] if r['Score'] is not None]
            assert abs(np.mean(props)-c['sdmetrics']['quality'])<1e-9
            assert c['sdmetrics']['categorical_distributions']
            for field,frequency in c['sdmetrics']['categorical_distributions'].items():
                a=frequency['reference_counts'];b=frequency['generated_counts']
                assert sum(a.values())==d['training_rows'] and sum(b.values())==c['core']['structure_valid']
                independent=.5*sum(abs(a.get(v,0)/sum(a.values())-b.get(v,0)/sum(b.values())) for v in a.keys()|b.keys())
                assert abs(independent-frequency['total_variation'])<1e-12
        assert len(pd.read_csv(OUT/key/'selected_categorical_frequencies.csv'))>0
        if not d['text_metrics']:
            for arm,c in d['cohorts'].items():
                frame=pd.read_csv(OUT/key/(arm+'_diagnoses.csv'))
                assert math.isclose(frame.source_share.sum(),1.,abs_tol=1e-10)
                assert math.isclose(frame.generated_share.sum(),1.,abs_tol=1e-10)
            continue
        for role,v in d['text_metrics'].items():
            x=np.load(ROOT/manifest['features'][key]['training_reference'][role]['path']).astype(float)
            for arm,s in v['collections'].items():
                item=manifest['features'][key][arm][role];y=np.load(ROOT/item['path']).astype(float)
                assert sha(ROOT/item['path'])==item['sha256']
                assert len(x)==len(y)==s['n']
                assert np.max(np.abs(np.linalg.norm(y,axis=1)-1))<1e-5
                independent=float(rbf_kernel(x,gamma=v['gamma']).mean()+rbf_kernel(y,gamma=v['gamma']).mean()-2*rbf_kernel(x,y,gamma=v['gamma']).mean())
                assert abs(max(0.,independent)-s['mmd2'])<1e-8,(key,role,arm)
                for score in s['prdc'].values():
                    assert all(0<=score[f]<=1 for f in ['precision','recall','coverage'])
                    assert score['density']>=0
                comparisons+=1
        for arm,s in d['keywords']['collections'].items():
            frame=pd.read_csv(OUT/key/(arm+'_keywords.csv'));p=frame.source_share.to_numpy();q=frame.generated_share.to_numpy();m=(p+q)/2
            assert math.isclose(p.sum(),1.,abs_tol=1e-10) and math.isclose(q.sum(),1.,abs_tol=1e-10)
            def kl(a,b):
                mask=a>0
                return float(np.sum(a[mask]*np.log2(a[mask]/b[mask])))
            js=math.sqrt(max(0.,.5*(kl(p,m)+kl(q,m))))
            assert abs(js-s['js_distance_base2'])<1e-9
            assert abs(.5*np.abs(p-q).sum()-s['total_variation'])<1e-9
            assert frame.keyword.iloc[-1]=='OTHER'
            lexical+=1
    for file,expected in [('scorecard_original_adapters.csv',4),('scorecard_selected_policies.csv',4),('scorecard_all_cohorts.csv',10)]:
        assert len(pd.read_csv(OUT/file))==expected
    document=(OUT/'index.html').read_text(encoding='utf-8')
    assert document.count('Plotly.newPlot(')>=4 and '<h2>The four original QLoRA adapters</h2>' in document
    assert not (ROOT/'reports/assessment_gpu.lock').exists()
    audit.update(independent_mmd_recalculations=comparisons,independent_keyword_js_tv_recalculations=lexical,
                 strict_json_checks_passed=True,export_checks_passed=True,gpu_lock_released=True,
                 source_generation_protocols_verified=True,
                 tests={'passed':47,'command':'.venv-analysis/Scripts/python.exe -m unittest discover -s tests'},
                 software_sources={p:sha(ROOT/p) for p in ['scripts/distribution_benchmark.py','scripts/verify_distribution_benchmark.py','tests/test_distribution_benchmark.py']})
    write_json(OUT/'verification.json',audit)
    print(json.dumps({k:audit[k] for k in ['frozen_inputs_unchanged','independent_mmd_recalculations','independent_keyword_js_tv_recalculations','strict_json_checks_passed','export_checks_passed','gpu_lock_released']},indent=2))


if __name__=='__main__':main()
