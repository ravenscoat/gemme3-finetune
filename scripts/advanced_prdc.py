"""Use the released PRDC implementation on frozen independent text features."""
import importlib.metadata
from pathlib import Path
import numpy as np
from prdc import compute_prdc

from dataset_catalog import ROOT,read_json,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'


def main():
    manifest=read_json(OUT/'embedding_manifest.json');settings=manifest['protocol']
    results={'prdc_version':importlib.metadata.version('prdc'),'feature_manifest_sha256':sha(OUT/'embedding_manifest.json'),
             'datasets':{},'limits':['Experimental adaptation to GPT-2 text embeddings, originally developed for image generation.',
             'Small equal-size collections; dependent on embedding model, normalization, truncation and nearest-neighbor k.',
             'Conditional on structurally valid outputs; original/tuned failure denominators remain in core assessments.',
             'Density is not bounded by1; none of these scores is factual accuracy, privacy or a universal pass threshold.',
             'Validation-control scores are a same-source reference point, not a target guaranteeing equivalent distributions.']}
    results['limits'].append('Uses the same frozen feature model as existing Bitext MAUVE; complementary statistics, not independent embedding-model confirmation.')
    lines=['# Experimental text precision, recall, density and coverage','',
           'Frozen float32 GPT-2-large terminal-token embeddings, L2-normalized; maximum512 tokens. No classifier training.','',
           '| Dataset / field | Collection | k | Precision | Recall | Density | Coverage |','|---|---|---:|---:|---:|---:|---:|']
    for key,collections in manifest['features'].items():
        results['datasets'][key]={}
        for field in ['request','reply']:
            arrays={}
            for label in collections:
                item=collections[label][field];path=ROOT/item['path']
                if sha(path)!=item['sha256']:raise RuntimeError('Features changed')
                arrays[label]=np.load(path)
            values={}
            for label in ['base','tuned','validation_control']:
                values[label]={}
                for k in settings['prdc_nearest_k']:
                    score={name:float(value) for name,value in compute_prdc(arrays['training_reference'],arrays[label],nearest_k=k).items()}
                    values[label][str(k)]=score
                    lines.append(f'| {key} / {field} | {label} | {k} | '+ ' | '.join(f'{score[name]:.3f}' for name in ['precision','recall','density','coverage'])+' |')
            results['datasets'][key][field]={'n':len(arrays['base']),'scores':values,
                'truncation':{label:{k:collections[label][field][k] for k in ['truncated','n','max_original_tokens']} for label in collections}}
    write_json(OUT/'prdc_results.json',results)
    lines+=['',*['- '+note for note in results['limits']],'']
    (OUT/'prdc_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PRDC result saved.',flush=True)


if __name__=='__main__':main()
