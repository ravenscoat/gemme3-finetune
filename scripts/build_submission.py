"""Build a local review bundle, with selected adapters and explicit provenance."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile

from dataset_catalog import ROOT,CATALOG,data,read_json,sha
from json_io import write_json

DEST=ROOT/'submission/gemma-assessment-v1'


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def tree(source,target,extensions=None):
    for f in source.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts and (extensions is None or f.suffix in extensions):copy(f,target/f.relative_to(source))


def main():
    final=read_json(ROOT/'outputs/reliability-consistency-v1/decision.json')
    assert final['decision']=='reject_keep_existing_policy','Package selection needs explicit review before another decision'
    DEST.mkdir(parents=True,exist_ok=True);snapshot=read_json(ROOT/'reports/assessment_snapshot/snapshot.json')
    profiles={};adapters={}
    for key,item in snapshot['datasets'].items():
        adapter=Path(item['adapter']);tree(adapter,DEST/'adapters'/key)
        adapters[key]={'original_path':str(adapter),'weight_sha256':sha(adapter/'adapter_model.safetensors'),'config_sha256':sha(adapter/'adapter_config.json')}
        if key in ['bitext','reliability']:
            p=read_json(ROOT/'outputs/prefix-alignment-v1'/key/'protocol.json')
            profiles[key]={**{f:p[f] for f in ['prompts','assistant_prefixes','temperature','top_p','top_k','context']},'seed_start':p['baseline_batch_seed_start']}
        elif key=='insuranceqa':
            p=read_json(ROOT/CATALOG[key]['assessment']/'protocol.json');n=item['chosen']['attempted']
            profiles[key]={'prompts':[data(key)['prompt']]*n,'assistant_prefixes':['']*n,'seed_start':p['batch_seed_start'],**{f:p[f] for f in ['temperature','top_p','top_k','context']}}
        else:
            from mimic_conditioned_data import conditioned_prompt
            p=read_json(ROOT/'outputs/mimic-demo-fidelity-conditioned-v1-windows/comparison_plan.json')
            profiles[key]={'prompts':[conditioned_prompt(p['generic_prompt'],c) for c in p['conditions']],'assistant_prefixes':['']*p['n'],'seed_start':p['seed_start'],**{f:p[f] for f in ['temperature','top_p','top_k','context']}}
        source=ROOT/CATALOG[key]['prepared']
        for name in ['schema.json','generation_prompt.txt','preparation_audit.json']:
            if (source/name).exists():copy(source/name,DEST/'data_provenance'/key/name)
        selection=read_json(ROOT/CATALOG[key]['selection'])
        run=Path(selection['adapter']).parent
        for name in ['protocol.json','training_summary.json','selection.json','training_plan.json']:
            if (run/name).exists():copy(run/name,DEST/'training_protocols'/key/name)
    old=read_json(ROOT/CATALOG['mimic_demo']['selection']);oldpath=Path(old['adapter']);tree(oldpath,DEST/'adapters/mimic_demo_original')
    adapters['mimic_demo_original']={'original_path':str(oldpath),'weight_sha256':sha(oldpath/'adapter_model.safetensors'),'config_sha256':sha(oldpath/'adapter_config.json')}
    write_json(DEST/'generation_profiles.json',profiles)
    tree(ROOT/'reports/assessment_snapshot',DEST/'reports/assessment_snapshot')
    if (ROOT/'reports/criteria_evidence/v1').exists():
        tree(ROOT/'reports/criteria_evidence/v1',DEST/'reports/criteria_evidence/v1')
    if (ROOT/'reports/submission_portable_smoke.json').exists():copy(ROOT/'reports/submission_portable_smoke.json',DEST/'reports/submission_portable_smoke.json')
    tree(ROOT/'reports/reliability_workflow_audit/v1',DEST/'reports/reliability_workflow_audit/v1')
    tree(ROOT/'outputs/reliability-consistency-v1',DEST/'outputs/reliability-consistency-v1')
    tree(ROOT/'outputs/prefix-alignment-v1',DEST/'outputs/prefix-alignment-v1')
    if (ROOT/'reports/gemma_project_journey.md').exists():
        copy(ROOT/'reports/gemma_project_journey.md',DEST/'reports/gemma_project_journey.md')
        for name in ['gemma_project_journey.pdf','gemma_project_journey_verification.json']:
            copy(ROOT/'output/pdf'/name,DEST/'reports'/name)
        # Preserve the negative experiments discussed in the narrative, too.
        for name in ['fidelity-alignment-v1','schema-prompt-pilot-v1','prefix-prompt-pilot-v1']:
            tree(ROOT/'outputs'/name,DEST/'outputs'/name,{'.json','.jsonl','.md'})
        copy(ROOT/'data/mimic_demo/prepared/preparation_audit.json',DEST/'data/mimic_demo/prepared/preparation_audit.json')
    if (ROOT/'reports/gemma_report.md').exists():
        copy(ROOT/'reports/gemma_report.md',DEST/'reports/gemma_report.md')
        for name in ['gemma_report.pdf','gemma_report_verification.json']:
            copy(ROOT/'output/pdf'/name,DEST/'reports'/name)
    bitext=ROOT/'outputs/bitext-4b-full-pass-evaluation-500-windows'
    for name in ['base_generations.jsonl','tuned_generations.jsonl','generation_protocol.json','evaluation.json','evaluation_unfenced.json','extended_evaluation.json','mauve_evaluation.json','mauve_protocol.json']:
        if (bitext/name).exists():copy(bitext/name,DEST/'outputs'/bitext.name/name)
    advanced=ROOT/'reports/advanced_analysis/v1'
    for name in ['protocol.json','ai_review_summary.json','ai_judgments.jsonl','prdc_results.json','sdmetrics_results.json','exploratory_results.json','README.md']:
        if (advanced/name).exists():copy(advanced/name,DEST/'reports/advanced_analysis/v1'/name)
    for key in ['insuranceqa','reliability','mimic_demo']:
        source=ROOT/CATALOG[key]['assessment'];target=DEST/CATALOG[key]['assessment']
        for name in ['assessment.json','protocol.json','test_freeze.json','base_generations.jsonl','tuned_generations.jsonl','tuned_records.jsonl']:
            if (source/name).exists():copy(source/name,target/name)
    source=ROOT/'outputs/mimic-demo-fidelity-conditioned-v1-windows'
    for f in source.iterdir():
        if f.is_file() and f.suffix in ['.json','.jsonl','.md']:copy(f,DEST/'outputs'/source.name/f.name)
    for name in ['final_test_assessment.md','training_stage_comparison.md','all_datasets_assessment.md']:
        if (ROOT/'reports'/name).exists():copy(ROOT/'reports'/name,DEST/'reports'/name)
    for name in ['advanced-analysis-environment.txt','final_experiment_freeze.json','demo_simplification_verification.json']:
        if (ROOT/'reports'/name).exists():copy(ROOT/'reports'/name,DEST/'reports'/name)
    for folder in ['scripts','demo','tests']:
        tree(ROOT/folder,DEST/'source'/folder,{'.py','.ps1','.sh','.html','.js','.css'})
    for f in ROOT.glob('requirements*.txt'):copy(f,DEST/'source'/f.name)
    for name in ['decision.md','workflow.md','NEXT_SESSION.md']:copy(ROOT/name,DEST/name)
    copy(ROOT/'SUBMISSION_TALK.md',DEST/'SUBMISSION_TALK.md')
    copy(ROOT/'scripts/portable_generate.py',DEST/'run_model.py')
    base=read_json(ROOT/'.cache/models/gemma3-4b-training/source_manifest.json')
    write_json(DEST/'base_model_provenance.json',base)
    write_json(DEST/'adapters.json',adapters)
    write_json(DEST/'environment.json',{'python':sys.version,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}})
    report=(ROOT/'reports/assessment_snapshot/report.md').read_text(encoding='utf-8')
    (DEST/'REPORT.md').write_text(report,encoding='utf-8')
    readme=f'''# Gemma synthetic-data assessment — submission

Start with the [assessment report](reports/gemma_report.pdf). The longer
[project journey](reports/gemma_project_journey.pdf), [REPORT.md](REPORT.md)
and [decision log](decision.md) keep the detailed evidence.
The criterion is similarity to each training dataset, not insurance or clinical
accuracy. Improvements and regressions are both retained. No pass is guaranteed.

## Included

- Selected Gemma 3 4B QLoRA adapters for all four datasets, plus the original MIMIC
  adapter needed for its old-versus-candidate comparison. Weight files are unchanged.
- Frozen generic/controlled outputs, selected metrics, all64 Reliability AI judgments,
  the rejected final24-attempt fix, and offline distribution charts.
- Source code, data preparation/split provenance, environment versions and hashes.
- Exact selected generation prompts, assistant prefixes, settings and seed schedules.

Open `outputs/prefix-alignment-v1/distributions.html` directly in a browser.
Open `outputs/reliability-consistency-v1/report.md` for the final rejected change.

## Verify without a GPU

```powershell
python verify_submission.py
```

This checks every packaged file against SHA256SUMS.json. It does not re-score models.

## Generate using the packaged adapters

The base model is not bundled. Obtain the publisher-licensed base repository
`{base['repo']}` at revision `{base['revision']}`.
Its original file checksums are in base_model_provenance.json. Use a compatible
Windows CUDA/Unsloth environment; the tested stack is in environment.json and
source/requirements-windows.txt. Linux requires its own compatible GPU packages.
We did not claim validation of a fresh-machine installation.

```powershell
python run_model.py --dataset bitext --base-model D:\\path\\to\\base-model --batch-start 0 --output generated.jsonl
```

This generates four records, matching the study batch size. Use another multiple
of four to reproduce another recorded profile batch. Other dataset keys are
insuranceqa, reliability and mimic_demo. Supplied prefix/diagnosis/null conditions
are external controls; their compliance is not independent model quality.
Exact sampled text may vary with GPU/library versions. The runner copies the
adapter into a temporary local folder and changes only that copy's base path;
the original adapter/config files and their hashes remain unchanged.

The run_model.py runner needs no source training records. Raw training datasets
and the multi-GB base weights are intentionally not bundled. Download/preparation
scripts and split hashes are supplied in source/scripts and data_provenance.
Do not overwrite the original frozen workspace to recreate data or training.

## Reproduce analysis in the original workspace

These commands are for the original D:\\gemma-finetune project, where source data
and cached base weights already exist. They rebuild derived reports, not training:

```powershell
cd D:\\gemma-finetune
.\\.venv-analysis\\Scripts\\python.exe scripts\\record_reliability_workflow_review.py
.\\.venv-analysis\\Scripts\\python.exe scripts\\report_consistency_trial.py
.\\.venv-analysis\\Scripts\\python.exe scripts\\build_assessment_snapshot.py
.\\scripts\\run_demo_windows.ps1
```

Open http://127.0.0.1:8765/#evaluation and refresh an existing tab.
The frontend's live defaults remain unchanged; selected development cohorts are
shown separately. Source code is included, but its data explorer requires the
prepared source datasets; it is not a standalone data-free web deployment.

## Limits to state in the presentation

- Distinguish original Gemma, fine-tuned generic generation, and externally
  controlled generation. Do not credit supplied label frequencies to learning.
- The latest cohorts are development evidence, including pilot reuse. No new
  independent final test or proof of distribution equivalence is claimed.
- Reliability's45 Pass/15 Unsure/4 Issue are one unblinded AI review, not expert
  accuracy. Four concrete problems remain. The final fix failed and was rejected.
- InsuranceQA still has factual/coherence concerns. Clinical plausibility and
  complete ICD-code validity are unverified. MIMIC is the official100-patient demo,
  not full MIMIC or the U.S. population. Source resemblance is not privacy assurance.
'''
    if (ROOT/'reports/submission_portable_smoke.json').exists():readme+='\n## Local runner check\n\nThe packaged Bitext adapter was loaded through the portable runner and generated\nfour one-token completions as a smoke check. This verifies loading, not output\nquality or portability of every adapter to a fresh installation. Evidence:\nreports/submission_portable_smoke.json.\n'
    if (ROOT/'reports/criteria_evidence/v1/index.html').exists():
        readme+='\n## Start with the assignment criterion\n\nOpen reports/criteria_evidence/v1/index.html in a browser; all charts work offline.\nThe matching report.md and reference_bands.json contain the numeric evidence.\nReproduce in the original workspace with:\n`.venv-analysis/Scripts/python.exe scripts/build_criteria_evidence.py`.\nThe 200-repeat training-subsample bands are descriptive context, not confidence\nintervals, pass thresholds, or a new independent test.\n'
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    verifier='''import hashlib,json\nfrom pathlib import Path\nr=Path(__file__).resolve().parent\nm=json.loads((r/'SHA256SUMS.json').read_text(encoding='utf-8'))\nfor name,expected in m.items():\n p=(r/name).resolve()\n assert p.is_relative_to(r.resolve()),name\n with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()\n assert actual==expected,name\nprint(f'Verified {len(m)} packaged files.')\n'''
    (DEST/'verify_submission.py').write_text(verifier,encoding='utf-8')
    manifest={}
    for f in sorted(DEST.rglob('*')):
        if f.is_file() and f.name!='SHA256SUMS.json' and '.runtime-cache' not in f.parts:
            with f.open('rb') as stream:manifest[str(f.relative_to(DEST)).replace('\\','/')]=hashlib.file_digest(stream,'sha256').hexdigest()
    write_json(DEST/'SHA256SUMS.json',manifest)
    archive=ROOT/'submission/gemma-assessment-v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in [*manifest,'SHA256SUMS.json']:z.write(DEST/name,arcname='gemma-assessment-v1/'+name)
    review_archive=ROOT/'submission/gemma-assessment-review.zip'
    review_readme=('# Gemma assessment — review-only archive\n\n'
        'Start with reports/gemma_report.pdf for the assessment and methods appendix,\n'
        'or reports/gemma_project_journey.pdf for the longer narrative,\n'
        'then REPORT.md, SUBMISSION_TALK.md and the offline visual page\n'
        'reports/criteria_evidence/v1/index.html. Inspect raw outputs, protocols,\n'
        'the decision log and offline charts. Adapter binaries are intentionally\n'
        'omitted from this small review archive; they are in gemma-assessment-v1.zip.\n'
        'adapters.json records their provenance and hashes. No base model is bundled.\n'
        'python verify_submission.py verifies the files in this archive.\n'
        'The model runner needs the full bundle and the pinned, licensed base model.\n').encode()
    review_manifest={name:digest for name,digest in manifest.items() if not name.startswith('adapters/')}
    review_manifest['README.md']=hashlib.sha256(review_readme).hexdigest()
    with zipfile.ZipFile(review_archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in review_manifest:
            if name=='README.md':z.writestr('gemma-assessment-review/README.md',review_readme)
            else:z.write(DEST/name,arcname='gemma-assessment-review/'+name)
        z.writestr('gemma-assessment-review/SHA256SUMS.json',json.dumps(review_manifest,indent=2))
    print('Bundle:',DEST,'files:',len(manifest),'ZIP MB:',round(archive.stat().st_size/1e6,1),flush=True)
    print('Review ZIP MB:',round(review_archive.stat().st_size/1e6,1),flush=True)


if __name__=='__main__':main()
