"""Pin and download the remaining public insurance sources onto D:."""
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
QA_REVISION = '5c380dd086067adacc2fec3bfa920a78e57bdcf6'


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'GemmaAssessment/1.0'}), timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def download(name, revision, source, files):
    directory = ROOT / 'data' / name / 'raw'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'revision': revision, 'source': source, 'files': []}
    for filename, url in files:
        path = directory / filename
        if not path.exists():
            data = fetch(url)
            path.write_bytes(data)
        data = path.read_bytes()
        manifest['files'].append({'name': filename, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        print(name, filename, len(data), flush=True)
    (directory.parent / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    filenames = ['vocabulary', 'README.md', 'InsuranceQA.label2answer.raw.encoded.gz']
    filenames += [f'InsuranceQA.question.anslabel.raw.100.pool.solr.{split}.encoded.gz' for split in ['train', 'valid', 'test']]
    base = f'https://raw.githubusercontent.com/shuzi/insuranceQA/{QA_REVISION}/V2/'
    download('insuranceqa', QA_REVISION, 'https://github.com/shuzi/insuranceQA/tree/master/V2', [(name, base + name) for name in filenames])
    repo = 'pashas/insurance-ai-reliability-benchmark'
    revision = json.loads(fetch('https://huggingface.co/api/datasets/' + repo))['sha']
    base = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/'
    names = ['README.md', 'data/benchmark_v1.jsonl', 'data/train.jsonl', 'data/validation.jsonl', 'data/test.jsonl']
    download('reliability', revision, 'https://huggingface.co/datasets/' + repo, [(name.split('/')[-1], base + name) for name in names])


if __name__ == '__main__':
    main()
