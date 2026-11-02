"""Read-only and invalid-request smoke checks; never starts a generation job."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8765'
checks = []


def request(path, expected=200, payload=None, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    req = Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    try:
        response = urlopen(req, timeout=30)
    except HTTPError as exc:
        response = exc
    with response:
        assert response.status == expected, (path, response.status, expected)
        data = response.read()
        result = json.loads(data) if response.headers.get_content_type() == 'application/json' else data
    checks.append({'path': path, 'status': expected})
    return result


assert b'Synthetic Data Studio' in request('/')
assert request('/api/meta')['training_rows'] == 31200
assert request('/api/dataset?intent=check_coverage&limit=2')['total'] == 800
assert len(request('/api/dataset?offset=8&limit=8')['records']) == 8
assert request('/api/dataset?q=not_a_real_phrase_791641')['total'] == 0
assert request('/api/saved?index=12')['live'] is False
scores = request('/api/metrics')
assert scores['stages'][2]['valid'] == 500
assert scores['test']['test_records'] == 3900
request('/api/jobs/not-a-real-job', 404)
request('/api/generate', 400, payload=[])
request('/api/generate', 400, payload={'prompt': ''})
request('/api/generate', 400, payload={'prompt': 'test', 'intent': 'unknown'})
request('/api/generate', 400, payload={'prompt': 'test', 'temperature': 99})
request('/api/generate', 403, payload={'prompt': 'test'}, origin='https://example.com')
request('/missing', 404)
evidence = {'checks': checks, 'passed': len(checks), 'live_generation': 'Verified separately through the browser; these checks do not generate.'}
(ROOT / 'reports/demo_smoke_checks.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(f'{len(checks)} demo HTTP checks passed.')
