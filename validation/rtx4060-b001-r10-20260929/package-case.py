"""Create and validate a native case and bundle without rewriting evidence."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
index = json.loads((ROOT / 'run-index.json').read_text(encoding='utf-8'))
review = []
for path in sorted((ROOT / 'logs').glob('*.runtime-log')):
    raw = path.read_bytes()
    decoded = raw.decode('utf-8')
    suspects = re.findall(
        r'(?i)(?:bearer\s+\S+|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|'
        r'(?:password|access_token|api_key)\s*[=:]\s*[^\s,}]+)', decoded
    )
    if suspects:
        raise RuntimeError(f'Manual sensitive-data review needed: {path.name}')
    review.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'bytes': len(raw), 'credential_pattern_matches': len(suspects)})
if not (ROOT / 'log-review.json').exists():
    with (ROOT / 'log-review.json').open('x', encoding='utf-8') as handle:
        json.dump({'automated_check': 'credential-pattern scan, not a guarantee of absence',
                   'manual_review': 'Fixed validation AI prompt and benign output inspected; benchmark raw output inspected.',
                   'files': review}, handle, indent=2)

os.environ['UV_PROJECT_ENVIRONMENT'] = str(PROJECT / '.venv/rtx4060-py312')
os.environ['UV_NO_SYNC'] = '1'


def run(name: str, arguments: list[str]) -> dict:
    saved = ROOT / f'{name}.stdout.json'
    if saved.exists():
        return json.loads(saved.read_text(encoding='utf-8'))
    command = ['uv', 'run', 'jaull', 'experiments', 'case', *arguments]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', check=False)
    with (ROOT / f'{name}.stdout.json').open('x', encoding='utf-8') as handle:
        handle.write(result.stdout)
    with (ROOT / f'{name}.stderr.log').open('x', encoding='utf-8') as handle:
        handle.write(result.stderr)
    print(name, 'exit', result.returncode, result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f'{name} failed; real output retained')
    return json.loads(result.stdout)


created = run('case-create', ['create', '--experiment', index['experiment_id'],
                            '--benchmark', index['benchmarks'][0]['id'],
                            '--evidence', f"logs/{Path(index['experiment_log_path']).name}:runtime_log",
                            '--evidence', f"logs/{Path(index['benchmarks'][0]['log_path']).name}:runtime_log",
                            '--json'])
print('CREATE_KEYS', list(created), flush=True)
case_id = created['case']['identity']['case_id']
run('case-validate', ['validate', case_id, '--json'])
run('case-export', ['export', case_id, './bundle', '--evidence-root', '.', '--json'])
run('bundle-validate', ['bundle', 'validate', './bundle', '--json'])
print('CASE_ID', case_id, flush=True)
