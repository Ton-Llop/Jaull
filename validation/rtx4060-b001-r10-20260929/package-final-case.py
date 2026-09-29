"""Preserve both real benchmark attempts in a new case and bundle."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['UV_PROJECT_ENVIRONMENT'] = str(ROOT.parent.parent / '.venv/rtx4060-py312')
os.environ['UV_NO_SYNC'] = '1'
first = json.loads((ROOT / 'run-index.json').read_text('utf-8'))
control = json.loads((ROOT / 'control-benchmark-index.json').read_text('utf-8'))
benchmarks = [*first['benchmarks'], *control]
logs = [Path(first['experiment_log_path']), *[Path(b['log_path']) for b in benchmarks]]
with (ROOT / 'log-review-final.json').open('x', encoding='utf-8') as handle:
    json.dump({'review': 'Original experiment and both benchmark raw logs manually inspected.',
               'files': [{'path': str(p.relative_to(ROOT)),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                          'bytes': p.stat().st_size} for p in logs]}, handle, indent=2)


def run(name: str, args: list[str]) -> dict:
    command = ['uv', 'run', 'jaull', 'experiments', 'case', *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', check=False)
    with (ROOT / f'final-{name}.stdout.json').open('x', encoding='utf-8') as handle:
        handle.write(result.stdout)
    with (ROOT / f'final-{name}.stderr.log').open('x', encoding='utf-8') as handle:
        handle.write(result.stderr)
    print(name, 'EXIT', result.returncode, result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError(f'{name} failed; output retained')
    return json.loads(result.stdout)


arguments = ['create', '--experiment', first['experiment_id']]
for benchmark in benchmarks:
    arguments.extend(['--benchmark', benchmark['id']])
for log in logs:
    arguments.extend(['--evidence', f'logs/{log.name}:runtime_log'])
arguments.append('--json')
created = run('case-create', arguments)
case_id = created['case']['identity']['case_id']
run('case-validate', ['validate', case_id, '--json'])
run('case-export', ['export', case_id, './bundle-final', '--evidence-root', '.', '--json'])
run('bundle-validate', ['bundle', 'validate', './bundle-final', '--json'])
print('FINAL_CASE_ID', case_id, flush=True)
