"""Rerun the recorded command directly to distinguish runtime from capture issues."""

import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / 'records/bench-491757e3-204b-4956-b89e-61d39ce2f76b.json').open(
    encoding='utf-8'
) as handle:
    benchmark = json.load(handle)['benchmark']
command = benchmark['observation']['command']
print('DIRECT_DIAGNOSTIC_COMMAND', command, flush=True)
started = time.perf_counter()
result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
duration = time.perf_counter() - started
with (root / 'logs/direct-benchmark-control.runtime-log').open('x', encoding='utf-8') as handle:
    json.dump(
        {'command': command, 'exit_code': result.returncode, 'duration_seconds': duration,
         'stdout': result.stdout, 'stderr': result.stderr,
         'methodology': 'diagnostic direct subprocess capture; not a BenchmarkRecord'},
        handle, indent=2,
    )
    handle.write('\n')
print('DIRECT_EXIT_CODE', result.returncode)
print('DIRECT_DURATION', duration)
print('DIRECT_STDOUT', result.stdout)
print('DIRECT_STDERR', result.stderr)
