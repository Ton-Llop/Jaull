"""Bounded direct control of the recorded benchmark, with runtime progress traces."""

import argparse
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-bench', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and greater than zero')
    if not args.llama_bench.is_file():
        parser.error('--llama-bench must name an existing executable')
    if args.output.exists():
        parser.error('--output already exists; refusing to overwrite evidence')
    root = Path(__file__).resolve().parent
    record = root / 'records/bench-491757e3-204b-4956-b89e-61d39ce2f76b.json'
    benchmark = json.loads(record.read_text(encoding='utf-8'))['benchmark']
    command = list(benchmark['observation']['command'])
    command[0] = str(args.llama_bench.resolve())
    command.extend(['--progress', '-v'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print('DIRECT_DIAGNOSTIC_COMMAND', command, flush=True)
    timestamp = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding='utf-8', errors='replace',
            check=False, timeout=args.timeout,
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code, timed_out = None, True
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
    except OSError as exc:
        exit_code, timed_out, stdout, stderr = None, False, '', str(exc)
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(
            {'command': command, 'timestamp': timestamp, 'exit_code': exit_code,
             'duration_seconds': time.perf_counter() - started, 'timed_out': timed_out,
             'timeout_seconds': args.timeout, 'stdout': stdout, 'stderr': stderr,
             'cuda_graphs_disabled': os.environ.get('GGML_CUDA_DISABLE_GRAPHS'),
             'methodology': 'diagnostic direct subprocess capture; not a BenchmarkRecord'},
            handle, indent=2,
        )
        handle.write('\n')
    print('DIRECT_EXIT_CODE', exit_code, 'TIMED_OUT', timed_out, flush=True)
    print('SAVED', args.output, flush=True)


def _text(value: str | bytes | None) -> str:
    return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''


if __name__ == '__main__':
    main()
