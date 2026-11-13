#!/usr/bin/env bash
# Prepare one workspace of the agentic mini-benchmark (docs/TFG/mini-benchmark-agentic.md).
#
# The workspace is Jaull as it was just before the fix, plus the test files the fix added or
# changed, plus TASK.md. It is a fresh git repository whose first commit is tagged `base`,
# so grade.sh can tell exactly what the agent changed.
#
#   scripts/agentic-bench/prepare.sh T1 qwen36
#
# Environment: JAULL_BENCH (default ~/jaull-bench) holds the clone and the runs;
# JAULL_SRC overrides the clone (any Jaull repository with full history).
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 TASK LABEL   (e.g. $0 T1 qwen36)" >&2
  exit 2
fi
task=$1
label=$2
here=$(cd "$(dirname "$0")" && pwd)
bench=${JAULL_BENCH:-$HOME/jaull-bench}
src=${JAULL_SRC:-$bench/jaull-src}

line=$(awk -F'\t' -v t="$task" '$1 == t' "$here/tasks.tsv")
if [ -z "$line" ]; then
  echo "unknown task: $task" >&2
  exit 2
fi
commit=$(cut -f2 <<<"$line")
test_files=$(cut -f3 <<<"$line")
ws=$bench/runs/$task-$label
if [ -e "$ws" ]; then
  echo "$ws already exists: remove it or use another label" >&2
  exit 1
fi

mkdir -p "$bench/runs"
if [ ! -d "$src/.git" ]; then
  git clone --quiet https://github.com/Ton-Llop/Jaull.git "$src"
fi
if ! git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null; then
  git -C "$src" fetch --quiet origin
fi

mkdir -p "$ws"
git -C "$src" archive "$commit^" | tar -x -C "$ws"
# Every file the fix changed under tests/, at its fixed version: the new tests and their fixtures.
git -C "$src" diff --name-only "$commit^" "$commit" -- tests | while read -r f; do
  mkdir -p "$ws/$(dirname "$f")"
  git -C "$src" show "$commit:$f" > "$ws/$f"
done
cp "$here/tasks/$task.md" "$ws/TASK.md"

cd "$ws"
git init --quiet
git add -A
git -c user.name=bench -c user.email=bench@localhost commit --quiet \
  -m "base: $task (parent of $commit, with the tests of $commit)"
git tag base
uv sync --frozen --quiet
printf '%s\t%s\t%s\t%s\n' "$task" "$commit" "$label" "$(date -Is)" > "$bench/runs/$task-$label.meta"

echo "Workspace: $ws"
echo "Target tests before the agent (they should fail):"
# shellcheck disable=SC2086
uv run pytest -p no:cacheprovider $test_files 2>&1 | tail -1 || true
echo
echo "Next: cd $ws && opencode"
