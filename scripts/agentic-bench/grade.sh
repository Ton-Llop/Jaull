#!/usr/bin/env bash
# Grade one workspace of the agentic mini-benchmark (docs/TFG/mini-benchmark-agentic.md).
#
# Saves the agent's change as a patch, puts tests/ back to the base version (so a changed test
# can never make a task pass), runs the task's test files and the whole suite, and appends one
# row to results.csv.
#
#   scripts/agentic-bench/grade.sh T1 qwen36
set -uo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 TASK LABEL   (e.g. $0 T1 qwen36)" >&2
  exit 2
fi
task=$1
label=$2
here=$(cd "$(dirname "$0")" && pwd)
bench=${JAULL_BENCH:-$HOME/jaull-bench}
ws=$bench/runs/$task-$label
if [ ! -d "$ws/.git" ]; then
  echo "no workspace at $ws (run prepare.sh first)" >&2
  exit 1
fi
line=$(awk -F'\t' -v t="$task" '$1 == t' "$here/tasks.tsv")
commit=$(cut -f2 <<<"$line")
test_files=$(cut -f3 <<<"$line")
mkdir -p "$bench/results"
cd "$ws" || exit 1

# What the agent changed, including new files, against the base commit.
git add -A
git diff --cached base > "$bench/results/$task-$label.patch"
read -r files_changed insertions deletions < <(
  git diff --cached --shortstat base -- . ':(exclude)tests' |
    awk '{f=$1; i=0; d=0; for (n=1; n<=NF; n++) {if ($n ~ /insertion/) i=$(n-1); if ($n ~ /deletion/) d=$(n-1)} print f+0, i+0, d+0}'
)
files_changed=${files_changed:-0}
insertions=${insertions:-0}
deletions=${deletions:-0}
if [ -n "$(git diff --cached --name-only base -- tests)" ]; then
  tests_touched=yes
else
  tests_touched=no
fi

# Grade against the original tests, whatever the agent did to them.
git reset --quiet
git checkout --quiet base -- tests
git clean --quiet -fd -- tests

# shellcheck disable=SC2086
target_out=$(uv run pytest -p no:cacheprovider $test_files 2>&1)
target_rc=$?
suite_out=$(timeout 2400 uv run pytest -p no:cacheprovider 2>&1)
suite_rc=$?
summary() { tail -1 <<<"$1" | tr -d '=' | sed 's/^ *//; s/ *$//; s/,/;/g'; }
target_ok=$([ $target_rc -eq 0 ] && echo yes || echo no)
suite_ok=$([ $suite_rc -eq 0 ] && echo yes || echo no)

csv=$bench/results.csv
if [ ! -f "$csv" ]; then
  echo "graded_at,task,label,commit,target_ok,target_summary,suite_ok,suite_summary,tests_touched,files_changed,insertions,deletions" > "$csv"
fi
echo "$(date -Is),$task,$label,$commit,$target_ok,$(summary "$target_out"),$suite_ok,$(summary "$suite_out"),$tests_touched,$files_changed,$insertions,$deletions" >> "$csv"
printf '%s\n' "$suite_out" > "$bench/results/$task-$label.suite.log"

echo "task $task ($label)"
echo "  task tests : $target_ok  ($(summary "$target_out"))"
echo "  full suite : $suite_ok  ($(summary "$suite_out"))"
echo "  tests/ touched by the agent: $tests_touched"
echo "  outside tests/: $files_changed files, +$insertions -$deletions"
echo "  patch: $bench/results/$task-$label.patch"
