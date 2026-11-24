#!/bin/bash
# Fill the context step by step on one B60 case and record prompt speed, generation speed and RAM.
# Question it answers: does the model still generate when the context is really full, and how
# fast? The KV cache is reserved at load, so no new VRAM is expected; this checks it.
#
# usage: WIN_USER=<windows user> [STEPS="4000 16000 32000 45000"] ./b60-fill.sh P1..P6
#
# It stops the ramp when a step fails or one prompt takes longer than SLOW_MS (240000 ms).
# Same safety as b60-case.sh: refuses to start if another llama-server.exe is running, stops its own
# server below MIN_FREE_MIB of Windows RAM, and kills only the process it launched.
set -u
cd /mnt/c
CODER="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
Q32="Qwen2.5-32B-Instruct-Q4_K_M.gguf"
case "${1:-}" in
  P1) M=$CODER; CTX=49152;; P2) M=$CODER; CTX=51200;; P3) M=$CODER; CTX=53248;;
  P4) M=$Q32; CTX=12288;;   P5) M=$Q32; CTX=14336;;   P6) M=$Q32; CTX=15360;;
  *) echo "usage: WIN_USER=<windows user> $0 P1..P6"; exit 2;;
esac
: "${WIN_USER:?set WIN_USER to your Windows user name (the folder under C:\\Users)}"
ID=fill-$1
STEPS=${STEPS:-"4000 16000 32000 45000"}
for s in $STEPS; do
  [ $((s + 300)) -le "$CTX" ] || { echo "step $s does not leave room to generate inside -c $CTX"; exit 2; }
done
LLAMA=/mnt/c/llama-win/llama-server.exe
MODELS_W='C:\llama-win'
MODELS_L=/mnt/c/llama-win
SYS=/mnt/c/Windows/System32
PS=$SYS/WindowsPowerShell/v1.0/powershell.exe
CURL=$SYS/curl.exe; TASKLIST=$SYS/tasklist.exe; TASKKILL=$SYS/taskkill.exe
TMP_L=/mnt/c/Users/$WIN_USER/AppData/Local/Temp
TMP_W="C:\\Users\\$WIN_USER\\AppData\\Local\\Temp"
PORT=8099
MIN_FREE=${MIN_FREE_MIB:-3072}; START_MIN=${START_MIN_FREE_MIB:-6144}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-900}; STEP_TIMEOUT=${STEP_TIMEOUT:-900}
SLOW_MS=${SLOW_MS:-240000}
LM_FLAG=${LM_FLAG--lm dio}
OUT=$HOME/b60-results/$ID-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
log(){ echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
free_ram(){ "$PS" -NoProfile -Command "[int]((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes)" 2>/dev/null | tr -d '\r'; }
win_pid(){ "$PS" -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { \$_.CommandLine -like '*--port $PORT*' }).ProcessId" 2>/dev/null | tr -d '\r'; }
alive(){ "$TASKLIST" /FI "PID eq $1" /NH 2>/dev/null | grep -qw "$1"; }
SPID=""
stop(){
  [ -n "$SPID" ] || return 0
  local p=$SPID tries=0
  while [ $tries -lt 5 ]; do
    "$TASKKILL" /PID "$p" /F >/dev/null 2>&1
    sleep 1
    alive "$p" || { SPID=""; rm -f "$TMP_L/fill-"*.json; return 0; }
    tries=$((tries+1))
  done
  echo "!!! COULD NOT STOP the server PID=$p: run  /mnt/c/Windows/System32/taskkill.exe /PID $p /F" | tee -a "$OUT/run.log"
}
trap 'log "interrupted"; stop; exit 130' INT TERM HUP
trap stop EXIT

[ -f "$LLAMA" ] || { log "MISSING $LLAMA"; exit 1; }
[ -f "$MODELS_L/$M" ] || { log "MISSING model $MODELS_L/$M"; exit 1; }
n=$("$TASKLIST" /FI "IMAGENAME eq llama-server.exe" /NH | grep -ci "llama-server")
[ "$n" -eq 0 ] || { log "$n llama-server.exe already running: not starting"; exit 1; }
f=$(free_ram)
[ -n "$f" ] || { log "cannot read Windows RAM: not running blind"; exit 1; }
[ "$f" -ge "$START_MIN" ] || { log "available RAM $f MiB < $START_MIN: not starting"; exit 1; }
log "fill test $1: $M ctx=$CTX; steps: $STEPS; RAM available at start=$f MiB; abort threshold=$MIN_FREE MiB"

cat > "$OUT/helper.py" <<'PY'
import json
import sys

BASE = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. ")
cmd = sys.argv[1]
if cmd == "tok":  # tok OUT: the tokenize request for one repetition of BASE
    json.dump({"content": BASE}, open(sys.argv[2], "w"))
elif cmd == "count":  # count RESPONSE: tokens the server returned for BASE
    print(len(json.load(open(sys.argv[2]))["tokens"]))
elif cmd == "req":  # req OUT TOKENS_PER_REPEAT TARGET
    per, target = int(sys.argv[3]), int(sys.argv[4])
    prompt = BASE * max(1, target // per) + "\n\nIn one sentence, what is the text above made of?"
    json.dump({"prompt": prompt, "n_predict": 128, "temperature": 0, "ignore_eos": True,
               "cache_prompt": False}, open(sys.argv[2], "w"))
elif cmd == "parse":  # parse RESPONSE: one line with the numbers that matter
    try:
        d = json.load(open(sys.argv[2]))
        t = d.get("timings", {})
        print(f"prompt_tokens={d.get('tokens_evaluated')} prompt_ms={t.get('prompt_ms')} "
              f"prompt_tok_s={t.get('prompt_per_second')} gen_tokens={d.get('tokens_predicted')} "
              f"gen_tok_s={t.get('predicted_per_second')} stop={d.get('stop_type')}")
    except Exception as exc:
        print(f"no valid response ({exc})")
PY

# shellcheck disable=SC2086
"$LLAMA" -m "$MODELS_W\\$M" -dev "${DEV:-Vulkan1}" -ngl 99 -c "$CTX" -np 1 --jinja -lv 4 --host 127.0.0.1 --port $PORT $LM_FLAG > "$OUT/server.log" 2>&1 &
for i in $(seq 1 30); do SPID=$(win_pid); [[ "$SPID" =~ ^[0-9]+$ ]] && break; SPID=""; sleep 1; done
[ -n "$SPID" ] || { log "cannot find the launched process (see server.log)"; exit 1; }
log "Windows server PID=$SPID"

t0=$(date +%s); LOADED=0; ABORT=""
while :; do
  f=$(free_ram); echo "$(date +%T),$f" >> "$OUT/ram.csv"
  if [ -n "$f" ] && [ "$f" -lt "$MIN_FREE" ]; then ABORT="low RAM ($f MiB < $MIN_FREE)"; break; fi
  alive "$SPID" || { ABORT="the server exited (see server.log)"; break; }
  h=$("$CURL" -s -m 3 http://127.0.0.1:$PORT/health | tr -d '\r')
  case "$h" in *'"ok"'*) LOADED=1; break;; esac
  [ $(( $(date +%s) - t0 )) -gt "$LOAD_TIMEOUT" ] && { ABORT="load timeout"; break; }
  sleep 2
done
log "load: LOADED=$LOADED ABORT=${ABORT:-no}"
grep -m1 "projected to use" "$OUT/server.log" | tee -a "$OUT/run.log" >/dev/null

RESULTS="$OUT/steps.txt"; : > "$RESULTS"
if [ "$LOADED" -eq 1 ]; then
  python3 "$OUT/helper.py" tok "$TMP_L/fill-tok.json"
  "$CURL" -s -m 60 -X POST -H "Content-Type: application/json" --data-binary "@$TMP_W\\fill-tok.json" \
    "http://127.0.0.1:$PORT/tokenize" > "$OUT/tok.json"
  PER=$(python3 "$OUT/helper.py" count "$OUT/tok.json" 2>/dev/null)
  [[ "$PER" =~ ^[0-9]+$ ]] && [ "$PER" -gt 0 ] || { ABORT="cannot count tokens (see tok.json)"; LOADED=0; }
fi
if [ "$LOADED" -eq 1 ]; then
  log "one repetition of the filler text is $PER tokens"
  for target in $STEPS; do
    python3 "$OUT/helper.py" req "$TMP_L/fill-req.json" "$PER" "$target"
    log "step: asking for a prompt of about $target tokens"
    "$CURL" -s -m "$STEP_TIMEOUT" -X POST -H "Content-Type: application/json" \
      --data-binary "@$TMP_W\\fill-req.json" "http://127.0.0.1:$PORT/completion" > "$OUT/step_$target.json" &
    CP=$!
    while kill -0 $CP 2>/dev/null; do
      f=$(free_ram); echo "$(date +%T),$f" >> "$OUT/ram.csv"
      if [ -n "$f" ] && [ "$f" -lt "$MIN_FREE" ]; then
        ABORT="low RAM during step $target ($f MiB)"; kill $CP 2>/dev/null; break
      fi
      alive "$SPID" || { ABORT="the server died during step $target"; kill $CP 2>/dev/null; break; }
      sleep 3
    done
    wait $CP 2>/dev/null
    line=$(python3 "$OUT/helper.py" parse "$OUT/step_$target.json")
    echo "target=$target $line" | tee -a "$RESULTS" | tee -a "$OUT/run.log"
    [ -n "$ABORT" ] && break
    ms=$(echo "$line" | sed -E 's/.*prompt_ms=([0-9.]+).*/\1/')
    case "$ms" in *[!0-9.]*|"") ABORT="step $target returned no timing"; break;; esac
    # A slow step stops the ramp, but only if another step is still to come: after the last one it is
    # just a result, not an abort.
    if [ "${ms%.*}" -gt "$SLOW_MS" ] && [ "$target" != "${STEPS##* }" ]; then
      ABORT="step $target took ${ms%.*} ms, longer than $SLOW_MS: stopping the ramp"; break
    fi
  done
fi
stop

MINRAM=$(cut -d, -f2 "$OUT/ram.csv" 2>/dev/null | sort -n | head -1)
{ echo "case=$1 model=$M ctx=$CTX steps=$STEPS"; echo "loaded=$LOADED abort=${ABORT:-no}";
  echo "--- steps"; cat "$RESULTS";
  echo "minimum available RAM=${MINRAM:-?} MiB (threshold $MIN_FREE)"; echo "folder: $OUT"; } | tee "$OUT/summary.txt"
