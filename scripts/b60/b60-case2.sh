#!/bin/bash
# Memory-fit run for the second B60 case set (Q1 to Q8): large contexts.
# If llama.cpp projects that the model does NOT fit in VRAM, the server is stopped at
# llama.cpp's fit step, before the KV cache is reserved, so nothing is loaded.
#
# usage: WIN_USER=<windows user> [MIN_FREE_MIB=3072] [MIN_LEFT_MIB=0] ./b60-case2.sh Q1..Q8
#
# MIN_LEFT_MIB also stops a run that would fit but leave fewer than that many MiB free
# (it was 300 for Q7 and 500 for Q8 on 2026-11-24, to avoid loading with almost no margin).
# Reconstructed from the version run on 2026-11-24; the only change is that the Windows
# user name is now WIN_USER instead of being written in the temp paths.
set -u
cd /mnt/c
CODER="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
Q32="Qwen2.5-32B-Instruct-Q4_K_M.gguf"
Q27="Qwen3.8-27B-UD-Q4_K_M.gguf"
case "${1:-}" in
  Q1) M=$CODER; CTX=8192;;  Q2) M=$CODER; CTX=98304;; Q3) M=$CODER; CTX=131072;;
  Q4) M=$Q32; CTX=32768;;   Q5) M=$Q32; CTX=65536;;
  Q6) M=$Q27; CTX=98304;;   Q7) M=$Q27; CTX=114688;;  Q8) M=$Q27; CTX=131072;;
  *) echo "usage: WIN_USER=<windows user> $0 Q1..Q8"; exit 2;;
esac
: "${WIN_USER:?set WIN_USER to your Windows user name (the folder under C:\\Users)}"
ID=$1
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
LOAD_TIMEOUT=${LOAD_TIMEOUT:-900}
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
    alive "$p" || { SPID=""; rm -f "$TMP_L/b60req-$ID.json"; return 0; }
    tries=$((tries+1))
  done
  echo "!!! COULD NOT STOP the server PID=$p: run  /mnt/c/Windows/System32/taskkill.exe /PID $p /F" | tee -a "$OUT/run.log"
}
trap 'log "interrupted"; stop; exit 130' INT TERM HUP
trap stop EXIT

[ -f "$LLAMA" ] || { log "MISSING $LLAMA"; exit 1; }
[ -f "$MODELS_L/$M" ] || { log "MISSING model $MODELS_L/$M"; exit 1; }
n=$("$TASKLIST" /FI "IMAGENAME eq llama-server.exe" /NH | grep -ci "llama-server")
[ "$n" -eq 0 ] || { log "$n llama-server.exe already running: not starting (the measurement would not be valid and I will not touch someone else's server)"; exit 1; }
f=$(free_ram)
[ -n "$f" ] || { log "cannot read Windows RAM: not running blind"; exit 1; }
[ "$f" -ge "$START_MIN" ] || { log "available RAM $f MiB < $START_MIN: not starting"; exit 1; }
log "case $ID: $M ctx=$CTX; RAM available at start=$f MiB; abort threshold=$MIN_FREE MiB; extra flags: $LM_FLAG"

# shellcheck disable=SC2086
"$LLAMA" -m "$MODELS_W\\$M" -dev "${DEV:-Vulkan1}" -ngl 99 -c "$CTX" -np 1 --jinja -lv 4 --host 127.0.0.1 --port $PORT $LM_FLAG > "$OUT/server.log" 2>&1 &
for i in $(seq 1 30); do SPID=$(win_pid); [[ "$SPID" =~ ^[0-9]+$ ]] && break; SPID=""; sleep 1; done
[ -n "$SPID" ] || { log "cannot find the launched process (see server.log)"; exit 1; }
log "Windows server PID=$SPID"

# Phase 1: wait for llama.cpp's fit line (printed before the KV cache is reserved)
ABORT=""; STOPPED=""; FIT_P=""; FIT_F=""; k=0; t0=$(date +%s)
while :; do
  FITLINE=$(grep -m1 "common_params_fit_impl: projected to use" "$OUT/server.log" 2>/dev/null)
  [ -n "$FITLINE" ] && break
  k=$((k+1))
  if [ $((k % 10)) -eq 0 ]; then alive "$SPID" || { ABORT="the server exited before the fit step (see server.log)"; break; }; fi
  [ $(( $(date +%s) - t0 )) -gt 90 ] && { ABORT="no fit line within 90 s"; break; }
  sleep 0.3
done
if [ -n "${FITLINE:-}" ]; then
  FIT_P=$(echo "$FITLINE" | sed -E 's/.*projected to use ([0-9]+) MiB.*/\1/')
  FIT_F=$(echo "$FITLINE" | sed -E 's/.*vs\. ([0-9]+) MiB.*/\1/')
  log "llama.cpp fit: projected $FIT_P MiB, free $FIT_F MiB, would leave $((FIT_F-FIT_P)) MiB"
  if [ $((FIT_F-FIT_P)) -lt "${MIN_LEFT_MIB:-0}" ]; then
    STOPPED="stopped at the fit step: would leave less than ${MIN_LEFT_MIB:-0} MiB free; KV cache not reserved"
    log "$STOPPED"
    stop
  fi
fi

LOADED=0
if [ -z "$ABORT" ] && [ -z "$STOPPED" ]; then
  t0=$(date +%s)
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
fi

if [ "$LOADED" -eq 1 ]; then
  cat > "$TMP_L/b60req-$ID.json" <<J
{"prompt":"Explain in detail how a transformer language model generates text, step by step.","n_predict":512,"temperature":0,"ignore_eos":true,"cache_prompt":false}
J
  "$CURL" -s -m 900 -X POST -H "Content-Type: application/json" --data-binary "@$TMP_W\\b60req-$ID.json" "http://127.0.0.1:$PORT/completion" > "$OUT/response.json" &
  CP=$!
  while kill -0 $CP 2>/dev/null; do
    f=$(free_ram); echo "$(date +%T),$f" >> "$OUT/ram.csv"
    if [ -n "$f" ] && [ "$f" -lt "$MIN_FREE" ]; then ABORT="low RAM during generation ($f MiB)"; kill $CP 2>/dev/null; break; fi
    sleep 3
  done
  wait $CP 2>/dev/null
fi
stop

grep -iE "will leave|cannot meet|projected|fit|breakdown|Vulkan1|KV buffer|compute buffer|model buffer|offloaded|error|fail" "$OUT/server.log" > "$OUT/memlines.txt" 2>/dev/null
MINRAM=$(cut -d, -f2 "$OUT/ram.csv" 2>/dev/null | sort -n | head -1)
if [ "$LOADED" -eq 1 ]; then
GEN=$(python3 - "$OUT/response.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    t=d.get("timings",{})
    print(f"tokens={d.get('tokens_predicted')} tok/s={t.get('predicted_per_second')} stop={d.get('stop_type')}")
except Exception as e:
    print(f"no valid response ({e})")
PY
)
else
  GEN="not generated"
fi
{ echo "case=$ID model=$M ctx=$CTX";
  echo "fit: projected=${FIT_P:-?} MiB free=${FIT_F:-?} MiB";
  echo "stopped at fit=${STOPPED:-no} loaded=$LOADED abort=${ABORT:-no}";
  echo "generation: $GEN";
  echo "minimum available RAM=${MINRAM:-?} MiB (threshold $MIN_FREE)"; echo "folder: $OUT"; } | tee "$OUT/summary.txt"
