#!/bin/bash
# Memory-fit run for the first B60 case set (P1 to P6): load one GGUF at one context,
# generate 512 tokens and record llama.cpp's own memory breakdown.
#
# usage: WIN_USER=<windows user> [MIN_FREE_MIB=3072] ./b60-case.sh P1..P6
#
# Run it from WSL2 on a Windows PC. It starts llama-server.exe (native Windows, Vulkan)
# through WSL interop, so the model runs on the Windows driver, not on WSL's.
# Reconstructed from the version run on 2026-11-24; the only change is that the Windows
# user name is now WIN_USER instead of being written in the temp paths.
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
SPID=""
stop(){ [ -n "$SPID" ] && "$TASKKILL" /PID "$SPID" /F >/dev/null 2>&1; SPID=""; rm -f "$TMP_L/b60req-$ID.json"; }
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

t0=$(date +%s); LOADED=0; ABORT=""
while :; do
  f=$(free_ram); echo "$(date +%T),$f" >> "$OUT/ram.csv"
  if [ -n "$f" ] && [ "$f" -lt "$MIN_FREE" ]; then ABORT="low RAM ($f MiB < $MIN_FREE)"; break; fi
  "$TASKLIST" /FI "PID eq $SPID" /NH | grep -qw "$SPID" || { ABORT="the server exited (see server.log)"; break; }
  h=$("$CURL" -s -m 3 http://127.0.0.1:$PORT/health | tr -d '\r')
  case "$h" in *'"ok"'*) LOADED=1; break;; esac
  [ $(( $(date +%s) - t0 )) -gt "$LOAD_TIMEOUT" ] && { ABORT="load timeout"; break; }
  sleep 2
done
log "load: LOADED=$LOADED ABORT=${ABORT:-no}"

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

grep -iE "will leave|cannot meet|free|memory breakdown|breakdown|Vulkan1|projected|KV buffer|compute buffer|model buffer|offloaded|recurrent|error|fail" "$OUT/server.log" > "$OUT/memlines.txt" 2>/dev/null
MINRAM=$(cut -d, -f2 "$OUT/ram.csv" 2>/dev/null | sort -n | head -1)
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
{ echo "case=$ID model=$M ctx=$CTX"; echo "loaded=$LOADED abort=${ABORT:-no}"; echo "generation: $GEN";
  echo "minimum available RAM=$MINRAM MiB (threshold $MIN_FREE)"; echo "extra flags: $LM_FLAG"; echo "folder: $OUT"; } | tee "$OUT/summary.txt"
