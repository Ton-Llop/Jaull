# Velocidad con llama-bench: 5 modelos, pp512 y tg128, 6 hilos, 3 repeticiones. Medida descriptiva.
param([string]$Root = 'C:\cpu-bench')
$ErrorActionPreference = 'Stop'
$bench = Get-ChildItem "$Root\llama" -Recurse -Filter llama-bench.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $bench) { throw 'No encuentro llama-bench.exe' }
if (Get-Process llama-server -ErrorAction SilentlyContinue) { throw 'Hay un llama-server en marcha. Cierralo antes.' }

function Find-Model([string]$name) {
  $hit = Get-ChildItem "$Root\models" -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $hit) { $hit = Get-ChildItem "$env:LOCALAPPDATA\jaull\models" -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1 }
  if ($hit) { return $hit.FullName } else { return $null }
}
$models = @(
  @('C14', 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'),
  @('C12', 'qwen2.5-3b-instruct-q4_k_m.gguf'),
  @('C05', 'qwen2.5-7b-instruct-q3_k_m.gguf'),
  @('C01', 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'),
  @('C15', 'Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf'),
  @('C23', 'Ornith-1.5-9B-Q6_K.gguf'),
  @('C20', 'qwen2.5-coder-14b-instruct-q3_k_m.gguf')
)
$out = Join-Path $Root 'resultats\benchmark'
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($e in $models) {
  $m = Find-Model $e[1]
  if (-not $m) { Write-Host "FALTA $($e[1]) (se salta)" -ForegroundColor Yellow; continue }
  Write-Host "llama-bench $($e[0]) $($e[1]) ..." -ForegroundColor Cyan
  & $bench -m $m -t 6 -p 512 -n 128 -r 3 -o jsonl 2> "$out\$($e[0])-stderr.txt" | Set-Content "$out\$($e[0]).jsonl" -Encoding Ascii
  Start-Sleep -Seconds 20
}
Write-Host "Listo. Siguiente: 6-empaquetar.ps1" -ForegroundColor Green
