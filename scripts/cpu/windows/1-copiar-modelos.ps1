# Copia los GGUF desde WSL a C:\cpu-bench\models. Ejecutar ANTES de "wsl --shutdown" (WSL debe estar encendido).
$ErrorActionPreference = 'Stop'
$Root = 'C:\cpu-bench'
$Src = '\\wsl.localhost\Ubuntu-24.04\home\antoni\.local\share\jaull\models'
New-Item -ItemType Directory -Force "$Root\models" | Out-Null

$files = @(
  'Qwen\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q3_k_m.gguf',
  'Qwen\Qwen2.5-Coder-7B-Instruct-GGUF\qwen2.5-coder-7b-instruct-q3_k_m.gguf',
  'bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-IQ2_M.gguf',
  'Qwen\Qwen2.5-3B-Instruct-GGUF\qwen2.5-3b-instruct-q4_k_m.gguf',
  'Qwen\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q5_k_m.gguf',
  'ThomasBaruzier\Qwen2.5-0.5B-Instruct-GGUF\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'
)
foreach ($f in $files) {
  $from = Join-Path $Src $f
  $to = Join-Path "$Root\models" (Split-Path $f -Leaf)
  if (-not (Test-Path $from)) { Write-Host "NO EXISTE: $from" -ForegroundColor Red; continue }
  if ((Test-Path $to) -and ((Get-Item $to).Length -eq (Get-Item $from).Length)) { Write-Host "ya copiado: $(Split-Path $f -Leaf)"; continue }
  Write-Host "copiando $(Split-Path $f -Leaf) ..."
  Copy-Item $from $to -Force
}
Write-Host ""
Get-ChildItem "$Root\models" -Filter *.gguf | ForEach-Object { "{0,7:N2} GiB  {1}" -f ($_.Length/1GB), $_.Name }
Write-Host "Listo. Siguiente: 2-preparar.ps1" -ForegroundColor Green
