# Ejecuta la campana. Primero la prueba de humo:   .\4-campanya.ps1 -Smoke
# Despues la campana entera:                       .\4-campanya.ps1
# Solo algunos casos (por ejemplo para reanudar):  .\4-campanya.ps1 -Only C01,C02
param(
  [switch]$Smoke,
  [string]$Only = '',
  [string]$Root = 'C:\cpu-bench',
  [string]$Cases = 'cases.csv',
  [string]$Out = 'resultats',
  [switch]$Warmup,
  [int]$LoadTimeoutS = 400,
  [int]$GenTimeoutS = 300,
  [int]$PauseS = 15
)
$ErrorActionPreference = 'Stop'
$scripts = Join-Path $Root 'scripts'
$caso = Join-Path $scripts '3-caso.ps1'

function Find-Model([string]$name) {
  $hit = Get-ChildItem "$Root\models" -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $hit) { $hit = Get-ChildItem "$env:LOCALAPPDATA\jaull\models" -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1 }
  if ($hit) { return $hit.FullName } else { return $null }
}
function Get-AvailMiB { [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes }

if ($Smoke) {
  $m = Find-Model 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'
  if (-not $m) { throw 'Falta Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (ejecuta 2-preparar.ps1)' }
  & $caso -Id 'S00' -Model $m -Ctx 2048 -Mode 'nommap' -Out $Out -LoadTimeoutS $LoadTimeoutS -GenTimeoutS $GenTimeoutS
  Write-Host "Prueba de humo terminada. Si arriba pone carregat=1 generat=1, todo funciona." -ForegroundColor Green
  return
}

# De menor a mayor memoria predicha: los casos peligrosos van al final.
$orden = @('C14', 'C13', 'C12', 'C09', 'C05', 'C06', 'C10', 'C07', 'C11', 'C01', 'C02', 'C03', 'C08', 'C04',
           'C18', 'C19', 'C20', 'C23', 'C15', 'C16', 'C17', 'C24', 'C21', 'C22')
$cases = @{}
$rows = @(Import-Csv (Join-Path $scripts $Cases))
$rows | ForEach-Object { $cases[$_.id] = $_ }
# con cases.csv se usa el orden de arriba; con otro fichero (p. ej. cases-v2.csv), el orden del propio fichero
if ($Cases -ne 'cases.csv') { $orden = @($rows | ForEach-Object { $_.id }) }
$sel = if ($Only) { $Only.Split(',') | ForEach-Object { $_.Trim() } } else { $orden }

$baseline = Get-AvailMiB
Write-Host "RAM disponible al empezar: $baseline MiB" -ForegroundColor Cyan
$total = 0
foreach ($id in $sel) { $total += $cases[$id].modes.Split(';').Count }
$n = 0
foreach ($id in $sel) {
  $c = $cases[$id]
  $m = Find-Model $c.file
  foreach ($mode in $c.modes.Split(';')) {
    $n++
    if (-not $m) { Write-Host "[$n/$total] ${id} ${mode}: FALTA el fichero $($c.file) (se salta)" -ForegroundColor Yellow; continue }
    Write-Host "[$n/$total] $id $mode" -ForegroundColor Cyan
    try {
      & $caso -Id $id -Model $m -Ctx ([int]$c.ctx) -Mode $mode -Out $Out -Warmup:$Warmup -LoadTimeoutS $LoadTimeoutS -GenTimeoutS $GenTimeoutS
    } catch {
      Write-Host "   ERROR en ${id} ${mode}: $($_.Exception.Message) (se sigue con el siguiente)" -ForegroundColor Red
      Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    Write-Host "   pausa de $PauseS s para que Windows recupere memoria ..."
    Start-Sleep -Seconds $PauseS
  }
}
Write-Host "Campana terminada." -ForegroundColor Green
