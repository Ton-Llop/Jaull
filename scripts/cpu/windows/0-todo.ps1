# TODO EN UNO: prepara, prueba de humo, campana, velocidad y zip. Lanzar y olvidarse.
#   powershell -ExecutionPolicy Bypass -File C:\cpu-bench\scripts\0-todo.ps1
# Si se corta, se puede volver a lanzar: no vuelve a descargar lo que ya esta.
#
# Version corta (solo los casos cerca del limite, sin velocidad; ~1 h):
#   powershell -ExecutionPolicy Bypass -File C:\cpu-bench\scripts\0-todo.ps1 -Only C18,C19,C20,C23,C15,C16,C17,C24,C21,C22 -SinBenchmark
param([string]$Root = 'C:\cpu-bench', [string]$Only = '', [switch]$SinBenchmark)
$ErrorActionPreference = 'Stop'
$scripts = Join-Path $Root 'scripts'
New-Item -ItemType Directory -Force "$Root\resultats" | Out-Null
Start-Transcript -Path (Join-Path $Root ("resultats\transcript-{0}.txt" -f (Get-Date -Format 'yyyyMMdd-HHmm'))) -Append | Out-Null

# que el portatil no se duerma mientras corre
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win.Power]::SetThreadExecutionState([uint32]2147483651) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED

try {
  if (Get-Process vmmemWSL -ErrorAction SilentlyContinue) { throw 'WSL sigue encendido y ocupa RAM. Ejecuta "wsl --shutdown", espera 10 s y vuelve a lanzar.' }
  if (Get-Process llama-server -ErrorAction SilentlyContinue) { throw 'Hay un llama-server en marcha. Cierralo y vuelve a lanzar.' }
  $avail = [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes
  Write-Host "RAM disponible al empezar: $avail MiB" -ForegroundColor Cyan
  if ($avail -lt 7000) { Write-Host 'AVISO: hay menos de 7000 MiB disponibles; cierra aplicaciones si puedes (la campana sigue igualmente).' -ForegroundColor Yellow }

  Write-Host '=== 1/5 Preparar y descargar ===' -ForegroundColor Green
  & (Join-Path $scripts '2-preparar.ps1')

  Write-Host '=== 2/5 Prueba de humo ===' -ForegroundColor Green
  & (Join-Path $scripts '4-campanya.ps1') -Smoke
  $last = Import-Csv (Join-Path $Root 'resultats\resum.csv') | Select-Object -Last 1
  if ($last.id -ne 'S00' -or $last.carregat -ne '1' -or $last.generat -ne '1') { throw 'La prueba de humo ha fallado: no sigo. Mira C:\cpu-bench\resultats\S00-*\server.log' }

  Write-Host '=== 3/5 Campana ===' -ForegroundColor Green
  & (Join-Path $scripts '4-campanya.ps1') -Only $Only

  if (-not $SinBenchmark) {
    Write-Host '=== 4/5 Velocidad ===' -ForegroundColor Green
    & (Join-Path $scripts '5-benchmark.ps1')
  }

  Write-Host '=== 5/5 Empaquetar ===' -ForegroundColor Green
  & (Join-Path $scripts '6-empaquetar.ps1')

  Write-Host 'TERMINADO. Ya puedes decirme "listo".' -ForegroundColor Green
  try { [console]::Beep(880, 400); [console]::Beep(1175, 600) } catch { }
} finally {
  [Win.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null   # ES_CONTINUOUS: vuelve al comportamiento normal
  Stop-Transcript | Out-Null
}
