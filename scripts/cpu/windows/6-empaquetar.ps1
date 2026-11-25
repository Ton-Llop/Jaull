# Junta todos los resultados en un zip en Descargas.
param([string]$Root = 'C:\cpu-bench', [string]$Out = 'resultats')
$ErrorActionPreference = 'Stop'
$dest = Join-Path $env:USERPROFILE ("Downloads\cpu-{0}-{1}.zip" -f $Out, (Get-Date -Format 'yyyyMMdd-HHmm'))
$tmp = Join-Path $env:TEMP 'cpu-bench-zip'
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Force $tmp | Out-Null
Copy-Item "$Root\$Out" "$tmp\$Out" -Recurse
Copy-Item "$Root\estat-maquina.txt" $tmp -ErrorAction SilentlyContinue
# los logs pueden ser grandes: se dejan, pero se quitan los stdout vacios
Get-ChildItem "$tmp\$Out" -Recurse -Filter stdout.log | Where-Object { $_.Length -eq 0 } | Remove-Item
Compress-Archive -Path "$tmp\*" -DestinationPath $dest -Force
Remove-Item $tmp -Recurse -Force
Write-Host ("Creado {0} ({1:N1} MiB)" -f $dest, ((Get-Item $dest).Length / 1MB)) -ForegroundColor Green
Write-Host "Ya puedes decirme que esta listo."
