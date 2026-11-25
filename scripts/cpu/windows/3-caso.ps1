# Ejecuta UN caso: arranca llama-server (solo CPU), espera la carga, genera 128 tokens y lo guarda todo.
# Lo llama 4-campanya.ps1; tambien se puede ejecutar a mano.
param(
  [Parameter(Mandatory = $true)][string]$Id,
  [Parameter(Mandatory = $true)][string]$Model,
  [Parameter(Mandatory = $true)][int]$Ctx,
  [string]$Mode = 'nommap',
  [int]$Tokens = 128,
  [int]$MinFreeMiB = 400,
  [int]$MaxPageGrowMiB = 3000,
  [int]$LoadTimeoutS = 900,
  [int]$GenTimeoutS = 900,
  [string]$Root = 'C:\cpu-bench',
  [int]$Port = 8099
)
$ErrorActionPreference = 'Stop'
$ci = [System.Globalization.CultureInfo]::InvariantCulture

$server = Get-ChildItem "$Root\llama" -Recurse -Filter llama-server.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $server) { throw 'No encuentro llama-server.exe en C:\cpu-bench\llama (ejecuta 2-preparar.ps1)' }
if (Get-Process llama-server -ErrorAction SilentlyContinue) { throw 'Ya hay un llama-server en marcha: no empiezo (la medida no seria valida).' }
if (-not (Test-Path $Model)) { throw "No existe el modelo: $Model" }

function Get-AvailMiB { [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes }
function Get-PageMiB { [int](Get-CimInstance Win32_PageFileUsage | Measure-Object -Property CurrentUsage -Sum).Sum }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$dir = Join-Path $Root "resultats\$Id-$Mode-$stamp"
New-Item -ItemType Directory -Force $dir | Out-Null
$ram = Join-Path $dir 'ram.csv'
'seg,disponible_mib,pagefile_mib,servidor_ws_mib,servidor_privat_mib' | Set-Content $ram -Encoding Ascii

$avail0 = Get-AvailMiB
$page0 = Get-PageMiB
Write-Host "[$Id $Mode ctx=$Ctx] disponible antes: $avail0 MiB"

$script:minAvail = $avail0
$script:maxPage = $page0
$script:maxWS = 0
$script:abort = ''
$proc = $null
$curl = $null
$t0 = Get-Date

function Sample {
  $a = Get-AvailMiB
  $pg = Get-PageMiB
  $ws = 0; $pv = 0
  if ($proc -and -not $proc.HasExited) {
    try {
      $p = Get-Process -Id $proc.Id -ErrorAction Stop
      $ws = [int]($p.WorkingSet64 / 1MB); $pv = [int]($p.PrivateMemorySize64 / 1MB)
    } catch { }
  }
  $seg = [int]((Get-Date) - $t0).TotalSeconds
  Add-Content $ram "$seg,$a,$pg,$ws,$pv" -Encoding Ascii
  if ($a -lt $script:minAvail) { $script:minAvail = $a }
  if ($pg -gt $script:maxPage) { $script:maxPage = $pg }
  if ($ws -gt $script:maxWS) { $script:maxWS = $ws }
  if ($a -lt $MinFreeMiB) { $script:abort = "disponible $a MiB < $MinFreeMiB MiB" }
  elseif (($pg - $page0) -gt $MaxPageGrowMiB) { $script:abort = "pagefile +$($pg - $page0) MiB" }
}

$loaded = 0; $generated = 0; $availLoaded = 0; $exitCode = ''
$tokOut = ''; $ppOut = ''; $tgOut = ''
try {
  $srvArgs = @('-m', ('"' + $Model + '"'), '-c', "$Ctx", '-np', '1', '-t', '6', '-lv', '4', '--host', '127.0.0.1', '--port', "$Port")
  if ($Mode -eq 'nommap') { $srvArgs += '--no-mmap' }
  $proc = Start-Process -FilePath $server -ArgumentList $srvArgs -RedirectStandardOutput "$dir\stdout.log" -RedirectStandardError "$dir\server.log" -PassThru -WindowStyle Hidden
  $t0 = Get-Date

  # 1) esperar la carga
  while (((Get-Date) - $t0).TotalSeconds -lt $LoadTimeoutS) {
    Sample
    if ($script:abort) { break }
    if ($proc.HasExited) { break }
    try {
      $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
      if ($r.StatusCode -eq 200) { $loaded = 1; break }
    } catch { }
    Start-Sleep -Seconds 1
  }
  if ($loaded -eq 0 -and -not $script:abort -and -not $proc.HasExited) { $script:abort = "tiempo de carga superado ($LoadTimeoutS s)" }
  $availLoaded = Get-AvailMiB

  # 2) generar
  if ($loaded -eq 1 -and -not $script:abort) {
    $body = '{"prompt":"Explain in detail how a transformer language model generates text, step by step.","n_predict":' + $Tokens + ',"temperature":0,"ignore_eos":true,"cache_prompt":false}'
    Set-Content -Path "$dir\request.json" -Value $body -Encoding Ascii
    $curlArgs = @('-s', '-m', "$GenTimeoutS", '-X', 'POST', '-H', '"Content-Type: application/json"', '--data-binary', "@$dir\request.json", "http://127.0.0.1:$Port/completion")
    $curl = Start-Process -FilePath 'curl.exe' -ArgumentList $curlArgs -RedirectStandardOutput "$dir\response.json" -PassThru -WindowStyle Hidden
    while (-not $curl.HasExited) {
      Sample
      if ($script:abort) { break }
      Start-Sleep -Seconds 1
    }
    Sample
  }
} finally {
  if ($curl -and -not $curl.HasExited) { Stop-Process -Id $curl.Id -Force -ErrorAction SilentlyContinue }
  if ($proc) {
    if ($proc.HasExited) { $exitCode = "$($proc.ExitCode)" }
    else { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; $proc.WaitForExit(20000) | Out-Null }
  }
}
Start-Sleep -Seconds 3
$availEnd = Get-AvailMiB

# 3) resultado de la generacion
if (Test-Path "$dir\response.json") {
  $raw = Get-Content "$dir\response.json" -Raw -ErrorAction SilentlyContinue
  if ($raw) {
    try {
      $j = $raw | ConvertFrom-Json
      if ($j.tokens_predicted -ge 1) { $generated = 1 }
      $tokOut = "$($j.tokens_predicted)"
      $tgOut = [string]::Format($ci, '{0:N2}', [double]$j.timings.predicted_per_second)
      $ppOut = [string]::Format($ci, '{0:N2}', [double]$j.timings.prompt_per_second)
    } catch { }
  }
}

if ($loaded -eq 0 -and -not $script:abort) { $script:abort = "el servidor ha acabado sin cargar (exit $exitCode)" }
if ($loaded -eq 1 -and $generated -eq 0 -and -not $script:abort) { $script:abort = 'sin respuesta de generacion (timeout o caida)' }

# 4) buffers de memoria que imprime llama.cpp
$bm = 0.0; $bk = 0.0; $bc = 0.0; $br = 0.0
if (Test-Path "$dir\server.log") {
  $lines = Get-Content "$dir\server.log"
  $lines | Select-String -Pattern 'buffer size|KV buffer|compute buffer|model buffer|n_ctx|offloaded|error|failed|out of memory' | ForEach-Object { $_.Line } | Set-Content "$dir\memlines.txt" -Encoding Ascii
  foreach ($l in $lines) {
    $m = [regex]::Match($l, '(model|KV|compute|RS) buffer size\s*=\s*([0-9\.]+)\s*MiB')
    if ($m.Success) {
      $v = [double]::Parse($m.Groups[2].Value, $ci)
      switch ($m.Groups[1].Value) { 'model' { $bm += $v } 'KV' { $bk += $v } 'compute' { $bc += $v } 'RS' { $br += $v } }
    }
  }
}
$bt = $bm + $bk + $bc + $br
function F($x) { [string]::Format($ci, '{0:N1}', $x) }

$summary = @(
  "id=$Id", "mode=$Mode", "ctx=$Ctx", "model=$(Split-Path $Model -Leaf)",
  "disponible_abans_mib=$avail0", "disponible_carregat_mib=$availLoaded", "disponible_min_mib=$($script:minAvail)", "disponible_final_mib=$availEnd",
  "pagefile_abans_mib=$page0", "pagefile_max_mib=$($script:maxPage)", "servidor_ws_max_mib=$($script:maxWS)",
  "carregat=$loaded", "generat=$generated", "tokens=$tokOut", "tok_s_generacio=$tgOut", "tok_s_prompt=$ppOut",
  "buffer_model_mib=$(F $bm)", "buffer_kv_mib=$(F $bk)", "buffer_compute_mib=$(F $bc)", "buffer_rs_mib=$(F $br)", "buffer_total_mib=$(F $bt)",
  "aturat=$($script:abort)", "exit_code=$exitCode", "carpeta=$dir"
)
$summary | Set-Content "$dir\summary.txt" -Encoding Ascii

# una fila por ejecucion en un CSV unico
$csv = Join-Path $Root 'resultats\resum.csv'
if (-not (Test-Path $csv)) {
  'id,mode,ctx,disponible_abans,disponible_min,pagefile_abans,pagefile_max,carregat,generat,tok_s_generacio,tok_s_prompt,buffer_model,buffer_kv,buffer_compute,buffer_total,aturat,carpeta' | Set-Content $csv -Encoding Ascii
}
"$Id,$Mode,$Ctx,$avail0,$($script:minAvail),$page0,$($script:maxPage),$loaded,$generated,$tgOut,$ppOut,$(F $bm),$(F $bk),$(F $bc),$(F $bt),`"$($script:abort)`",$dir" | Add-Content $csv -Encoding Ascii

Write-Host ("[{0} {1}] carregat={2} generat={3} tok/s={4} buffers={5} MiB min disponible={6} MiB {7}" -f $Id, $Mode, $loaded, $generated, $tgOut, (F $bt), $script:minAvail, $script:abort)
