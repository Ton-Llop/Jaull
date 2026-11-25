# Descarga llama.cpp (build fijado, con SHA-256) y TODOS los modelos que falten (con comprobacion de tamano),
# y guarda el estado de la maquina. Se puede repetir: salta lo que ya esta y reanuda las descargas cortadas.
$ErrorActionPreference = 'Stop'
$Root = 'C:\cpu-bench'
New-Item -ItemType Directory -Force "$Root\llama", "$Root\models", "$Root\resultats" | Out-Null

# 1) llama.cpp b11889, solo CPU
$exe = Get-ChildItem "$Root\llama" -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) {
  $zip = "$Root\llama.zip"
  Write-Host "Descargando llama.cpp b11889 (CPU) ..."
  curl.exe -L --fail --retry 5 -o $zip https://github.com/ggml-org/llama.cpp/releases/download/b11889/llama-b11889-bin-win-cpu-x64.zip
  $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
  $esperado = 'bf563dffb502a230ae06c9dc455b54cbc079e7e0bc270cd4d6191209b28d175d'
  if ($hash -ne $esperado) { throw "SHA-256 incorrecto: $hash (esperado $esperado). Borra $zip y repite." }
  Expand-Archive $zip "$Root\llama" -Force
  Remove-Item $zip
  $exe = Get-ChildItem "$Root\llama" -Recurse -Filter llama-server.exe | Select-Object -First 1
}
Write-Host "llama-server: $($exe.FullName)"

# 2) modelos: repo, fichero, tamano exacto en bytes (los del Hub). Los Mellum ya estan en AppData\Local\jaull\models.
$models = @(
  @('ThomasBaruzier/Qwen2.5-0.5B-Instruct-GGUF', 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf', 397808160),
  @('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q5_k_m.gguf', 1285494304),
  @('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', 2104932768),
  @('bartowski/Qwen2.5-7B-Instruct-GGUF', 'Qwen2.5-7B-Instruct-IQ2_M.gguf', 2780342976),
  @('Qwen/Qwen2.5-7B-Instruct-GGUF', 'qwen2.5-7b-instruct-q3_k_m.gguf', 3808391072),
  @('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF', 'qwen2.5-coder-7b-instruct-q3_k_m.gguf', 3808391104),
  @('bartowski/Qwen2.5-7B-Instruct-GGUF', 'Qwen2.5-7B-Instruct-Q4_K_M.gguf', 4683074240),
  @('Qwen/Qwen2.5-Coder-14B-Instruct-GGUF', 'qwen2.5-coder-14b-instruct-q3_k_m.gguf', 7339203968),
  @('ornith-ai/Ornith-1.5-9B-GGUF', 'Ornith-1.5-9B-Q6_K.gguf', 7558901696)
)
$falta = 0
foreach ($m in $models) {
  $dest = Join-Path "$Root\models" $m[1]
  if (-not ((Test-Path $dest) -and ((Get-Item $dest).Length -eq [int64]$m[2]))) { $falta += [int64]$m[2] - $(if (Test-Path $dest) { (Get-Item $dest).Length } else { 0 }) }
}
$libre = (Get-PSDrive C).Free
Write-Host ("Por descargar: {0:N1} GiB; libre en C: {1:N1} GiB" -f ($falta / 1GB), ($libre / 1GB))
if ($libre -lt ($falta + 8GB)) { throw 'No hay espacio suficiente en C: (hace falta lo que falta por descargar mas 8 GiB de margen).' }

foreach ($m in $models) {
  $name = $m[1]; $size = [int64]$m[2]
  $dest = Join-Path "$Root\models" $name
  if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $size)) { Write-Host "ya esta: $name"; continue }
  $url = "https://huggingface.co/$($m[0])/resolve/main/$name"
  Write-Host ("Descargando {0} ({1:N2} GiB) ..." -f $name, ($size / 1GB)) -ForegroundColor Cyan
  for ($i = 1; $i -le 4; $i++) {
    curl.exe -L --fail -C - --retry 5 --retry-delay 10 -o $dest $url
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $size)) { break }
    Write-Host "  intento $i incompleto, reintento ..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
  }
  if (-not ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $size))) { throw "No se ha podido descargar completo: $name" }
}
$mellum = @(Get-ChildItem "$env:LOCALAPPDATA\jaull\models" -Recurse -Filter 'Mellum*.gguf' -ErrorAction SilentlyContinue)
Write-Host "Mellum encontrados: $($mellum.Count) (se esperan 2)"
if ($mellum.Count -lt 2) { Write-Host 'AVISO: faltan Mellum; sus casos se saltaran.' -ForegroundColor Yellow }

# 3) Estado de la maquina y hashes
$out = "$Root\estat-maquina.txt"
$os = Get-CimInstance Win32_OperatingSystem
$lines = @()
$lines += "fecha=$(Get-Date -Format s)"
$lines += "windows=$($os.Caption) build $($os.BuildNumber)"
$lines += "cpu=$((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
$lines += "ram_total_mib=$([int]($os.TotalVisibleMemorySize/1024))"
Get-CimInstance Win32_PhysicalMemory | ForEach-Object { $lines += "dimm=$([int]($_.Capacity/1GB)) GB $($_.Speed) MT/s" }
Get-CimInstance Win32_PageFileUsage | ForEach-Object { $lines += "pagefile=$($_.Name) tam=$($_.AllocatedBaseSize) MiB" }
$lines += "plan_energia=$((powercfg /getactivescheme) -join ' ')"
$lines += "llama_server=$(& $exe.FullName --version 2>&1 | Select-Object -First 3 | Out-String)"
$lines += "--- sha256 (hash, bytes, fichero)"
$gguf = @(Get-ChildItem "$Root\models" -Filter *.gguf) + $mellum
foreach ($f in $gguf) {
  Write-Host "SHA-256 de $($f.Name) ..."
  $lines += "$((Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower())  $($f.Length)  $($f.Name)"
}
$lines | Set-Content $out -Encoding Ascii
Write-Host "Guardado $out" -ForegroundColor Green
