# Builds the Jaull desktop installer: the app, a portable Python with Jaull and every dependency
# uv.lock pins, packed with Inno Setup. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File desktop\installer\build.ps1
# Needs: Rust (cargo), uv, and Inno Setup 6 (ISCC.exe). Output: desktop\target\installer\Jaull-Setup-<version>.exe
param([string]$PythonVersion = '3.12', [string]$Version = '0.1.0')
$ErrorActionPreference = 'Stop'
$desktop = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $desktop
$out = Join-Path $desktop 'target\installer'
$stage = Join-Path $out 'stage'

$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw 'Inno Setup 6 was not found (winget install JRSoftware.InnoSetup).' }
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

Write-Host '1/5 Building the app' -ForegroundColor Cyan
Push-Location $desktop
cargo build --release
if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
Pop-Location

Write-Host '2/5 Fresh staging folder' -ForegroundColor Cyan
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force $stage | Out-Null
Copy-Item (Join-Path $desktop 'target\release\jaull-desktop.exe') (Join-Path $stage 'Jaull.exe')

Write-Host "3/5 Portable Python $PythonVersion (python-build-standalone, through uv)" -ForegroundColor Cyan
$pyroot = Join-Path $out 'pyroot'
if (Test-Path $pyroot) { Remove-Item $pyroot -Recurse -Force }
# --no-bin: do not put python3.x.exe on this PC's PATH, only the copy we package.
uv python install $PythonVersion --install-dir $pyroot --no-bin
if ($LASTEXITCODE -ne 0) { throw 'uv python install failed' }
# uv also makes a junction named after the minor version; take the real folder, not the link.
$pydir = Get-ChildItem $pyroot -Directory |
    Where-Object { -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -and (Test-Path (Join-Path $_.FullName 'python.exe')) } |
    Select-Object -First 1
if (-not $pydir) { throw "no python.exe under $pyroot" }
Move-Item $pydir.FullName (Join-Path $stage 'python')
Remove-Item $pyroot -Recurse -Force
$python = Join-Path $stage 'python\python.exe'

Write-Host '4/5 Jaull and its locked dependencies into that Python' -ForegroundColor Cyan
$req = Join-Path $out 'requirements.txt'
Push-Location $repo
uv export --frozen --no-dev --no-hashes --no-emit-project -o $req | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'uv export failed' }
uv pip install --python $python --break-system-packages -r $req
if ($LASTEXITCODE -ne 0) { throw 'installing the dependencies failed' }
uv pip install --python $python --break-system-packages --no-deps $repo
if ($LASTEXITCODE -ne 0) { throw 'installing Jaull failed' }
Pop-Location
& $python -m jaull --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'the bundled Python cannot run jaull' }
# Bytecode caches are rebuilt on the user's PC; leaving them out keeps the installer smaller.
Get-ChildItem (Join-Path $stage 'python') -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
$size = (Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("   staged: {0:N2} GiB" -f ($size / 1GB))

Write-Host '5/5 Installer (Inno Setup)' -ForegroundColor Cyan
& $iscc "/DStage=$stage" "/DAppVersion=$Version" (Join-Path $PSScriptRoot 'jaull.iss')
if ($LASTEXITCODE -ne 0) { throw 'ISCC failed' }
$setup = Join-Path $out "Jaull-Setup-$Version.exe"
Write-Host ("Done: {0} ({1:N0} MiB)" -f $setup, ((Get-Item $setup).Length / 1MB)) -ForegroundColor Green
