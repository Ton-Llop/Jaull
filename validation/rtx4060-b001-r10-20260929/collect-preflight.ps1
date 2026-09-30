$ErrorActionPreference = 'Continue'
$campaignDir = $PSScriptRoot
if (Test-Path -LiteralPath (Join-Path $campaignDir 'preflight.log')) {
    throw 'Refusing to overwrite an existing evidence log.'
}
Start-Transcript -LiteralPath (Join-Path $campaignDir 'preflight.log')
function Invoke-LoggedCheck {
    param([string]$Label, [scriptblock]$Action)
    Write-Output "CHECK: $Label"
    Write-Output (Get-Date -Format o)
    $global:LASTEXITCODE = 0
    & $Action
    Write-Output "EXIT_CODE: $global:LASTEXITCODE"
}
Invoke-LoggedCheck 'git rev-parse HEAD' { git rev-parse HEAD }
Invoke-LoggedCheck 'git status --short (campaign script already created)' { git status --short }
Invoke-LoggedCheck 'OS and Python environment' {
    .venv\Scripts\python.exe -c "import sys,platform,importlib.metadata as m; print(sys.executable); print(platform.platform()); print(sys.version); print('jaull',m.version('jaull'))"
    uv --version
    uv python list --only-installed
    wsl --list --verbose
}
Invoke-LoggedCheck 'GPU identity and memory at preflight timestamp' {
    nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv
}
Invoke-LoggedCheck 'Reference artifact local SHA-256 and cached download metadata' {
    $artifactPath = 'C:\Users\USER\AppData\Local\jaull\models\bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf'
    Get-Item -LiteralPath $artifactPath | Format-List FullName,Length
    $artifactHash = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output "SHA256: $artifactHash"
    Write-Output ('EXPECTED_HASH_MATCH: ' + ($artifactHash -eq '65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423'))
    Get-ChildItem -LiteralPath (Join-Path (Split-Path $artifactPath) '.cache\huggingface\download') -Filter '*.metadata' -File | ForEach-Object {
        Write-Output $_.FullName
        Get-Content -LiteralPath $_.FullName
    }
}
$env:UV_OFFLINE = '1'
$env:UV_PYTHON_DOWNLOADS = 'never'
$env:UV_NO_SYNC = '1'
Invoke-LoggedCheck 'uv run jaull scan (existing environment; UV_NO_SYNC=1)' { uv run jaull scan }
Invoke-LoggedCheck 'uv run jaull doctor (existing environment; UV_NO_SYNC=1)' { uv run jaull doctor }
Invoke-LoggedCheck 'Jaull native runtime discovery' {
    .venv\Scripts\python.exe -c "from jaull.runtime.locator import RuntimeLocator; print([x.model_dump(mode='json') for x in RuntimeLocator().discover_llama_cpp()])"
}
Remove-Item Env:UV_NO_SYNC
Invoke-LoggedCheck 'uv run --python 3.12 ruff check . (offline; no Python downloads)' { uv run --python 3.12 ruff check . }
Invoke-LoggedCheck 'uv run --python 3.12 mypy src (offline; no Python downloads)' { uv run --python 3.12 mypy src }
Invoke-LoggedCheck 'uv run --python 3.12 pytest (offline; no Python downloads)' { uv run --python 3.12 pytest }
$env:PATH = (Join-Path (Get-Location) '.venv\Scripts') + ';' + $env:PATH
Invoke-LoggedCheck 'python -m compileall -q src (existing Python 3.14.7)' { python -m compileall -q src }
Invoke-LoggedCheck 'git diff --check' { git diff --check }
$env:UV_NO_SYNC = '1'
Invoke-LoggedCheck 'Supplementary ruff on existing Python 3.14 environment' { uv run ruff check . }
Invoke-LoggedCheck 'Supplementary mypy on existing Python 3.14 environment' { uv run mypy src }
Invoke-LoggedCheck 'Supplementary full pytest including architecture tests on existing Python 3.14 environment' { uv run pytest }
Invoke-LoggedCheck 'git diff --stat' { git diff --stat }
Invoke-LoggedCheck 'git status --short' { git status --short }
Stop-Transcript
