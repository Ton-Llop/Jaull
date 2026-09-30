$ErrorActionPreference = 'Continue'
$repoDir = (Get-Location).Path
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repoDir '.venv\rtx4060-py312'
$env:UV_OFFLINE = '1'
$env:UV_PYTHON_DOWNLOADS = 'never'
$env:UV_FROZEN = '1'
$env:UV_LINK_MODE = 'copy'
$env:SSL_CERT_FILE = Join-Path $repoDir '.venv\rtx4060-trust.pem'
$env:PATH = (Join-Path $repoDir '.venv\rtx4060-py312\Scripts') + ';' + (Join-Path $repoDir '.venv\rtx4060-runtime-b10357\bin') + ';' + $env:PATH
$env:PYTEST_ADDOPTS = '--junitxml=validation/rtx4060-b001-r10-20260929/pytest.xml'
$script:Results = @()
function Invoke-LoggedGate {
    param([string]$Name, [string]$Command, [scriptblock]$Action)
    $logPath = Join-Path $PSScriptRoot ($Name + '.log')
    if (Test-Path -LiteralPath $logPath) { throw "Refusing to overwrite $logPath" }
    Write-Output "COMMAND: $Command" | Tee-Object -FilePath $logPath
    Write-Output (Get-Date -Format o) | Tee-Object -FilePath $logPath -Append
    $global:LASTEXITCODE = 0
    & $Action 2>&1 | Tee-Object -FilePath $logPath -Append
    $gateExit = $global:LASTEXITCODE
    Write-Output "EXIT_CODE: $gateExit" | Tee-Object -FilePath $logPath -Append
    $script:Results += [PSCustomObject]@{name=$Name;command=$Command;exit_code=$gateExit;log=$logPath}
}
Invoke-LoggedGate 'scan-ready' 'uv run jaull scan' { uv run jaull scan }
Invoke-LoggedGate 'doctor-ready' 'uv run jaull doctor' { uv run jaull doctor }
Invoke-LoggedGate 'gate-ruff' 'uv run --python 3.12 ruff check .' { uv run --python 3.12 ruff check . }
Invoke-LoggedGate 'gate-mypy' 'uv run --python 3.12 mypy src' { uv run --python 3.12 mypy src }
Invoke-LoggedGate 'gate-pytest' 'uv run --python 3.12 pytest' { uv run --python 3.12 pytest }
Invoke-LoggedGate 'gate-compileall' 'python -m compileall -q src' { python -m compileall -q src }
Invoke-LoggedGate 'gate-diff-check' 'git diff --check' { git diff --check }
$script:Results | ConvertTo-Json -Depth 5 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'gates-results.json')
git diff --stat
git status --short
