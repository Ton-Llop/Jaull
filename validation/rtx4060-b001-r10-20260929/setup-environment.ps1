$ErrorActionPreference = 'Continue'
$repoDir = (Get-Location).Path
$runtimeDir = Join-Path $repoDir '.venv\rtx4060-runtime-b10357'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repoDir '.venv\rtx4060-py312'
Write-Output ('Timestamp: ' + (Get-Date -Format o))
Write-Output "Isolated Python environment: $env:UV_PROJECT_ENVIRONMENT"
uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
uv sync --locked --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'Locked environment synchronization failed' }
$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b10357'
Write-Output ($release | Select-Object tag_name,target_commitish,published_at | ConvertTo-Json)
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
foreach ($name in @('llama-b10357-bin-win-cuda-13.3-x64.zip','cudart-llama-bin-win-cuda-13.3-x64.zip')) {
    $asset = $release.assets | Where-Object name -eq $name
    if ($null -eq $asset) { throw "Release asset unavailable: $name" }
    Write-Output ($asset | Select-Object name,size,digest,browser_download_url | ConvertTo-Json)
    $archivePath = Join-Path $runtimeDir $name
    if (Test-Path -LiteralPath $archivePath) { throw "Archive already exists: $archivePath" }
    curl.exe --fail --location --retry 3 --output $archivePath $asset.browser_download_url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output "Downloaded SHA256: $archiveHash"
    if ($asset.digest -and $asset.digest -ne "sha256:$archiveHash") { throw "Release digest mismatch: $name" }
    Expand-Archive -LiteralPath $archivePath -DestinationPath (Join-Path $runtimeDir 'bin')
}
Get-ChildItem -LiteralPath (Join-Path $runtimeDir 'bin') -Recurse -File | Select-Object FullName,Length | Format-Table -AutoSize
Write-Output 'SETUP_COMPLETE'
