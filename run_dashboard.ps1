$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create .venv and install requirements-lock.txt first. See README.md.'
}
& $projectPython (Join-Path $PSScriptRoot 'dashboard_server.py') @args
exit $LASTEXITCODE
