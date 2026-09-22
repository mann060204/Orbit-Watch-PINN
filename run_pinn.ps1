$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Install the project virtual environment first.' }
& $projectPython (Join-Path $PSScriptRoot 'run_pinn.py') @args
exit $LASTEXITCODE
