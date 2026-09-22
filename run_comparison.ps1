$ErrorActionPreference = 'Stop'
$comparisonPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $comparisonPython)) {
    throw 'Create .venv and install requirements-lock.txt plus requirements-pinn.txt first.'
}
& $comparisonPython (Join-Path $PSScriptRoot 'run_comparison.py') @args
exit $LASTEXITCODE
