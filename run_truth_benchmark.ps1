$ErrorActionPreference = 'Stop'
$truthPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $truthPython)) {
    throw 'Install the base, PINN, and truth benchmark dependencies first.'
}
& $truthPython (Join-Path $PSScriptRoot 'run_truth_benchmark.py') @args
exit $LASTEXITCODE
