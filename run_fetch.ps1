# Use installed Python, or the Python runtime bundled with this Codex installation.
$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if ($pythonCommand) {
    $pythonExecutable = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $bundledPython) {
    $pythonExecutable = $bundledPython
} else {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pythonLauncher) {
        throw 'Python 3.10 or newer is required. Install Python and run python fetch_celestrak.py.'
    }
    $pythonExecutable = $pythonLauncher.Source
}
& $pythonExecutable (Join-Path $PSScriptRoot 'fetch_celestrak.py') @args
exit $LASTEXITCODE
