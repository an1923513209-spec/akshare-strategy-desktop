$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "desktop_strategy_app.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found. Run: python -m venv .venv"
}

& $python $script
