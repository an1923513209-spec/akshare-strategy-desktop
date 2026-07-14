$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
& ".\.venv\Scripts\python.exe" ".\check_install.py"
& ".\.venv\Scripts\python.exe" ".\scripts\run_backtesting.py" --symbol 002472 --start 20200101 --fast 10 --slow 30
& ".\.venv\Scripts\python.exe" ".\scripts\scan_vectorbt.py" --symbol 002472 --start 20200101
& ".\.venv\Scripts\python.exe" ".\scripts\visualize_strategy.py" --symbol 002472 --start 20200101
