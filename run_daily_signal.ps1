$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
& ".\.venv\Scripts\python.exe" ".\daily_signal_report.py" --symbol 002472 --horizon short --strategy-type auto --shares 100 --buy-price 43.54 --buy-date 20260710
