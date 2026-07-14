$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "desktop_strategy_app.py"

Start-Process -FilePath $python -ArgumentList @("`"$script`"") -WorkingDirectory $root -WindowStyle Hidden
