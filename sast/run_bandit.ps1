# Run OWASP-recommended SAST (Bandit) over the Python source and save reports.
# Usage:  powershell -File sast/run_bandit.ps1
# (Bandit logs to stderr; allow that without aborting.)
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here

Write-Host "[sast] running Bandit ..."
bandit -r $root -x ./.venv,./models,./data,./results -f txt  -o (Join-Path $here "bandit_report.txt")
bandit -r $root -x ./.venv,./models,./data,./results -f json -o (Join-Path $here "bandit_report.json")
Write-Host "[sast] reports written to $here\bandit_report.txt and bandit_report.json"

Write-Host "[sast] severity summary:"
Get-Content (Join-Path $here "bandit_report.txt") | Select-String -Pattern "Severity:|Low:|Medium:|High:|No issues"
