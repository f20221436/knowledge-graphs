# Runs two eval configurations back to back on the same code, then regenerates docs/RESULTS.md.
#   powershell -ExecutionPolicy Bypass -File evals\run_pair.ps1 -Model qwen2.5:3b
param([string]$Model = "qwen2.5:3b", [string]$Log = "evals\runs\run_pair.log")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
"start $(Get-Date -Format s)" | Out-File $Log -Encoding utf8
& $py evals\run_eval.py --model $Model --label "fixes v2: outcome guard + amount guard; cypher on" 2>&1 | Out-File $Log -Append -Encoding utf8
& $py evals\run_eval.py --model $Model --no-cypher --label "fixes v2: fixed tools only (no cypher)" 2>&1 | Out-File $Log -Append -Encoding utf8
& $py scripts\make_results.py 2>&1 | Out-File $Log -Append -Encoding utf8
"done $(Get-Date -Format s)" | Out-File $Log -Append -Encoding utf8
