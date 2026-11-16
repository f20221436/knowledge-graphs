# Waits for the step 5 pipeline to finish (it uses the CPU), then re-measures the agent (v4) and regenerates RESULTS.md.
param([string]$Model = "qwen2.5:3b", [string]$Wait = "evals\runs\followup_step5.log", [string]$Log = "evals\runs\v4.log")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
while (-not (Select-String -Path $Wait -Pattern "^done" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 30 }
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
"start $(Get-Date -Format s)" | Out-File $Log -Encoding utf8
& $py evals\run_eval.py --model $Model --label "v4: v3 + explicit amount takes precedence; cypher on" 2>&1 | Out-File $Log -Append -Encoding utf8
& $py scripts\make_results.py 2>&1 | Out-File $Log -Append -Encoding utf8
"done $(Get-Date -Format s)" | Out-File $Log -Append -Encoding utf8
