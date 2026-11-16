# Follow-up eval (v3: schema tool) then step 5 (LLM as predictor), then regenerates docs/RESULTS.md.
#   powershell -ExecutionPolicy Bypass -File evals\run_followup_and_step5.ps1
param([string]$Model = "qwen2.5:3b", [string]$Log = "evals\runs\followup_step5.log")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
"start $(Get-Date -Format s)" | Out-File $Log -Encoding utf8
& $py evals\run_eval.py --model $Model --label "v3: schema tool + examples, schema on Cypher errors; cypher on" 2>&1 | Out-File $Log -Append -Encoding utf8
"eval done $(Get-Date -Format s)" | Out-File $Log -Append -Encoding utf8
& $py evals\llm_predictor.py --model $Model --n 400 --subset 60 --repeats 3 --determinism 20 2>&1 | Out-File $Log -Append -Encoding utf8
& $py scripts\make_results.py 2>&1 | Out-File $Log -Append -Encoding utf8
"done $(Get-Date -Format s)" | Out-File $Log -Append -Encoding utf8
