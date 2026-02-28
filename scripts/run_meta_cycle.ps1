#!/usr/bin/env pwsh

# Wrapper for VSCode/PowerShell to normalize business exits from meta_cycle.
# Usage: .\scripts\run_meta_cycle.ps1 --csv ... --reason ...
& python -m scripts.meta_cycle @args

$originalExit = 0
if ($null -ne $LASTEXITCODE) {
    $originalExit = [int]$LASTEXITCODE
}

$exitTaxonomy = @{
    0  = "SUCCESS"
    10 = "SUCCESS_WITH_REJECT"
    20 = "BUSINESS_FAIL"
}

$meaning = "CRASH/ERROR"
if ($exitTaxonomy.ContainsKey($originalExit)) {
    $meaning = $exitTaxonomy[$originalExit]
}

$normalizedExit = $originalExit
if ($originalExit -eq 0 -or $originalExit -eq 10 -or $originalExit -eq 20) {
    $normalizedExit = 0
}

Write-Host "[run_meta_cycle] original_exit=$originalExit meaning=$meaning normalized_exit=$normalizedExit"
exit $normalizedExit
