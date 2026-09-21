<#
run_tui_ablation.ps1

Runs the target_update_interval ablation (FIX 7 in train_cql.py) as four
sequential full training runs (3 seeds each -> 12 trainings total), all
other knobs (alpha=4.0, qfunc, oversample-factor, seeds, split) held
identical.

Order: 8000 (baseline) and 500 (extreme case) first, since if 500 doesn't
move FQE at all relative to 8000, the target-update-interval hypothesis
is probably dead and 2000/4000 aren't needed to confirm that. Reorder the
$Intervals array below if you'd rather run all four regardless, or if you
already have a saved reports/cql_results.json for the 8000 baseline and
want to skip re-running it (see -SkipBaseline below).

Stops immediately (does not run the remaining intervals) if any run exits
non-zero, so a broken/incomplete run never silently gets treated as a
valid comparison point later.

Usage:
    .\run_tui_ablation.ps1
    .\run_tui_ablation.ps1 -SkipBaseline
    .\run_tui_ablation.ps1 -Alpha 4.0 -Config config/rl_config.yaml
#>

param(
    [double]$Alpha = 4.0,
    [string]$Config = "config/rl_config.yaml",
    [switch]$SkipBaseline
)

# Ordered: baseline (8000) and the extreme case (500) first; 2000/4000
# only matter if those two disagree. Remove entries or reorder as needed.
$Intervals = @(8000, 500, 2000, 4000)

if ($SkipBaseline) {
    Write-Host "[run_tui_ablation] -SkipBaseline set: skipping target_update_interval=8000 (assuming reports/cql_results_tui8000.json or reports/cql_results.json already exists from a prior run)."
    $Intervals = $Intervals | Where-Object { $_ -ne 8000 }
}

$StartTime = Get-Date
$Completed = @()

foreach ($tui in $Intervals) {
    $ModelDir = "models_tui$tui"
    $OutFile  = "reports/cql_results_tui$tui.json"

    Write-Host ""
    Write-Host "==================================================================="
    Write-Host "[run_tui_ablation] Starting target_update_interval=$tui (alpha=$Alpha)"
    Write-Host "[run_tui_ablation]   model-dir: $ModelDir"
    Write-Host "[run_tui_ablation]   out:       $OutFile"
    Write-Host "==================================================================="

    $RunStart = Get-Date

    python src/rl/train_cql.py `
        --config $Config `
        --alpha $Alpha `
        --target-update-interval $tui `
        --model-dir $ModelDir `
        --out $OutFile

    $ExitCode = $LASTEXITCODE
    $RunElapsed = (Get-Date) - $RunStart

    if ($ExitCode -ne 0) {
        Write-Host ""
        Write-Host "[run_tui_ablation] ERROR: run for target_update_interval=$tui exited with code $ExitCode after $($RunElapsed.ToString('hh\:mm\:ss'))." -ForegroundColor Red
        Write-Host "[run_tui_ablation] Stopping here -- NOT running the remaining intervals ($(($Intervals | Where-Object { $_ -notin ($Completed + $tui) }) -join ', '))." -ForegroundColor Red
        Write-Host "[run_tui_ablation] Completed successfully before this: $(if ($Completed.Count -gt 0) { $Completed -join ', ' } else { '(none)' })" -ForegroundColor Red
        exit $ExitCode
    }

    Write-Host "[run_tui_ablation] target_update_interval=$tui finished OK in $($RunElapsed.ToString('hh\:mm\:ss')) -> $OutFile"
    $Completed += $tui
}

$TotalElapsed = (Get-Date) - $StartTime
Write-Host ""
Write-Host "==================================================================="
Write-Host "[run_tui_ablation] All runs complete: $($Completed -join ', ') (total $($TotalElapsed.ToString('hh\:mm\:ss')))"
Write-Host "[run_tui_ablation] Results written to reports/cql_results_tui<N>.json for each N above."
Write-Host "[run_tui_ablation] Reminder: compare these via run_fqe_check.py / compare_policies.py,"
Write-Host "[run_tui_ablation] NOT via val_action_match/test_action_match in these summary files --"
Write-Host "[run_tui_ablation] those are diagnostic-only (see train_cql.py module docstring)."
Write-Host "==================================================================="