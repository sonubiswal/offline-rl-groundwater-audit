# Phase-5 Provenance

## Historical numbers (pre-fix)
See the explainer's §9.6 table and git history of `reports/policy_comparison.json`.

## What changed
A code edit changed observation standardisation before being fed to learned policies.
Hand-written policies are unchanged; only learned agents moved.

## Current code
- scaler sidecars stored next to each checkpoint (`*.d3.scaler.json`)
- `src/rl/policy_io.py` is the single loader
- `tests/test_scaler_pipeline.py` asserts load -> score is reproducible

## Canonical numbers
The JSON/CSV files in this commit are canonical.
Historical numbers remain in git history and must be cited as superseded.

## Alpha
`revision_experiments/e1_canonical_vs_sweep.json` and `e6_sweep_config_audit.csv`
document canonical (alpha=1.0) vs sweep-best (alpha=4.0).
The canonical run is classified exploratory, per the explainer.
