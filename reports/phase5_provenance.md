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

## Explainer Section 9.5f is superseded

The internal explainer (Trishna-OPAL-Team-Explainer.pdf) claims that the
CQL advantage "reverses" out-of-distribution, citing numbers like
CQL-BC = -16.01 in-distribution and -38.40 in the severe band.

No committed artefact reproduces this. All four committed OOD files
(ood_scaled_result.json, ood_scaled_meanq_result.json,
ood_w3_full_eval_only.json, ood_w3_zero_eval_only.json) show CQL-BC
positive in every band, with the gap growing with distance in three of
four files.

The explainer's Section 9.5f table must be marked superseded and must not
be cited in the final report. Its provenance is not recoverable from the
committed code or data. Most likely explanation: it described an earlier
code branch or an FQE-only estimator that the current artefacts have
replaced.

The honest framing, in final_report_claims.md Claim 4, is:

  Both policies collapse catastrophically in the severe OOD band
  (BC about -90, CQL about -86). CQL fails slightly less on average.
  This is not a "CQL wins OOD" claim; it is "both fail, with wide
  per-seed variance."
