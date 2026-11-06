# Open Items - Trishna OPAL (post-revision)

## Closed by this commit
- [x] BCQ, IQL, BCQ-matched trainers
- [x] 10-seed retraining and cross-combination
- [x] Conformal prediction study
- [x] Alpha canonical-vs-sweep audit (e1, e6)
- [x] Within-reward / OOD re-tests (e2, e5, e8)
- [x] Checkpoint audit and manifest (e6, e7)
- [x] Scaler sidecars per checkpoint + test
- [x] Phase-5 provenance note

## Still open (P0)
- [x] ROI written justification: reports/roi_justification.md (bbox contains 9 states; Maharashtra-led 3-state core; Vidarbha-heavy district density)
- [x] Final report citations grounded: see reports/final_report_claims.md
- [x] Q-value -> plain-language advice module: src/explainability/qvalue_translation.py, tests/test_qvalue_translation.py (8 tests), reports/explainability_methodology.md

## Still open (P1)
- [ ] Strict region-level holdout with re-kriging
- [ ] Depth-stratified per-well LSTM
- [ ] Drop vegetation from Phase 3
- [ ] Sequence-length sweep
- [ ] Ablate row_norm/col_norm from v2
- [ ] Verify both spatial-block partition functions
- [ ] Fix simulator seasonal decoupling
- [ ] Re-seed simulator from v2

## Cleanup
- [ ] After freeze, verify reports_2/ is a duplicate of reports/bc_sweep/ then delete (~95 MB)

## Documented side effects (not defects)

- Commit 80c0a18 committed 230 `.npy` return arrays under `reports_10seed/*/returns/`
  (~1 MB total). Reproducible from seed + config. The `reports/**/returns/` rule does
  not match the `reports_10seed/` root. Not worth amending. `.gitignore` now includes
  `reports_*/**/returns/` for future runs.

## Cleanup items (post-freeze, non-blocking)

- [ ] Confirm reports_2/ is a duplicate of reports/bc_sweep/ and delete to reclaim ~95 MB
- [ ] Confirm _archive/ contents are all reproduced elsewhere and delete on disk
- [ ] Confirm figs/ contents are superseded by figures/ and figures_final/


## Closed during Step 2 verification

- [x] Calibrated prediction intervals: empirical coverage 0.8936 (C0) / 0.8918 (C1)
  at nominal 0.90, well-disjoint calibration/test split enforced in run_conformal.py.
  Caveats for report: interval width ~13.5 m vs RMSE 5.30 m; mild degradation at
  10-20 km (0.874); pred_y out-of-sample is guaranteed by the held-out split test,
  not by run_conformal.py itself.

## Revision findings recorded during Step 2 verification

### 10-seed CQL vs BC: complete picture

From reports_10seed/<family>/policy_comparison_summary.csv:

  family          dataset                  alpha   BC      CQL     diff
  base            offline_rl_dataset.h5    1.0     11.924  9.451   -2.473
  epsilon         ..._epsilon.h5           1.0      9.088  9.135   +0.047
  no_oracle       ..._no_oracle.h5         4.0      9.094  9.167   +0.073
  no_oracle_a1    ..._no_oracle.h5         1.0      9.094  9.244   +0.150
  no_oracle_a10   ..._no_oracle.h5         10.0     9.094  9.134   +0.040

Interpretation:
- Four of five families: CQL advantage +0.04 to +0.15. No CI excludes zero.
- base family: BC dominates by 2.47 return units. Dataset-specific, not
  evidence that CQL is worse in general (BC=11.92 on base vs 9.09 elsewhere;
  CQL is stable across families).
- Historical 3-seed +0.1096 (epsilon) does not replicate at 10 seeds.

### e1 resolves the "e1 vs reports_10seed" scale mismatch

e1_sweep_vs_canonical.py rolls on rf_grid starts; reports_10seed uses an
unrecorded start set. Same comparison gives:

  e1 canonical (rf_grid):      CQL - BC = -0.0129 [-0.101, +0.069]   null
  e1 sweep_default (rf_grid):  CQL - BC = -0.0574 [-0.128, +0.004]   null
  e1 BC canonical - BC sweep:  -0.0445     [-0.104,  0.000]          null

Conclusions:
- BC training recipe (canonical vs sweep-default hyperparameters) does not
  matter: -0.045, CI spans zero.
- The +0.047 vs -0.057 headline gap in the paper is start-set choice, not
  BC recipe. Both are null either way.
- e1 is not a contradiction to reports_10seed; it is the diagnostic that
  explains the discrepancy.

### Sweep-config audit passed for what it checks (e6_sweep_config_audit.csv)

- 220 rows, all match=True.
- Columns audited: intended_lr, intended_hidden, loaded_lr, loaded_hidden.
- Alpha is NOT in this audit; see separate alpha provenance note.




## Step 2 closed

All four claims verified against committed artefacts:

1. Downscaling: R^2 = 0.187 vs kriging 0.155; +0.104 m, CI [0.051, 0.157].
   CLOSED.

2. Conformal intervals: empirical 0.8936 (C0) / 0.8918 (C1) at nominal
   0.90, well-disjoint split enforced. CLOSED.

3. In-distribution 10-seed CQL vs BC: null on all four no_oracle/epsilon
   families (+0.040 to +0.150, no CI excludes zero). Historical 3-seed
   +0.1096 does not replicate. base family shows BC wins by 2.47 (dataset-
   specific). CLOSED with revised framing.

4. OOD behaviour: CQL-BC positive in every band across all four committed
   OOD files; gap grows with distance in 3 of 4. Explainer Section 9.5f
   sign-reversal does NOT reproduce and is marked superseded. CLOSED with
   revised framing.

Remaining P0 items:

- [x] Q-value -> plain-language advice module: src/explainability/qvalue_translation.py, tests/test_qvalue_translation.py (8 tests), reports/explainability_methodology.md


