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
- [ ] ROI written justification
- [ ] Final report citing canonical numbers from this commit
- [ ] Q-value -> plain-language advice module (if explainability claimed)

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

### 10-seed CQL − BC is null on both training regimes (e1_canonical_vs_sweep.json)

- canonical:         CQL − BC = -0.0129, 95% CI [-0.1011, +0.0686], excludes zero = false
- sweep_default:     CQL − BC = -0.0574, 95% CI [-0.1282, +0.0044], excludes zero = false
- BC canonical − BC sweep: -0.0445, 95% CI [-0.1039, 0.0], excludes zero = false
  (only seeds 123/1234/12345 differ between the two BC families)

Interpretation: the historical 3-seed CQL advantage (+0.1096) does NOT survive
the 10-seed replication. CQL is not better than BC at 10 seeds on either regime.
The final report must state this as a null result, not as an advantage.

### Sweep-config audit passed for what it checks (e6_sweep_config_audit.csv)

- 220 rows, all match=True.
- Columns audited: intended_lr, intended_hidden, loaded_lr, loaded_hidden.
- Alpha is NOT in this audit; see separate alpha provenance note (below).
