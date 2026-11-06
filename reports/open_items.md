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
