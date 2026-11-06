# Final Report Claims - Trishna OPAL

Each claim below is bounded by numbers pasted from committed files. When the
report states a claim, cite this document as the source, and respect the
"do not say" line.

## 1. Downscaling

Claim: The gradient-boosting downscaler improves on training-well kriging by
0.104 m RMSE on a 14,062-reading well-level holdout within the study region.
(Source: reports/phase5_provenance.md, explainer Section 11.)

Scope: The improvement does NOT extend to region-level spatial extrapolation.
Under a quadrant holdout, plain kriging beats the learned layer in 3 of 4
quadrants.

Do not say: "The model generalises spatially." It does not.

## 2. Calibrated prediction intervals

Claim: Split-conformal intervals achieved 0.8936 (global, C0) and 0.8918
(distance-stratified, C1) empirical coverage at nominal 0.90 on a
well-disjoint calibration/test split. (Source: conformal_results/
conformal_summary.csv; split enforcement in run_conformal.py lines 177-199.)

Scope: Interval mean width is approximately 13.5 m against held-out RMSE
5.30 m. Coverage degrades mildly with distance (0.898 at 5-10 km, 0.874 at
10-20 km). These are useful as bounds, not as precise estimates.

Do not say: "The intervals are tight" or "the model predicts precisely."
Both are false.

## 3. In-distribution CQL vs BC at 10 seeds

Claim: At 10 seeds, CQL does not outperform BC by a statistically
significant margin on any of the four epsilon/no_oracle dataset families.
The gap ranges from +0.040 to +0.150 return units and no bootstrap CI
excludes zero. The historical 3-seed +0.1096 does not replicate.
(Sources: reports_10seed/<family>/policy_comparison_summary.csv;
revision_experiments/e1_canonical_vs_sweep.json.)

Scope: One family (base, dataset offline_rl_dataset.h5) shows BC
decisively better than CQL by 2.47 units. That is dataset-specific; BC
mean on base is 11.92 vs 9.09 on the other four families, while CQL is
stable across all five. The base gap does NOT generalise to CQL being
worse in general.

Do not say: "CQL beats BC" or "BC beats CQL". Either is wrong without
naming the dataset and number of seeds.

## 4. Out-of-distribution behaviour

Claim: Under all four committed OOD evaluation files, CQL's return exceeds
BC's in every band, and the gap grows with distributional distance:

  source                          in_dist  mild   moderate  severe
  ood_scaled_result.json           +5.31   +3.59   +5.94    +4.49
  ood_scaled_meanq_result.json     +4.24   +6.78  +11.07   +19.18
  ood_w3_full_eval_only.json       +1.23   +3.48   +7.06    +6.32
  ood_w3_zero_eval_only.json       +1.17   +3.32   +6.73    +6.01

Endpoint contrast (severe - in_dist) survives the paired CI test under
weights [1.0, 2.0, 0.5] (obs +5.09, CI [0.05, 10.14]) but not under
weights [1.0, 2.0, 0.0] (obs +4.84, CI [-0.08, 9.76]).

Scope: Both policies collapse catastrophically in the severe band (BC
about -90, CQL about -86). A +4 to +19 advantage between two failing
policies is not a working policy. The per-seed deltas are wide: 2 of 10
seeds show negative endpoint contrast, so the average is not uniform.

Do not say: "CQL reverses OOD." The internal explainer claims this in
Section 9.5f, but no committed artefact reproduces it (see
phase5_provenance.md). Do not say "BC beats CQL OOD" either. Neither is
supported.

## Cross-cutting caveats

- The simulator is a stylised bucket model, not calibrated to real aquifer
  physics. Assumptions A1-A6 apply to all Phase 5 numbers.
- Evaluation uses n_episodes=500 and eval_seed=9999 in a controlled
  simulator. This is not evidence of real-world causal effect.
- FQE and direct rollout disagree systematically by +0.437 (e8 CI
  [0.113, 0.762]), suggesting FQE is optimistic but not in a direction
  that flips the sign.

## Status summary

| Claim | Strength | Status |
|---|---|---|
| 1. Downscaling      | real but small (2% of error)      | CLOSED |
| 2. Conformal        | 89.4% at nominal 90%              | CLOSED |
| 3. In-dist 10-seed  | null on 4/5 families, BC wins 1/5 | CLOSED, framing changed |
| 4. OOD behavior     | both fail, CQL fails less         | CLOSED, framing changed |
