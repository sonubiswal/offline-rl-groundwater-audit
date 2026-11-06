# Trishna OPAL: A Satellite-to-Policy Pipeline for Groundwater in Central India

**Status:** Draft. Every number in this report traces to `reports/final_report_claims.md`.
No claim is made that is not sourceable from a committed artefact.

---

## Abstract

TODO (~250 words). Structure:
- Problem: satellite signals are coarse; decision-making for pumping is unvalidated.
- What we built: end-to-end pipeline, well-disjoint validation, offline RL policy evaluation.
- What we found: downscaling adds 2% of error over kriging; CQL advantage is null at 10 seeds;
  OOD both policies fail catastrophically.
- What we contribute: the audit, the negative results, the calibration procedure.

## 1. Introduction

TODO (~1000 words). Structure:
- 1.1 The problem: groundwater depletion in central India
- 1.2 Why satellite data alone is not enough
- 1.3 Why offline RL rather than online
- 1.4 What this report claims (cite final_report_claims.md)
- 1.5 What this report does not claim (the "do not say" list from final_report_claims.md)

## 2. Data and Region of Interest

TODO (~1500 words). Cite:
- reports/roi_justification.md (all numbers there)
- The nine-state bbox, Maharashtra-led 3-state core
- The four satellite layers (GRACE, Sentinel-1, Sentinel-2, CHIRPS, GLDAS, SRTM)
- The CGWB well data: 92,227 readings, 4,063 wells, 2013-01-01 to 2022-07-28
- The manifest with SHA-256 and retrieval URL

## 3. Validation Methodology

TODO (~1500 words). The single most important section. Structure:
- 3.1 The circularity problem
- 3.2 The held-out-well design (710 wells, 17.5%)
- 3.3 The automated test that enforces zero overlap
- 3.4 The held-out-never-rasterised rule
- 3.5 The cross-validation protocol (spatially blocked, not date-grouped)
- 3.6 What this design can and cannot prove

## 4. Downscaling (Phase 2)

TODO (~1000 words). Cite final_report_claims.md Claim 1.
- The kriging baseline and its role
- The gradient-boosting layer
- The R^2 = 0.187 vs kriging R^2 = 0.155 result
- The 2% of error framing
- The quadrant holdout and why it matters
- The three rejected feature experiments (Phase 2.5 / 2.6)

## 5. Time-Series Model (Phase 3)

TODO (~800 words). Cite final_report_claims.md Claim 5 (per-well LSTM).
- The grid ConvLSTM infeasibility (7 contiguous sequences)
- The pivot to per-well LSTM
- R^2 = 0.575 vs persistence 0.439
- The +0.136 above persistence is the defensible contribution, not the raw 0.575

## 6. Synthetic RL Environment (Phase 4)

TODO (~1200 words). Cite assumptions A1-A6 in reports/open_items.md and final_report_claims.md cross-cutting caveats.
- Why a synthetic environment is necessary
- The water-balance equation
- Each of the six assumptions, what it costs
- The directional validation against real CGWB wells (sign correct, 2.8x too weak)
- The nine pre-training sanity checks

## 7. Policy Evaluation (Phase 5)

TODO (~1500 words). Cite final_report_claims.md Claim 3.
- The five-family 10-seed comparison table
- The historical 3-seed +0.1096 and why it does not replicate
- The base family's -2.47 gap and why it is dataset-specific
- The e1 diagnostic: start-set sensitivity, not BC recipe
- FQE vs direct rollout disagreement (e8: +0.437, CI [0.113, 0.762])

## 8. Out-of-Distribution Behaviour

TODO (~1000 words). Cite final_report_claims.md Claim 4.
- The four OOD files, their per-band numbers
- The endpoint contrast that survives the paired CI test (weights 1,2,0.5) and the one that does not (1,2,0.0)
- The explainer Section 9.5f claim and its supersession
- The framing: both fail catastrophically, CQL fails slightly less

## 9. Explainability

TODO (~800 words). Cite reports/explainability_methodology.md.
- The Q-gap confidence rule
- The five action labels tied to the code line 148
- What the module can and cannot do (four items)

## 10. Limitations

TODO (~1500 words). This section is mandatory. Structure:
- 10.1 Depth compression in downscaling
- 10.2 Kriging dominates
- 10.3 The holdout is weaker than it looks (median 5.8 km to nearest training well)
- 10.4 The simulator is a bucket model (A1-A6, especially A4 and A5)
- 10.5 The simulator was seeded from v1, not v2
- 10.6 The historical 3-seed result does not replicate at 10 seeds
- 10.7 OOD: both policies fail; FQE mis-ranks
- 10.8 Explainability inherits critic errors
- 10.9 The explainer Section 9.5f does not reproduce
- 10.10 Data drift, environment, reproduction notes

## 11. What We Can Say and What We Cannot Say

This section is the boundary between what the committed artefacts support
and what they do not. Every sentence below traces to
`reports/final_report_claims.md`. When a later section of this report
makes a claim, that claim is bounded by this section.

### 11.1 What we can say

**On the downscaler.** The gradient-boosting downscaler improves on
training-well kriging by 0.104 m RMSE on a 14,062-reading well-level
holdout within the study region. The improvement does not extend to
region-level spatial extrapolation: under a quadrant holdout, plain
kriging beats the learned layer in three of four quadrants. The
pipeline is reproducible, the evaluation is coverage-controlled and
bias-audited, and the primary remaining uncertainty is upstream
kriging quality.

**On the prediction intervals.** Split-conformal intervals achieved
0.8936 (global) and 0.8918 (distance-stratified) empirical coverage at
nominal 0.90 on a well-disjoint calibration and test split. Interval
mean width is approximately 13.5 m against a held-out RMSE of 5.30 m.
The intervals are useful as bounds, not as precise estimates.

**On the in-distribution policy comparison.** At 10 seeds, CQL does not
outperform BC by a statistically significant margin on any of the four
no_oracle and epsilon dataset families. The gap ranges from +0.040 to
+0.150 return units, and no bootstrap confidence interval excludes
zero. The historical 3-seed result of +0.1096 does not replicate. One
family (base, offline_rl_dataset.h5) shows BC decisively better than
CQL by 2.47 return units, but that gap is dataset-specific and does not
generalise to CQL being worse in general.

**On out-of-distribution behaviour.** Under all four committed
out-of-distribution evaluation files, CQL's return exceeds BC's in
every band, and the gap grows with distributional distance. The
endpoint contrast survives the paired confidence interval test under
reward weights [1.0, 2.0, 0.5] but not under [1.0, 2.0, 0.0]. Both
policies collapse catastrophically in the severe band (BC approximately
-90, CQL approximately -86). A gap of four to nineteen return units
between two failing policies is not a working policy.

**On explainability.** The Q-value-to-advice module translates the
critic's preferences into plain language, with a confidence level
derived from the Q-gap between the best and second-best actions. It
does not validate the critic, the reward function, or the simulator.
Its advice inherits every limitation of those three layers.

### 11.2 What we cannot say

The following claims do not appear anywhere in this report because no
committed artefact supports them.

- Strong spatial generalisation. The model generalises to held-out
  wells within a region that also contains training wells. It does not
  generalise spatially, as the quadrant holdout demonstrates.
- Calibrated prediction intervals narrow enough to serve as point
  predictions. The intervals are honest but wide (13.5 m against 5.30 m
  RMSE).
- State-of-the-art groundwater prediction. The R-squared of 0.187 sits
  in the middle of published well-level results once holdout design is
  accounted for. It is not a state-of-the-art figure.
- That the machine-learning layer is the primary source of predictive
  skill. Kriging carries approximately 98 percent of the signal.
- That the model forecasts the future. Date-grouped validation is
  negative in every configuration.
- That CQL beats BC, or that BC beats CQL, without qualifying which
  dataset, which number of seeds, and which start set.
- That CQL reverses out-of-distribution. The internal explainer claimed
  this in Section 9.5f, but no committed artefact reproduces it (see
  `reports/phase5_provenance.md`).
- That the simulator is physically calibrated. It is a stylised bucket
  model with six numbered assumptions.
- That the ConvLSTM approach was shown not to work as a technique. The
  limit is data volume (only seven contiguous grid sequences project-
  wide), not the method.
- Any real-world pumping, aquifer, economic, or sustainability benefit.
  Every Phase 5 number is a statement about a simulation.
## 12. Future Work

TODO (~600 words). Pull from reports/open_items.md "Still open (P1)" section:
- Strict region-level holdout with re-kriging
- Depth-stratified per-well LSTM analysis
- Drop vegetation from Phase 3
- Sequence-length sweep
- Ablate row_norm/col_norm from v2
- Fix simulator seasonal decoupling
- Re-seed simulator from v2
- More seeds beyond 10

## References

TODO. Cite:
- The CGWB dataset (URL from data/raw/cgwb_raw.manifest.csv)
- GRACE, Sentinel-1/2, CHIRPS, GLDAS, SRTM
- d3rlpy (version 2.8.1)
- kriging reference (ordinary kriging, spherical variogram)
- conformal prediction reference
- CQL paper
- BC paper

## Appendix A: Reproducibility

TODO. Include:
- Environment: Python 3.10.20, d3rlpy 2.8.1, pytest 9.1.1
- Git tags: opal-validated-v1 (freeze), opal-revision-v1 (Step 1), opal-report-v1 (after 3E)
- Test suite: pytest tests/ (69 tests, 68 pass, 1 skip)
- Every number comes from a committed file; no manual transcription

## Appendix B: Open Items at Time of Submission

TODO. Paste reports/open_items.md as-is. This is a statement of what remains open,
not an apology.

