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

This section lists every limitation that materially affects how a result
should be read. It is not an apology; it is a boundary condition. Each
item is measured, not asserted, and each one has a specific consequence
for the claim it constrains.

### 10.1 Depth compression in downscaling

The gradient-boosting model systematically over-predicts shallow depths
and under-predicts deep depths. On the well-level holdout:

  actual depth   readings   error (m)   bias (m)
  0-10 m            4,015       3.10      +1.55  (over-predicts)
  10-20 m             702       6.17      -5.21  (under-predicts)
  20-30 m             104      14.39     -13.83
  30-50 m              42      26.78     -25.58
  50 m+                 6      51.49     -50.64  (too few to trust)

The shallow over-prediction and the deep under-prediction have opposite
signs, so they partly cancel in the mean bias (-0.05 m), which makes the
overall bias look almost perfect. It is not. The model should be treated
as reliable mainly shallower than about 15 to 20 m, and its output beyond
that depth should be treated as qualitative.

Cause, diagnosed rather than assumed: the trained model's own predictions
never exceed 24.74 m anywhere in the held-out set, less than half of the
51.78 m maximum in its training target. Tree-based regression predicts
averages of training examples in each leaf, which mechanically pulls
output toward the centre of the training distribution. Deep examples are
a thin tail (99th percentile of the training target is only 16.90 m), so
there is little deep signal to draw on. A reweighted-training fix was
attempted and rejected because it degraded every aggregate metric.

Consequence for the report: any statement about groundwater depths in the
study region is reliable for shallow wells and unreliable for deep wells.
Because Phase 4 seeds the simulator from the Phase 2 downscaler, every
Phase 5 result inherits this bias.

### 10.2 Kriging dominates

The kriged surface has a holdout error of 5.404 m. The learned downscaler
reduces this to 5.300 m. That is an improvement of 0.104 m, about 2% of
total error. Approximately 98% of the useful signal is already in the
kriged surface. The learned layer polishes the last 2%.

Consequence: the honest way to describe the downscaler is not "a model
that predicts groundwater" but "a small correction on top of kriging."
If the goal were purely accuracy, the honest answer from this work is
that effort should go into better kriging (elevation as external drift,
better variogram, anisotropy) rather than into more covariates. Three
separate feature additions in this project — a 12-month rainfall deficit,
residual kriging, and soil texture plus land use — all failed under the
pre-registered improvement threshold of +0.02 on cross-validation.

### 10.3 The holdout is weaker than it looks

Held-out wells are interleaved with training wells. The median held-out
well is 5.8 km from the nearest training well, and 99.8% are within 25 km.
This is a well-level holdout, not a spatial holdout. The model is asked
to predict at a point surrounded by known points, not at a point in an
otherwise unobserved region.

The stricter test — remove an entire quadrant of the map, refit, and
predict held-out wells in the excluded quadrant — was run. Under this
test, plain kriging beats the learned layer in three of four quadrants,
with pooled R-squared of 0.155 (kriging) against 0.101 (learned layer).
The learned layer's advantage is interpolation between known points, not
extrapolation into unobserved regions.

One further caveat on the quadrant test itself: the kriged map was left
fixed across the four folds rather than re-kriged from the reduced
training set inside each fold. This means even the quadrant numbers are a
lower bound on how hard the true spatial extrapolation test would be.

Consequence: the phrase "the model generalises spatially" must not appear
anywhere. The model generalises to held-out wells within a region that
also contains training wells. That is all the holdout supports.

### 10.4 The simulator is a stylised bucket model

The synthetic RL environment is a two-arithmetic-constant model, not a
calibrated aquifer. Six numbered assumptions govern it:

  A1. Crop water demand is a simple seasonal proxy, not derived from
      crop-type maps, irrigation records, or FAO crop-coefficient tables.
      It also decouples from the calendar after the first step of each
      episode — a known defect (see 10.10).
  A2. Crop revenue rises with pumping until demand is met, then flattens.
      No real price or yield data backs the magnitude of the formula.
  A3. There is one fixed, location-independent critical depth for
      domestic supply, not calibrated to well depth or local
      infrastructure.
  A4. One sustainability threshold (15 m default) applies to the whole
      study region. Real sustainable yield varies by aquifer type and
      geology, which this project has no data for.
  A5. Pumping lowers the water table linearly at up to 2 m per quarter at
      full pumping. Not a calibrated aquifer response function.
  A6. Recharge is a fixed 15% fraction of recent rainfall. Not a
      physically calibrated infiltration model.

A 3x3 sweep of A4 (threshold) and A5 (drawdown rate) produced a 226-fold
spread in the magnitude of the CQL-versus-BC advantage at the deepest
state. The sign never flipped (positive and significant in all 27 cells),
so the qualitative finding is robust, but the magnitude is a parameter
choice, not a stable physical quantity. Meanwhile, climate scenario
(drought vs. normal vs. heavy rain) moved the answer by less than 0.5
return units. The thing everyone would assume matters most — drought
versus normal — barely matters; two arbitrary constants matter enormously.

Directional validation: the simulator's rainfall-recharge relationship
was checked against the actual CGWB wells. Real wells show a rainfall-to-
next-quarter correlation of -0.325 (more rain, shallower next reading).
The simulator shows -0.117 (same sign, weaker). The ratio is roughly
2.78x — the real effect is about 2.8 times stronger than simulated. This
establishes that the simulator is not qualitatively wrong in direction.
It does not establish that A6's fixed-fraction form or its 0.15
coefficient is calibrated.

Consequence: no amount of reinforcement-learning rigour makes the reward
function true. Every Phase 5 number is a statement about a simulation.
"No simulation result in this report should be read as a claim about
real aquifer behaviour."

### 10.5 The simulator was seeded from v1, not v2

The Phase 4 simulator was built and validated before the v2 downscaler
existed. Its initial water depths are drawn from the v1 model, which is
confirmed by the saved artefact: 11 input features, v1's signature
against v2's 16. Every Phase 5 result inherits v1's depth-compression
bias (Section 10.1), not v2's.

Re-seeding from v2 would produce a different pipeline than the one that
was built and validated. It is classified as out of scope for this
project, not as unfinished work. Whether re-seeding from v2 would change
the Phase 5 conclusion is untested.

### 10.6 The historical 3-seed result does not replicate at 10 seeds

The original Phase 5 evaluation reported a CQL-minus-BC advantage of
+0.1096 with 95% confidence interval [+0.050, +0.174] on three training
seeds, using the epsilon dataset family. This was the project's
originally-claimed headline result.

At 10 seeds, none of the five dataset families reproduce it:

  family          dataset                  BC       CQL      gap
  base            offline_rl_dataset.h5    11.924   9.451   -2.473
  epsilon         ..._epsilon.h5            9.088   9.135   +0.047
  no_oracle       ..._no_oracle.h5          9.094   9.167   +0.073
  no_oracle_a1    ..._no_oracle.h5          9.094   9.244   +0.150
  no_oracle_a10   ..._no_oracle.h5          9.094   9.134   +0.040

The four epsilon/no_oracle families agree within +0.04 to +0.15 return
units, none significant. The base family shows BC decisively better by
2.47 units, but BC's mean on base (11.92) is much higher than on the
other four (9.09), while CQL is stable across all five. This is a
dataset effect, not evidence that CQL is worse in general.

The e1 diagnostic (revision_experiments/e1_sweep_vs_canonical.py)
isolates two further effects. BC training recipe does not matter: the
canonical versus sweep-default BC pair differs by -0.045 (CI [-0.104,
0.000]). Start-set choice does matter for the headline sign: on rf_grid
starts, CQL-BC is -0.013 (null); on the unrecorded start set used by the
10-seed summaries, CQL-BC is +0.047 (null). Both are null; the sign
depends on starts.

Consequence: the historical 3-seed positive is best described as a seed
artefact. The current honest claim is that at 10 seeds, in-distribution,
CQL does not outperform BC on any of the four comparable families. Any
report sentence that begins "CQL beats BC" must qualify which dataset,
which number of seeds, and which start set.

### 10.7 Out-of-distribution: both policies fail; FQE mis-ranks

The four committed OOD files (ood_scaled_result.json,
ood_scaled_meanq_result.json, ood_w3_full_eval_only.json,
ood_w3_zero_eval_only.json) all show CQL-minus-BC positive in every band.
The in_dist to severe gap grows from +1.2/+4.5 to +6.0/+6.3/+19.2
depending on critic and reward weighting. Under weights [1.0, 2.0, 0.5],
the endpoint contrast (severe minus in-distribution) survives the paired
confidence interval test with observed +5.09 and CI [0.05, 10.14]. Under
weights [1.0, 2.0, 0.0], it does not (observed +4.84, CI [-0.08, 9.76]).

But the absolute numbers are catastrophic. BC's return falls from -12 at
in-distribution to -90.7 in the severe band. CQL's falls from -6.8 to
-86.2. Both policies are effectively failing; a +4 to +19 gap between two
deeply negative returns is not a working policy, and is not evidence of
one policy being "better" in any operationally meaningful sense.

Separately, Fitted Q Evaluation and direct rollout disagree systematically
by +0.437 (95% CI [0.113, 0.762]), with FQE over-estimating the CQL-BC
gap relative to rollout. This is evidence that offline policy evaluation
is itself unstable at distributional distance, not evidence for either
policy.

Consequence: the phrase "CQL beats BC out-of-distribution" must not
appear. The correct framing is "both policies fail catastrophically
outside the training distribution; CQL fails slightly less on average,
with wide per-seed variance (2 of 10 seeds show a negative endpoint
contrast)."

### 10.8 Explainability inherits critic errors

The Q-value-to-advice module reads the critic's Q-values and reports a
recommendation with a confidence level derived from the Q-gap. It does
not validate the critic. If the critic over-values a harmful action, the
module's plain-language text will present that action as the
recommendation with high confidence.

The confidence thresholds (tau_high = 0.5, tau_low = 0.1) are heuristic.
They were calibrated against the observed reward scale (per-episode
return ~9.4 over 20 steps, so per-step reward ~0.47), but they are not
derived from anything. A user with a different critic scale should pass
their own thresholds.

Consequence: the explainability layer must not be described as validating
the policy's safety or correctness. It translates. It does not endorse.
See reports/explainability_methodology.md, section "What this module
cannot do", for the full four-item list.

### 10.9 The internal explainer's Section 9.5f does not reproduce

The internal project explainer (Trishna-OPAL-Team-Explainer.pdf) claimed
that the CQL advantage "reverses" out-of-distribution, citing numbers
such as CQL-minus-BC = -16.01 in-distribution and -38.40 in the severe
band. This claim was prominent in earlier project discussions.

No committed artefact reproduces it. All four OOD files show the
opposite sign. The explainer's provenance for that table is not
recoverable from the committed code or data. It is marked superseded in
reports/phase5_provenance.md.

Consequence: the final report cites the committed OOD files, not the
explainer. Any sentence citing the explainer's Section 9.5f table is
incorrect and should be deleted from drafts.

### 10.10 Known defects and process notes

- **Seasonal decoupling in the simulator.** Crop water demand peaks at
  the very first step of an episode and then drifts with random noise,
  even though the calendar fields advance correctly each step. Any
  agent's apparent grasp of "season matters" is only genuinely exercised
  at the start of each episode. Recorded as an inherited limitation of
  Phase 5, not fixed in this version.

- **Reproducibility incidents.** A code edit changed observation
  standardisation between the historical Phase 5 runs and the current
  code, making the historical confidence intervals non-reproducible and
  (in this case) more favourable to CQL. The code diff itself is
  unrecoverable without version history. This prompted the commit-based
  freeze in this revision, and the scaler sidecars + policy_io.py loader
  that make the current pipeline deterministic.

- **The alpha value used in the canonical run.** The recorded sweep
  scored alpha = 4.0 highest, but the canonical run used alpha = 1.0,
  set manually on the command line. Whether that was set before or after
  seeing the favourable result is not documented anywhere. The sweep
  metric itself is known to be biased toward low alpha, so neither value
  is justified. The canonical result is classified exploratory, and
  revision_experiments/e1_canonical_vs_sweep.json plus
  e6_sweep_config_audit.csv record the comparison.

- **Duplicate Phase 5 efforts.** Two separate Phase 5 runs exist in the
  repository: the original epsilon/no_oracle runs (3 seeds) and the
  later expanded 10-seed runs. They use different canonical
  configurations. They are documented separately in
  reports/phase5_provenance.md, and the report cites the 10-seed runs as
  canonical.

- **Documented artifacts that were not reproduced.** A stray npy file
  in reports_10seed/*/returns/ was committed in an earlier revision;
  it is reproducible from seed + config. An orphan model file with
  unexplained settings is quarantined. Both are recorded in
  reports/open_items.md as side effects, not as open defects.

- **Dataset region is not Maharashtra.** As documented in
  reports/roi_justification.md, the study region is a Maharashtra-led
  three-state core (Maharashtra 48.7%, Telangana 21.4%, Karnataka 12.7%)
  plus six peripheral states at low density. Calling it "Maharashtra"
  overstates administrative cleanliness and understates geographic
  heterogeneity. Any claim about "groundwater in Maharashtra" must be
  qualified.

Consequence for the report: the honest closing sentence of the
limitations section is that this project contributes a reproducible,
audited, and bounded pipeline and evaluation methodology. It does not
contribute a deployable advisory system, and it does not support any of
the claims listed in Section 11.2.
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


