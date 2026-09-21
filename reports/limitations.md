# Limitations

**Project:** Trishna-OPAL
**Status:** Living document, updated as each phase surfaces new documented
limitations. See `reports/validation_methodology.md` for the full
methodology this document assumes as context.

**Reconciliation note (this revision):** this file merges in two updates
that had drifted out of sync with the base document: a fuller set of
limitations for the revised ("v2") Phase 2 model (Section 1B, new), and a
revised Phase 5 limitations section (Sections 16–25, replacing an earlier,
simpler single-bootstrap version). Nothing below has been deleted for
being superseded — superseded content is kept and clearly labeled,
consistent with this document's own practice elsewhere (e.g. Section 10.1).

**Reconciliation note (added, this revision — Phase 5 expanded report):**
a further, substantially larger Phase 5 report has since been supplied
(multi-scenario robustness, reward-weight ablation, RF-perturbation
sensitivity, QR-vs-Mean-Q critic ablation, adaptation experiment,
provenance/leakage/CI audits — appended below as Sections 27–36). **This
expanded report is not a resolution of Sections 16–26 below; it is a
different, larger body of experiments that does not reference or resolve
the FQE-vs-direct-rollout disagreement documented in Section 19, and uses
a different canonical configuration (α=1.0 with a QR critic, 20,000
training steps, ~0.698 test action-match) than anything reported in
Sections 16–26.** Whether this expanded report supersedes, extends, or
runs alongside the earlier Phase 5 effort on the same checkpoints/dataset
has not been established from the material provided, and is recorded here
as an open item (see Section 36) rather than silently assumed.

**Reconciliation note (added, this revision — session integration):** a
subsequent working session established three further results not present
in any earlier draft of this document: (1) a genuine state-distribution
out-of-distribution (OOD) diagnostic, distinct from the expanded report's
scenario-shift experiment (Section 30), showing CQL's return advantage
reverses under true OOD states — recorded below as new Section 19.5; (2) a
direct re-confirmation of the region-level RF-vs-Kriging holdout numbers
already summarized in Section 1B.1 — recorded as a clarifying note there;
and (3) a behavioral validation of the Phase 4 simulator's rainfall
response against real CGWB observations, which is recorded in
`validation_methodology.md` (new section under the expanded Phase 5
material) rather than here, since it concerns simulator construction
methodology rather than a limitation per se. Per this document's own
stated practice, these are additive integrations, not replacements of

**Reconciliation note (added, this revision — end-of-session integration).** A final working session resolved or documented: (1) Phase-4 RF seed provenance confirmed from artifact (v1, `n_features_in_ == 11`); (2) reward-range and state-range discrepancies resolved (Section 25); (3) 61,913-row figure reconciles to the v1 training table (Section 27.1); (4) leakage upgraded from 4/6 to 5/6 (Section 35.3); (5) Phase-5 relationship between original and expanded reports resolved — **separate training efforts** (Section 36); (6) a **standardization drift** means historical Phase-5 CQL−BC CIs are not reproducible under current code, and current numbers are materially different (and stronger for CQL) — new Section 19.4b; (7) the OOD reversal in Section 19.5 **replicates under a Mean-Q critic**, closing open item (i); (8) the A4/A5 sensitivity of the 25 m scenario gap is now measured (Section 29); (9) per-episode return arrays are archived for the four original Phase-5 configs **and all 18 `paper_exp/` expanded-report configurations (216 array files total)** (Section 35.4).

anything above.

---

## 1. Depth-Dependent Prediction Bias (RF Extrapolation/Compression)

**Scope flag (updated):** this finding was first diagnosed against the
original ("v1") Phase 2 model (11 features, held-out R²=0.198,
N=4,869) and has since been **re-run against the current, adopted "v2"
model** (16 features, no Sentinel-2, R²=0.187, N=14,062) — see
`validation_methodology.md`, Section 12 for the v2 model itself and
Section 12a for the depth-stratified re-run. **The v2 re-run confirms
the same structural pattern**: deep-bin error is essentially identical
to v1 (−5.0 m at 10–20 m, −14.0 m at 20–30 m,
−26.6 m at 30–50 m, −60.9 m at 50+, n=15), and a
depth-balanced sample-weighting mitigation degraded every aggregate
metric without improving the deep bins. The failure is structural to
the covariate set, not a loss-function or model-selection problem. The
numbers reported below are **v1's**, retained here because they are the
more thoroughly covered version (larger per-bin sample sizes), and the
v2 confirmation is now closed rather than open.

**Finding:** Stratifying the v1 held-out validation result
(R²=0.198, RMSE=5.226m, MAE=3.371m, n=4,869) by actual well depth reveals
a systematic, monotonic degradation with depth that the aggregate metric
conceals:

| Actual depth range | n | RMSE (m) | MAE (m) | Mean bias (m) |
|---|---|---|---|---|
| 0–10 | 4,015 | 3.10 | 2.44 | +1.55 (overprediction) |
| 10–20 | 702 | 6.17 | 5.40 | −5.21 (underprediction) |
| 20–30 | 104 | 14.39 | 13.83 | −13.83 |
| 30–50 | 42 | 26.78 | 25.58 | −25.58 |
| 50+ | 6 | 51.49 | 50.64 | −50.64 *(n too small to be reliable — reported for completeness, not conclusions)* |

The model **overpredicts** shallow wells (depth < ~15m) and
**increasingly underpredicts** deeper wells, with the bias reaching
tens of metres at the deepest observed readings. Because these opposite-
signed errors partially cancel in aggregate, the overall mean residual
(−0.05m) looks nearly unbiased — this is misleading on its own and should
not be quoted without the stratified breakdown above.

**Note on R² at the stratified level:** every individual depth bin shows
*negative* R², despite the pooled aggregate R² being positive (0.198).
This is not a contradiction: R² measures error relative to a subset's own
variance, and that variance shrinks sharply within a narrow depth band,
making "beat the mean" a much harder bar to clear even at constant
absolute error. RMSE/MAE, not R², are the meaningful metrics for this
stratified comparison.

**Root cause, diagnosed directly (not assumed):** the Kriged training
target (built from training wells only) *does* contain deep values — its
maximum is 51.78m, comparable to the deepest held-out readings. However,
the trained Random Forest's own predicted output never exceeds 24.74m
across the entire held-out set — less than half of what its own training
target contained. This rules out Kriging-side smoothing as the primary
cause and instead points to a structural property of Random Forest
regression: predictions are formed by averaging training samples within a
leaf, which mechanically shrinks output toward the center of the training
distribution. This effect is compounded by class imbalance in the
training data itself — the 99th percentile of the Kriged training target
is only 16.90m against a maximum of 51.78m, meaning genuinely deep
training examples are a thin, rare tail rather than a well-represented
range, so RF's leaf-averaging has little deep signal to draw on relative
to the abundant shallow examples.

**Practical implication:** the model, as currently trained, should be
considered reliable primarily for shallow-to-moderate groundwater depths
(roughly under 15–20m) and should not be trusted for quantitative
predictions at depths beyond that range. Any downstream use of this
model's output (e.g. as an input to the Phase 3/4 advisory/RL system)
should account for this — a predicted moderate depth carries materially
less confidence once the true value plausibly lies in the deeper regime
this model was not able to represent well. **This caveat applies directly
to Phase 4's seed table, which was built from this same v1 model — see
Section 12 below.**

**Not pursued as a further fix:** correcting this would require either
revisiting the Random Forest approach itself (e.g. quantile regression
forests, or a target transform designed for heavy-tailed distributions)
or rebalancing the training data toward deeper examples — both legitimate
directions for future work, but neither was pursued here, since doing so
would require re-touching the held-out set beyond the single, final,
already-reported validation check this project's methodology commits to.
This limitation is reported as-is rather than iterated away.

### 1.5 Trade-off decomposition caveat (reward-component behavior)

A reward-component decomposition of the frozen BC and CQL rollouts at the
moderate-depletion start state (gw=15 m, rainfall=50, extraction=0.3)
showed CQL selecting higher-pumping actions than BC: mean per-step crop
revenue +0.294 for CQL seed42 vs +0.110 for BC seed42, with corresponding
mean depletion penalties +0.235 vs +0.090. All three CQL seeds show both
higher revenue and higher penalty than every BC seed.

This is not inconsistent with the earlier canonical-configuration finding
that CQL selected the 100% pumping action less often than BC (Section 17
above), but it must be stated explicitly: **CQL's conservatism is
state-dependent, not uniform.** At moderate depletion, the learned CQL
policy pumps more than BC; at severe depletion (the canonical scenario
cells at 25 m), CQL's conservatism dominates and it pumps less. The
"conservative policy" description should be qualified with the state
regime.

**Not yet done.** The trade-off decomposition was run at a single start
state and covers only the frozen canonical policies. A full sensitivity
of the trade-off to start-state depth and to reward weighting has not
been run. This is documented as an open item.

## 1B. Phase 2 (v2 Model): Comprehensive Limitations

The v2 model (`validation_methodology.md`, Section 12 — HistGradientBoosting,
16 features, no Sentinel-2 NDVI, held-out R²=0.187, RMSE=5.300m, N=14,062)
supersedes v1 as the project's adopted Phase 2 result. All limitations
below are quantified with the numbers actually produced by the v2
pipeline and should be read alongside — not instead of — Section 1 above,
which remains a real, verified finding for v1 and an open re-verification
item for v2.

### 1B.1 Spatial extrapolation is not demonstrated

**Status:** Documented, quantified, unavoidable with the current well
density.

Held-out wells are a median 5.82 km from the nearest training well; 99.8%
of evaluated readings are within 25 km. This is a well-level holdout, not
a spatial holdout: every held-out reading is in a region that has nearby
training wells.

Under region-level holdout — where an entire 32×32 quadrant of the grid
is excluded from training — kriging-only outperforms the RF in 3 of 4
quadrants (pooled R²=0.155 vs 0.101). **The RF's improvement over kriging
is confined to interpolation between observed wells, not extrapolation to
unobserved regions.**

**Clarifying note (session integration, this revision):** these pooled
figures (RF R²=0.101 vs Kriging R²=0.155, RF RMSE=5.573 vs Kriging
RMSE=5.404, on the same 14,062-reading region-holdout evaluation) were
independently re-confirmed against `fast_spatial_holdout.csv` in a
subsequent working session and match this section exactly — this is a
re-verification of an existing finding, not a new result, and does not
change the conclusion above.

**Consequence for the paper:** the result should be described as
"well-level held-out generalization within the study region," not
"spatial generalization."

**Future work:** a strict region-level holdout that also re-kriges per
fold would test whether the full pipeline extrapolates. Given the RF
already loses on the fast version, this is unlikely to change the
conclusion.

### 1B.2 Small margin over kriging

The RF improves on kriging-only by 0.104 m RMSE (95% CI [0.051, 0.157]).
This is statistically significant but small relative to the base error
of ~5.3 m. On individual folds, kriging wins (fold 1: 0.237 vs 0.212 R²).

**Consequence:** the paper should not overstate the RF's contribution.
The Kriged surface carries most of the predictive signal; the covariate
layer recovers ~2% of the residual.

### 1B.3 Uncertainty quantification is not calibrated

A split-conformal interval calibrated on spatial-block CV out-of-fold
residuals achieved **50.4% empirical held-out coverage at nominal 90%**.
The interval is systematically too narrow because CV residuals do not
reflect the true held-out difficulty (see Section 1B.4).

**Consequence:** prediction intervals are not reported. The paper reports
point predictions only. Calibrated uncertainty is identified as future
work.

**Possible fix (not pursued):** scale the interval half-width by the
ratio of held-out RMSE to CV RMSE (~2.9×), which is a heuristic requiring
validation on independent data.

### 1B.4 Cross-validation overstates held-out skill

Spatial-block CV R² ranges 0.50 (2×2 partition) to 0.64 (8×8 partition),
versus held-out R²=0.187. **The CV-to-held-out gap is ~3× in R² and ~2.9×
in RMSE.**

**Cause:** spatial-CV test cells sit within 1–2° of training cells; the
Kriged surface interpolates well there, and the RF inherits that
interpolation quality. Held-out wells are 5.82 km median from the nearest
training well, and the Kriged surface is less accurate at those
locations.

**Consequence:** CV values must not be reported as skill estimates. CV is
useful for model *selection* only. The paper should report CV and
held-out side by side and explicitly note that CV overestimates.

**Reference:** this pattern is documented in hydrology ML literature
(Nolan et al. 2015, *J. Hydrology*; Tule et al. 2026, *Mathematical
Geosciences*; various groundwater-quality papers using spatial CV).

### 1B.5 Seasonal and interannual variation

Held-out R² varies significantly by season and year:

- Pre-monsoon RMSE 6.62 m, monsoon 5.14 m, post-monsoon 4.71 m.
- Fold-to-fold R² range 0.114–0.247.

Pre-monsoon (driest, most-depleted period, when demand is highest) is the
hardest season. This is consistent with the physics: coarse monthly
covariates and a stale lag cannot capture the rapid recharge dynamics of
the wet season, and the Kriged surface is noisier there.

**Consequence:** the paper should report per-season metrics and note that
skill is highest in the season of lowest management urgency.

### 1B.6 Kriging quality is the binding constraint

The Kriged surface itself has held-out RMSE 5.404 m. The RF reduces this
by 0.104 m. **The remaining ~5.2 m of error lives in kriging, not in the
downscaler.** The covariate layer is not the lever that will produce a
materially better result.

**Consequence:** future work should target regression-kriging (elevation
as external drift, better variogram, anisotropy) rather than adding more
covariates to the RF.

### 1B.7 Data limitations

- **Well density:** 3,353 training wells over ~700,000 km² is
 ~1 well per 200 km². Median well-to-well distance is small, but
 coverage is uneven.
- **Temporal record:** 2013–2021; CHIRPS starts in 2015, so all 2013
 readings are unevaluable (2,011 of 16,073 = 12.5% of held-out).
- **Sentinel-2 cloud masking:** in the original (v1) configuration, only
 42.1% of held-out readings had a valid S2 pixel. Removing S2 recovered
 coverage to 87.5% but also removed an informative covariate whose
 ablation showed it was not helpful anyway (see 1B.8).
- **2013 exclusion:** documented, unavoidable without CHIRPS data for
 2013.

### 1B.8 Synthetic target structure

The Kriged target is a smooth interpolation built from training wells.
The RF is therefore learning to reproduce a smooth spatial field. The
resulting R² should be interpreted in that context: the target itself is
not raw observation; it is a spatially interpolated reconstruction.

**Consequence:** the paper's validation tests the RF's ability to
reproduce the Kriged surface at held-out wells, not its ability to
predict raw CGWB readings directly. The Kriged surface is a modeling
choice.

### 1B.9 Model selection is within-noise

All four candidates (RF/HGBR × depth/residual) perform within 0.035 m
RMSE of each other on held-out (5.285–5.320 m). The spatial-CV-selected
model (`hgbr_residual`) is not significantly better than the others.

**Consequence:** the choice of model family and target mode is arbitrary
for this problem. The paper should state that the four candidates are
equivalent, not that one is superior.

### 1B.10 Region-holdout caveat

The region-level holdout reported in 1B.1 holds the Kriged surface fixed
across folds — only the RF training rows are region-restricted. Cells
inside the excluded quadrant still carry information from training wells
via kriging's spatial smoothing.

A strict version that re-kriges per fold would test whether the full
pipeline extrapolates. It was not run because the RF already loses on the
fast version, so the strict version is unlikely to change the conclusion.

### 1B.11 Sentinel-2 removal, S2-reintroduced row/col coordinates

v2 dropped Sentinel-2 NDVI (1B.7) but reintroduced `row_norm`/`col_norm`
spatial-coordinate features, which an earlier Phase 2 iteration had
removed after they caused severe location-memorization overfitting
(training R²=0.980 vs. held-out R²=0.20; see
`validation_methodology.md`, Section 10). v2 uses a much larger training
table and stronger regularization than that earlier iteration, and does
not reproduce that extreme training/held-out gap — but 1B.1's region-level
holdout result (kriging beats RF in 3 of 4 quadrants once whole regions
are excluded) is at least partially consistent with some residual
location-dependence from these coordinate features, not purely
coincidence. This has not been isolated (e.g. by an ablation with
`row_norm`/`col_norm` removed from v2) and is an open item.

### 1B.12 Cross-phase dependence

The Phase 2 result feeds Phase 4 (RL dataset generation) and Phase 5
(offline RL). **However, Phase 4's seed table was built from the v1
model, not v2 — see Section 12 below.** The downstream RL agent consumes
the downscaled groundwater state; Phase 2's prediction uncertainty (v1's
R²=0.198, or v2's R²=0.187 depending on which model is eventually used to
reseed) and its spatial-extrapolation failure (1B.1) propagate into the
RL policy's decision quality.

**Consequence:** Phase 5's inconclusive/mixed results (Section 19) may in
part reflect upstream prediction uncertainty rather than purely the RL
algorithm or evaluation methodology.

### 1B.13 Summary

The v2 Phase 2 result is **modest, statistically significant within its
scope, and honestly limited**. Its primary contribution is methodological:
rigorous coverage analysis, spatial-block CV, region-level holdout, and
documented failure modes. Its numerical skill (R²=0.187) is in the middle
of the published distribution for well-level groundwater prediction and
is not a headline result.

**What the paper should claim:**

> RF downscaling improves on training-well kriging by 0.104 m RMSE
> (95% CI [0.051, 0.157]) on a 14,062-reading well-level holdout within
> the study region. The improvement does not extend to region-level
> spatial extrapolation, where kriging-only is more robust. The pipeline
> is reproducible, the evaluation is coverage-controlled and
> bias-audited, and the primary remaining uncertainty is upstream
> kriging quality.

**What the paper should not claim:**

- Strong spatial generalization
- Calibrated prediction intervals
- State-of-the-art groundwater prediction
- That the RF layer is the primary source of predictive skill

## 2. Temporal Extrapolation

*(Carried forward from `validation_methodology.md`, Section 11.)* Date-
grouped cross-validation (holding out entire time periods rather than
spatial regions) consistently produced negative R² across every
hyperparameter configuration tested, for both v1 and v2 (v2's date-
grouped CV was re-run as a diagnostic under the same discipline and shows
the same pattern). The model has not been shown to reliably extrapolate
to genuinely unseen future time periods — its validated skill is for
spatial interpolation within the observed 2013–2022 historical record.

## 3. Held-Out Evaluation Coverage

**Current (v2, adopted) figure:** 14,062 of 16,073 held-out readings
(87.5%) are evaluated by the current Phase 2 model, after dropping
Sentinel-2 NDVI specifically to remove the coverage bottleneck it
imposed (see 1B.7 and `validation_methodology.md`, Section 12.2). All
2,011 skips are from 2013, blocked by missing CHIRPS coverage; the
evaluated subset shows no covariate SMD > 0.25 against the skipped
subset and is representative of 2014–2021.

**Historical (v1) figure, superseded:** only 4,869 of 16,073 held-out
readings (30.3%) could be evaluated under the original 11-feature,
S2-including configuration, since a complete feature vector required
simultaneous coverage across four monthly covariates, two rolling-window
features, and a prior-month Kriged lag term, compounded by S2's low and
seasonally-biased coverage. This evaluated subset was concentrated in
years with strong satellite/GRACE coverage and should not be assumed
representative of the full 2013–2022 record, particularly 2013–2014
(entirely excluded, pre-dating the satellite pull) and 2017–2018 (reduced
by the real GRACE mission gap). This historical coverage constraint is
retained here because it is the coverage basis for the Section 1
depth-bias finding and for the v1 numbers still referenced in Phase 3
(see `validation_methodology.md` Section 16.1's cross-reference note).

## 4. Other Carried-Forward Limitations (Phase 1)

- GRACE's native ~300km resolution means the downscaled 64×64 grid does
 not represent true sub-kilometer accuracy anywhere in this pipeline —
 the fine grid resolution should not be mistaken for fine-grained
 physical measurement precision.
- Sentinel-1/Sentinel-2 monthly composites frequently have partial
 spatial coverage (as low as single-digit percentages in some months),
 since a single month's satellite passes rarely cover the full ~900km-
 wide study region.

## 5. Grid-to-Grid ConvLSTM: Structural Data-Volume Infeasibility

### 5.1 Finding

The roadmap-specified grid-to-grid ConvLSTM (rolling multi-quarter
sequences over the full 64×64 covariate stack, predicting the next
quarter's target grid) could not be trained reliably. This is reported as
a **structural data-availability finding, independently diagnosed and
quantified** — not an assumed explanation, a code defect, or a tuning
failure. Full architectural and training detail is in
`validation_methodology.md`, Section 14.

### 5.2 Diagnostic Evidence

Of 38 available Kriged quarterly surfaces, a per-date coverage check
(`check_coverage.py`) found only 25 have a complete covariate stack across
all channels. Critically, these 25 usable dates are **not contiguous** —
they fall into five isolated runs:

| Run | Date range | Length (quarters) |
|---|---|---|
| A | 2015-08 → 2016-01 | 3 |
| B | 2016-05 → 2017-01 | 4 |
| C | 2017-05 → 2017-08 | 2 |
| **D** | **2018-05 → 2020-01** | **13** |
| E | 2020-08 → 2021-01 | 3 |

With a 6-quarter lookback window, only Run D exceeds the minimum length
needed to contribute even one training sequence, yielding just **7**
usable sequences project-wide. An initial training attempt at this sample
size — with spatial-patch augmentation applied to multiply gradient
updates per real sequence — produced a held-out R² of **−0.0655** (worse
than predicting the mean), with training loss continuing to fall while
validation loss rose after roughly the midpoint of training, a textbook
overfitting signature at this sample size (full training curve:
`validation_methodology.md`, Section 14.4).

### 5.3 Root Cause, Diagnosed Directly (Not Assumed)

**Channel-level attrition.** Among the 25 nominally "usable" dates,
`sentinel2_ndvi` is the dominant and most erratic source of missingness,
ranging from 10% to **100%** NaN on individual dates — far exceeding any
other channel, and consistent with the cloud-cover-driven Sentinel-2 gaps
already documented in Section 4 of this document. **This is one of the
independent findings that later motivated dropping Sentinel-2 NDVI
entirely from the v2 Phase 2 feature set (Section 1B.7,
`validation_methodology.md` Section 12.2).**

**Gap-level attrition.** The breaks between runs correspond to
identifiable, largely independent causes rather than one systemic
failure:

| Gap | Duration | Diagnosed cause |
|---|---|---|
| 2013-01 → 2015-05 | ~2.5 years | Sentinel-2A launched June 2015 — imagery genuinely does not exist beforehand |
| 2017-11 → 2018-01 | ~2–3 months | Plausibly linked to the documented GRACE mission transition gap (Jul 2017–Apr 2018, Section 4) |
| 2020-01 → 2020-08 | ~7 months | Plausibly COVID-19 field-campaign disruption — **not independently confirmed** — stated as plausible, not verified |

This rules out a single, fixable data-pull bug as the explanation: the
constraint is the intersection of several independent, mostly-real-world
gaps (satellite launch timing, a documented mission transition, and a
plausible pandemic disruption), not a preprocessing error concentrated in
one place.

### 5.4 Practical Implication

A full-grid spatiotemporal deep learning approach is **not viable at the
currently available covariate density and time range**, regardless of
architecture, hidden-size, or hyperparameter choices — the ceiling is set
by data availability, not model capacity. This is a scope boundary of the
current data pull and should be stated as such in the final report, not
framed as a negative result about ConvLSTMs as a technique in general.

The comparatively minor feature importance of `sentinel2_ndvi` in the
v1 Phase 2 RF model (a smaller contributor than `grace`, `target_lag1`,
`month_sin`, and `chirps_roll3`) suggests that even a full recovery of
NDVI coverage would not, by itself, have been the deciding factor in this
model's viability; the genuine multi-month structural gaps (satellite
launch timing, mission transition, plausible pandemic disruption) are the
binding constraint, not per-date sensor noise. **v2 has since removed
Sentinel-2 from Phase 2 entirely on this same basis (Section 1B.7).**

### 5.5 Not Pursued as a Further Fix (Quantified)

Shrinking the sequence lookback window was considered as a way to admit
the shorter runs (A, B, C, E) into the training set, and was rejected on
quantitative grounds rather than by assumption:

| `seq_len` | Total sequences (all runs combined) |
|---|---|
| 6 (used) | 7 |
| 5 | 8 |
| 4 | 9 |
| 3 | 11 |
| 2 | 15 |

Even the most aggressive reduction evaluated (`seq_len=2`) — which would
also undercut the entire rationale for using a sequence model over RF's
existing single-lag feature — tops out at an estimated 15 sequences,
still one to two orders of magnitude below what spatiotemporal deep
learning typically requires for reliable generalization. Extending the
historical covariate pull was also considered and rejected: the binding
constraint is *contiguous* coverage, not total date count, and the
existing pull already spans the full CGWB data-availability window
(Section 2 of the methodology document). Both alternatives are recorded
here as legitimate directions ruled out by calculation, not by
assumption, consistent with this document's Section 1 precedent for
documenting rejected fixes explicitly rather than omitting them.

---

## 6. Point-Scale LSTM: Task-Scope and Baseline-Comparison Caveats

### 6.1 Finding

The revised, point-scale (per-well) LSTM achieved R²=0.575 against true
held-out CGWB wells, substantially exceeding the **v1** Phase 2 RF
baseline's R²=0.198 — this comparison predates the v2 Phase 2 rebuild and
has not been re-run against v2 (`validation_methodology.md`, Section 16.1
cross-reference note). **This headline number must not be reported
without the two caveats below** — both materially change how the result
should be interpreted, and omitting either would overstate what has
actually been demonstrated.

### 6.2 Caveat (a): Different Task Scope — Not a Direct Substitute for RF

RF predicts groundwater depth at **any** location, including places with
no monitoring history at all — this is the project's actual downscaling
objective, and remains necessary for producing spatial coverage maps
(e.g. for the advisory system planned in later phases). The point-scale
LSTM can only be applied at locations with an **existing reading
history**; it cannot answer "what is the depth at this unmonitored
point?" — only "given this well's own history, what will it read
next?"

**Implication:** the two models solve different problems and are
complementary, not competing. Reporting this result as "the LSTM beat RF"
without this qualification would misrepresent both what was measured and
what each model can actually be used for.

### 6.3 Caveat (b): Persistence, Not Raw R², Is the Valid Comparison

The LSTM's delta-target design (predicting change from the well's own
last true reading, rather than absolute depth — chosen specifically to
avoid the depth-compression failure mode diagnosed in Section 1 of this
document) means the model has access to the same anchor information a
trivial **persistence baseline** ("predict no change") would use. A
dedicated check (`persistence_baseline_check.py`, run on the identical
held-out sequences) was performed specifically to isolate how much of the
LSTM's apparent skill is attributable to groundwater's natural
quarter-to-quarter autocorrelation versus genuinely learned dynamics:

| Metric | Persistence baseline | Point-scale LSTM | Improvement |
|---|---|---|---|
| R² | 0.439 | 0.575 | +0.136 absolute (24.2% reduction in remaining/unexplained variance) |
| RMSE (m) | 4.283 | 3.726 | −13.0% |
| MAE (m) | 2.500 | 2.023 | −19.1% |
| N | 5,597 | 5,597 | (identical held-out sequences) |

**The defensible, reportable contribution of the model is this gap above
persistence — not the raw R²=0.575 figure taken alone**, which would
overstate the model's genuinely learned contribution if quoted without
this baseline.

### 6.4 Practical Implication for Downstream Use

If the point-scale LSTM's output is used as an input to a later phase
(e.g. a Phase 4+ advisory or RL system), it should be applied **only** at
locations with sufficient reading history (≥ `seq_len`+1 = 5 quarters, at
the current configuration), and its predictions should not be conflated
with RF's location-agnostic downscaled surface.

### 6.5 Not Pursued as Further Verification (Open)

- **Depth-stratified error analysis**, analogous to Section 1's
 stratified breakdown for RF — it has not been verified whether the
 delta-target design actually resolved the depth-dependent compression
 bias diagnosed in Section 1, or merely relocated it.
- **`seq_len` sensitivity** (2, 3, 5, 6 quarters) against the point-scale
 task specifically — not independently validated.

---

## 7. Point-Scale LSTM: Held-Out Coverage

### 7.1 Finding

Of 16,073 held-out CGWB readings, only 8,127 (50.5%) retained a usable
covariate sample when building the well-level feature table
(`build_well_level_dataset.py`) — the remaining 7,946 readings fell on
dates with no available covariate stack at all. Of the 710 held-out
wells, 609 (85.8%) had enough remaining readings (> `seq_len`=4) to
contribute at least one evaluation sequence; the other 101 wells were
excluded from Phase 3 evaluation entirely for having too short a
surviving reading history.

### 7.2 Root Cause

This is directly consistent with — and somewhat less severe than — the
per-date coverage attrition diagnosed in Section 5.2, since the
well-level pipeline tolerates a single missing channel per reading
(typically `sentinel2_ndvi`) rather than requiring simultaneous
completeness across the entire covariate stack the way the grid pipeline
does.

### 7.3 Practical Implication

This evaluated subset (609 of 710 held-out wells) should **not be assumed
representative of all held-out wells uniformly**. Any claim about the
point-scale LSTM's real-world applicability should be scoped to "wells
with sufficient covariate-period overlap," not silently generalized to
the full held-out set.

## 8. Point-Scale LSTM: Untested Design Choices (Carried Forward as Open Limitations)

- **`seq_len=4`** was adopted after the grid model's lookback requirements
 proved infeasible (Section 5.5), not selected via a sensitivity sweep.
- **`sentinel2_ndvi`** remains in the Phase 3 covariate set despite being
 the largest single source of missingness (a sensitivity run with this
 channel dropped has not been performed for Phase 3, even though Phase 2
 v2 has separately dropped it entirely).
- **The COVID-19 field-campaign disruption explanation** (Section 5.3) is
 stated as plausible based on timing alone and has not been verified.

## 9. Missing Hydrogeological Covariates (Geology, Soil Type, Land Use)

**Finding:** the covariate set used throughout this project includes
surface and near-surface signals (rainfall, soil moisture, vegetation,
terrain) and temporal persistence, but **no direct representation of
subsurface aquifer material** — lithology/geology, soil type, or land
use/land cover.

**Why this matters:** groundwater depth is fundamentally governed by the
rock/soil medium water moves through. Much of the study region likely
sits on Deccan Trap basalt, a fractured-rock aquifer system — a
distinction the current model has no way to represent.

**Update (Phase 2.6):** the soil type and land use/land cover portion of
this hypothesis has since been tested (against the v1 feature set) and
found not to improve training-only CV (Section 10). Lithology/geology
remains untested. **This test has also not been re-run against v2's
16-feature set** — see Section 10.

## 10. Soil Texture + LULC Covariates: No Improvement (Third Negative Result)

### 10.1 Finding

Soil texture class (OpenLandMap, 12 USDA classes) and land use/land cover
(ESA WorldCover, 11 classes), added as 23 one-hot static covariates
alongside the existing terrain features, produced **no measurable
improvement** in training-only, spatially-blocked cross-validation: best
candidate CV R² = 0.639, identical to the v1 baseline's best CV R² of
0.639. This is the **third** documented negative result for an RF
enhancement attempt, alongside the Phase 2.5 rainfall-deficit feature and
residual Kriging (both tested against v1).

### 10.2 Pre-Registration Discipline Followed

A pre-registered decision rule required CV R² to exceed baseline by more
than +0.02 before touching the held-out set. The actual improvement was
+0.000, so the held-out set was not touched for this experiment.

### 10.3 Verification Against a False-Negative Risk

New feature columns were confirmed present (`len(FEATURE_COLUMNS) == 35`)
before accepting the CV result, ruling out a silent wiring/import bug.

### 10.4 Interpretation

A real, verified negative result. Plausible (unconfirmed) explanations
include export-resolution mismatch, single-depth-slice/single-snapshot
limitations, and RF's depth-compression bias dominating held-out error.

### 10.5 What Remains Untested

**Lithology/geology** was not tested in this round. **This experiment was
also run only against the v1 feature set/CV baseline, not v2 — an open
item, added here and in `validation_methodology.md` Section 20.6.**

## 11. Phase 4 Synthetic Environment: Explicit Assumption Log

**Finding:** the offline RL dataset (Phase 4) is generated by a
**stylized bucket-model simulator**, not a calibrated hydrological model.
`synthetic_reward.py`'s own docstring states this plainly. Six numbered
assumptions govern the simulator's behavior:

- **A1.** `crop_water_demand` is a simple seasonal proxy, not derived
 from real crop-type maps, irrigation records, or FAO crop-coefficient
 tables.
- **A2.** `crop_revenue_proxy` assumes revenue rises with pumping only up
 to the point demand is met, then flattens; no real price/yield data
 backs the formula's magnitude.
- **A3.** `domestic_supply_score` assumes one fixed, location-independent
 critical depth for domestic-access failure. Real adequacy depends on
 well depth, pump capacity, and local infrastructure — none modeled.
- **A4.** `depletion_penalty` uses one fixed `sustainability_threshold_m`
 for the entire study region. Real sustainable yield varies by aquifer
 type and geology (Section 9's untested hydrogeology direction).
- **A5.** The action→`gw_level` relationship is a simple linear-plus-
 recharge model, not a calibrated aquifer response function.
- **A6.** Rainfall recharge is a fixed fraction of `recent_rainfall`, not
 a physically calibrated infiltration model (which would itself depend
 on the soil-type covariate this project's own Phase 2.6 test (Section
 10) found did not improve RF's spatial predictions).

**Update (session integration, this revision):** a behavioral validation
of A6's rainfall-recharge assumption against real CGWB observations has
since been performed and is recorded in `validation_methodology.md`
(new subsection under the expanded Phase 5 material). Summary: the
simulator's rainfall-response sign matches the sign observed in real CGWB
wells (both negative — more rain, shallower next reading), but the
simulated correlation magnitude differs from the observed one. This is
evidence the simulator's qualitative direction is not fabricated from
nothing, but it does not establish that A6's specific fixed-fraction
functional form or magnitude is calibrated — see the cross-referenced
section for the exact figures and the units caveat that limits how far
this comparison can be pushed.

**Practical implication:** every quantity these assumptions touch —
transitions, rewards, and therefore anything a Phase 5 policy learns from
them — inherits these simplifications. This is an accepted, explicitly
documented scope boundary for using synthetic data to generate an offline
RL dataset, not a defect to be fixed before proceeding. It should not be
described as physically calibrated in the final report.

## 12. Phase 4: RF Seed-Model — Version Is v1, Not v2 (Resolved)

**Finding.** Direct artifact inspection confirms `data/processed/rf_downscale_model.joblib` has `n_features_in_ == 11`, matching v1's exact feature list from `validation_methodology.md` §9; the v2 model `rf_v2_noS2_model.joblib` has `n_features_in_ == 16`. **Phase 4 is confirmed seeded from v1, not v2.**

**Consequence.** Phase 4's seed table (and every Phase-5 result) inherits v1's prediction behavior — including v1's depth-compression bias (Section 1) — not v2's. Whether re-seeding Phase 4 from v2 would materially change Phase-5 results remains untested and is a future-work item, not a defect. **This is now verified rather than inferred.**

## 13. Phase 4: Weak Rainfall→Recharge Correlation (r = −0.13)

**Finding:** an out-of-band Pearson correlation between rainfall and
subsequent simulated groundwater change, computed directly from the raw
offline-dataset arrays, came back at **r = −0.13** — the correct sign
(more rain → shallower next state), but weak.

**Practical implication:** the simulator's rainfall-driven recharge
signal is real but modest relative to whatever combination of pumping
effects and injected noise dominates the remaining variance. This is
consistent with, and quantifies, A6's own acknowledged crudeness. **See
also the update to Section 11 above and the corresponding new section of
`validation_methodology.md`: this internal simulator-to-simulator
correlation (r = −0.13, simulated rainfall vs. simulated next-state
change) is a distinct quantity from the simulator-vs-real-CGWB behavioral
comparison performed later, which should not be confused with this one.**

## 14. Phase 4: `crop_water_demand` Decoupled From Season After the First Step

**Finding:** `synthetic_reward.step()` updates `crop_water_demand` via
pure Gaussian noise around its previous value; it is never re-derived
from the calendar fields (`month_sin`/`month_cos`), even though those
fields correctly advance one quarter per step in the trajectory
generator. A1's seasonal demand pattern therefore only genuinely holds at
a trajectory's initial state — every subsequent quarter, demand
random-walks independent of season.

**Practical implication:** any learned policy relationship between
season and demand-driven reward is weaker across a trajectory's later
steps than its first. Not yet fixed. **The expanded Phase 5 report's
reward-weight ablation (Section 30 below) varies the reward *weights*
but does not report re-testing or fixing this decoupling — it remains an
inherited limitation of any policy in that report as well.**

## 15. Phase 4: Untested/Unverified Design Claims

- **Spatial-block partition equivalence.** `src/utils/spatial_block_cv.py`
 asserts it reuses "the same" 8×8 partition as `cv_tune.py`'s original
 `assign_spatial_block()`, but implements it with a different function
 signature. Whether the two produce identical block assignments has not
 been checked.
- **Policy-diversity verification.** A first-action distribution check
 across 200 episodes came back near-uniform, but this is a crude
 spot-check, not confirmation that per-episode policy labels exactly
 match the configured `n_trajectories_per_policy`.

---

## 16. Phase 5 (Original Report): Offline Dataset Scale

**Finding:** the CQL/BC experiments in Phase 5 are trained and evaluated
against a single offline dataset of 30,000 transitions across 1,500
episodes (Phase 4, seeded from the **v1** RF model — see Section 12
above). This is workable for a research prototype but is a small dataset
relative to the complexity of a real-world groundwater-pumping control
problem.

**Practical implication:** conclusions from Phase 5 — including the CQL
vs. BC comparison itself (Section 19 below) — should be read as evidence
about what this specific, modestly-sized synthetic dataset supports, not
as evidence of how either algorithm would behave with substantially more
data, or with real (non-synthetic) transitions. **Note: the expanded
Phase 5 report (Section 27 onward) reports the same dataset size (1,500
episodes / 30,000 transitions), so this scale limitation applies there
too.**

## 17. Phase 5 (Original Report): Action-Support Imbalance

**Finding:** the offline dataset's action distribution is heavily
skewed: 0% and 25% pumping together account for 60.8% of all transitions
(8,417 + 9,820 of 30,000), while 100% pumping — the least-represented
action — accounts for only 8.3% (2,478 transitions).

| Action | Pumping | Count | Share |
|---|---|---|---|
| 0 | 0% | 8,417 | 28.1% |
| 1 | 25% | 9,820 | 32.7% |
| 2 | 50% | 5,482 | 18.3% |
| 3 | 75% | 3,803 | 12.7% |
| 4 | 100% | 2,478 | 8.3% |

**Practical implication:** policy value estimates (via FQE) are least
trustworthy for the actions with the thinnest support — principally 75%
and 100% pumping. Even though CQL is specifically designed to penalize
out-of-distribution action selection, this does not eliminate
distributional extrapolation error; it only discourages the learned
policy from relying on poorly-supported actions. Any downstream use of
either policy's recommendations should weight recommendations toward
less-pumped actions with correspondingly more confidence than
recommendations toward heavier pumping. **This same imbalance applies to
the expanded Phase 5 report's dataset (same 30,000-transition dataset,
Section 27.1 below), and is the direct explanation for that report's
own finding that 100% pumping is the least-supported action in every
sensitivity/ablation result.**

## 18. Phase 5 (Original Report): Synthetic Reward and Environment — Inherited, Not New, Assumptions

**Finding:** Phase 5's learned policies are trained entirely against the
Phase 4 synthetic simulator and its reward function (`synthetic_reward.py`,
numbered assumptions A1–A6, Section 11 of this document). Phase 5
introduces no new physical modeling — it inherits every one of those
assumptions unchanged, including the fixed sustainability threshold (A4),
the uncalibrated linear-plus-recharge aquifer response (A5), the crude
fixed-fraction rainfall recharge (A6, independently quantified at
r = −0.13 in Section 13), and the post-first-step decoupling of
`crop_water_demand` from season (Section 14). It also inherits the v1
RF-seeding question flagged in Section 12.

**Practical implication, restated specifically for Phase 5:** whatever
BC and CQL "learned," they learned to optimize a **defined, synthetic
reward function** — not directly observed farmer welfare, measured crop
yield, electricity cost, or any other real economic or hydrological
outcome. Neither policy's recommendations should be read as validated
against real groundwater dynamics, real crop economics, or real
farmer behavior. This limitation propagates from the environment into
every Phase 5 result without exception, **including every result in the
expanded Phase 5 report (Section 27 onward) — its own §5.64.1 makes the
same point independently, in the same terms.**

## 19. Phase 5 (Original Report): Two Evaluation Methods, Different Conclusions (Supersedes Earlier Draft)

**This section replaces an earlier draft of Phase 5's headline limitation,
which reported only a single FQE-based bootstrap comparison with a CI
containing zero and described α=1.0 as "selected" from a sweep. That
earlier framing is incomplete: it omits the direct-rollout result (which
does show a significant CQL advantage on the ε configuration) and
mischaracterizes α=1.0 as sweep-selected when it was in fact a manual
override — the sweep's highest-scoring value was α=4.0. Do not cite the
single-bootstrap version.**

**Note added, this revision:** the expanded Phase 5 report (Section 27
onward) reports α=1.0 as the "canonical production" configuration and
states it was independently provenance-verified as α=1.0 across
checkpoints, without repeating this section's caveat about α=4.0 scoring
higher on the recorded sweep. This does not resolve the caveat — it
means the expanded report's provenance check confirms the checkpoints
*match their own stated config* (α=1.0), not that α=1.0 was the
best-performing value found. Both facts can be true simultaneously and
should both be stated if this material is cited.

**Finding:** the CQL-vs-BC comparison produced **different conclusions
under two different evaluation methods** run against the same trained
policies. Both are reported below; neither is treated as the sole
"headline" result.

### 19.1 Direct simulator rollout (ε configuration)

| Quantity | Value |
|---|---|
| Observed difference (CQL − BC) | **+0.1096** |
| 95% paired-bootstrap CI | **[+0.0498, +0.1742]** |
| Bootstrap seed | 2026 |
| Bootstrap replicates | 5,000 |
| CI contains zero? | **No** |
| Probability bootstrap difference > 0 | **1.0** |

Source: `reports/epsilon/policy_comparison.json`.

**Reading:** on the ε direct-rollout evaluation, CQL outperformed BC by
a statistically significant margin.

### 19.2 FQE estimate (ε configuration, same trained policies)

| Quantity | Value |
|---|---|
| Observed difference (CQL − BC) | −0.0018 |
| 95% bootstrap CI | [−0.0416, +0.0385] |
| CI contains zero? | **Yes** |

Source: `reports/fqe_epsilon.json`, `cql_minus_bc_fqe_bootstrap`.

**Reading:** FQE on the same trained policies returns a statistical tie.
The FQE bootstrap itself carries an explicit caveat that it resamples
across only 3 paired seed-level values, so its interval is illustrative
of seed variability, not a replacement for the per-episode bootstrap
used in direct rollout.

### 19.3 Direct simulator rollout (no_oracle variants)

| Run | CQL − BC | 95% CI | Contains zero? |
|---|---|---|---|
| no_oracle (α=4.0) | +0.0805 | [−0.1940, +0.3517] | Yes |
| no_oracle (α=1.0) | +0.1307 | [−0.1324, +0.3873] | Yes |
| no_oracle (α=10.0) | −0.0109 | [−0.2586, +0.2355] | Yes |

Sources: `reports/no_oracle*/policy_comparison.json`.

**Reading:** the no_oracle variants span positive and negative point
estimates, with all three confidence intervals containing zero. They do
not independently confirm the ε direct-rollout result.

### 19.4 Discrepancy and α-provenance caveat

- Direct rollout (ε): significant CQL advantage.
- FQE (ε): statistical tie.
- no_oracle variants (all three α): CIs contain zero.

**The two evaluation methods disagree in sign and significance on the ε
configuration, and this disagreement is unresolved.**

Additionally, the **α = 1.0** value used for the ε run was set manually
via a `--alpha` command-line override — **not selected from any recorded
sweep**. The α sweep recorded on disk
(`reports/cql_results_epsilon.json`, `alpha_sweep.grid_results`) reports:

| α | `init_state_value` (validation) |
|---|---|
| 0.5 | 1.1213 |
| 1.0 | 1.2548 |
| **4.0** | **1.3143** |
| 10.0 | 1.2957 |

The highest-scoring value in the recorded sweep is **α = 4.0, not 1.0**.
The sweep metric itself (`InitialStateValueEstimationEvaluator`) carries
a known structural bias toward low α (documented as `BUGFIX 2` in
`train_cql.py`), and auto-selection from it was disabled. Whether α = 1.0
was chosen before or after observing the ε direct-rollout result is not
documented in any retained artifact.

**Practical implication:** the current evidence **does not support a
single-method claim** that CQL outperforms BC on this task and dataset,
and does not support a claim that it does not. The defensible statement
is:

> *"On the ε configuration's direct simulator rollout, CQL outperformed
> BC by 0.1096 (95% CI [+0.050, +0.174]). FQE on the same trained
> policies returned a statistical tie (CQL−BC = −0.0018, 95% CI [−0.042,
> +0.039]). The two methods disagree, and the α = 1.0 configuration used
> for the ε run was set by manual override rather than chosen from a
> post-BUGFIX sweep. The ε result is therefore reported as exploratory
> pending α-provenance verification."*

Any report language describing Phase 5 as showing "CQL is the better
policy" or "CQL did not beat BC" would oversimplify the actual result
and should be corrected before publication. **The expanded Phase 5
report (Section 27 onward) does use exactly this kind of unqualified
language ("CQL provides a consistent simulated-return advantage... the
central Phase 5 finding") for its own, differently-scoped experiments —
readers should not conflate that report's conclusions with a resolution
of the discrepancy documented in this section, which concerns a
different evaluation methodology (FQE vs. direct rollout) that the
expanded report does not mention.**

### 19.4b — Phase-5 CQL-vs-BC CIs Not Reproducible Across Code Revisions

**Finding.** Running the current `compare_policies.py` against the same models and dataset does **not** reproduce the historical CQL−BC CIs:

| Config | Historical [95% CI] | Current [95% CI] |
|---|---|---|
| ε | +0.1096 [+0.0498, +0.1742] | **+0.3402 [+0.2433, +0.4407]** |
| no_oracle | +0.0805 [−0.1940, +0.3517] | **+0.5843 [+0.3236, +0.8534]** |
| no_oracle α=1 | +0.1307 [−0.1324, +0.3873] | **+0.9504 [+0.6976, +1.2181]** |
| no_oracle α=10 | −0.0109 [−0.2586, +0.2355] | **+0.5144 [+0.2676, +0.7783]** |
| base `models/` | *(no historical backup)* | **−1.1959 [−1.6651, −0.7465]** |

**Mechanism.** Localized to the observation standardization fed to learned policies:
- Hand-coded policy returns reproduce bit-identically (`Δ = 0.000000` across all configs) — ruling out any change to simulator, RNG stream, RF table, initial-state sampler, or the hand-coded policy implementations.
- `bc_baseline.py` was modified at **2026-09-10 13:31**, after the ε report (10:11) and after all no_oracle reports (09-09).
- `obs_mean[gw_level]` shifted from **15.075** (historical ε) / **15.025** (historical no_oracle) to **13.5857** (current, all configs). Historical ε and historical no_oracle also disagree with each other — the pre-fix code was itself unstable.
- `SPLIT_SEED = 1`, `split_episode_indices`, and `episodes_to_arrays` are unchanged in the current source; testing alternative split seeds (1, 2, 42, 9999) does not reproduce 15.07, so the shift is not a seed issue.

**Current code is deterministic:** repeat runs produce byte-identical CIs.

**Practical implication.** The four original Phase-5 CQL−BC CIs must be **marked superseded** wherever cited. **Post-fix, three of three no_oracle-family configs show a significant CQL advantage** (+0.51 to +0.95, across no_oracle, no_oracle α=1, and no_oracle α=10), materially strengthening CQL over the historical position in which only one of three was significant. The ε configuration shows a separate, smaller significant CQL advantage (+0.34) on a different dataset and is not part of the no_oracle family; it is reported separately in §19.1. The base `models/` pair shows the reverse, but a cross-combination diagnostic confirms this reflects a genuinely stronger base BC policy (mean 10.89 vs 8.28), not a CQL-algorithm failure.

**What cannot be recovered.** The specific code diff is not recoverable without git history. Preserving the historical JSONs as `.bak` files is the only protection against future silent drift. **Recommendation:** archive `bc_baseline.py` and `compare_policies.py` at their historical revisions (or init `git`).

### 19.5 [New, session integration] CQL's Advantage Does Not Extend Outside the Training-State Distribution

**Provenance of this section:** this is a new finding, established in a
working session subsequent to both the original and expanded Phase 5
reports summarized above and in Sections 27–36 below. It is distinct
from — and must not be conflated with — the expanded report's
scenario-shift/"OOD" experiment (Section 30 below), whose own OOD
fraction came back at 0.0 (i.e., it did not actually reach
out-of-distribution states). The diagnostic here was constructed
specifically to produce genuine state-distribution OOD, using a
four-band Mahalanobis-distance construction against the training state
distribution, so that in-distribution (ID) and progressively
further-out-of-distribution bands could be compared directly under the
same trained policies used elsewhere in Phase 5.

**Finding — action distributions:** across the four Mahalanobis bands,
the total-variation distance between the BC and CQL action distributions
remained roughly stable (TV ≈ 0.62–0.65) from the in-distribution band out
to the most severe out-of-distribution band tested. **This means the two
policies' qualitative disagreement about which action to take does not
itself widen much under OOD conditions** — the divergence documented
elsewhere (Sections 28/31 below) is present even in-distribution and
does not obviously worsen further out.

**Finding — realized return is negative in every band of the four-band
state-distance partition:** the same CQL − BC return comparison that
shows a **positive** CQL advantage in the expanded report's in-distribution
scenario and reward experiments (Sections 29/31/32 below) **is negative
in every band of the four-band state-distance partition, and widens
monotonically with distance from the training-state distribution**:

| Band | CQL − BC (return) |
|---|---|
| In-distribution (ID) | −16.01 |
| Severe OOD | −38.40 |

(Intermediate bands fall between these two values, widening
monotonically from ID to severe OOD in the negative direction.) A clean,
fixed 20-step rollout horizon was used specifically to rule out an
artifact where policies simply run for different effective lengths
before terminating or hitting a state-space boundary; the reversal
persists under this controlled horizon.

**Candidate explanations (not independently confirmed):** CQL's
conservatism term is trained to penalize actions poorly supported by the
offline dataset's *action* distribution (Section 17), not by its *state*
distribution — so CQL has no built-in mechanism protecting it once the
input state itself is unfamiliar, only once the chosen action is. A
plausible reading is that CQL's Q-function extrapolates unreliably in
unfamiliar state regions in a way that happens to be more costly than
BC's simpler, more direct imitation of the training data's action
choices — but this is offered as a plausible mechanism, not a verified
one; no ablation isolating this specific explanation has been run.

**Practical implication — this is a boundary condition on the entire
CQL-over-BC narrative, in both the original and expanded reports:** every
CQL advantage reported in this document and in Sections 27–36 below
(and every corresponding result in `validation_methodology.md`) should be
read as applying **within the training-state distribution**, not as a
general property of CQL relative to BC on this problem. Any final-report
or viva claim that "CQL provides a consistent return advantage" must be
qualified with this reversal, since it is the single most direct
counter-evidence to a general claim of CQL superiority. This finding
should be read alongside, not instead of, the expanded report's own
scenario-shift experiment (Section 30) — that experiment tested
robustness to shifted *transition/reward parameters* under conditions
that were not actually OOD in the state-distribution sense; this section
tests genuine state-distribution OOD directly and finds the opposite
qualitative result.

**Open items (end-of-session integration).** ~~(i) no ablation has isolated whether the reversal is driven specifically by the QR critic formulation~~ — **resolved**: the four-band diagnostic reproduces under a Mean-Q critic with the same sign and monotonic gradient:

| Band | QR | Mean-Q |
|---|---|---|
| in-distribution | −16.01 | −8.64 |
| mild | — | −13.63 |
| moderate | — | −15.76 |
| severe OOD | −38.40 | −18.17 |

Both critics show CQL−BC negative in every band. **The reversal is not QR-specific.** Magnitude roughly halves under Mean-Q — effect size is critic-dependent, qualitative finding is critic-independent. A per-step termination diagnostic (20.00 mean steps, 0.000 terminated fraction across all 24 cells) rules out an early-termination confound. Remaining open items: (ii) the four Mahalanobis bands have not been cross-referenced against which specific `SimState` fields drive membership in the most severe band; (iii) the diagnostic has been run once per critic formulation, on the same three seeds as every other Phase-5 robustness check.

### 19.6 [New] Reward-Weight Ablation Under Retraining — One Non-Significant Cell

**Provenance.** This is an addendum to the reward-weight ablation
originally reported in `validation_methodology.md` §47. The original
froze the canonical CQL policy and re-scored it under seven weight
vectors; the addendum retrains BC and CQL from scratch under each weight
vector. Both are documented; the retrained version is the stronger test.

**Finding.** Under retraining, six of seven reward-weight configurations
show a statistically significant CQL advantage over BC (paired bootstrap
95% CI excludes zero):

| (w1, w2, w3) | CQL − BC | 95% CI | Sig? |
|---|---:|---|:---:|
| (1, 0, 0) | +0.069 | [+0.051, +0.087] | ✓ |
| (1, 2, 0) | +0.302 | [+0.157, +0.450] | ✓ |
| (0, 2, 0.5) | +0.886 | [+0.694, +1.088] | ✓ |
| (1, 0, 0.5) | +0.031 | [+0.015, +0.047] | ✓ |
| (1, 1, 0.5) | +0.062 | **[−0.010, +0.133]** | **✗** |
| (1, 4, 0.5) | +1.470 | [+1.128, +1.828] | ✓ |
| (1, 2, 0.5) canonical | +0.345 | [+0.185, +0.507] | ✓ |

**The one exception is `(1, 1, 0.5)`**, where the depletion-penalty weight
is halved relative to canonical and the CQL advantage is not statistically
robust when the policy is retrained against that reward.

**Practical implication.** The paper should not claim "CQL outperformed
BC under all tested reward weightings" without qualification. The correct
statement is: "CQL's advantage holds under six of seven tested reward
weightings when the policy is retrained; the exception is `(1, 1, 0.5)`,
where the advantage is not statistically robust." Whether this reflects a
genuine boundary at intermediate penalty weights or is specific to the
three seeds tested is not established — a finer sweep around w2=1.0 would
be needed to distinguish.

**Two provenance caveats.** (i) The per-config `training_manifest.json`
does **not** carry `(w1, w2, w3)` as a first-class field; the reward vector
is recorded indirectly via the reward-conditioned dataset name and its
`dataset_sha1`. Six of the seven configs have distinct dataset hashes; the
canonical `(1, 2, 0.5)` config reuses the shared canonical dataset
`0ba217a2...`. (ii) The ablation was run at the code's default CQL learning
rate (6.25×10⁻⁵), not the canonical Phase-5 override at 3×10⁻⁴ — the
absolute numbers are not directly comparable to the scenario evaluations
in `validation_methodology.md` §45. The internal CQL-vs-BC comparison
within each cell is unaffected.

### 19.7 [New] Phase-2 RF Error Propagation Through BC and CQL

**Finding.** RF error injection into the RL state's `gw_level` (sampled
from the held-out error distribution: mean −0.04 m, std 4.99 m,
p05/p95 = −8.5/+5.4 m, n=13,989) produces return changes of less than 2
units on a 20-step horizon. Signs vary across seeds; no systematic
difference between BC and CQL:

| Policy | Seed | Clean | Noisy | Δ |
|---|---:|---:|---:|---:|
| BC | 42 | −0.089 | −0.159 | +0.070 |
| BC | 123 | +9.953 | +10.026 | −0.074 |
| BC | 2024 | +3.673 | +3.356 | +0.318 |
| CQL | 42 | +8.990 | +8.916 | +0.073 |
| CQL | 123 | −30.189 | −28.366 | −1.823 |
| CQL | 2024 | −0.322 | −0.053 | −0.269 |

**Interpretation.** At the evaluated 20-step horizon, policy return is not
primarily limited by RF state error. BC's mean degradation (+0.105) and
CQL's mean degradation (−0.673, i.e. net improvement under noise) are both
small, and the CQL sign is driven entirely by seed 123, whose clean return
is anomalously bad. **No statistically robust BC-vs-CQL difference in RF-
error sensitivity is demonstrated by this experiment.**

**Limitations.** (i) The injected error distribution is the RF's typical
held-out error; the RF's depth-dependent worst-case error (−60 m at 50+ m)
is outside the practical range of the sampled pool, so this test does not
bear on deep-well propagation. (ii) The horizon is 20 steps; longer
horizons would allow the injected error to compound. (iii) The RF error
is injected only into the observation channel; the transition dynamics
themselves are unchanged, so the test measures observation-noise
sensitivity, not simulator-mismatch sensitivity.

## 20. Phase 5 (Original Report): Offline Training, Synthetic-Simulator Evaluation

**Finding:** all Phase 5 training (BC, CQL, α sweep, FQE) was computed
offline against the static Phase 4 dataset. Policy evaluation for
CQL-vs-BC was performed by **two methods**:

1. **FQE** — offline critic-based value estimate, no simulator
 interaction.
2. **Direct simulator rollout** — the trained policy was executed inside
 the controlled Phase 4/5 synthetic simulator and evaluated by realized
 return.

**No real-world pumping intervention occurred.** No actual aquifer
manipulation occurred. All rollout evaluation is against the synthetic
simulator that generated the Phase 4 dataset.

**Practical implication:** Phase 5 establishes offline-RL feasibility and
an evaluation methodology that includes both critic-based and
simulator-rollout measurements on the same trained policies. It does
**not** establish — and should not be described as establishing —
real-world pumping-policy superiority, real-world aquifer improvement,
deployment safety, or long-term real-world sustainability outcomes for
either the BC or CQL policy. Direct simulator rollout measures what the
policy does inside a stylized, uncalibrated environment (Section 18),
which is a simulator result, not a real-world one. **This applies
identically to the expanded Phase 5 report's simulator-based scenario,
reward-ablation, and adaptation experiments (Sections 27–34 below), and
to the new OOD diagnostic in Section 19.5 above.**

## 21. Phase 5 (Original Report): Both Evaluation Methods Have Independent Uncertainty

**Finding:** the two policy-value estimation methods used in Phase 5
each carry their own source of uncertainty:

- **FQE** is a learned critic-based estimate. Its approximation error is
 a source of uncertainty that compounds with, rather than substitutes
 for, the bootstrap interval it reports.
- **Direct simulator rollout** uses the Phase 4 reward function to define
 its return. It is a simulator result, and its validity is bounded by
 the reward function's assumptions (Section 18) — not by the accuracy
 of any learned estimator.

Additionally, FQE's ε bootstrap interval is computed on **3 paired
seed-level values** (seeds 42, 123, 2024), which is a very small sample
for a bootstrap; the file itself carries a caveat stating this. Direct
rollout's bootstrap (from `compare_policies.py`) resamples at the
episode level, which provides tighter intervals but only against the
same simulator reward.

**Practical implication:** the two evaluation methods are not
interchangeable instruments measuring the same quantity. When they
disagree (Section 19), that disagreement is a real finding about the
limitations of current offline-policy evaluation on this problem, not a
tiebreak to be resolved by picking the more convenient method.

## 22. Phase 5 (Original Report): Convergence Not Conclusively Established

**Finding:** an extended training/convergence investigation, including a
diagnostic comparison of target-update intervals (500, 2,000, 4,000,
8,000 — the last of which was retained as the final configuration),
showed training curves moving toward convergence but did **not** produce
a clean, unequivocal convergence advantage for CQL over BC.

**Related diagnostic from the same project (Phase 5 stability ablation,
reported separately in the offline-RL record):** QR critics at
target-update-interval = 500 exhibited severe TD-loss escalation (4.4 →
30.0 over 4 epochs across three seeds), while mean-Q critics remained
stable at both 500 and 8,000. This ablation is the reason the 8,000
default was not replaced with a lower value.

**Practical implication:** neither algorithm's training should be
described as demonstrably fully converged to an optimal solution under
the current schedule. The 8,000-step target-update interval was retained
because nothing examined overturned it — this is a weaker claim than
having shown it to be optimal via the same explicit best-of-N sweep
procedure used for α (Section 35 of `validation_methodology.md`), and
should not be conflated with that stronger standard when reported.

## 23. Phase 5 (Original Report): Robustness Evidence Is Limited to Three Seeds

**Finding:** multi-seed diagnostics were run across three seeds (42, 123,
2024) to check whether the BC/CQL results were dependent on a single
random initialization.

**Practical implication:** this constitutes robustness *evidence*, not
proof of universal generalization across arbitrary seeds, dataset
resamples, or environment configurations. Individual per-seed results
have not yet been transcribed into `validation_methodology.md` (see that
document's Section 42, open items) — until they are, the multi-seed
check should be cited only as "robustness diagnostics were performed,"
not as a quantified stability guarantee. In particular, three seeds is
not sufficient to make confident bootstrap-coverage claims; the FQE
bootstrap caveat in Section 19.2 already notes this. **The expanded
Phase 5 report also uses only three seeds (42, 123, 2024) throughout —
the same limitation applies to every seed-level claim in Section 27
onward below, and to the new OOD diagnostic in Section 19.5 above.**

## 24. Phase 5 (Original Report): Limited Algorithmic Scope

**Finding:** Phase 5 compares exactly two algorithms — Behavioral Cloning
and Conservative Q-Learning — against a single offline dataset.

**Practical implication:** this does not constitute a broad benchmark
against the wider offline-RL literature. In particular, no comparison was
made against other established offline-RL methods (e.g. IQL, TD3+BC,
AWAC, Decision Transformer, BCQ). Any claim that offline RL "works well"
for this problem, or that CQL-style conservatism is the right approach in
general, should be scoped to "relative to BC on this dataset" — the
comparison performed — not generalized to offline RL as a field. **This
scope limitation is unchanged by the expanded Phase 5 report, which also
compares only BC and CQL (plus a QR-vs-Mean-Q critic-formulation
ablation within CQL itself, Section 31 below — not a new algorithm).**

## 25. Phase 5 (Original Report): Unresolved Numeric Discrepancies

**Finding — resolved (end-of-session integration).** Direct read of `data/processed/offline_rl_dataset.h5`:

- **Reward range:** min −8.6625, max +1.5000, mean 0.5095, std 0.6988. `validation_methodology.md` §27.1 was correct; the ingestion summary's ≈±8 was a transcription error.
- **State range:** `gw_level` ∈ [1.81, 46.23], mean 13.35 (not the −100 to +100 reported at ingestion). `recent_rainfall` ∈ [0, 313.68]; `crop_water_demand` ∈ [0, 1]; `month_sin`/`month_cos` ∈ [−1, 1]; `extraction_rate` ∈ [0.0002, 0.9986]. All within physically plausible bounds for CGWB monitoring wells.

Both figures should be treated as final; no further re-read is needed.

## 26. Phase 5 (Original Report): Crop-Water-Demand Coupling Limitation (Carried Forward from Phase 4)

**Finding, restated for Phase 5's context:** because `crop_water_demand`
decouples from the calendar after a trajectory's first simulated step
(Section 14 of this document; `validation_methodology.md`, Section 24.2),
any policy trained on this dataset — including both BC and CQL — is
learning against a reward signal whose seasonal-demand component is only
genuinely present at trajectory initialization and effectively random
thereafter.

**Practical implication:** claims about either learned policy's ability
to respond appropriately to seasonal irrigation demand over a multi-step
horizon should be treated with corresponding caution — the environment
itself does not consistently represent that seasonal signal beyond the
first step of any given episode. This is an inherited environment
limitation, not a new failure introduced by the BC or CQL training
procedures themselves. **This applies without modification to the
expanded Phase 5 report's scenario/reward-ablation results below, since
that report reuses the same simulator (its own §5.64.1 confirms the
simulator was not changed).**

---

# Phase 5 (Expanded Report): Additional Robustness, Ablation, and Provenance Findings

**Provenance of this section:** the material in Sections 27–36 below is
transcribed from a separate, later, substantially larger Phase 5 report
covering nine-cell scenario robustness, a scenario-shift/OOD experiment,
a seven-configuration reward-weight ablation, an RF-perturbation
sensitivity check, a QR-vs-Mean-Q critic ablation, a frozen-vs-adapted
policy comparison, and provenance/leakage/CI audits of the archived
checkpoints. **As flagged in the reconciliation note at the top of this
document, the relationship between this material and Sections 16–26
above is not established** — this report does not reference the
FQE-vs-rollout disagreement (Section 19), reports a different canonical
configuration (α=1.0 with a 32-quantile QR critic, 20,000 training
steps), and reports substantially different summary statistics (e.g.
mean test action-match ≈0.698, vs. no action-match metric reported in
Sections 16–26). Both bodies of evidence are retained below. **Section
19.5 above is a separate, later addition again, and applies to both
bodies of evidence.**

## 27. Phase 5 (Expanded): Scope and Environment Reconfirmation

**Finding:** the expanded report reconfirms the same six-dimensional
observation space, five discrete pumping actions (0/25/50/75/100%), and
depth-below-ground sign convention (larger = deeper = worse) documented
in Section 22 of `validation_methodology.md`, and states explicitly that
the RL environment is simulated rather than a real-world intervention
environment.

**Practical implication:** none of the findings in this expanded report
establish real-world intervention effectiveness — the report's own
language is that it evaluates "policy-learning performance within the
constructed environment," which is consistent with, and does not
strengthen, this document's existing Section 18/20/24 caveats about the
synthetic reward and simulator.

## 27.1 Offline Dataset (Expanded Report's Description)

**Finding:** the expanded report describes the same offline dataset
scale as the original report — 1,500 episodes, 30,000 transitions, 500
episodes each of random/greedy-extraction/conservative behavior policy,
6-dimensional observations, 5 discrete actions — plus a "61,913 groundwater-grid prediction rows" figure as the RF prediction source.

**Resolved (end-of-session integration).** The 61,913-row figure matches `data/processed/rf_training_table.csv` exactly (61,913 rows), which matches `reports/rf_grid_predictions.csv` (61,913 rows). The training table's columns are v1's 11 features with no `row_norm`/`col_norm` and no `*_age_days` columns — consistent with confirmed v1 seeding (Section 12). This also explains `validation_methodology.md` §51.3's `*_age_days` mismatch: the Phase-5 RF table is v1.

## 28. Phase 5 (Expanded): BC and CQL Action-Match Performance

**Finding:** across three seeds, both BC and CQL reached a mean test
action-match (i.e., agreement with the dataset's own behavior-policy
actions) of approximately 0.698 (BC: 0.6982; CQL: 0.6980) — effectively
tied.

**Practical implication:** the expanded report itself draws the correct
conclusion here — that CQL's downstream return advantage (Sections 29–30
below) is not explained by better imitation of the training data's
action distribution, since the two methods imitate that distribution
about equally well. This is a useful, well-supported negative finding
and is retained as such. **This return advantage is itself bounded by
the new Section 19.5 finding above — it holds in-distribution and
reverses under genuine OOD.**

## 29. Phase 5 (Expanded): Nine-Cell Scenario Robustness — Caveats

**Finding:** a 3×3 grid (groundwater depth 5/15/25 m × normal/drought/
high-rainfall condition) reported CQL outperforming BC in all nine
cells, with the CQL advantage growing sharply with depth (~+0.16 at 5 m
to ~+20.3 at 25 m).

**Caveats that must accompany this finding if cited:**

- **Simulator-internal result only.** As with every other Phase 5 result
 in this document (Section 20), this is a synthetic-simulator return
 comparison, not a real-world or even an independently-reproduced
 groundwater outcome. The physical interpretation offered for why the
 gap widens at depth (conservative policies avoid compounding
 depletion penalties) is plausible given the reward structure (A4's
 fixed depletion threshold, Section 11) but is a post-hoc explanation
 of simulator behavior, not an independently verified physical

- **Parameter sensitivity (resolved, end-of-session integration).** A 3×3 A4/A5 sweep (`reports/a4a5_sensitivity.json`) shows the 25 m CQL−BC gap varies **+0.35 to +79.3** across the parameter grid — a **226× spread**. The canonical cell (+19.72 [+17.25, +22.27] at A4=15, A5=2) reproduces the §45 headline of +20.33 to within 0.6 units. The scenario factor contributes **less than 0.5 return units** within any cell. CQL−BC is positive and CI-significant in all 27 cells tested, so the qualitative finding survives; the single-number "+20.3" is a canonical-parameter value, not a stable physical magnitude. **See `validation_methodology.md` §56 for full results.** The 226× spread is itself a finding about how strongly these two uncalibrated assumption constants interact with the deep-depletion regime.

## 30. Phase 5 (Expanded): Scenario-Shift Experiment Is Not Genuine OOD

**Finding, stated plainly by the expanded report itself and retained
here as a limitation:** the scenario-shift/OOD experiment (policies
trained on normal-condition data only, evaluated under drought/extreme
conditions) reported an **OOD fraction of 0.0** for both BC and CQL
across all evaluated cells.

**Practical implication:** despite being labeled an "OOD" or
"scenario-shift" experiment, it does not establish genuine
state-distribution out-of-distribution generalization — the states
encountered under the shifted conditions were not, in fact, outside the
training distribution. The correct, narrower claim is robustness to
scenario-level transition/reward-parameter shifts, not OOD
generalization. Any report language should use this narrower framing
explicitly rather than the word "OOD" unqualified. **The genuinely
OOD diagnostic that this experiment's name suggests, but does not
perform, is reported separately above as Section 19.5 — and reaches the
opposite qualitative conclusion (CQL advantage reverses) from this
experiment's finding of small, uniformly positive degradation under
non-OOD shifts. The two should never be cited interchangeably.**

## 31. Phase 5 (Expanded): Reward-Weight Ablation — Scope Caveat

**Finding:** CQL outperformed BC across seven tested reward-weight
configurations (varying w1/w2/w3), with all seven pooled 95% CIs
excluding zero.

**Practical implication:** this shows the CQL-over-BC advantage (within
this simulator) is not an artifact of one specific weight choice. It
does **not** show robustness to changes in the reward function's
*structure* — every configuration still uses A1–A6's underlying
functional forms (Section 11 of this document): the same fixed
sustainability threshold, the same linear-plus-recharge dynamics, and
the same seasonally-decoupled demand process after the first step
(Section 14). Varying scalar weights on a fixed structural form is a
narrower robustness claim than varying the reward function's structure
itself, and should not be described as testing "the reward function" in
general. **Nor does this experiment test robustness to state-distribution
shift — all seven configurations are still evaluated in-distribution;
see Section 19.5.**

## 32. Phase 5 (Expanded): RF-Perturbation Sensitivity — Return vs. Action Stability

**Finding:** CQL showed a lower action-flip rate than BC at all six
tested perturbation levels of the RF-derived groundwater state input.
However, the expanded report's own analysis notes the return response
was asymmetric — at a −2.0 m perturbation, BC's return changed by
roughly +1.0 while CQL's changed by roughly −1.4.

**Practical implication:** lower *action*-flip sensitivity to upstream
RF prediction error does not imply lower *return* sensitivity — the two
should not be conflated. Given that the upstream RF model (v1, per
Section 12 above) has a documented depth-dependent compression bias of
tens of metres at deeper wells (Section 1), and this sensitivity check
only tested perturbations of ±0.5 to ±2.0 m, the tested perturbation
range is small relative to the RF model's own documented worst-case
error at depth. This sensitivity result should not be read as bounding
CQL's or BC's robustness to the RF model's actual error distribution,
only to the smaller synthetic perturbations tested. **This asymmetric,
CQL-unfavorable return response at even a modest −2.0 m perturbation is
directionally consistent with, though far smaller in magnitude than, the
much larger CQL-unfavorable reversal found under genuine OOD states
(Section 19.5) — both point toward CQL's return advantage being fragile
outside a narrow, well-supported region of state space.**

## 33. Phase 5 (Expanded): QR vs. Mean-Q Critic — Not Established as Superior

**Finding:** the canonical CQL configuration uses a 32-quantile QR critic
(confirmed by direct checkpoint inspection: `QRQFunctionFactory
(n_quantiles=32)`). A dedicated ablation against a Mean-Q critic found QR
significantly outperformed Mean-Q on 2 of 3 seeds; on the third seed
(42), Mean-Q was numerically higher but the CI crossed zero.

**Practical implication:** the report's own conclusion — that QR
superiority is "promising but seed-sensitive" and not universally
established — is the correct, defensible framing and is retained
verbatim in spirit here. Given that all of Phase 5's robustness evidence
(both reports) is limited to exactly these same three seeds (Section 23
above), the QR-vs-Mean-Q result should be treated with the same
seed-count caution as every other three-seed claim in this document.
**The OOD diagnostic in Section 19.5 has since been repeated under a
Mean-Q critic (see the Mean-Q replication table there), and both
critics show the same negative sign and monotonic gradient — the
critic-formulation caveat that applied when §19.5 was first added
is now closed.**

## 34. Phase 5 (Expanded): Scenario-Specific Adaptation Can Be Harmful

**Finding:** a frozen (normal-condition-trained) policy substantially
outperformed a scenario-specifically-retrained policy at 25 m
groundwater depth, across all three environmental conditions tested
(frozen-minus-adapted gap of roughly +10.1 to +10.4 return units).

**Practical implication:** this is a genuine, useful negative finding —
naive retraining on scenario-specific data is not guaranteed to improve
an offline-RL groundwater policy, and may substantially harm it,
plausibly through overfitting to a narrower data slice at the deep-
depletion end of the state space where the dataset's action support is
already thinnest (Section 17 above: 100% pumping is only 8.3% of
transitions, and deep-depletion states plausibly co-occur with heavier
extraction). The report is correct to scope this finding to the specific
adaptation procedure tested, not to domain adaptation, transfer
learning, or meta-RL methods in general.

## 35. Phase 5 (Expanded): Provenance, Leakage, and CI Audits — What Was and Was Not Verified

### 35.1 Provenance audit — genuinely verified, but bounded

**Finding:** a corrected, recursive (`rglob`-based) checkpoint audit
covered 114/114 archived `.d3` checkpoints across 40/40 expected
(group, algorithm) coverage cells, verifying CQL α (1.0), CQL learning
rate (3e-4), BC learning rate (1e-3), training steps (20,000), and
target-update interval (8,000) against each checkpoint's live-loaded
configuration, with zero mismatches found.

**Practical implication:** this is a real and useful configuration-
consistency check — it confirms the archived checkpoints match their
claimed hyperparameters. It does **not** verify that those hyperparameter
choices were themselves well-justified (see Section 19's α-provenance
caveat above, which this audit does not address) or that the checkpoints
correspond to a specific, documented RF seed version (Section 27.1
above, unresolved) or dataset split seed independent of the training
manifest's own self-report.

### 35.2 Checkpoint-internal seed — not verifiable

**Finding:** the d3rlpy checkpoint format does not store the training
random seed as an inspectable attribute; direct inspection of loaded
algorithm objects for seed/RNG fields returned an empty result. Seed
provenance for the 114 checkpoints instead relies entirely on filename
conventions (e.g. `cql_seed42.d3`) and training manifests where present.

**Practical implication:** any claim of the form "checkpoint X was
trained with seed Y" is only as reliable as the filename/manifest
bookkeeping, not an independently, cryptographically verified fact. This
is an archival limitation of the artifact format, not a specific error
found, and should be reported as such rather than omitted.

### 35.3 Leakage verification — 4 of 6 checks, not 6 of 6

**Finding:** of six structural leakage checks, two could not be executed
as designed: the feature-age check was not applicable (no `*_age_days`
columns exist in the RF training table used for this dataset — note this
is inconsistent with `validation_methodology.md` Section 9's v2 feature
set, which *does* include `lag_age_days`/`chirps_age_days`/etc., implying
this Phase 5 dataset's RF training table predates or otherwise does not
match v2 — a further, new data point relevant to the open Section 12/27.1
RF-version question), and the held-out-ID hash comparison could not run
because the required hash manifest was never generated. The remaining
four checks passed, including a source-level inspection confirming
`target_lag1` is constructed from a strictly earlier calendar-month
Kriged surface (consistent with the leakage safeguard already documented
in `validation_methodology.md` Section 9, and independently re-verified
at the source-code level — see `validation_methodology.md`'s note on
`find_nearest_prior_kriged` under the same section).

**Upgraded to 5/6 (end-of-session integration).** A well-level hash manifest now exists at `data/held_out_wells/held_out_ids.manifest.json` (SHA1 file-hash `029865a1…`, well-level SHA256 `5c69b622…`, n=710). With `--manifest-for-hash-check` pointing at it, CHECK 6 flips from SKIP to PASS. **The correct summary is 5 of 6 verified, 1 not applicable** (feature-age check, structurally blocked by the v1 training table's lack of `*_age_days` columns).

**Tool reporting defect.** `check_leakage.py` prints "6/6 checks passed" when 1 is a SKIP — skip branches return `True` and are counted as PASS. The correct reading is **5 PASS / 1 SKIP**.

### 35.4 CI internal-consistency audit — verified; independent recomputation not possible

**Finding:** an automated CI-consistency tool reported 24 PASS / 0 FAIL /
6 SKIP across per-seed identity checks, pooled-summary consistency,
CI-ordering, zero-containment, and sign-consistency checks. The 6 skips
were all `recompute_ci` checks, skipped because no raw per-episode return
arrays (`*returns*.npy`/`.npz`) were found in the archived report
directory.

**Resolved for the original Phase-5 report (end-of-session integration).** Per-episode return arrays are archived at `reports/<config>/returns/` for ε, no_oracle, no_oracle_alpha1, no_oracle_alpha10, and root. **However, independent recomputation from those arrays produces the current numbers, not the historical ones** — see Section 19.4b. The historical numbers are preserved only in `.bak` files. The `paper_exp/` expanded-report configurations are archived as well — 216 array files across 18 paper_exp configurations (end-of-session integration).

### 35.5 Freeze manifest is post-hoc

**Finding:** a 280-file freeze manifest was generated after the
experiments described in this section, not contemporaneously with model
training.

**Practical implication:** it should be described as a post-hoc
reproducibility snapshot, not a training-time immutable freeze — it
documents what currently exists on disk, not a cryptographic guarantee
that nothing was altered between training and archiving.

## 36. Phase 5 (Expanded): Summary of What Remains Open

- **Relationship to the original Phase-5 report — resolved (end-of-session integration).** The expanded report is a **separate training effort**: different report directory (`reports/paper_exp` vs `reports/epsilon`+`no_oracle*`), different underlying datasets (`dataset_normal_train.h5`, `dataset_adapt_*.h5`, `dataset_w*.h5` vs the original H5s), different dates (13 Sep 2026 vs 09–10 Sep 2026). The expanded report never references FQE, ε/no_oracle naming, or the +0.1096/−0.0018 figures. **Both bodies of evidence should be cited separately, each with its own caveats.** The weights-ablation runs in the expanded report use `learning_rate = 6.25e-05` (the code default), not the canonical `3e-04`, consistent with `validation_methodology.md` §47's training-regime caveat.

### 36.7 Session completion record

Completed in the end-of-session working session:

- **RF seed provenance** — v1 confirmed from artifact (Section 12).
- **Reward/state ranges** — resolved (Section 25).
- **61,913-row figure** — reconciled to v1 training table (Section 27.1).
- **Held-out-ID manifest** — created; leakage audit upgraded to 5/6 (Section 35.3).
- **Raw return arrays** — archived for the four original Phase-5 configs and all 18 `paper_exp/` configurations (216 array files total; Section 35.4).
- **Standardization drift** — diagnosed and documented (Section 19.4b, new).
- **OOD reversal under Mean-Q** — replicates; open item (i) closed (Section 19.5).
- **A4/A5 sensitivity** — measured (Section 29).
- **Phase-5 reconciliation** — original and expanded reports are separate efforts (Section 36).
- **3-seed CQL−BC per-seed values** — transcribed into `validation_methodology.md` §40.
- **Three-policy baseline numbers** — transcribed into §33.1.
- **Reward-component trade-off decomposition** — documented in §1.5.
- **Reward-weight ablation under retraining** — documented in §19.6.
- **Phase-2 error propagation** — documented in §19.7.
- **RF depth diagnostic v2** — closed in `validation_methodology.md` §12a.

**Remaining open items:** none outstanding.

Re-seeding Phase 4 from v2 describes a different pipeline than the one this
project built and validated; it is out of scope, not an unfinished item.
Standardization-drift root cause is not recoverable from artifacts
without git history; the historical `.bak` files preserve the record, and
version-controlling the codebase is a process recommendation, not an open
scientific question. Per-episode return arrays are archived for all
configurations whose numbers underpin the paper's quantitative claims
(the four original Phase-5 configs plus all 18 paired-checkpoint configs
under `reports/paper_exp/` ? 216 array files total). The SOTA comparison
draft lives at ?12.9; the abstract/intro framing is captured at ?43.0.
No item on this list blocks the paper's defensibility.

## Session Ledger — End-of-Session Integration

Values established this session, all verified from artifacts or fresh runs:

| Field | Value | Source |
|---|---|---|
| v1 RF features | 11 | `rf_downscale_model.joblib`, `n_features_in_` |
| v2 RF features | 16 | `rf_v2_noS2_model.joblib` |
| Reward range | −8.6625 to +1.5000 | `offline_rl_dataset.h5` |
| gw_level range | 1.81 to 46.23 m | same |
| Held-out wells | 710 | well-level manifest |
| Leakage audit | 5/6 verified | `check_leakage.py` with manifest |
| 61,913 rows | v1 `rf_training_table.csv` | matches `rf_grid_predictions.csv` |
| Return arrays | archived, 18 per config | `reports/<config>/returns/*.npy` |
| Environment | d3rlpy 2.8.1, numpy 2.2.6, sklearn 1.7.2 | runtime |
| OOD reversal (QR) | id −16.01 → severe −38.40 | `reports/ood_return_test.json` |
| OOD reversal (Mean-Q) | id −8.64 → severe −18.17 | `reports/ood_return_test_meanq.json` |
| A4/A5 sweep range | +0.35 to +79.3 (226× spread) | `reports/a4a5_sensitivity.json` |
| A4/A5 canonical cell | +19.72 [+17.25, +22.27] | same |
| CQL−BC (adopted, ε) | +0.3402 [+0.2433, +0.4407] | current `compare_policies.py` |
| CQL−BC (adopted, no_oracle) | +0.5843 [+0.3236, +0.8534] | same |
| CQL−BC (adopted, α=1) | +0.9504 [+0.6976, +1.2181] | same |
| CQL−BC (adopted, α=10) | +0.5144 [+0.2676, +0.7783] | same |
| CQL−BC (adopted, base) | −1.1959 [−1.6651, −0.7465] | same |

**Known defects to record, not fix:**
- `check_leakage.py` prints "6/6 passed" when 1 is SKIP (return-True-from-skip).
- `data/processed/rf_v2_nocoords_model.joblib` has 16 features (not 14) and no source reference in any `.py/.json/.yaml` — orphaned artifact; quarantine or trace source.
- `reports/policy_comparison.json` (root) was overwritten during this session with no `.bak` made — historical value lost unless git tracks it.
- `verify_ci_from_arrays.py` (drafted during session) compares new arrays against the new JSON — a self-consistency check, not reproducibility; do not cite its "OK" output as reproducibility evidence.
