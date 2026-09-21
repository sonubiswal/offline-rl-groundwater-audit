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
validation loss rose after roughly the midpoint of training, a
textbook overfitting signature at this sample size (full training curve:
`validation_methodology.md`, Section 14.4).

### 5.3 Root Cause, Diagnosed Directly (Not Assumed)

**Channel-level attrition.** Among the 25 nominally "usable" dates,
`sentinel2_ndvi` is the dominant and most erratic source of missingness,
ranging from 10% to **100%** NaN on individual dates — far exceeding any
other channel, and consistent with the cloud-cover-driven Sentinel-2 gaps
already documented in Section 4 of this document.

**Gap-level attrition.** The breaks between runs correspond to identifiable,
largely independent causes rather than one systemic failure:

| Gap | Duration | Diagnosed cause |
|---|---|---|
| 2013-01 → 2015-05 | ~2.5 years | Sentinel-2A launched June 2015 — imagery genuinely does not exist beforehand |
| 2017-11 → 2018-01 | ~2–3 months | Plausibly linked to the documented GRACE mission transition gap (Jul 2017–Apr 2018, Section 4) |
| 2020-01 → 2020-08 | ~7 months | Plausibly COVID-19 field-campaign disruption — **not independently confirmed**; stated as plausible, not verified |

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
Phase 2 RF model (a smaller contributor than `grace`, `target_lag1`,
`month_sin`, and `chirps_roll3` — Section 12 of the methodology document)
suggests that even a full recovery of NDVI coverage would not, by itself,
have been the deciding factor in this model's viability; the genuine
multi-month structural gaps (satellite launch timing, mission transition,
plausible pandemic disruption) are the binding constraint, not per-date
sensor noise.

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
held-out CGWB wells, substantially exceeding the Phase 2 RF baseline's
R²=0.198. **This headline number must not be reported without the two
caveats below** — both materially change how the result should be
interpreted, and omitting either would overstate what has actually been
demonstrated.

### 6.2 Caveat (a): Different Task Scope — Not a Direct Substitute for RF

RF predicts groundwater depth at **any** location, including places with
no monitoring history at all — this is the project's actual downscaling
objective, and remains necessary for producing spatial coverage maps
(e.g. for the advisory system planned in later phases). The point-scale
LSTM can only be applied at locations with an **existing reading
history**; it cannot answer "what is the depth at this unmonitored
point?" — only "given this well's own history, what will it read next?"

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
this baseline. This mirrors the same discipline this project already
applied when the Phase 2.5 residual-Kriging correction's initial
cross-validation (random `GroupKFold`) was found to be measuring an
easier problem than the true held-out geometry, and was corrected with a
fairer, spatially-blocked comparison before the technique's benefit was
assessed.

### 6.4 Practical Implication for Downstream Use

If the point-scale LSTM's output is used as an input to a later phase
(e.g. a Phase 4+ advisory or RL system), it should be applied **only** at
locations with sufficient reading history (≥ `seq_len`+1 = 5 quarters, at
the current configuration), and its predictions should not be conflated
with RF's location-agnostic downscaled surface — the two serve different
purposes, cover different locations, and carry different reliability
profiles. A downstream system that needs coverage everywhere (including
unmonitored locations) must still rely on RF (or a future improvement to
it); a system that specifically needs a forecast at an already-monitored
well can draw on the LSTM instead.

### 6.5 Not Pursued as Further Verification (Open)

Two checks that would strengthen this result further were identified but
not yet performed, and are recorded here rather than left implicit:

- **Depth-stratified error analysis**, analogous to Section 1's
  stratified breakdown for RF — it has not been verified whether the
  delta-target design actually resolved the depth-dependent compression
  bias diagnosed in Section 1, or merely relocated it to a different part
  of the depth distribution.
- **`seq_len` sensitivity** (2, 3, 5, 6 quarters) against the point-scale
  task specifically — the current value of 4 was carried over as a
  practical default from the grid-model pivot (`validation_methodology.md`,
  Section 15.4) and has not been independently validated as optimal for
  the point-scale formulation.

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
does. The improvement from ~34% usable dates (Section 5.2's 25/38 at the
*date* level) to ~50.5% usable *readings* reflects this more permissive,
per-channel tolerance rather than any actual increase in underlying data
availability.

### 7.3 Practical Implication

This evaluated subset (609 of 710 held-out wells) should **not be assumed
representative of all held-out wells uniformly**. As with Phase 2
(Section 3 of this document), coverage gaps concentrate in specific
years/channels rather than being spread evenly across the well network —
a well whose readings happen to fall in a well-covered period is more
likely to be evaluated than one whose readings cluster in a gap period,
independent of any property of the well or location itself. Any claim
about the point-scale LSTM's real-world applicability should be scoped to
"wells with sufficient covariate-period overlap," not silently
generalized to the full held-out set.

## 8. Point-Scale LSTM: Untested Design Choices (Carried Forward as Open Limitations)

The following were practical defaults adopted to complete Phase 3 within
scope, not independently validated choices, and should be treated as open
limitations rather than settled methodology if this work is extended:

- **`seq_len=4`** was adopted after the grid model's lookback requirements
  proved infeasible (Section 5.5), not selected via a sensitivity sweep
  against the point-scale task itself (see Section 6.5).
- **`sentinel2_ndvi`** remains in the covariate set despite being the
  largest single source of missingness at both the grid level (Section
  5.2) and, to a lesser extent, the well level (Section 7.1–7.2); a
  sensitivity run with this channel dropped has not been performed.
- **The COVID-19 field-campaign disruption explanation** (Section 5.3) is
  stated as plausible based on timing alone and has not been verified
  against an actual CGWB campaign-schedule source. This should be
  upgraded to a cited claim, or removed, before appearing in the final
  report's discussion section.