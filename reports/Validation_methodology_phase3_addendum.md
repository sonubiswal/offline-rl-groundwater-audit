# Phase 3 — Temporal Modeling: ConvLSTM (Infeasible) and Point-Scale LSTM (Revised)

## 14. Grid-to-Grid ConvLSTM

### 14.1 Architecture Implemented

`src/models/convlstm.py` — a self-contained ConvLSTM cell implementation
(no external ConvLSTM library dependency, per the roadmap's explicit
requirement), verified against the roadmap's exact tensor-shape
specification via an automated smoke test:

- **Input:** `(batch, seq_len, channels, 64, 64)`
- **Output:** `(batch, 1, 64, 64)`
- **Cell design:** a single `Conv2d` producing all four LSTM gates
  (input, forget, output, candidate) at once from the concatenation of
  the current input and previous hidden state — standard ConvLSTM gating,
  substituting 2D convolutions for the fully-connected gates of a
  classical LSTM cell, to preserve spatial structure across the sequence.
- **Config-driven:** `hidden_dim`, `num_layers`, `kernel_size` read from
  `config/model_config.yaml`, supporting either a single scalar (applied
  to every layer) or a per-layer list.
- **Read-out head:** a `1×1` convolution collapsing the final layer's
  hidden state to a single-channel output grid.

### 14.2 Sequence Assembly Pipeline

`src/models/prepare_sequences.py` builds rolling windows over the Kriged
surface dates:

1. Loads all available Kriged target surfaces (`.npy` files in
   `data/interim/kriged_target_monthly/`).
2. For each date, builds the covariate stack via
   `build_feature_stack_for_date` (the same function used by
   `residual_kriging.py` in Phase 2.5) — channels: `chirps`, `gldas`,
   `sentinel2_ndvi`, `grace`, `srtm_elevation`, `srtm_slope`,
   `chirps_roll3`, `gldas_roll3`.
3. Slides a `SEQ_LEN`-quarter window across **consecutive** valid dates,
   rejecting any window whose internal gaps (or gap to its target date)
   exceed `MAX_GAP_DAYS`.

**Revision from monthly to quarterly framing:** the initial version
assumed monthly cadence (`SEQ_LEN=12`, implicit ~30-day gap tolerance).
Because CGWB campaigns are actually quarterly (Section 2), this was
revised to `SEQ_LEN=6` (six quarters ≈ 1.5 years) and `MAX_GAP_DAYS=120`
(allowing up to ~4-month gaps between consecutive quarters, matching
observed real campaign spacing per Section 9's lag-window finding for
Phase 2).

### 14.3 Root-Cause Diagnosis: Coverage Analysis (`check_coverage.py`)

The initial run under the revised quarterly framing produced only **7
usable training sequences** — an unexpectedly low count against 38 total
Kriged surfaces. Rather than treat this as an unexplained data problem,
a dedicated diagnostic script checked every one of the 38 Kriged dates
individually for covariate-stack completeness.

**Result: 25/38 dates have a usable stack; 13/38 fail outright** (stack
returns `None`, meaning at least one covariate source has zero coverage
for that date). Critically, the 25 usable dates are **not contiguous** —
they fall into five isolated runs:

| Run | Date range | Length (quarters) | Sequences contributed (seq_len=6) |
|---|---|---|---|
| A | 2015-08 → 2016-01 | 3 | 0 |
| B | 2016-05 → 2017-01 | 4 | 0 |
| C | 2017-05 → 2017-08 | 2 | 0 |
| **D** | **2018-05 → 2020-01** | **13** | **7** |
| E | 2020-08 → 2021-01 | 3 | 0 |

A run of length *L* contributes `max(L − seq_len, 0)` sequences. Only Run
D (13 quarters) exceeds the 7-quarter minimum (6 input + 1 target) needed
to contribute even one sequence — exactly reproducing the observed count
of 7.

**Per-channel attrition detail:** among the 25 "usable" dates, NaN
fraction varies sharply by channel. `sentinel2_ndvi` is the dominant
source of both hard failures and soft degradation, ranging from 10% to
**100%** NaN on individual dates (e.g. 2016-08-01, 2020-08-01) — far
higher and more erratic than any other channel. This is consistent with
Sentinel-2's known cloud-cover sensitivity, already documented as a
Phase 1 limitation (Section 4/Section 5).

**Gap-by-gap explanation:**

| Gap | Duration | Likely cause |
|---|---|---|
| 2013-01 → 2015-05 | ~2.5 years, near-total failure | Sentinel-2A launched June 2015 — imagery genuinely does not exist before this |
| 2017-11 → 2018-01 (and surrounding attrition through 2018-01) | ~2–3 months of failures | Plausibly linked to the documented GRACE mission transition gap (Jul 2017–Apr 2018, Section 4) |
| 2020-01 → 2020-08 | ~7 months | Plausibly COVID-19 field-campaign disruption — **not independently confirmed** against a CGWB campaign-schedule source; stated here as a plausible, not verified, explanation |

### 14.4 Training Attempt and Result

Given the severe sample-size constraint, a patch-augmentation strategy
(`patch_augment.py`) was tried before abandoning the grid approach
entirely: each 64×64 grid is cut into non-overlapping 16×16 spatial
patches, multiplying the number of training examples per real sequence
without claiming to create new independent information (documented
explicitly in the module's own docstring as a way to extract more
gradient signal from a small dataset, not a fix for the underlying
scarcity).

**Critical correctness safeguard:** the train/validation split was
performed at the **original sequence level first** (last 1 of 7
sequences reserved for validation), and patches were extracted
**separately** for each side afterward — never pooling all patches
together before splitting. This prevents patches from the same
underlying grid (which are spatially correlated, sharing the same date
and regional context) from leaking across the train/validation boundary.

**Configuration used:** `hidden_dim=8`, `num_layers=1` (deliberately
minimal, given the sample size), `patch_size=16` (stride = patch_size,
non-overlapping), batch size = 8, Adam optimizer (lr=1e-3,
weight_decay=1e-4), `SmoothL1Loss`, early stopping patience = 15.

**Data split:** 6 training sequences → 96 patches; 1 validation sequence
(2020-01-01) → 16 patches.

**Training curve (selected epochs):**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 10 | 25.362 | 19.507 |
| 20 | 18.200 | 13.816 |
| 30 | 13.921 | 10.863 |
| 40 | 11.583 | 9.695 |
| 50 | 9.762 | 14.372 |
| 55 (early stop) | — | 9.695 (best) |

**Final result:** validation R² = **−0.0655** (n=4,096 pixels, all drawn
from the single held-out sequence's 16 patches) — worse than predicting
the mean. Training loss continued to fall while validation loss began
rising after epoch ~40, a classic overfitting signature consistent with
having only 6 independent training examples regardless of patch count.

**Held-out CGWB well evaluation (Metric 2) was never computed for this
model.** Per the project's cross-validation discipline (Section 11), the
held-out set is touched only once a model has already demonstrated
viability on training-only validation. Since Metric 1 (against the
Kriged surface, the easier of the two required metrics) had already
failed decisively, running Metric 2 would have provided no additional
information and would have used up part of the project's single-check
budget against the true held-out wells for no benefit.

### 14.5 Alternatives Considered and Rejected (Quantified)

Before abandoning the grid approach, shrinking `seq_len` was considered
as a way to admit Runs A/B/C/E into the training set. This was rejected
on quantitative grounds, computed directly from the run-length table in
Section 14.3:

| `seq_len` | Sequences from Run A (3) | Run B (4) | Run C (2) | Run D (13) | Run E (3) | **Total** |
|---|---|---|---|---|---|---|
| 6 (used) | 0 | 0 | 0 | 7 | 0 | **7** |
| 5 | 0 | 0 | 0 | 8 | 0 | **8** |
| 4 | 0 | 0 | 0 | 9 | 0 | **9** |
| 3 | 0 | 1 | 0 | 10 | 0 | **11** |
| 2 | 1 | 2 | 0 | 11 | 1 | **15** |

Even the most aggressive reduction tested in this analysis (`seq_len=2`,
which would also weaken the model's temporal memory to just two
quarters, undermining the entire motivation for a sequence model over
RF's single-lag feature) tops out at an estimated 15 sequences —
still one to two orders of magnitude below what spatiotemporal deep
learning typically requires for reliable generalization. **This
quantitative ceiling, not just the 7-sequence result at `seq_len=6`, is
the basis for treating grid-level ConvLSTM as infeasible at the current
data volume**, rather than a problem `seq_len` tuning could solve.

Two further alternatives were identified but not implemented, for the
reasons stated:
- **Relaxing `sentinel2_ndvi`'s completeness requirement** (e.g. allowing
  a date through with partial NDVI coverage, imputing the gap) — could
  recover some of the 13 hard-failure dates, but would not bridge the
  genuine, multi-month structural gaps (Sentinel-2 pre-launch, the GRACE
  transition, the 2020 gap), which are the dominant constraint on run
  length, not per-date NDVI noise.
- **Extending the historical covariate pull** beyond the current
  2015–2022 range — would not help, since the binding constraint is
  *contiguous* coverage, not total date count, and the existing record
  already spans the full CGWB data availability window (Section 2).

### 14.6 Decision

Grid-to-grid ConvLSTM was **not pursued further**. This is documented as
a structural, data-availability finding — a property of the underlying
satellite/campaign coverage record, independently diagnosed and
quantified rather than assumed — not a code defect, an architecture
choice, or a tuning failure. The project pivoted to a point-scale
architecture (Section 15) that does not require whole-grid temporal
contiguity. Full practical implications and downstream recommendations
are recorded in `reports/limitations.md`, Section 5.

---

## 15. Point-Scale (Per-Well) LSTM: Revised Methodology

### 15.1 Rationale

The constraint diagnosed in Section 14 is specific to *grid-level*
sequences, which require an unbroken run of fully-covered calendar
quarters across the **entire study region simultaneously**. A
*well-level* sequence only requires covariate coverage at **one well's
single pixel** for each of that well's own reading dates — a
categorically weaker requirement. Isolated "OK" dates that could not
support a grid-wide sequence (Runs A, B, C, E — 12 of the 25 usable
dates, 48% of usable coverage) still contribute usable rows at the well
level, since each well-date row is evaluated independently rather than
requiring simultaneous region-wide coverage.

This also better matches the data granularity that made Phase 2's RF
model viable in the first place: RF trained on 22,838 independent
(grid-cell, date) samples, not 23 full-grid snapshots. The point-scale
LSTM restores that same granularity while adding sequential memory RF's
single `target_lag1` feature could not capture.

### 15.2 Data Preparation Pipeline (`build_well_level_dataset.py`)

For every well reading — built as two **separate** invocations, one for
the training-well set and one for the held-out-well set, preserving the
Section 3 spatial split exactly rather than re-deriving it — the script:

1. Groups readings by their exact date to avoid rebuilding the same
   covariate stack redundantly (cache pattern, same as
   `compute_rf_predictions_at_wells` in Phase 2.5).
2. Samples the covariate stack (`build_feature_stack_for_date`,
   `kriged_dir` passed through for the `target_lag1`-equivalent signal)
   at the well's exact grid cell via `latlon_to_grid_cell`.
3. Retains the row unless **every** channel is NaN at that pixel — a
   single noisy channel (typically `sentinel2_ndvi`) does not disqualify
   an otherwise-usable reading, unlike the grid pipeline's stricter
   whole-stack requirement.
4. Joins the resulting feature vector with the well's own raw `depth_m`
   reading.

**Output granularity:** one row per (well, date) — matching Phase 2 RF's
training granularity exactly.

**Coverage result (held-out set):** of 16,073 held-out readings, **8,127
rows (50.6%) retained a usable sample** — 7,946 skipped for having no
covariate stack at all on their date. This is consistent with (and
somewhat better than) the ~66% date-level unusability found in Section
14.3, since the well-level pipeline's per-channel tolerance recovers
some rows that the grid pipeline's all-channels-complete requirement
would have discarded.

### 15.3 Model Architecture (`train_point_lstm.py`, `WellLSTM`)

- **Sequence input:** a 2-layer LSTM (`hidden_size=64`, `dropout=0.2`)
  over six time-varying covariates: `chirps`, `gldas`, `sentinel2_ndvi`,
  `grace`, `chirps_roll3`, `gldas_roll3`.
- **Static input:** two terrain covariates constant per well —
  `srtm_elevation`, `srtm_slope` — concatenated with the LSTM's final
  hidden state.
- **Output head:** `Linear(66 → 32) → ReLU → Dropout(0.2) → Linear(32 → 1)`,
  predicting a scalar delta.
- **Target formulation — delta, not absolute depth:** the model predicts
  `next_depth − last_known_depth`, reconstructing the absolute prediction
  by adding this delta back to the well's own true previous reading. This
  design choice was made specifically in response to the depth-dependent
  compression bias diagnosed in Phase 2 (`limitations.md`, Section 1),
  where RF's leaf-averaging structurally shrank predictions toward the
  training distribution's center, especially at depth. Starting from the
  well's own true anchor point removes the need for the model to
  reconstruct absolute level from covariates alone — it only has to
  predict the *change*, which is a narrower and better-conditioned
  regression target.

### 15.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Sequence length (`seq_len`) | 4 quarters |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-5 |
| Loss function | `SmoothL1Loss` (Huber-style; more robust to depth outliers than plain MSE) |
| Batch size | 128 |
| Max epochs | 80 |
| Early stopping patience | 10 epochs (on validation loss) |
| Train/val well split | 85% / 15%, by well ID |

**`seq_len=4` rationale:** chosen as a practical default after the grid
model's 6-quarter (and even 2-quarter, per Section 14.5's table) lookback
proved infeasible at the grid level. **Not separately tuned** against
alternative lengths (2, 3, 5, 6) as of this writing — see Section 17,
open items.

### 15.5 Anti-Leakage Safeguards

- **Spatial split reuse, not re-derivation:** sequences are built
  independently per well, and the training/held-out well partition reuses
  the exact Section 3 spatial split
  (`train_well_ids.csv` / `held_out_ids.csv` → `held_out_readings.csv`,
  joined against the full CGWB readings table) rather than performing any
  new splitting logic that could silently diverge from the project's
  single canonical held-out set.
- **Normalization fit boundary:** feature normalization (`StandardScaler`,
  per-channel, fit via `fillna(column mean)` before fitting to avoid
  NaN-propagation into the scaler) is fit on **training wells only** and
  applied unchanged to both the internal validation split and the true
  held-out set — no statistics from held-out data ever influence scaling.
- **Sequence-level well grouping:** within a well, sequences are built
  from that well's own chronologically-sorted readings only; no
  cross-well sequence mixing occurs at any point.
- **Internal validation split (85/15, by well ID)** provides an
  early-stopping signal without touching the true held-out wells,
  mirroring the same discipline as Phase 2's training-only
  cross-validation (Section 11).

---

## 16. Point-Scale LSTM: Final Held-Out Validation Result

Evaluated once against the true held-out CGWB wells
(`train_point_lstm.py`'s printed headline metric, independently confirmed
with well_id/date metadata attached via `eval_point_lstm.py`).

### 16.1 Headline Comparison Table

| Model | Task | R² | RMSE (m) | MAE (m) | N |
|---|---|---|---|---|---|
| RF (Phase 2) | Spatial downscaling — any location, including unmonitored | 0.198 | 5.226 | 3.371 | 4,869 |
| Naive persistence | Predict next reading = last reading, at monitored wells only | 0.439 | 4.283 | 2.500 | 5,597 |
| **Point-scale LSTM** | Forecast next reading from covariates + history, at monitored wells only | **0.575** | **3.726** | **2.023** | 5,597 |

### 16.2 Framing Caveats (Critical — Do Not Omit When Citing This Result)

**(a) Task-scope mismatch with RF.** RF solves a structurally harder
problem: predicting groundwater depth at *any* location, including places
with no well history at all. Its R²=0.198 must be read in that context —
not as an inferior version of the task the LSTM solves. Conversely, the
point-scale LSTM **cannot** answer "what is the depth at this unmonitored
location?" — only "given this well's own history, what will it read
next?" The two are complementary results, not directly substitutable, and
should never be presented as a simple "LSTM beats RF" claim.

**(b) Persistence, not RF, is the valid baseline for isolating the LSTM's
learned contribution.** Because the delta-target design gives the LSTM
access to the same anchor information a naive persistence baseline uses
(the well's own last true reading), a meaningful share of its apparent
skill could, in principle, be pure autocorrelation rather than learned
dynamics. The dedicated persistence check
(`persistence_baseline_check.py`, run on the identical held-out
sequences) shows persistence alone reaches R²=0.439. The LSTM's
defensible, reportable contribution is the **gap above persistence**:

| Metric | Persistence | LSTM | Improvement |
|---|---|---|---|
| R² | 0.439 | 0.575 | +0.136 absolute (24.2% reduction in remaining/unexplained variance: (1−0.439) → (1−0.575)) |
| RMSE (m) | 4.283 | 3.726 | −13.0% |
| MAE (m) | 2.500 | 2.023 | −19.1% |

This gap — not the raw R²=0.575 figure in isolation — is the number that
represents genuinely learned, covariate-driven predictive skill beyond
what quarter-to-quarter groundwater persistence already provides for
free.

**(c) Consistency with Phase 2's own feature importances.** RF's final
model (Section 12) already ranked `target_lag1` as its second-largest
contributor (0.247, essentially tied with `grace` at 0.253) using only a
*single* prior reading as a feature, with no sequential structure. The
point-scale LSTM's result — a substantial improvement over even a perfect
single-lag baseline (persistence) — is consistent with, and quantitatively
extends, that earlier finding: temporal memory is a real and substantial
source of signal for this problem, and a sequence model can extract more
of it than a single-lag feature can.

### 16.3 Held-Out Coverage for This Evaluation

Of 710 held-out wells, 609 (85.8%) retained enough surviving readings
(> `seq_len`=4, after the coverage attrition described in Section 15.2)
to contribute at least one evaluation sequence, yielding 5,597 evaluable
sequences from those wells. The other 101 wells were excluded from Phase
3 evaluation entirely. This evaluated subset should not be assumed
representative of all 710 held-out wells uniformly — consistent with
Phase 2's own coverage caveat (Section 3), attrition here is driven by
which wells happen to fall on well-covered dates, not by any property of
the wells themselves.

### 16.4 Conclusion

The point-scale LSTM should be reported as a **complementary result** to
the RF baseline, not a replacement for it: RF provides the spatial
coverage the project's original downscaling objective requires; the
point-scale LSTM provides a well-scoped, honestly-caveated demonstration
that temporal memory carries real, learnable predictive signal beyond
what a static, single-lag regression model can extract — directly
motivated by, and quantitatively confirming, Phase 2's own feature
importance findings.

---

## 17. Bugs Found and Fixed During Phase 3

Documented here in the same spirit as Phase 2's Section 9/10 — the
diagnostic process is itself part of this project's defensibility record.

**Import path mismatch (`ModuleNotFoundError: feature_utils`).** Multiple
Phase 3 scripts (`prepare_sequences.py`, `eval_convlstm.py`) initially
inserted `src/data/` onto `sys.path` when importing `feature_utils`,
following an assumption from the roadmap's repository layout. In the
actual implemented repository, `feature_utils.py` resides in
`src/downscaling/` (alongside `rf_downscale.py`, which depends on it).
Fixed by aligning every Phase 3 script's `sys.path` insertion with the
pattern already working in `residual_kriging.py`
(`parent.parent / "downscaling"`).

**`ConvLSTM.__init__` crash on list-valued `kernel_size`
(`TypeError: unsupported operand type(s) for //: 'list' and 'int'`).**
The original architecture only normalized `hidden_dim` to a per-layer
list, not `kernel_size`; a `model_config.yaml` specifying `kernel_size` as
a per-layer list (mirroring `hidden_dim`'s own list format) reached
`ConvLSTMCell.__init__` unnormalized, where `kernel_size // 2` fails
against a list. Identified during the first `train_convlstm.py` run,
before any training compute was spent past model construction.

**Patch-level train/validation leakage risk (identified and designed
against proactively, not caught as a live bug).** Because patch
augmentation (Section 14.4) extracts multiple spatially-correlated
patches from each original sequence, splitting patches directly (rather
than splitting whole sequences first) could have let validation patches
sit immediately adjacent to training patches from the same date/region —
a spatial-leakage pattern directly analogous to the `GroupKFold` issue
found and fixed in Phase 2.5's residual-Kriging cross-validation
(`limitations.md` is silent on this since it predates Phase 3, but see
`validation_methodology.md` Section 11's spatial-block CV correction for
the precedent). `patch_augment.py` and the patched training script were
written with the sequence-level split enforced *before* patch extraction
specifically to avoid reintroducing this same failure mode in a new form.

---

## 18. Open Items Before Phase 3 Is Fully Closed

- [x] Grid-to-grid ConvLSTM architecture implemented per roadmap spec,
      verified via shape smoke test (Section 14.1)
- [x] Grid-to-grid ConvLSTM attempted, diagnosed as data-volume
      infeasible, root cause quantified via per-date and per-channel
      coverage analysis (Section 14.3)
- [x] Rejected alternatives (shrinking `seq_len`, relaxing NDVI
      completeness, extending the historical pull) quantified and
      documented rather than dismissed without analysis (Section 14.5)
- [x] Point-scale LSTM implemented as the revised approach, with
      anti-leakage safeguards matching the rest of the project's
      discipline (Section 15)
- [x] Held-out validation run once, against true held-out wells
      (Section 16)
- [x] Naive persistence baseline computed for fair comparison, and
      reported as the primary basis for interpreting the LSTM's result
      rather than the raw R² alone (Section 16.2)
- [x] Predicted-vs-actual scatter plot generated
      (`eval_point_lstm.py` → `reports/point_lstm_vs_persistence_scatter.png`)
- [x] Bugs encountered during implementation documented (Section 17)
- [ ] `sentinel2_ndvi`-dropped sensitivity run — given it is the largest
      single source of covariate-stack attrition at both the grid level
      (Section 14.3) and, to a lesser extent, the well level
      (Section 15.2) — not yet run
- [ ] `seq_len` sensitivity sweep (2, 3, 5, 6) against the point-scale
      LSTM — current choice of 4 was a practical default carried over
      from the grid-model pivot, not independently validated as optimal
      for the point-scale task
- [ ] COVID-19 field-campaign disruption (Section 14.3, 2020-01→2020-08
      gap) stated as a plausible but **unverified** explanation — would
      benefit from a citable CGWB campaign-schedule source before being
      stated as fact in the final report
- [ ] Depth-stratified error analysis for the point-scale LSTM (analogous
      to Phase 2's Section 12a) has not been performed — worth checking
      whether the delta-target design actually resolved the depth-
      dependent compression bias, or merely relocated it