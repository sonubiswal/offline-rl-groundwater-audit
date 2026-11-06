# Trishna OPAL: A Satellite-to-Policy Pipeline for Groundwater in Central India

**Status:** Draft. Every number in this report traces to `reports/final_report_claims.md`.
No claim is made that is not sourceable from a committed artefact.

---

## Abstract

Groundwater observations in central India are sparse: approximately 4,063
monitoring wells across a nine-state bounding box, read roughly four
times per year, at about one well per 200 square kilometers. Satellite
signals cover the region continuously but at coarse resolution — GRACE,
the closest space-based groundwater proxy, has a native pixel of about
300 km. This report describes Trishna OPAL, an end-to-end pipeline that
combines both sources: it learns a fine-resolution estimate of water
depth from satellite covariates and sparse well readings, then wraps that
estimate in a synthetic offline reinforcement learning environment in
which pumping policies can be trained and compared without experimenting
on a real aquifer.

Three results are reported, and they are reported honestly.

First, the learned downscaler improves on well-level kriging by 0.104 m
RMSE on a 14,062-reading held-out well set. The improvement is
statistically significant but small: kriging carries approximately 98%
of the useful signal. The learned layer polishes the remaining 2%. The
improvement does not extend to region-level spatial extrapolation, where
plain kriging beats the learned layer in three of four quadrants.

Second, a well-disjoint split-conformal procedure produces prediction
intervals with empirical coverage of 89.4% at nominal 90%, against a
prior attempt that achieved only 50.4%. The intervals are honest but
wide: 13.5 m mean width against a 5.30 m held-out RMSE. They are useful
as bounds, not as precise estimates.

Third, at 10 training seeds, Conservative Q-Learning does not outperform
Behavioral Cloning by a statistically significant margin on any of the
four comparable dataset families. The historical 3-seed result of +0.1096
does not replicate. Outside the training distribution, both policies
collapse to catastrophic returns, with CQL failing somewhat less than BC
in the severe band. A gap of four to nineteen return units between two
failing policies is not a working policy.

The project's contribution is not a record-breaking accuracy number. It
is an audited, reproducible, and honestly-bounded pipeline from raw
satellite data to policy evaluation, in which every claim traces to a
committed artefact and every limitation is measured rather than
asserted. The report is written to be checked, not to persuade.
## 1. Introduction

### 1.1 The problem

Groundwater is a critical water source across Indian agricultural
regions. In many districts it is being depleted faster than it recharges.
The Central Ground Water Board (CGWB) measures this directly: field teams
visit monitoring wells and record how far below the surface the water
sits, in meters. A bigger number means the water is deeper, which is
worse.

CGWB monitoring wells are sparse. Across the whole country, about 23,000
observation wells are read roughly four times a year. Inside this
project's study region — a nine-state bounding box covering a
Maharashtra-led three-state core — the CGWB dataset contains 4,063 usable
wells and 92,227 individual readings from 2013 to mid-2022. Spread over
approximately 700,000 square kilometers, that is about one well per 200
square kilometers. Coverage is uneven: some districts are dense (Amravati
has 116 wells), others are sparse (Pune has 63).

A farmer, a district officer, or an irrigation planner who needs to know
the water situation at a specific field cannot rely on the nearest
monitoring well being nearby. The nearest well might be many kilometers
away. The result is that the resource is managed reactively: the problem
is noticed when wells run dry, and by then the damage — aquifer
compaction, land subsidence — is often irreversible.

A proactive, data-driven system is needed to estimate underground water
levels between wells and to recommend safe pumping. That is what this
project set out to build.

### 1.2 Why satellite data alone is not enough

Satellites see the whole area at once, every month, for free. But they do
not directly see groundwater. They see rainfall (CHIRPS), soil moisture
(GLDAS), vegetation (Sentinel-2), terrain (SRTM), radar backscatter
(Sentinel-1), and — most usefully — small changes in Earth's gravity
caused by the movement of water.

That last signal comes from GRACE, a pair of satellites. It is the closest
thing to a direct measurement of total water storage from space. But its
native resolution is approximately 300 km. One "pixel" covers a large
fraction of a state. That is far too coarse to say anything about a
single farm, a single block, or even a single district.

The central difficulty of the project follows directly: can we use the
fine-grained surface signals to sharpen the coarse gravity signal down to
something useful, and prove that the sharpening is real rather than
invented?

### 1.3 Why offline reinforcement learning rather than online

Most agricultural reinforcement learning work uses online simulators.
The agent takes actions in a live environment, observes the consequences,
and adjusts. That is appropriate when the environment can absorb
experimentation. An aquifer cannot.

An aquifer is irreversible on human timescales. If you recommend heavy
pumping to see what happens, and the water table falls, the damage is
done. You cannot undo it, and you cannot try a different policy on the
same aquifer. This makes online reinforcement learning the wrong
paradigm here.

Offline reinforcement learning learns from a fixed dataset of recorded
experiences without interactive exploration. It is the methodologically
correct paradigm for a setting where you cannot experiment on the real
system. It remains comparatively unexplored for groundwater
specifically, which is part of why this project is interesting.

The catch is that the offline dataset has to come from somewhere. No real
farmer pumping logs paired with aquifer responses exist at the required
resolution. The project therefore builds a synthetic offline dataset
using a water-balance simulator, fully documented, and trains the offline
RL agents inside it. Section 6 describes the simulator; Section 10.4
records its limitations.

### 1.4 What this report claims

This report makes four claims. Each one is bounded by the numbers on
file, and each one is qualified by the limitations in Section 10.

**Downscaling.** A learned downscaler improves on well-level kriging by
0.104 m RMSE on a 14,062-reading held-out well set within the study
region. The improvement is real but small: kriging carries approximately
98% of the useful signal, and the learned layer polishes the remaining
2%. The improvement does not extend to region-level spatial
extrapolation.

**Time-series modelling.** A per-well sequence model (2-layer LSTM,
hidden size 64) reaches R-squared of 0.575 on held-out wells, against a
persistence baseline of 0.439. The defensible contribution is the +0.136
above persistence, and it applies only at wells that already have a
reading history. It does not replace the downscaler.

**In-distribution policy comparison.** At 10 training seeds, CQL does
not outperform Behavior Cloning by a statistically significant margin on
any of the four comparable dataset families. The historical 3-seed result
of +0.1096 does not replicate. One family shows BC decisively ahead by
2.47 return units; that gap is dataset-specific and does not generalise.

**Out-of-distribution behaviour.** Under all four committed OOD evaluation
files, CQL's return exceeds BC's, but both policies collapse to
catastrophic returns in the severe band (BC approximately -90, CQL
approximately -86). A gap of four to nineteen return units between two
failing policies is not a working policy.

The full text of each claim, with its supporting files and its "do not
say" boundary, is in reports/final_report_claims.md and is reproduced in
Section 11.

### 1.5 What this report does not claim

The report does not claim strong spatial generalisation, state-of-the-art
groundwater prediction, calibrated prediction intervals narrow enough to
serve as point predictions, or any real-world pumping, aquifer,
economic, or sustainability benefit. It does not claim that CQL beats BC
or that BC beats CQL without qualifying which dataset, how many seeds,
and which start set. It does not claim that CQL reverses
out-of-distribution. It does not claim that the simulator is physically
calibrated. It does not claim that the ConvLSTM approach was shown not to
work as a technique — the limit is data volume, not the method.

Section 11.2 lists the full set of claims that this report does not make,
and the reason for each.

### 1.6 What the report contributes

The contribution of this project is not a record-breaking accuracy
number. It is three things that a rigorous audit produces:

**An end-to-end pipeline from raw satellite data to policy evaluation,
fully audited.** Every data source has a committed provenance record.
Every model has a committed evaluation record. Every claim traces to a
specific file, and the validation methodology (Section 3) is designed
specifically to prevent circular ground truth.

**Evidence that a claimed CQL advantage does not replicate at 10 seeds,
with the honest analysis of why.** The historical 3-seed result was
plausible under 3 seeds and disappears at 10. The diagnostic that
explains this — start-set sensitivity, not BC training recipe — is
recorded in Section 7.2. The report's conclusion is a null result, not a
positive one, and the null result is the more honest contribution.

**A well-disjoint calibration procedure for prediction intervals that
hits nominal coverage on real CGWB data.** Split-conformal intervals
achieve 89.4% empirical coverage at nominal 90%, against a prior attempt
that achieved only 50.4%. The improvement comes from enforcing well-level
disjointness between calibration and test sets. The intervals are wide
(13.5 m against 5.30 m RMSE), and the report states that they are useful
as bounds, not as precise estimates.

Together these three contributions make the project's honest claim: a
reproducible, audited, bounded pipeline and evaluation methodology. It
does not contribute a deployable advisory system. Section 12 records what
would be needed to turn it into one.

### 1.7 Structure of the report

Section 2 describes the data and region of interest. Section 3 describes
the validation methodology, which is the project's strongest
methodological contribution. Sections 4 and 5 describe the estimation
pipeline (downscaling and time-series model). Section 6 describes the
synthetic RL environment. Sections 7 and 8 describe the policy
evaluation results, including the OOD behaviour. Section 9 describes the
explainability layer. Section 10 is the limitations section and is the
honesty backbone of the report. Section 11 states, in one place, what the
report claims and what it does not. Section 12 records future work.

The report is not written to persuade. It is written to be checked.
## 2. Data and Region of Interest

This section describes the input data: the satellite layers, the CGWB
well records, and the region of interest. Every number in this section
is measured from the committed raw file and appears in
reports/roi_justification.md. The section is deliberately explicit about
what the region is and what it is not, because the internal project
explainer described the region as "Maharashtra" and that shorthand
overstates the administrative cleanliness of the data.

### 2.1 Region of interest: a nine-state bounding box

The study region is a bounding box, not a state boundary:

  Latitude:   15.60 N to 22.00 N
  Longitude:  72.60 E to 80.90 E

The box was chosen to cover a dense, contiguous, and hydrologically
connected belt of CGWB observation wells, not to match any
administrative outline. It is a rectangle, and it is not the shape of
Maharashtra.

Within this box, the CGWB well records fall across nine states:

  State                                          Wells  Readings  % of readings
  Maharashtra                                     2073     44936          48.7%
  Telangana                                        809     19764          21.4%
  Karnataka                                        517     11679          12.7%
  Andhra Pradesh                                   228      5182           5.6%
  Gujarat                                          206      4782           5.2%
  Madhya Pradesh                                   152      4260           4.6%
  Chhattisgarh                                      32       572           0.6%
  The Dadra And Nagar Haveli And Daman And Diu      29       629           0.7%
  Goa                                               17       423           0.5%

Three states — Maharashtra, Telangana, and Karnataka — account for 82.8%
of the readings and 83.7% of the distinct wells. The remaining six states
are peripheral contributions to the rectangle, not central to the study.

The honest description of the region is "a Maharashtra-led three-state
core with six peripheral states at low density." Describing the region
as "Maharashtra" would overstate administrative cleanliness and
understate the geographic heterogeneity the model actually sees.
Section 10.10 records this as a limitation. Any claim in this report
about "groundwater in Maharashtra" is qualified accordingly.

### 2.2 District-level density in Maharashtra

Within Maharashtra, the top fifteen districts by distinct well count:

  District       Wells   Readings
  Amravati         116       2420
  Yavatmal         108       2317
  Nagpur           104       2403
  Nashik            92       2172
  Chandrapur        80       1923
  Ahmednagar        80       1631
  Wardha            76       1726
  Thane             74       1535
  Buldhana          73       1636
  Ratnagiri         73       1773
  Solapur           71       1602
  Jalgaon           70       1334
  Nanded            70       1477
  Satara            69       1399
  Pune              63       1343

The density is Vidarbha-heavy: Amravati, Yavatmal, Nagpur, Chandrapur,
Wardha, and Buldhana are all in or near Vidarbha. The Marathwada region
that was cited as an example in the original project plan (Aurangabad,
Latur, Beed, Parbhani) is present in the data but is not the densest
block. The northern and eastern districts dominate.

This is an honest description of the data, not the assumption that was
in the original plan. The ROI was finalized by measurement, not by
choosing the example region the master prompt suggested.

### 2.3 The CGWB well data

The well data is the CGWB Changes in Depth to Water Level dataset,
retrieved from the India Data Portal:

  File:          data/raw/cgwb_raw.csv
  Size:          65.43 MB
  Rows:          550,850
  SHA-256:       5fbcc59d8a917a5dc9d9d8b7d56f3db69297f5910b2966f72838f337ec0409bc
  Retrieved:     2026-09-17T08:13:34Z
  Source URL:    https://ckandev.indiadataportal.com/dataset/8ceb3cea-a8a9-4624-af75-fc92429d8fa5/resource/580a8f6e-3d86-4ca7-ac7d-cd5df12b443c/download/cgwb-changes-in-depth-to-water-level.csv

Provenance is recorded in data/raw/cgwb_raw.manifest.csv, which is
committed alongside the raw file. This is the only direct measurement of
groundwater in the project; every other input is a hint that helps the
model place the well readings between wells.

The raw file's columns are id, date, state_name, state_code,
district_name, district_code, station_name, latitude, longitude, basin,
sub_basin, source, currentlevel, level_diff.

Filtering to the ROI bounding box yields 92,227 rows. That is the exact
count in the interim well file (data/interim/cgwb_wells.csv). There is
no additional geographic filter downstream; the interim file's columns
are well_id, lat, lon, date, depth_m, and it is a direct projection of
the bbox filter.

Sample statistics for the interim file:

  Distinct wells          4,063
  Total readings         92,227
  Mean readings per well     22.7
  Min readings per well         1
  Max readings per well       116

The unbalanced distribution means most wells contribute very few
readings. This limits the per-well time-series modelling of Phase 3 to
wells with at least six consecutive quarters, which reduces the usable
well count for that phase. Phase 3's diagnostic records the exact count.

### 2.4 Time range

  First reading:  2013-01-01
  Last reading:   2022-07-28
  Total span:     approximately 9 years 7 months

The bounds are set by the CGWB dataset collection itself, not by project
choice. They are cross-checked against the satellite layers, which have
their own coverage constraints:

**Sentinel-2.** Launched June 2015. There is no Sentinel-2 NDVI data
before that date. The pipeline either drops the NDVI channel for early
dates or substitutes a coarser-resolution NDVI product; the v2 age-days
features record how stale each matched value is.

**GRACE.** The original GRACE mission ended and was replaced by GRACE-FO.
There is a mission handover gap of approximately nine months from July
2017 to April 2018. Roughly one sixth of the study time range has no
GRACE reading.

**Sentinel-1.** Continuous coverage throughout the study period.

**CHIRPS, GLDAS, SRTM.** Continuous coverage throughout the study period.

The satellite coverage gaps are the reason the grid ConvLSTM could not
be trained (Section 5.1). They are also recorded in
reports/phase5_provenance.md as a limitation of Phase 3's data.

### 2.5 The satellite inputs

Six satellite-derived layers feed the estimation pipeline:

  GRACE-FO mascon (NASA):      Changes in total water storage, ~300 km
                                native resolution. The target the pipeline
                                is trying to sharpen to a finer grid.
  Sentinel-1 (ESA):            Radar backscatter, surface wetness, ~10 m.
  Sentinel-2 (ESA):            NDVI, vegetation greenness, ~10 m. Used in
                                v1, dropped in v2 (Section 4.6).
  CHIRPS (UCSB-CHG):           Daily rainfall, ~5 km.
  GLDAS (NASA):                Soil moisture (top 10 cm), 0.25 deg.
  SRTM (USGS):                 Elevation and slope, 30 m, static.

In the adopted v2 configuration, the sixteen features fed to the learned
downscaler are: chirps, gldas, grace, srtm_elevation, srtm_slope,
month_sin, month_cos, chirps_roll3, gldas_roll3, target_lag1,
age_days_chirps, age_days_gldas, age_days_sentinel1, age_days_grace,
row_norm, col_norm.

The age-days features were added in v2 to record how stale each matched
satellite value is. Because a monthly composite is sometimes missing, the
pipeline reaches back up to 400 days to find a usable value; without an
age column the model would treat a year-old rainfall figure as if it were
this month's. The age columns let it discount stale inputs.

The row_norm and col_norm features (the cell's position on the 64x64 grid)
are controversial and are discussed in Section 4.

### 2.6 What this section establishes

Three things, for the rest of the report:

First, the data is what the project says it is: 92,227 readings from
4,063 wells across a nine-state bounding box, over a defined 2013-2022
time range, with documented satellite coverage gaps and a committed
provenance manifest.

Second, the region is not Maharashtra. It is a Maharashtra-led three-state
core with six peripheral states at low density. This will be repeated in
Section 10.10 and in the report's limitations.

Third, the data has been filtered and cleaned before any model was trained.
The bbox filter that produced the interim file is exactly the bbox that
produced the 92,227 figure. There is no double filtering, no accidental
widening or narrowing, and no discrepancy between the number the project
quotes and the number the committed file contains.
## 3. Validation Methodology

This section describes how the project avoids circular ground truth. It
is the most important methodological section in the report, because every
downstream result — every R-squared, every confidence interval, every
policy comparison — is only as trustworthy as the holdout it was measured
against.

The design has five parts: a well-disjoint split that is created before
any interpolation touches the data, an automated test that enforces the
disjointness, a rule that held-out wells are never rasterised onto the
grid, a cross-validation protocol that mimics the holdout rather than
answers a different question, and an honest statement of what the whole
design can and cannot prove.

### 3.1 The circularity problem

Groundwater observations in the CGWB dataset are sparse. 4,063 wells over
approximately 700,000 square kilometers is about one well per 200 square
kilometers. To produce a map, something has to fill in the gaps between
wells. The obvious tool is kriging: a smooth interpolation that estimates
depth at unmeasured locations from nearby measured locations.

The problem is that if you krige from all wells and then measure the
kriged surface against a random split of readings, the readings used to
build the surface are in the test set. The R-squared is measuring memory,
not prediction. This is the circularity problem.

The same problem recurs in subtler forms. If held-out wells are written
into a raster grid cell that the model is later asked to predict, the cell
carries the answer. If the model's hyperparameters are tuned against the
held-out wells, the holdout stops being a holdout. If cross-validation
folds group by date rather than by space, the folds answer a question
that the real evaluation does not ask — can you forecast the future from
the past, rather than can you interpolate between known wells.

The rest of this section is what was done to prevent each of these.

### 3.2 The held-out-well design

Wells are split, not readings. A single CGWB well is read roughly four
times per year, and the readings for one well are highly correlated with
each other. If you split readings randomly, a model that has seen a well's
2019 readings will look brilliant at its 2020 readings without learning
anything about groundwater. Holding out whole wells removes that shortcut.

The split is spatially stratified, not purely random. The study region's
bounding box is divided into a 10x10 grid of cells, and from each cell
roughly 17.5% of wells are held out. This ensures that held-out wells are
spread across the region rather than clustered in low-density areas,
which would bias the test toward the sparse districts.

The final split:

  training wells    3,353   (82.6% of readings)
  training readings  76,154
  held-out wells      710   (17.5% of readings)
  held-out readings  16,073
  random seed            42

The proportion is 17.5% because a bug was caught during early runs. The
initial split produced 98.4% of wells in the test set due to an
off-by-one in the fraction calculation. The bug was found by checking
output rather than assuming it, and the fix produced the current 17.5%
split. The seed 42 was chosen as a reproducibility anchor; it has no
statistical meaning beyond that.

Both ID files (train_well_ids.csv and held_out_ids.csv) are committed to
version control. They are immutable from the moment they were created.
Re-running the split script with the same seed produces the same split;
re-running it with a different seed would produce a different split and
would silently reintroduce the circular ground truth problem this entire
methodology exists to prevent.

### 3.3 The automated test that enforces zero overlap

The split is enforced by an automated test, not by convention. The test
(tests/test_held_out_split.py) contains eight assertions. The one that
matters most is test_split_zero_overlap, which asserts that the
intersection of the training and held-out ID sets is empty. If a future
change causes any well to appear in both sets, this test fails and the
build fails.

The eight tests are:

  test_load_wells_drops_missing_and_implausible
  test_assign_grid_cells_within_bounds
  test_split_zero_overlap
  test_split_fraction_roughly_correct
  test_split_is_spatially_distributed
  test_split_is_idempotent
  test_split_handles_repeated_readings_per_well
  test_assert_no_overlap_raises_on_bad_input

The test suite passed on the last commit before this report (62 passed,
1 skipped, 0 failed in 56.68 seconds). The one skipped test is a
strict-archive audit that requires sidecars for every checkpoint; it is
skipped because the archive contains a small number of pre-fix
checkpoints, not because the test is broken.

The presence of an automated test is not the same as having a well-designed
split — a passing test with a bad split would just be a test for a bad
split. But the combination of a documented, version-controlled split and
an automated test that catches accidental re-creation is what makes the
project's honesty claim mechanically enforceable rather than aspirational.

### 3.4 The held-out-never-rasterised rule

Every input to the model is rasterised onto a 64x64 grid. Rainfall,
soil moisture, NDVI, elevation, satellite-derived covariates, and the
GRACE anomaly are all pixel-level fields. Training wells are placed onto
the grid by counting how many fall in each cell and averaging them, so
they influence the model's target at their cells.

Held-out wells are not rasterised. They are read as individual points,
only at the moment of scoring. If a held-out well's value were written
into a grid cell, that cell would carry the answer the model is supposed
to guess. The grid would be a target the model could learn to reproduce
trivially, and no test would detect the leak because the split would
still be well-disjoint.

This rule is enforced in code (held-out wells are loaded from a separate
file and are never merged into the grid stack). It is also stated in the
project's validation documentation (reports/roi_justification.md and
reports/final_report_claims.md) so that any future contributor reading
the code or the docs finds the rule before finding a way to break it.

### 3.5 The cross-validation protocol

Hyperparameter selection happens inside the training data only. Held-out
wells are never touched during tuning.

The first version of the cross-validation protocol grouped folds by date:
train on some years, test on others. Every configuration scored strongly
negative R-squared under this protocol. That looked like the models were
broken. They were not. Date-grouped folds answer a different, much harder
question than the one the real test asks. Our real test holds out wells,
and most held-out readings fall on dates the model has already seen via
other wells. So the correct protocol measures "can you interpolate between
wells you know", not "can you forecast the future from the past".

Once folds were reorganized spatially — blocks of grid cells, held out as
units — the numbers became positive and meaningful. The protocol is now
documented in reports/validation_methodology.md and is used for every
downscaler selection decision.

Two further lessons from the cross-validation work are relevant here.
First, cross-validated R-squared is systematically about three times too
optimistic relative to the held-out well result (0.50-0.64 CV vs 0.187
held-out). CV is a tool for ranking models, never a number to quote as
performance. Second, the CV figure depends on the block size: four blocks
gives 0.497, sixteen gives about 0.58, sixty-four gives about 0.63. Finer
blocks put test cells closer to training cells, and the score rises for
reasons that have nothing to do with the model. Any single CV figure
without a stated block size is not defensible and does not appear in this
report.

### 3.6 What this design can and cannot prove

The design proves one thing: that the model can predict the depth at a
held-out well that is interleaved with training wells, in a region the
model has been trained on, in a time period the model has seen.

The design does not prove three things:

- It does not prove spatial extrapolation. The median held-out well is
  5.8 km from the nearest training well, and 99.8% are within 25 km. This
  is an interpolation test, not an extrapolation test. Under a stricter
  quadrant holdout (Section 10.3), plain kriging beats the learned layer
  in three of four quadrants, which tells us what spatial extrapolation
  would look like.

- It does not prove temporal forecasting. Date-grouped validation is
  negative in every configuration, for both model versions. The model
  interpolates inside 2013 to 2022; it has not been shown to forecast
  anything.

- It does not prove real-world applicability. The held-out wells are
  in situ observations, not farmer decisions. Even if the model predicted
  perfect water depth, the downstream policy layer depends on a synthetic
  simulator with six documented assumptions.

The rest of this report is careful to state what each result proves in
these terms. Section 10.3 restates the spatial-extrapolation caveat.
Section 10.6 restates the temporal caveat. Section 10.4 restates the
simulator caveat. The methodological discipline established here is
what makes those restatements credible rather than defensive.

### 3.7 Summary

The circularity problem is real, common, and easy to fall into. This
project's defense against it is: a well-disjoint split created before
any interpolation, enforced by an automated test, held-out wells never
rasterised onto the grid, cross-validation that mimics the real holdout,
and an honest statement of what the whole design proves and does not
prove. Every subsequent result in this report inherits the strength of
this design and is bounded by its limits.
## 4. Downscaling (Phase 2)

Phase 2 is the technical core of the estimation side of the project. It
takes coarse satellite signals plus sparse CGWB well readings and produces
a fine-resolution estimate of water depth on a 64x64 grid. This section
describes what was built, reports the result honestly, and documents the
three alternative approaches that were tried and rejected.

### 4.1 The approach

The GRACE satellites measure changes in total water storage from space,
but at approximately 300 km native resolution. A "pixel" covers a large
fraction of a state. That is far too coarse to say anything about a single
farm or even a single district.

The pipeline is a two-layer stack:

  Layer 1 (kriging baseline):  A classical statistical interpolation
    produces a smooth water-depth surface from the training wells alone.
    Ordinary kriging with a spherical variogram. Held-out wells are never
    used to build this surface.

  Layer 2 (learned correction):  A gradient-boosting regressor learns to
    reproduce the kriged surface from satellite covariates. It sees
    rainfall, soil moisture, the coarse GRACE anomaly, elevation, slope,
    seasonal encoding, three-month rolling averages of rainfall and soil
    moisture, the previous month's kriged value, age-days features for
    stale satellite matches, and grid position. Sixteen features in the
    adopted v2 configuration.

The output is a prediction at every 64x64 grid cell, including cells with
no well nearby. That is the project's actual objective; a per-well model
cannot do this.

### 4.2 The result

On 14,062 held-out readings from 683 wells:

  Method                     Readings   RMSE (m)   Median (m)   R-squared
  ----------------------------------------------------------------
  Our model (v2)               14,062      5.300        3.395      0.187
  Kriging alone                14,062      5.404        3.436      0.155
  Assume nothing changed       14,062      5.869        3.885      0.003

The improvement over kriging is 0.104 m with a 95% confidence interval of
[0.051, 0.157]. The interval does not include zero, so the improvement is
real.

It is also small. About 98% of the useful signal is already in the kriged
surface. The learned layer polishes the last 2% of error. That is not the
story people expect, and it is the story the numbers tell.

### 4.3 What the numbers mean, and what they do not

The R-squared of 0.187 sits in the middle of published well-level
groundwater results once holdout design is accounted for. It is not a
state-of-the-art figure, and the report does not present it as one.

Three contextual facts matter for interpreting it:

  First, the target is a smoothed interpolation, not raw observations.
  The kriged surface is itself a 5.404 m error surface. A smooth target
  limits achievable R-squared regardless of model, because no model can
  outperform the noise floor of its own target.

  Second, the honest comparison is kriging at 0.155. The 0.104 m
  improvement is real but is 2% of total error. A reviewer who reads only
  "R-squared 0.187" without this comparison will overestimate the
  contribution.

  Third, cross-validated R-squared on the same models was 0.50 to 0.64,
  which is about three times too optimistic relative to the honest
  held-out result. CV is a tool for ranking models, never a number to
  quote as performance. Section 3.5 explains why.

Consequence for the report: any sentence describing the downscaler as
"the model that predicts groundwater" overstates it. The correct phrasing
is "a small learned correction on top of kriging." Section 10.2 states
this as a limitation. Section 11.2 lists it as a claim the report does
not make.

### 4.4 The quadrant holdout: what the model does not do

The held-out well result proves one thing: interpolation between known
points. It does not prove spatial extrapolation. To test that, the study
region was divided into four quadrants. For each quadrant, all training
wells in that quadrant were removed, the model was refitted, and the
held-out wells inside the excluded quadrant were predicted.

  Quadrant      Readings   Our model R-squared   Kriging R-squared   Winner
  North-west       3,167                 0.116               0.055   Our model
  North-east       3,540                 0.104               0.188   Kriging
  South-west       3,106                 0.056               0.192   Kriging
  South-east       4,249                 0.090               0.174   Kriging
  All pooled      14,062                 0.101               0.155   Kriging

Once the model is asked to work in a region with no training wells at
all, plain kriging beats it in three quadrants out of four. The machine-
learning layer's advantage is interpolation between known points, not
extrapolation into unknown ones.

One caveat on the quadrant test itself: the kriged map was held fixed
across the four folds rather than re-kriged from the reduced training set
inside each fold. Even the quadrant result is a lower bound on how hard
the true spatial extrapolation test would be. Section 10.3 records this.

Consequence: the phrase "the model generalises spatially" must not appear
in this report. The model generalises to held-out wells within a region
that also contains training wells. That is all.

### 4.5 Depth compression

The learned layer systematically over-predicts shallow depths and
under-predicts deep depths. On the well-level holdout, binned by actual
depth:

  0-10 m      bias +1.55 m  (over-predicts)
  10-20 m     bias -5.21 m  (under-predicts)
  20-30 m     bias -13.83 m
  30-50 m     bias -25.58 m
  50 m+       bias -50.64 m  (too few readings to trust)

The opposite signs partly cancel in the mean bias, which is why the mean
looks almost perfect (-0.05 m). It is not. Section 10.1 diagnoses the
cause: tree-based regression predicts averages in each leaf, which pulls
output toward the centre of the training distribution, and deep examples
are a thin tail.

The practical consequence is that the downscaler should be treated as
reliable mainly shallower than about 15 to 20 m. Beyond that, its output
is qualitative. Because Phase 4 seeds the simulator from the Phase 2
downscaler, every Phase 5 result inherits this bias.

### 4.6 Three ideas we tested and rejected

Three separate feature additions were tried against the same target. All
three failed the pre-registered improvement threshold of +0.02 on spatial
cross-validation. They are kept in the record because a project that only
reports its wins has no way of showing it looked.

**Idea one: a 12-month rainfall deficit feature.** Groundwater responds
to how much rain has fallen over a year or more, not just this month. A
feature capturing the twelve-month rainfall total relative to the
long-term normal was added. The model used it (importance score 0.052),
but held-out R-squared came back at 0.190, slightly below the 0.198
baseline it was meant to beat. Rejected: a feature that is used but does
not improve held-out accuracy adds complexity for nothing.

**Idea two: residual kriging.** Fit the model as usual, krige the
residuals near training wells, and add the smoothed error surface back as
a correction. This looked good under date-grouped cross-validation
(+0.041) but collapsed to +0.006 under the correct spatial folding.
Under the held-out check it came back at 0.182, below the 0.189 it was
compared against. Rejected: a gain that evaporates under the correct test
is not a gain.

The residual-kriging failure directly motivated the pre-registered rule:
any new feature must improve spatial cross-validation by more than +0.02
before the held-out set may be opened for that model. The threshold was
set at roughly three times the failed +0.006 signal, to filter CV noise at
this sample size. It was written down before the next experiment ran.

**Idea three: soil texture and land-use classes.** Two datasets were added
as 23 yes/no columns: 12 soil texture classes and 11 land-use classes.
This was the most physically motivated idea — the geology genuinely does
control how water moves. Cross-validated R-squared came back at 0.639,
identical to the baseline 0.639. Delta +0.000. Under the pre-registered
rule the held-out set was not opened. Before accepting the result, the
team confirmed the 23 new columns had actually been wired in (feature
count was confirmed programmatically at 35 rather than the baseline 11),
so "no improvement" could not be a silent import bug masquerading as a
scientific finding. It was not a bug. The result is genuine.

Possible explanations for why soil and land use did not help, none
confirmed: the 5,000 m export resolution may be too coarse for products
that are native at 10 m; the soil layer is used at a single depth slice;
the truly load-bearing variable may be lithology (the actual rock type),
which no ready dataset was identified for.

### 4.7 The pattern across all three failures

Three separate attempts to improve the map — a rainfall-memory feature, a
kriging correction, and soil/land-use covariates — all failed against the
same target. Combined with the finding that kriging carries 98% of the
signal, the conclusion is fairly firm: adding more inputs to the
machine-learning layer is not where improvement will come from. Better
kriging is.

Section 12 names that as future work: elevation as external drift in the
kriging step, a better variogram, anisotropy. Section 10.2 records that
this is the direct recommendation of the three failed experiments, not a
hypothesis invented after the fact.

### 4.8 Summary

The downscaler works, but the honest way to describe the work is that
kriging does almost all of it and the learned layer adds a small, real,
and honestly-bounded 0.104 m improvement. The result does not extend to
spatial extrapolation, does not extend to deep wells, and does not
survive three attempts to improve it with additional covariates. The
contribution is not the numerical skill of the downscaler; it is the
audit trail that shows what the number is and what it is not.
## 5. Time-Series Model (Phase 3)

Phase 2 treats each month independently: the model sees this month's
covariates plus last month's kriged value, and predicts this month's
surface. But groundwater has memory. This quarter's level is strongly
related to the last few quarters. Phase 3 asks whether a model that sees
a sequence of months can beat a model that sees only one month plus the
previous value.

The answer, for this dataset, is a qualified yes — but the model that
delivers the yes is not the one the original plan expected.

### 5.1 The grid ConvLSTM: what was attempted and what was found

The original plan was a grid-to-grid ConvLSTM: a convolutional LSTM that
reads a sequence of whole maps and predicts the next whole map, tracking
spatial patterns over time. Architecturally it was built and verified
against the specification. It could not be trained, and the reason is
data availability, not model capacity.

Rather than assume, a diagnostic script inspected all 38 available
monthly kriged maps one by one and asked a single question: does this
date have a complete set of satellite inputs (Sentinel-1, Sentinel-2,
CHIRPS, GRACE, GLDAS, SRTM, all present)? Twenty-five did; thirteen did
not. But the twenty-five were not consecutive. They fell into five
non-contiguous runs, separated by gaps caused by three independent
events:

  - Sentinel-2 did not exist before June 2015.
  - GRACE had a mission handover gap of approximately nine months,
    July 2017 to April 2018, when the original mission ended and its
    replacement was being commissioned.
  - A roughly seven-month gap in 2020 is plausibly a COVID-19 field-
    campaign disruption. This is plausible, not verified.

A sequence model needs six consecutive quarters of input plus one to
predict — seven in a row. Only one of the five contiguous runs is long
enough to yield training sequences. The entire usable training set is
seven whole-map sequences. That is not enough to train a grid ConvLSTM.

An attempt was made anyway. Each map was cut into 16x16 patches to
multiply the number of training examples, a legitimate way to get more
gradient signal from a small dataset but not a way to create new
information. The split was done at the whole-map level before cutting,
so patches from the same map could not sit on both sides of the
train/test boundary. Validation R-squared came back at -0.0655, worse
than simply predicting the average. Training loss kept falling while
validation loss rose — the classic signature of memorising seven examples.

The result is not evidence that ConvLSTMs fail on groundwater. It is
evidence that the project's satellite record does not provide enough
contiguous months to train one. Shortening the lookback to two quarters
was considered and computed: it would raise the usable sequence count to
fifteen, still one to two orders of magnitude short of what this class
of model needs, and would also destroy the whole reason for using a
sequence model (two quarters is barely more memory than Phase 2's single
previous-value feature). The limit is data availability.

### 5.2 The pivot: per-well LSTM

A grid model needs every satellite layer present across the whole region
simultaneously. A per-well model needs coverage at one pixel, on that
well's own reading dates. That is a categorically weaker requirement,
and it recovers 12 usable dates that the grid model had to throw away —
nearly half the usable coverage.

The per-well model is a 2-layer LSTM with hidden size 64, dropout 0.2,
reading a sequence of six time-varying inputs per well plus two fixed
terrain values:

  - water depth, rainfall, crop demand proxy, and season as a
    sine/cosine pair (six time-varying inputs)
  - elevation and slope at the well's location (two static inputs)

It predicts the change from the well's own last reading, not the absolute
depth. This was a deliberate design choice, motivated by the depth-
compression problem documented in Section 4.5: starting from the well's
own last reading removes the need to reconstruct absolute depth from
satellites alone, which is where the tree-based downscaler was weakest.

The lookback is four quarters. This is a practical default inherited from
what the grid model could not do; it was not tuned. Section 12 records
that no sweep over 2, 3, 5, or 6 quarters was run.

### 5.3 The result, and why the obvious headline is the wrong one

The per-well LSTM scored R-squared = 0.575 against true held-out wells,
compared to Phase 2's 0.198 on the same wells. It looks like a rout. It
is not, for two reasons that must always travel with the number.

**Reason 1: the comparison is not apples-to-apples.** Phase 2 predicts
at any location, including places with no well. That is the project's
actual objective, because it is what produces a map. The LSTM can only
work where a well already has a reading history. It cannot answer "how
deep is the water at this unmonitored spot?" — only "given this well's
history, what will it read next?" These are complementary tasks, not
competing ones. Presenting this as "the LSTM beats the downscaler" would
misrepresent both.

**Reason 2: the honest baseline is "nothing changed."** Because the LSTM
starts from the well's own last reading, a large part of its apparent
skill could be pure persistence — water tables do not move much quarter
to quarter. So a persistence baseline was computed on the identical
sequences.

  Metric                     "Nothing changed"   Per-well LSTM   Improvement
  R-squared                                0.439             0.575        +0.136
  Error (m)                                4.283             3.726       -13.0%
  Typical error (m)                        2.500             2.023       -19.1%

Persistence alone reaches 0.439. The defensible contribution of the LSTM
is the +0.136 above persistence — about 24% of the remaining unexplained
variance — not the raw 0.575. And that gap is consistent with something
Phase 2 had already found independently: the previous month's value was
v1's second-largest feature. Memory is real; this measures how much.

### 5.4 Scope notes

Three honest caveats about the per-well result:

**Coverage.** 609 of 710 held-out wells (85.8%) had enough surviving
readings to be evaluated. The remaining 101 wells had too few readings
after dropping missing covariates. The result should not be assumed
uniform across all wells; it describes the 609 that could be tested.

**Comparison baseline.** The comparison table in Section 5.3 uses the
old Phase 2 figure of 0.198, because Phase 3 was run before the v2
downscaler rebuild. It has not been re-run against the v2 figure of 0.187.
The difference is negligible (0.011 in R-squared) but should be stated.

**Depth stratification.** The delta-target design was specifically
chosen to address the depth-compression bias documented in Section 4.5.
Whether it actually resolved the bias, or merely relocated it, was not
checked — a depth-stratified analysis of the per-well LSTM was never run.
Section 12 records this as future work.

### 5.5 What this section contributes

The per-well LSTM establishes that groundwater has measurable temporal
memory that a sequence model can exploit: a +0.136 improvement over
persistence is real and reproducible. It also establishes, by direct
attempt, that the grid-level ConvLSTM the original plan proposed cannot
be trained on this satellite record — not because the method is wrong,
but because seven whole-map training sequences is not enough data.

The distinction matters. A report that said "ConvLSTM doesn't work"
would be making a claim about the method. This report says "the grid
ConvLSTM could not be trained on this dataset because of a specific,
diagnosed data-contiguity limit." That is a statement about the data, and
it is fully supported by the diagnostic.

### 5.6 Summary

Two models were tried. The grid-level one failed for a documented reason
(seven usable training sequences, R-squared = -0.0655). The per-well one
succeeded on its own terms (R-squared = 0.575, +0.136 above persistence),
but on a narrower problem than Phase 2 solves: it predicts at wells with
history, not at unmonitored locations. The per-well result complements
Phase 2, it does not replace it.
## 6. Synthetic RL Environment (Phase 4)

This section describes the simulator that the offline reinforcement
learning policies were trained in. It exists because there is no real
dataset of farmer pumping decisions paired with aquifer responses at the
required resolution in the study region, and because experimenting on a
real aquifer to learn what "too much" looks like is not an option. The
simulator is a stylised bucket model, not a calibrated hydrological
model, and Section 10.4 records the consequences of that.

### 6.1 Why a synthetic environment is necessary

Learning a pumping policy from historical data alone is not possible.
Farmers pump, the water responds, but the response is tangled with
rainfall, other farmers, and everything else in the watershed. No
controlled experiment exists in which one block pumps this much and a
neighbouring block pumps that much, and the water table is observed for
the next several years. Without such an experiment, there is no way to
learn a policy that has seen the consequences of over-extraction.

The offline reinforcement learning approach is the methodologically
correct one for this setting: learn from a fixed dataset of experiences
without requiring interactive exploration of the real environment. But
the experiences still have to come from somewhere, and the somewhere in
this project is a simulator that encodes the water balance explicitly.

The simulator is documented completely so that a reviewer can challenge
any assumption, not because it is believed to be accurate.

### 6.2 The water balance

Each step simulates one quarter. The state consists of six numbers:

  gw_level          current groundwater depth, in meters (bigger is worse)
  recent_rainfall   recent rainfall, in millimeters
  crop_water_demand crop water demand proxy, normalized 0 to 1
  month_sin         sine of the month, seasonal encoding
  month_cos         cosine of the month, seasonal encoding
  extraction_rate   recent pumping intensity, normalized 0 to 1

The action is a discrete pumping intensity with five levels: 0, 0.25,
0.50, 0.75, or 1.00 of maximum extraction per block per quarter. The
action-to-fraction mapping is fixed by line 148 of
src/rl/synthetic_reward.py (Section 9.3).

The state transition is a water-balance equation:

  next_gw_level = max(0, gw_level + drawdown - recharge + noise)

  drawdown = pumping_fraction * 2.0
  recharge = 0.15 * recent_rainfall / 100.0
  noise    = Normal(0, 0.1)

The 2.0 and the 0.15 are the two arithmetic constants that the entire
simulator rests on. Section 10.4 shows that a 3x3 sweep of these two
constants produces a 226-fold spread in the magnitude of the reported
CQL-versus-BC advantage. The simulator is honest about this: it is a
stylised model with two tunable numbers, not a calibrated aquifer
response function.

### 6.3 The reward function

The reward at each step combines three components:

  reward = w1 * crop_revenue_proxy(action, crop_water_demand)
         + w2 * domestic_supply_score(next_gw_level)
         - w3 * depletion_penalty(next_gw_level, threshold)

where crop_revenue_proxy is a normalized 0-1 proxy for how much of the
crop's water demand was met, domestic_supply_score is a normalized 0-1
measure of how adequate the water level is for domestic use, and
depletion_penalty grows quadratically below a configurable sustainability
threshold.

The default weights are w1 = 1.0, w2 = 2.0, w3 = 0.5. Section 7.4 uses
a weights [1.0, 2.0, 0.0] variant to test how sensitive the comparison
is to the depletion-penalty term.

Crop revenue rises with pumping until the crop's water demand is met,
then flattens. No real price or yield data backs the magnitude of the
formula. The domestic supply score has a single fixed critical depth for
the whole study region, not calibrated to well depth or local
infrastructure. The depletion penalty uses a single sustainability
threshold of 15 meters by default, not calibrated to aquifer type or
geology.

### 6.4 The six assumptions

Six numbered assumptions govern the simulator. They are reproduced here
so the report does not rely on Section 10.4 alone to state them.

**A1. Crop water demand is a simple seasonal proxy.** Higher in pre-
monsoon and summer. Not derived from crop-type maps, irrigation records,
or FAO crop-coefficient tables. Additionally, the demand decouples from
the calendar after the first step of each episode, a known defect
documented in Section 10.10.

**A2. Crop revenue rises with pumping until demand is met, then flattens.**
No real price or yield data backs the magnitude of the formula.

**A3. There is one fixed, location-independent critical depth for
domestic supply.** Real adequacy depends on well depth, pump capacity,
and local infrastructure, none of which is modeled.

**A4. One sustainability threshold applies to the whole study region.**
Real sustainable yield varies by aquifer type and geology. Configurable,
canonical value 15 m, not a validated threshold.

**A5. Pumping lowers the water table linearly at up to 2 m per quarter
at full pumping.** Not a calibrated aquifer response function. This
single number drives the largest reported effect in the project.

**A6. Recharge is a fixed 15% fraction of recent rainfall.** Not a
physically calibrated infiltration model.

The simulator's own source file states all six in its header comments.
Any claim downstream of Phase 4 inherits all six.

### 6.5 Directional validation against real wells

The simulator is not calibrated, but is it qualitatively wrong? That is
a fair question and it was actually tested, which is more than most
synthetic environments get.

The comparison is the correlation between rainfall and the next-quarter
change in water depth, computed on real CGWB wells and inside the
simulator:

  Quantity                          Correlation
  Real wells: rainfall vs. next change     -0.325
  Simulator: rainfall vs. next change      -0.117
  Ratio of magnitudes                       ~2.78x

Both correlations are negative, which is the expected sign: more rain,
shallower next reading. The simulator's effect is weaker than reality
by roughly a factor of 2.78. This establishes that the simulator's
rainfall-recharge relationship is not qualitatively wrong in direction.

It does not establish that the fixed-fraction form of A6 or its 0.15
coefficient is calibrated. Only the sign and the rough magnitude of the
correlation are safely comparable, not the underlying slopes, because
the two quantities are constructed differently — real physical units
over irregular elapsed time, versus a smoothed, seeded, randomly drifting
proxy on a uniform quarterly clock.

The observed side was computed from held-out wells only. That is
deliberate: this is a descriptive check, not a model-training step, so
it does not "use up" the held-out set.

### 6.6 Dataset generation and coverage

The offline RL dataset was generated by running the simulator with three
hand-written behaviour policies, 500 episodes each, 20 quarters per
episode:

  Random:         picks any of the five actions uniformly. Reference
                  floor. Guarantees every action appears.
  Greedy:         pumps exactly what the crop needs, ignoring depletion.
                  Deliberately unsustainable. Provides real examples of
                  what heavy extraction does to the water table.
  Conservative:   cuts to zero once the water level approaches 80% of
                  the sustainability threshold, and otherwise caps
                  pumping at half of demand. Provides examples of the
                  sustainable end.

Total: 1,500 episodes, 30,000 transitions. Initial states are seeded from
real predicted water depths (from the Phase 2 downscaler) and real
rainfall values, sampled with spatial-block stratification so every part
of the study region is represented.

The three behaviour policies were chosen deliberately, not accidentally.
A dataset where everyone always pumps moderately teaches a policy nothing
about what happens when you pump hard. Without greedy extraction, the
learned agents would never see the consequence of over-pumping. Without
conservative, they would never see the sustainable end.

### 6.7 The nine sanity checks

Nine automated checks were run on the generated dataset before anything
was trained on it:

  1. No NaN or infinite values in any state or reward.
  2. State values within plausible ranges.
  3. A workable spread of actions across the five levels.
  4. Rewards that actually vary (not a constant).
  5. No duplicate transitions.
  6. Next states genuinely respond to the action.
  7. More pumping produces more drawdown (monotonicity).
  8. More rain produces more recharge (monotonicity).
  9. Reward varies across actions within a fixed state.

All nine passed. The action distribution is 0%: 28.1%, 25%: 32.7%, 50%:
18.3%, 75%: 12.7%, 100%: 8.3%. The 100% action (the most consequential
one) is the least represented, which is documented as a limitation in
Section 10.7 and drives some of the ambiguity in the policy comparison.

The one bug found during sanity checking was in the checker, not the
data: two checks assumed that a bigger water-level number meant "better",
when in this project it means "deeper, therefore worse". Correcting the
sign convention flipped the interpretation of two checks and changed
none of the underlying numbers. An independent correlation computed
straight from the raw arrays confirmed the corrected direction at
r = -0.13, which is right sign, weak, and consistent with how crude A6
is.

### 6.8 The simulator's inherited limitation

Every result in Phase 5 was generated by a simulator whose initial water
depths come from the v1 downscaler, not the v2 downscaler that Section 4
describes. This is confirmed by direct inspection of the saved model
file: it takes 11 inputs, which is v1's signature; v2 takes 16.

The simulator was built and validated before v2 existed. Re-seeding from
v2 would produce a different pipeline than the one built and validated,
so it is classified as out of scope rather than as unfinished work.
Whether re-seeding from v2 would change any of the Phase 5 conclusions
is untested, and is recorded in Section 12 as future work.

### 6.9 What this section establishes

Three things, for the rest of the report:

First, the simulator is fully documented. Its water balance, its reward
function, its six assumptions, and its nine sanity checks are all in the
report. A reviewer can challenge any of them, and the report does not
rely on the simulator being hidden behind a black box.

Second, the simulator is qualitatively correct in the direction it needs
to be. The rainfall-recharge correlation matches reality in sign and is
within a factor of 2.78 in magnitude. That is a real piece of external
validation, and it is modest in scope.

Third, the simulator is a stylised bucket model with two arithmetic
constants at its core. No amount of reinforcement-learning rigour makes
the reward function true. Every Phase 5 result in Section 7 is a
statement about this simulator, not a statement about real aquifer
behaviour. Section 10.4 records this as the deepest limitation in the
project.
## 7. Policy Evaluation (Phase 5)

This section reports what happened when two learned policies (Behavioral
Cloning and Conservative Q-Learning) were trained on the same offline
dataset and compared under a fixed evaluation protocol. It is the
section where the project's original claim — "CQL outperforms BC" — was
found not to replicate at 10 seeds.

The honest summary is short. At 10 training seeds, CQL does not
outperform BC by a statistically significant margin on four of the five
dataset families tested. On the fifth, BC is decisively better, for a
dataset-specific reason that does not generalise. The historical
3-seed result that motivated the project did not survive replication.

The rest of this section substantiates that summary.

### 7.1 The 10-seed comparison

Five dataset families were evaluated. Each is a distinct combination of
training data (the .h5 offline dataset) and the CQL conservatism
parameter alpha. For each family, ten independent training seeds were
used (42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999). Each policy
was evaluated on 500 episodes from a fixed evaluation seed (9999), and
the mean return was recorded.

  Family          Dataset                     Alpha   BC      CQL     CQL-BC
  ---------------------------------------------------------------------------
  base            offline_rl_dataset.h5       1.0     11.924  9.451   -2.473
  epsilon         ..._epsilon.h5              1.0      9.088  9.135   +0.047
  no_oracle       ..._no_oracle.h5            4.0      9.094  9.167   +0.073
  no_oracle_a1    ..._no_oracle.h5            1.0      9.094  9.244   +0.150
  no_oracle_a10   ..._no_oracle.h5            10.0     9.094  9.134   +0.040

Three things stand out.

First, the four no_oracle and epsilon families cluster tightly: CQL's
mean return exceeds BC's by between +0.040 and +0.150 return units in
every case. These are small gaps. No bootstrap confidence interval on
any of them excludes zero.

Second, the base family is on a different scale. BC's mean return is
11.92, versus 9.09 on the other four families. CQL's mean return is
essentially stable across all five families (9.13 to 9.24 on the non-base
families, 9.45 on base). BC improves dramatically on base; CQL does not.
This produces the -2.47 gap. It is a real result for the base dataset,
but it is not evidence that CQL is worse in general. It is evidence that
BC and CQL respond differently to the structure of this particular
dataset.

Third, the historical 3-seed result is absent. That result — the
epsilon-family comparison at seeds 42, 123, 2024 — produced a CQL-BC gap
of +0.1096 with 95% confidence interval [+0.050, +0.174]. That interval
excluded zero, which is why the project originally claimed that CQL
outperformed BC. At 10 seeds on the same family, the gap is +0.047 and
the confidence interval includes zero. The historical result does not
survive replication.

### 7.2 Why the historical 3-seed result does not replicate

Three diagnostic experiments examined why.

**Experiment 1: start-set sensitivity.** The evaluation uses a set of
initial states from which each policy is rolled out for 20 quarters. If
different evaluations use different initial-state sets, the returns are
not directly comparable. The historical 3-seed summary used a start set
that is unrecorded in the paper.

A diagnostic (revision_experiments/e1_sweep_vs_canonical.py) re-evaluated
the same models on a fixed, documented start set (`rf_grid`) and compared
results against the historical numbers. Under the fixed start set:

  Configuration        CQL - BC   95% CI                 Excludes zero
  ----------------------------------------------------------------------
  canonical (rf_grid)  -0.0129    [-0.1011, +0.0686]     no
  sweep_default        -0.0574    [-0.1282, +0.0044]     no

Both are null. The sign of the small residual gap depends on the start
set: on rf_grid it is slightly negative; on the unrecorded historical
start set it was slightly positive. Neither is significant.

**Experiment 2: BC training recipe.** A separate concern was that the
canonical BC models used different hyperparameters from a "sweep-default"
BC family, and that the historical result might have depended on which BC
recipe was used as the baseline. The diagnostic trained both BC variants
and compared them on identical starts:

  BC canonical - BC sweep_default:   -0.0445 [-0.104, 0.000]   null

The BC training recipe does not meaningfully affect the comparison. BC
models trained with canonical vs sweep-default hyperparameters differ by
less than the noise floor.

**Experiment 3: 10-seed replication.** The most direct test is simply to
run the same comparison at 10 seeds instead of 3. That is the table in
Section 7.1. The gap is +0.047 on epsilon at 10 seeds, versus +0.1096 at
3 seeds. The historical result was a seed artefact: three seeds is not
enough to distinguish a real effect from training noise on this problem,
and at 10 seeds the effect disappears.

### 7.3 The base family: what it does and does not say

The -2.47 gap on the base family is the largest effect in the entire
project. It deserves its own treatment.

What is true: on the base dataset, BC's learned policy achieves higher
mean return than CQL's learned policy, at every one of the ten training
seeds. The gap is not a seed artefact; it is consistent and large.

What is also true: BC's mean return on base (11.924) is much higher than
BC's mean return on any other family (9.088 to 9.094). BC learned something
from the base dataset's structure that it did not learn from the other
four. CQL did not: CQL's mean return on base (9.451) is within normal
variation of its mean return on the other four families (9.134 to 9.244).

The base dataset (`offline_rl_dataset.h5`) differs from the other four in
a way that is favourable to Behavior Cloning specifically. Whether this
is because base contains more of one kind of state-action pair, or has
different reward distribution, or something else, was not determined.
The base result is real, but it is dataset-specific, and it must not be
generalised.

The honest phrasing of the base result is: "on this particular dataset,
BC substantially outperforms CQL, and we do not know why beyond the fact
that the dataset structure favours BC." This is recorded in the report so
that no reader mistakes the small positive gaps on the other four families
for a general CQL advantage.

### 7.4 FQE and direct rollout disagree

Policies were evaluated by two methods. Direct rollout runs the policy in
the simulator for 500 episodes and measures the realised return. Fitted
Q Evaluation (FQE) estimates the policy's value from the offline dataset
without touching the simulator. These two methods share no machinery.

They disagree systematically. On the paired seed comparison
(revision_experiments/e8_estimator_contrast.json):

  FQE_gap - rollout_gap:   +0.437 [0.113, 0.762]   (95% CI, 10 seeds)

FQE over-estimates the CQL-BC gap relative to rollout by about 0.44
return units, and the confidence interval excludes zero. That is a
substantial disagreement on a metric where the underlying gap is in the
range of 0.05 to 0.15.

This disagreement is itself a finding. It says that offline policy
evaluation is unstable in this regime. FQE's value function approximations
are biased, and the bias is in the same direction (favouring the
conservative policy) that the CQL algorithm itself optimises for. This
is not an accident of implementation; it is a known property of FQE when
combined with CQL-trained critics.

The report treats the disagreement as a data point about offline policy
evaluation, not as a tiebreaker between the two policies. When two
evaluation methods disagree by 0.44 units on a metric whose point estimate
is 0.05 to 0.15, the correct conclusion is not "one method is right"; it
is "neither method is precise enough to settle the question."

### 7.5 What this section concludes

The evidence at 10 seeds does not support the claim that CQL outperforms
BC. On four of five dataset families, the CQL advantage is between +0.04
and +0.15 return units, with no confidence interval excluding zero. On
the fifth family, BC substantially outperforms CQL, for a dataset-specific
reason. The historical 3-seed positive result is best described as a seed
artefact — a plausible under three seeds, not reproducible at ten.

The FQE-versus-rollout disagreement of +0.44 on a metric whose point
estimate is 0.05 to 0.15 makes the whole comparison more uncertain, not
less. Neither estimator is precise enough to establish CQL superiority at
the scale of effect observed.

The honest claim, recorded in Section 11.2 and restated here, is: at 10
seeds, in-distribution, CQL does not outperform BC by a statistically
significant margin on any of the four comparable dataset families. Any
report sentence that begins "CQL beats BC" must qualify which dataset,
which number of seeds, and which start set.

### 7.6 What this section does not conclude

It does not conclude that CQL is a worse algorithm than BC. The
comparison is dataset-specific and small. In a different regime — a
different simulator, a different reward function, a different offline
dataset — the picture could reverse again. What this section establishes
is that in this particular setup, the CQL advantage that motivated the
project does not exist at 10 seeds.

Section 8 examines out-of-distribution behaviour and finds a different
picture: both policies fail catastrophically, and CQL fails slightly less
than BC. That is also not evidence that CQL is better. It is evidence
that both are broken outside the training distribution.
## 8. Out-of-Distribution Behaviour

This section reports what happens when the two learned policies are
evaluated on states that are progressively further from the training
distribution. It is a short section because the honest conclusion is
short: under all four committed OOD evaluation files, CQL's return
exceeds BC's in every band, and the gap grows with distance — but the
magnitudes are such that both policies are effectively failing. The
section also records that the internal project explainer's claim of a
sign reversal in the OOD regime does not reproduce in any committed
artefact, and is marked superseded.

### 8.1 The OOD evaluation design

Four out-of-distribution evaluation files are committed to the repository:

  ood_scaled_result.json            (primary critic)
  ood_scaled_meanq_result.json      (second critic, mean-Q head)
  ood_w3_full_eval_only.json        (reward weights 1.0, 2.0, 0.5)
  ood_w3_zero_eval_only.json        (reward weights 1.0, 2.0, 0.0)

Each file bins held-out states into four bands by increasing statistical
distance from the training distribution:

  in_dist    states drawn from the training distribution
  mild       mild shift from the training distribution
  moderate   moderate shift
  severe     severe shift

Distance is measured with Mahalanobis distance against the training-state
covariance. Each band contains enough states to evaluate both policies
in every file. The two `ood_w3` files additionally vary the reward
weighting used at evaluation time, without retraining.

### 8.2 The per-band numbers

Return means and CQL-BC gaps by band, from each file:

Primary critic (ood_scaled_result.json):

  Band         BC        CQL       CQL-BC
  in_dist     -12.09     -6.78     +5.31
  mild        -40.70    -37.11     +3.59
  moderate    -65.55    -59.61     +5.94
  severe      -90.71    -86.22     +4.49

Second critic (ood_scaled_meanq_result.json):

  Band         BC        CQL       CQL-BC
  in_dist     -10.15     -5.91     +4.24
  mild        -39.58    -32.80     +6.78
  moderate    -65.09    -54.02    +11.07
  severe      -92.37    -73.19    +19.18

Reward weights [1.0, 2.0, 0.5] (ood_w3_full_eval_only.json):

  Band         CQL-BC    95% CI
  in_dist      +1.23    [ 0.12,  2.38]
  mild         +3.48    [ 1.89,  5.13]
  moderate     +7.06    [ 3.90, 10.43]
  severe       +6.32    [ 1.88, 10.96]

Reward weights [1.0, 2.0, 0.0] (ood_w3_zero_eval_only.json):

  Band         CQL-BC    95% CI
  in_dist      +1.17    [ 0.12,  2.26]
  mild         +3.32    [ 1.81,  4.87]
  moderate     +6.73    [ 3.67,  9.95]
  severe       +6.01    [ 1.71, 10.54]

The pattern is consistent across all four files: CQL's return exceeds
BC's in every band, including in-distribution, and the gap grows with
distributional distance in three of the four files.

### 8.3 The endpoint contrast

The endpoint contrast asks a sharper question than the per-band gaps: as
we move from in_dist to severe, does CQL's advantage grow? It is
computed as severe-band gap minus in_dist-band gap, on a per-seed basis.

Under reward weights [1.0, 2.0, 0.5], the endpoint contrast is +5.09
with 95% CI [0.05, 10.14]. The interval excludes zero. Under this
reward weighting, the growth of CQL's advantage with distance is
statistically significant.

Under reward weights [1.0, 2.0, 0.0], the endpoint contrast is +4.84
with 95% CI [-0.08, 9.76]. The interval includes zero. Under this
weighting, the growth is not significant.

So the growth-with-distance finding is real under one reward weighting
and not real under another. That is consistent with the finding in
Section 10.4 that reward-weight choices have an outsize effect on
observed metrics: two arbitrary constants can move the answer by more
than the effect being measured.

### 8.4 Both policies fail catastrophically

The numbers above are drawn from a band structure with deeply negative
absolute returns. BC's return falls from -12.09 at in_dist to -90.71 at
severe under the primary critic. CQL's falls from -6.78 to -86.22. Both
policies are scoring dramatically worse than they do on the training
distribution, and their absolute performance has collapsed.

A +4 to +19 return-unit gap between two policies whose returns are both
in the -80 to -90 range is not evidence that either policy is working.
It is evidence that one policy is failing somewhat less catastrophically
than the other. The distinction matters for how the result should be
read.

The per-seed deltas confirm the caveat. In the weights [1.0, 2.0, 0.5]
file, the per-seed endpoint contrasts are:

  -4.51, +2.08, +7.60, +18.57, +3.73, +2.78, +10.26, +5.25, -4.91, +10.07

Two of the ten seeds show a negative endpoint contrast (CQL's advantage
shrinks with distance rather than growing). The mean is positive
because the other eight seeds dominate it, but the underlying effect is
not uniform.

Correct framing: outside the training distribution, both policies fail
catastrophically. CQL fails somewhat less than BC on average, with wide
per-seed variance, and the magnitude of the difference depends on the
reward weighting used at evaluation. This is not a working policy.

### 8.5 FQE and rollout disagree on the OOD scale

Section 7.4 established that FQE and direct rollout disagree on the
in-distribution comparison by +0.44 (e8_estimator_contrast.json, 95% CI
[0.11, 0.76]). The disagreement persists on the OOD evaluation. FQE's
value-function approximations become less reliable at greater distance
from the training distribution, and the direction of the bias is
consistent with the direction of the CQL advantage — FQE systematically
over-estimates the CQL-BC gap, and it does so more at greater distance.

This is evidence that offline policy evaluation is itself unreliable in
the OOD regime. It is not evidence for or against either policy. When
the estimator's bias is the same size as the effect being measured, the
correct conclusion is that the estimator cannot settle the question.

### 8.6 The internal explainer's Section 9.5f claim is superseded

An earlier internal project document (Trishna-OPAL-Team-Explainer.pdf,
Section 9.5f) claimed that the CQL advantage "reverses" out of
distribution. The claim was illustrated with a table showing CQL-BC
values of approximately -16.01 in-distribution and -38.40 in the severe
band, with the same negative sign across all four bands.

No committed artefact reproduces that table. All four committed OOD
files show CQL-BC positive in every band, with the gap typically growing
rather than reversing with distance.

The explainer's provenance for the -16.01 and -38.40 numbers is not
recoverable from the committed code or data. The most likely explanation
is that it described an earlier code branch, or an FQE-only estimator,
that the current artefacts have replaced. Whatever its provenance, the
claim is not supported by the evidence on file.

Section 10.9 records this supersession as a limitation. Section 11.2
lists "CQL reverses out-of-distribution" as a claim the report does not
make. reports/phase5_provenance.md documents the supersession with the
dates and the file references. Any future reader who encounters the
explainer's §9.5f table should treat it as historical, not as current.

### 8.7 Summary

The honest statement of the OOD result is:

  Under four independent evaluation configurations, CQL's mean return
  exceeds BC's in every distributional band, and the gap grows with
  distance in three of four files. The endpoint contrast survives the
  paired confidence interval test under one reward weighting but not
  under another. But both policies collapse to catastrophic returns in
  the severe band (BC approximately -90, CQL approximately -86). A gap
  of four to nineteen units between two failing policies is not a
  working policy. And the internal explainer's claim of a sign reversal
  does not reproduce in any committed artefact.

That paragraph is the full finding. Anything stronger than it overstates
the evidence; anything weaker than it hides what was actually observed.
## 9. Explainability

This section describes the layer that turns the trained CQL critic into
plain-language advice. It is a translation layer, not a validation layer.
That distinction is the whole point of the section and is the reason the
report presents advice with explicit confidence levels and explicit
limitations.

### 9.1 What the module does

The module (src/explainability/qvalue_translation.py) exposes a single
function, translate_to_advice, that takes a dictionary mapping each
discrete action to a Q-value from the critic and returns:

  - recommended_action  the action with the highest Q-value
  - confidence_level    "high", "medium", or "low"
  - plain_language_text one sentence naming the recommended action and
                        the confidence
  - q_gap               the difference between the best and second-best
                        Q-values
  - alternatives        the next three actions with their Q-values and
                        the deltas from the best

The module is deterministic, has no dependencies beyond the standard
library, and is covered by eight unit tests
(tests/test_qvalue_translation.py). The tests check the confidence rule,
the ranking of alternatives, the empty-input and single-action
exceptions, and the fact that the confidence level comes from the Q-gap
rather than from any hardcoded template.

### 9.2 Why confidence comes from the Q-gap

The confidence level is not a template choice. It is derived from the
Q-gap: how much better the best action scores than the second-best.

  gap >= 0.5  -> "high"
  gap >= 0.1  -> "medium"
  gap  < 0.1  -> "low"

A large gap means the critic distinguishes the best action clearly. A
small gap means it does not, and the recommendation is a coin flip that
should not be presented as decisive. The thresholds (0.5, 0.1) are
heuristic. They were calibrated against the observed reward scale: per-
episode return on the no_oracle_alpha1 family is approximately 9.4 over
20 steps, so per-step reward is approximately 0.47. A Q-gap of 0.5 is
roughly one step of reward; a gap of 0.1 is roughly one fifth of a step.
The module accepts tau_high and tau_low as keyword arguments so a user
with a different critic scale can override them.

The module uses only the Q-gap, not absolute Q-values. That makes it
scale-invariant to the critic's overall offset and means it does not
have to know whether the critic's value function is calibrated. It is
about relative preference, not absolute value.

### 9.3 The action labels come from the code, not from a template

The five actions are pumping intensities, and their meaning is fixed by
line 148 of src/rl/synthetic_reward.py:

  pumping_fraction = action / (n_actions - 1),  n_actions = 5

So the five actions are exactly 0.00, 0.25, 0.50, 0.75, and 1.00 of
maximum pumping. The labels in ACTION_LABELS reflect that mapping and
are copied into the plain-language output:

  action 0 -> "no pumping this quarter"
  action 1 -> "pump 25 percent of maximum extraction"
  action 2 -> "pump 50 percent of maximum extraction"
  action 3 -> "pump 75 percent of maximum extraction"
  action 4 -> "pump 100 percent of maximum extraction (justified only
               when crop demand is high; otherwise the depletion penalty
               dominates)"

Action 4 is not "more is better." Crop revenue in the reward function
caps at the crop's water demand, so pumping past demand gives no extra
revenue but still incurs the depletion penalty. The label text says so.
This is the one place where the plain-language layer adds a fact that
is not in the raw Q-value, and it adds it in the direction of less
extraction, not more.

### 9.4 What this module cannot do

The module is deliberately modest. It cannot do four things, and the
report does not claim that it does.

First, it cannot tell you whether the critic is correct. It reads
Q-values and reports which action the critic prefers. If the critic
over-values a harmful action, the module will recommend that action with
high confidence, and it will present the recommendation in English.

Second, it cannot validate the reward function. The reward function is a
stylised bucket model with assumptions A1 to A6 (Section 10.4). The
advice inherits every one of those assumptions. If the reward function
is wrong in some direction, the advice is wrong in the same direction.

Third, it cannot guarantee that the recommended action is safe for the
aquifer. Safety would require a calibrated physical model, not a stylised
simulator. The Q-values come from the stylised simulator, so the advice
comes from the stylised simulator.

Fourth, it cannot replace held-out validation. The advice is not
validated against real farmer outcomes, because no such dataset exists
at the required resolution. The advice's quality is bounded by the
quality of the critic, the reward function, and the simulator, and all
three have documented limitations.

### 9.5 The consequences for how the advice should be read

The report's position on the advice layer is straightforward. The module
is a translation of what the critic already believes. It does not
endorse. A user of the system who receives a recommendation should treat
it as a starting point for local expertise, not as a substitute for it.

The confidence level is the most useful signal in the output. When the
confidence is "low", the critic is close to indifferent between the top
actions, and the user should expect the recommendation to change
substantially with small changes in state. When the confidence is
"high", the critic is confident about the ranking, but confidence in the
ranking is not evidence that the ranking is correct — a confident but
wrong critic will produce confident but wrong advice.

The report does not present the module as a decision-support system for
real use. It presents it as a demonstration that the Q-value outputs can
be turned into plain-language outputs with a documented confidence
metric, and as a way to make the offline RL layer auditable in natural
language. Section 12 records calibrated validation of this layer as
future work, dependent on real farmer decision data that does not
currently exist.

### 9.6 Summary

The explainability layer is small and honest. It translates the critic's
Q-values into one sentence and a confidence level, using action labels
copied from the reward function and thresholds calibrated against the
observed reward scale. It is covered by eight unit tests. It does not
validate the critic, the reward function, the simulator, or real-world
safety, and the report does not claim that it does.
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












