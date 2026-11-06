# Region of Interest - Written Justification

Closes the ROI TODO flagged as unfinished in the internal explainer and in
earlier versions of this project. Every number below is measured from the
committed raw CGWB file; none are asserted.

## 1. Source data

  File:          data/raw/cgwb_raw.csv
  Size:          65.43 MB
  Rows:          550,850
  SHA-256:       5fbcc59d8a917a5dc9d9d8b7d56f3db69297f5910b2966f72838f337ec0409bc
  Retrieved:     2026-09-17T08:13:34Z
  Source URL:    https://ckandev.indiadataportal.com/dataset/8ceb3cea-a8a9-4624-af75-fc92429d8fa5/resource/580a8f6e-3d86-4ca7-ac7d-cd5df12b443c/download/cgwb-changes-in-depth-to-water-level.csv

Provenance is recorded in data/raw/cgwb_raw.manifest.csv, which is
committed alongside the raw file.

Columns: id, date, state_name, state_code, district_name, district_code,
station_name, latitude, longitude, basin, sub_basin, source, currentlevel,
level_diff.

## 2. Bounding box

  Latitude:   15.60 N to 22.00 N
  Longitude:  72.60 E to 80.90 E

This is a rectangle, not a state boundary. It was chosen to cover a
hydrologically connected groundwater belt rather than to match an
administrative outline. It is not the shape of Maharashtra.

## 3. What the rectangle actually contains

Filtering the raw file by this bbox yields 92,227 rows, which equals the
interim file's reading count exactly. There is no additional geographic
filter downstream.

State composition inside the bbox:

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

Three states (Maharashtra, Telangana, Karnataka) account for 82.8% of the
readings and 83.7% of the distinct wells. The remaining six states are
peripheral contributions to the rectangle, not central to the study.

The internal explainer described the region as "Maharashtra." That is a
geographic shorthand, not the actual filter. The actual ROI is a
Maharashtra-led three-state core. The report must state this. Calling it
"Maharashtra" would overstate the state-level cleanliness of the data and
understate the geographic heterogeneity the model actually sees.

## 4. Maharashtra district-level density

Top 15 districts inside the bbox, by distinct well count:

  District      Wells  Readings
  Amravati        116      2420
  Yavatmal        108      2317
  Nagpur          104      2403
  Nashik           92      2172
  Chandrapur       80      1923
  Ahmednagar       80      1631
  Wardha           76      1726
  Thane            74      1535
  Buldhana         73      1636
  Ratnagiri        73      1773
  Solapur          71      1602
  Jalgaon          70      1334
  Nanded           70      1477
  Satara           69      1399
  Pune             63      1343

The density is Vidarbha-heavy (Amravati, Yavatmal, Nagpur, Chandrapur,
Wardha, Buldhana). The Marathwada region cited as an example in the
original master prompt (Aurangabad, Latur, Beed, Parbhani) is present but
is not the densest block. The northern and eastern districts dominate.

## 5. Time range

  First reading:  2013-01-01
  Last reading:   2022-07-28
  Total span:     approximately 9 years 7 months

Bounded by:
  - CGWB dataset collection start (2013)
  - CGWB dataset collection end (mid-2022)

Cross-checked against satellite layer coverage:
  - Sentinel-2 launched June 2015, so early 2013-2015 has no Sentinel-2
    NDVI; the pipeline uses MODIS-equivalent NDVI or drops the NDVI
    channel for early dates. Age-days features in v2 record staleness.
  - GRACE has a mission handover gap July 2017 to April 2018, roughly
    9 months of the record. Documented in reports/phase5_provenance.md.

## 6. Sample size distribution

  4,063 distinct wells
  92,227 readings
  22.7 readings per well on average
  Min: 1 reading for a well (many wells visited only once)
  Max: 116 readings for a well

The unbalanced distribution means most wells contribute very few readings.
This limits per-well time-series modelling (Phase 3) to wells with at
least 6 consecutive quarters; that count is recorded in the Phase 3
diagnostic, not here.

## 7. Justification summary

The bbox was chosen because it captures a dense, contiguous, and
hydrologically connected set of CGWB observation wells across a
Maharashtra-led three-state region, with adequate spatial and temporal
coverage for the 64x64 grid and 12-month sliding windows used downstream.
The bbox is a rectangle for convenience; it is not a claim that the region
is homogeneous, nor a claim that the study is "about Maharashtra."

Three limitations follow from this choice and are recorded here so they
cannot be forgotten:

  1. The region is not Maharashtra. It is a Maharashtra-led three-state
     core with six peripheral states at low density. Any claim about
     "groundwater in Maharashtra" must be qualified.

  2. District-level density is uneven (Amravati 116 wells, Pune 63).
     Models that use location implicitly will be more accurate in dense
     districts. The quadrant holdout test in the explainer Section 5.7
     measures this.

  3. The bbox extends slightly over the Arabian Sea on the west and into
     non-CGWB areas on the south and east. Kriging handles heterogeneous
     coverage; the report's limitations section must state that the model
     sees uneven support and its skill is uneven.

## 8. What this closes

The earlier version of this project left the ROI as a placeholder with a
TODO marker. That TODO is closed: the region is measured, not asserted,
and every number in this file is reproducible from the committed raw CSV
with the commands recorded in the Step 3A git history.
