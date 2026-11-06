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

TODO (~500 words). This section is a direct quote of the claims and "do not say" lines from
reports/final_report_claims.md. Do not paraphrase.

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
