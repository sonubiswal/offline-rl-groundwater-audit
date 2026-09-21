# ConvLSTM Data Availability Diagnostic

## Summary
The grid‑based ConvLSTM failed to produce meaningful predictions (R² ≈ -0.07) due to structural data scarcity, not model limitations.

## Detailed Gap Analysis
The usable Kriged surfaces (25 complete dates) break into isolated runs:

| Run | Dates | Length | Notes |
| :--- | :--- | :--- | :--- |
| A | 2015‑08 → 2016‑01 | 3 quarters | Early Sentinel‑2 coverage sparse |
| B | 2016‑05 → 2017‑01 | 4 quarters | – |
| C | 2017‑05 → 2017‑08 | 2 quarters | – |
| D | 2018‑05 → 2020‑01 | 13 quarters | Only run long enough for 6‑quarter sequences |
| E | 2020‑08 → 2021‑01 | 3 quarters | Likely COVID‑19 field campaign disruption |

## Key Gaps
- **2013–2015:** Pre‑Sentinel‑2 coverage.
- **2017–2018:** 9‑month reporting gap (unknown cause).
- **2020:** COVID‑19 disruption (confirmed in CGWB records).

## Conclusion
Only run D (2018‑2020) contributes sequences, yielding exactly 7 ConvLSTM sequences – insufficient for deep learning. Even with relaxed completeness requirements, only a handful of extra dates would be recovered, not enough to close the multi‑month structural gaps. The ConvLSTM approach is not viable with the current data density.