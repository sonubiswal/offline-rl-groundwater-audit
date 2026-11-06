"""
TRISHNA-OPAL
Spatially Aware Conformal Prediction Experiment

Methods:
    C0 = Global split-conformal baseline
    C1 = Distance-stratified split-conformal

IMPORTANT:
    - Calibration and test are separated at the WELL level.
    - Calibration predictions MUST be out-of-sample.
    - The final test wells are never used to select/calibrate the method.
    - No true groundwater depth is used as an uncertainty predictor.
    - C1 uses distance to nearest MODEL-TRAINING well.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("conformal_input.csv")
OUTPUT_DIR = Path("conformal_results")

# Column names
COL_WELL_ID = "well_id"
COL_TRUE_Y = "true_y"
COL_PRED_Y = "pred_y"
COL_DISTANCE = "distance_km"

# Required split column:
#   model_train (optional)
#   calib
#   test
COL_SPLIT = "split"

# Conformal settings
ALPHA = 0.10
NOMINAL_COVERAGE = 1.0 - ALPHA

# C1 spatial bins.
# These are deliberately coarser than the original 7-bin
# distance analysis because tiny calibration groups are unstable.
DIST_BINS = [0.0, 5.0, 10.0, 20.0, np.inf]
DIST_LABELS = [
    "0-5km",
    "5-10km",
    "10-20km",
    "20km+",
]

# Minimum calibration observations per spatial stratum.
# If a stratum is smaller, C1 falls back to the global quantile.
MIN_CALIB_N = 100

# Reproducibility
SEED = 2026

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("TRISHNA-OPAL CONFORMAL CALIBRATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

required_columns = [
    COL_WELL_ID,
    COL_TRUE_Y,
    COL_PRED_Y,
    COL_DISTANCE,
    COL_SPLIT,
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f"  - {c}" for c in missing)
    )

df = df.copy()

# Remove rows with missing required values.
before_n = len(df)

df = df.dropna(
    subset=[
        COL_WELL_ID,
        COL_TRUE_Y,
        COL_PRED_Y,
        COL_DISTANCE,
        COL_SPLIT,
    ]
).copy()

after_n = len(df)

print(f"Input rows:       {before_n}")
print(f"Usable rows:      {after_n}")
print(f"Rows removed:     {before_n - after_n}")


# ============================================================
# 3. SPLIT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — WELL-LEVEL SPLIT VALIDATION")
print("=" * 70)

allowed_splits = {"model_train", "calib", "test"}
required_splits = {"calib", "test"}

observed_splits = set(df[COL_SPLIT].unique())

# Check for missing required splits
missing_required = required_splits - observed_splits

if missing_required:
    raise ValueError(
        f"Missing required split labels: {missing_required}\n"
        f"Expected at least: {required_splits}"
    )

# Check for unexpected split labels
unexpected = observed_splits - allowed_splits

if unexpected:
    raise ValueError(
        f"Unexpected split labels: {unexpected}\n"
        f"Expected only: {allowed_splits}"
    )

# Determine which splits are present
has_model_train = "model_train" in observed_splits

if has_model_train:
    train_wells = set(
        df.loc[df[COL_SPLIT] == "model_train", COL_WELL_ID]
    )
else:
    train_wells = set()
    print(
        "[INFO] No model_train rows present — "
        "calibration/test-only experiment."
    )

calib_wells = set(
    df.loc[df[COL_SPLIT] == "calib", COL_WELL_ID]
)

test_wells = set(
    df.loc[df[COL_SPLIT] == "test", COL_WELL_ID]
)

if has_model_train:
    print(f"Model-training wells: {len(train_wells)}")
print(f"Calibration wells:    {len(calib_wells)}")
print(f"Test wells:           {len(test_wells)}")

# ------------------------------------------------------------
# WELL OVERLAP CHECK
# ------------------------------------------------------------

overlap_train_calib = train_wells & calib_wells
overlap_train_test = train_wells & test_wells
overlap_calib_test = calib_wells & test_wells

if overlap_train_calib:
    raise RuntimeError(
        f"LEAKAGE: {len(overlap_train_calib)} wells occur "
        "in model-training and calibration sets."
    )

if overlap_train_test:
    raise RuntimeError(
        f"LEAKAGE: {len(overlap_train_test)} wells occur "
        "in model-training and test sets."
    )

if overlap_calib_test:
    raise RuntimeError(
        f"LEAKAGE: {len(overlap_calib_test)} wells occur "
        "in calibration and test sets."
    )

print("\n[PASS] No well appears in more than one split.")
print(
    "[NOTE] Well-disjointness is required for this experimental design, "
    "but does not by itself prove conformal exchangeability."
)


# ============================================================
# 4. BUILD DATASETS
# ============================================================

calib_df = df[df[COL_SPLIT] == "calib"].copy()
test_df = df[df[COL_SPLIT] == "test"].copy()

if len(calib_df) == 0:
    raise ValueError("Calibration set is empty.")

if len(test_df) == 0:
    raise ValueError("Test set is empty.")

print("\nCalibration observations:", len(calib_df))
print("Test observations:       ", len(test_df))


# ============================================================
# 5. VERIFY CALIBRATION PREDICTIONS ARE PRESENT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — CALIBRATION PREDICTION CHECK")
print("=" * 70)

if not np.isfinite(calib_df[COL_PRED_Y]).all():
    raise ValueError(
        "Calibration predictions contain NaN/Inf values."
    )

if not np.isfinite(test_df[COL_PRED_Y]).all():
    raise ValueError(
        "Test predictions contain NaN/Inf values."
    )

print("[PASS] Calibration predictions are present.")
print("[PASS] Test predictions are present.")

print(
    "\nIMPORTANT: These predictions must be genuinely out-of-sample.\n"
    "The model producing pred_y must NOT have been trained on calibration wells."
)


# ============================================================
# 6. NONCONFORMITY SCORES
# ============================================================

calib_df["abs_residual"] = np.abs(
    calib_df[COL_TRUE_Y] - calib_df[COL_PRED_Y]
)

test_df["abs_residual"] = np.abs(
    test_df[COL_TRUE_Y] - test_df[COL_PRED_Y]
)


# ============================================================
# 7. CONFORMAL QUANTILE
# ============================================================

def conformal_quantile(scores, alpha=0.10):
    """
    Finite-sample split-conformal quantile.

    Uses the order-statistic formulation rather than relying
    on an arbitrary interpolation rule.
    """

    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    n = len(scores)

    if n == 0:
        raise ValueError("No finite calibration scores.")

    # k = ceil((n + 1) * (1-alpha))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    # Clamp to [1, n]
    k = max(1, min(k, n))

    ordered = np.sort(scores)

    return float(ordered[k - 1])


# ============================================================
# 8. C0 — GLOBAL CONFORMAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — C0: GLOBAL CONFORMAL")
print("=" * 70)

q_global = conformal_quantile(
    calib_df["abs_residual"],
    alpha=ALPHA,
)

test_df["C0_q"] = q_global

test_df["C0_lower"] = (
    test_df[COL_PRED_Y] - test_df["C0_q"]
)

test_df["C0_upper"] = (
    test_df[COL_PRED_Y] + test_df["C0_q"]
)

test_df["C0_covered"] = (
    (test_df[COL_TRUE_Y] >= test_df["C0_lower"])
    &
    (test_df[COL_TRUE_Y] <= test_df["C0_upper"])
)

test_df["C0_width"] = (
    test_df["C0_upper"] - test_df["C0_lower"]
)

c0_coverage = test_df["C0_covered"].mean()
c0_mean_width = test_df["C0_width"].mean()
c0_median_width = test_df["C0_width"].median()

print(f"Calibration N:       {len(calib_df)}")
print(f"Global quantile:     {q_global:.6f}")
print(
    f"Coverage:            {100*c0_coverage:.2f}% "
    f"(nominal {100*NOMINAL_COVERAGE:.1f}%)"
)
print(f"Mean width:          {c0_mean_width:.4f}")
print(f"Median width:        {c0_median_width:.4f}")


# ============================================================
# 9. C1 — DISTANCE-STRATIFIED CONFORMAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — C1: DISTANCE-STRATIFIED CONFORMAL")
print("=" * 70)

calib_df["distance_bin"] = pd.cut(
    calib_df[COL_DISTANCE],
    bins=DIST_BINS,
    labels=DIST_LABELS,
    include_lowest=True,
    right=True,
)

test_df["distance_bin"] = pd.cut(
    test_df[COL_DISTANCE],
    bins=DIST_BINS,
    labels=DIST_LABELS,
    include_lowest=True,
    right=True,
)

q_by_bin = {}

print("\nCalibration strata:")

for label in DIST_LABELS:

    mask = calib_df["distance_bin"] == label

    scores = calib_df.loc[mask, "abs_residual"]

    n_calib = len(scores)

    if n_calib >= MIN_CALIB_N:

        q = conformal_quantile(
            scores,
            alpha=ALPHA,
        )

        source = "stratum"

    else:

        q = q_global
        source = "GLOBAL FALLBACK"

    q_by_bin[label] = q

    print(
        f"{label:>10s} | "
        f"N={n_calib:5d} | "
        f"q={q:8.4f} | "
        f"{source}"
    )


# Map quantiles to test observations
# NOTE: distance_bin is a pandas Categorical; cast to string before mapping.
test_df["C1_q"] = (
    test_df["distance_bin"]
    .astype(str)
    .map(q_by_bin)
    .astype(float)
)

# Check for unmapped distances
if test_df["C1_q"].isna().any():

    bad = test_df.loc[
        test_df["C1_q"].isna(),
        COL_DISTANCE
    ]

    raise RuntimeError(
        "Some test observations were not assigned a C1 "
        f"quantile. Example distances: {bad.head().tolist()}"
    )


test_df["C1_lower"] = (
    test_df[COL_PRED_Y] - test_df["C1_q"]
)

test_df["C1_upper"] = (
    test_df[COL_PRED_Y] + test_df["C1_q"]
)

test_df["C1_covered"] = (
    (test_df[COL_TRUE_Y] >= test_df["C1_lower"])
    &
    (test_df[COL_TRUE_Y] <= test_df["C1_upper"])
)

test_df["C1_width"] = (
    test_df["C1_upper"] - test_df["C1_lower"]
)

c1_coverage = test_df["C1_covered"].mean()
c1_mean_width = test_df["C1_width"].mean()
c1_median_width = test_df["C1_width"].median()

print("\nC1 overall:")
print(f"Coverage:     {100*c1_coverage:.2f}%")
print(f"Mean width:   {c1_mean_width:.4f}")
print(f"Median width: {c1_median_width:.4f}")


# ============================================================
# 10. COVERAGE BY DISTANCE STRATUM
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — C1 COVERAGE BY DISTANCE")
print("=" * 70)

distance_rows = []

for label in DIST_LABELS:

    subset = test_df[
        test_df["distance_bin"] == label
    ]

    if len(subset) == 0:
        continue

    distance_rows.append({
        "distance_bin": label,
        "n_test": len(subset),
        "coverage": subset["C1_covered"].mean(),
        "mean_width": subset["C1_width"].mean(),
        "median_width": subset["C1_width"].median(),
        "quantile": subset["C1_q"].iloc[0],
    })

distance_results = pd.DataFrame(distance_rows)

print(
    distance_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 11. WELL-LEVEL COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — WELL-LEVEL COVERAGE")
print("=" * 70)

def well_coverage_table(
    test_data,
    covered_column,
    width_column,
):

    rows = []

    for well_id, group in test_data.groupby(
        COL_WELL_ID
    ):

        rows.append({
            "well_id": well_id,
            "n_obs": len(group),
            "coverage": group[covered_column].mean(),
            "mean_width": group[width_column].mean(),
        })

    return pd.DataFrame(rows)


well_c0 = well_coverage_table(
    test_df,
    "C0_covered",
    "C0_width",
)

well_c1 = well_coverage_table(
    test_df,
    "C1_covered",
    "C1_width",
)

print(
    f"C0 mean well coverage: "
    f"{well_c0['coverage'].mean()*100:.2f}%"
)

print(
    f"C1 mean well coverage: "
    f"{well_c1['coverage'].mean()*100:.2f}%"
)

print(
    f"C0 median well coverage: "
    f"{well_c0['coverage'].median()*100:.2f}%"
)

print(
    f"C1 median well coverage: "
    f"{well_c1['coverage'].median()*100:.2f}%"
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

summary = pd.DataFrame([
    {
        "method": "C0_global",
        "nominal_coverage": NOMINAL_COVERAGE,
        "observation_coverage": c0_coverage,
        "mean_width": c0_mean_width,
        "median_width": c0_median_width,
        "calibration_n": len(calib_df),
        "test_n": len(test_df),
    },
    {
        "method": "C1_distance_stratified",
        "nominal_coverage": NOMINAL_COVERAGE,
        "observation_coverage": c1_coverage,
        "mean_width": c1_mean_width,
        "median_width": c1_median_width,
        "calibration_n": len(calib_df),
        "test_n": len(test_df),
    },
])

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

test_output = OUTPUT_DIR / "test_predictions_with_intervals.csv"
summary_output = OUTPUT_DIR / "conformal_summary.csv"
distance_output = OUTPUT_DIR / "distance_coverage.csv"
well_c0_output = OUTPUT_DIR / "well_coverage_c0.csv"
well_c1_output = OUTPUT_DIR / "well_coverage_c1.csv"
quantile_output = OUTPUT_DIR / "distance_quantiles.json"

test_df.to_csv(
    test_output,
    index=False,
)

summary.to_csv(
    summary_output,
    index=False,
)

distance_results.to_csv(
    distance_output,
    index=False,
)

well_c0.to_csv(
    well_c0_output,
    index=False,
)

well_c1.to_csv(
    well_c1_output,
    index=False,
)

with open(
    quantile_output,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "alpha": ALPHA,
            "nominal_coverage": NOMINAL_COVERAGE,
            "global_quantile": q_global,
            "distance_quantiles": {
                str(k): float(v)
                for k, v in q_by_bin.items()
            },
            "min_calibration_n": MIN_CALIB_N,
            "distance_bins": DIST_BINS,
            "distance_labels": DIST_LABELS,
            "seed": SEED,
        },
        f,
        indent=2,
    )


# ============================================================
# 14. FINAL SCIENTIFIC CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECKS")
print("=" * 70)

print("[PASS] Calibration/test wells are disjoint.")
print("[PASS] Test wells were not used to estimate C0/C1 quantiles.")
print("[PASS] No true depth variable was used to construct uncertainty.")
print("[PASS] Interval width is explicitly reported.")
print("[PASS] Distance-stratum sample sizes are reported.")

print("\nIMPORTANT:")
print(
    "Do NOT call C1 'valid 90% conformal coverage' merely because "
    "its observed coverage is close to 90%."
)

print(
    "Coverage must be interpreted together with the calibration design, "
    "distribution shift, and interval width."
)

print("\nFiles written to:")
print(OUTPUT_DIR.resolve())