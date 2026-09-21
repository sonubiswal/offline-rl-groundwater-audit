"""
rf_downscale.py

Trains a RandomForestRegressor to predict fine-resolution groundwater
depth from coarse covariates (CHIRPS, GLDAS, Sentinel-2 NDVI, GRACE, SRTM
elevation/slope), targeting the Kriged surface built from TRAINING WELLS
ONLY (kriging.py).

--- FEATURE_COLUMNS DISCIPLINE (read before touching this file) ---
FEATURE_COLUMNS below is the PRODUCTION BASELINE: the exact 11-feature
set that produced the reported, defended Phase 2 result
(validation_methodology.md Section 12: R²=0.198, RMSE=5.226m,
MAE=3.371m). It must never be edited in place to add an experimental
column -- that is exactly what happened during both the Phase 2.5
rainfall-deficit test and the Phase 2.6 soil/LULC test, and in both
cases the change was never reverted afterward, leaving
rf_downscale_model.joblib silently out of sync with the reported number
(caught via n_features_in_ returning 12, not the documented 11, after
the Phase 2.6 work).

Rejected experimental candidates are kept as SEPARATE, clearly-labeled
constants (DEFICIT_CANDIDATE_COLUMN, SOIL_LULC_CANDIDATE_COLUMNS) and are
only ever added via the explicit `extra_feature_columns` parameter to
build_training_table()/train_rf()/run() -- never by editing
FEATURE_COLUMNS itself. A default `python rf_downscale.py` run with no
flags always trains the true 11-feature production model.

--- PHASE 2.5 CHANGE LOG (rejected, R²=0.190 vs 0.198 baseline) ---
Rainfall-deficit feature (chirps_cum12_anomaly) tested via
--with_rainfall_deficit. Kept as DEFICIT_CANDIDATE_COLUMN for
reproducibility of that experiment only.

--- PHASE 2.6 CHANGE LOG (rejected, CV delta +0.000 vs 0.639 baseline,
pre-registered +0.02 threshold, NO-GO before touching held-out) ---
Soil texture (12 classes) + LULC (11 classes) one-hot columns tested via
--with_soil_lulc. Kept as SOIL_LULC_CANDIDATE_COLUMNS for reproducibility
of that experiment only.
----------------------------
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (
    build_feature_stack_for_date, GRID_SIZE, MONTHLY_SOURCES, STATIC_SOURCES,
    spatial_features, seasonal_features, compute_monthly_climatology,
    SOIL_TEXTURE_CLASSES, LULC_CLASSES,
)


# =====================================================================
# PRODUCTION BASELINE -- matches validation_methodology.md Section 9/12
# EXACTLY. Do not edit this list to add experimental columns.
# =====================================================================
FEATURE_COLUMNS = (
    MONTHLY_SOURCES + STATIC_SOURCES
    + ["month_sin", "month_cos"]
    + ["chirps_roll3", "gldas_roll3", "target_lag1"]
)

# =====================================================================
# REJECTED EXPERIMENTAL CANDIDATES -- reproducible via extra_feature_columns,
# never merged into FEATURE_COLUMNS above.
# =====================================================================
DEFICIT_CANDIDATE_COLUMN = "chirps_cum12_anomaly"  # Phase 2.5, rejected

SOIL_LULC_CANDIDATE_COLUMNS = (
    [f"soil_texture_{c}" for c in SOIL_TEXTURE_CLASSES]
    + [f"lulc_{c}" for c in LULC_CLASSES]
)  # Phase 2.6, rejected


def build_training_table(
    kriged_dir: str | Path,
    interim_dir: str | Path,
    max_gap_days: int = 90,
    lag_max_gap_days: int = 120,
    extra_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    extra_feature_columns: optional list of additional column names to
    require/include beyond FEATURE_COLUMNS, for reproducing a specific
    rejected experiment (e.g. DEFICIT_CANDIDATE_COLUMN,
    SOIL_LULC_CANDIDATE_COLUMNS). Leave as None for the production
    baseline -- this is what a default run uses.
    """
    extra_feature_columns = extra_feature_columns or []
    full_feature_columns = FEATURE_COLUMNS + extra_feature_columns

    # Climatology is only needed if the deficit candidate is explicitly
    # requested -- tied directly to column inclusion so this can't drift
    # out of sync the way the old use_rainfall_deficit flag did.
    use_rainfall_deficit = DEFICIT_CANDIDATE_COLUMN in extra_feature_columns
    climatology = None
    if use_rainfall_deficit:
        print("[rf_downscale] Computing CHIRPS monthly climatology for the rainfall-deficit "
              "candidate feature (Phase 2.5 reproduction)...")
        climatology = compute_monthly_climatology(interim_dir, source="chirps")
        n_months_with_climatology = len(climatology)
        print(f"[rf_downscale] Climatology available for {n_months_with_climatology}/12 calendar months.")
        if n_months_with_climatology < 12:
            print("[rf_downscale] WARNING: climatology is missing at least one calendar month -- "
                  "any date requiring that month in its trailing 12-month window will be skipped.")

    kriged_dir = Path(kriged_dir)
    npy_files = sorted(kriged_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(
            f"No Kriged target files found in {kriged_dir}. Run kriging.py first."
        )

    rows = []
    n_dates_used, n_dates_skipped = 0, 0

    for npy_path in npy_files:
        date_str = npy_path.stem
        target_grid = np.load(npy_path)

        feature_stack = build_feature_stack_for_date(
            interim_dir, date_str,
            max_gap_days=max_gap_days,
            kriged_dir=kriged_dir,
            lag_max_gap_days=lag_max_gap_days,
            climatology=climatology,
        )
        if feature_stack is None:
            print(f"[rf_downscale] Skipping {date_str}: incomplete covariate/rolling/lag/deficit coverage.")
            n_dates_skipped += 1
            continue

        n_dates_used += 1
        season_feats = seasonal_features(date_str)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                target_val = target_grid[r, c]
                if np.isnan(target_val):
                    continue
                feat_vals_dict = {col: feature_stack[col][r, c] for col in feature_stack.keys()}
                feat_vals_dict.update(season_feats)

                # Only require completeness on the columns this run actually
                # needs -- avoids dropping rows over irrelevant extra keys
                # that might be present in feature_stack but unused.
                if any(col not in feat_vals_dict or np.isnan(feat_vals_dict[col])
                       for col in full_feature_columns):
                    continue

                rows.append({
                    "date": date_str, "row": r, "col": c,
                    **{col: feat_vals_dict[col] for col in full_feature_columns},
                    "target": target_val,
                })

    print(f"[rf_downscale] Used {n_dates_used} dates, skipped {n_dates_skipped} "
          f"(incomplete covariate/rolling/lag/deficit coverage).")
    print(f"[rf_downscale] Training table: {len(rows)} (date, cell) samples, "
          f"{len(full_feature_columns)} features "
          f"({'baseline only' if not extra_feature_columns else 'baseline + ' + ', '.join(extra_feature_columns[:3]) + ('...' if len(extra_feature_columns) > 3 else '')}).")

    return pd.DataFrame(rows)


def train_rf(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_estimators: int = 300,
    max_depth: int = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    feature_columns: which columns of df to train on. Defaults to the
    production baseline (FEATURE_COLUMNS) if not given -- pass
    FEATURE_COLUMNS + DEFICIT_CANDIDATE_COLUMN (etc.) explicitly to
    reproduce a specific rejected experiment instead.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS

    X = df[feature_columns].values
    y = df["target"].values

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    train_r2 = model.score(X, y)
    print(f"[rf_downscale] Training-set R² (against Kriged surface, NOT held-out wells): {train_r2:.3f}")

    importances = dict(zip(feature_columns, model.feature_importances_))
    print("[rf_downscale] Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.3f}")

    return model


def run(
    config_path: str = "config/data_config.yaml",
    kriged_dir: str = "data/interim/kriged_target_monthly",
    interim_dir: str = "data/interim",
    model_out_path: str = "data/processed/rf_downscale_model.joblib",
    table_out_path: str = "data/processed/rf_training_table.csv",
    max_gap_days: int = 90,
    lag_max_gap_days: int = 120,
    n_estimators: int = 300,
    max_depth: int | None = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    with_rainfall_deficit: bool = False,
    with_soil_lulc: bool = False,
) -> None:
    """
    Default (no flags): trains the true 11-feature PRODUCTION BASELINE --
    this is what should be run to reproduce validation_methodology.md
    Section 12's reported result, or to regenerate rf_downscale_model.joblib
    if it has drifted out of sync with the baseline (e.g. after an
    experimental run was never reverted).

    with_rainfall_deficit / with_soil_lulc: opt-in flags to reproduce the
    specific rejected Phase 2.5 / Phase 2.6 experiments instead. Using
    either flag trains and saves a NON-production model -- do not leave
    these on for a "normal" run, and do not commit the resulting
    .joblib as if it were the baseline.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    extra_feature_columns: list[str] = []
    if with_rainfall_deficit:
        extra_feature_columns.append(DEFICIT_CANDIDATE_COLUMN)
        print("[rf_downscale] NOTE: --with_rainfall_deficit is set -- this reproduces the "
              "REJECTED Phase 2.5 experiment (R²=0.190 vs 0.198 baseline). The resulting "
              "model is NOT the production baseline.")
    if with_soil_lulc:
        extra_feature_columns.extend(SOIL_LULC_CANDIDATE_COLUMNS)
        print("[rf_downscale] NOTE: --with_soil_lulc is set -- this reproduces the REJECTED "
              "Phase 2.6 experiment (CV delta +0.000, NO-GO). The resulting model is NOT "
              "the production baseline.")

    df = build_training_table(kriged_dir, interim_dir, max_gap_days, lag_max_gap_days,
                               extra_feature_columns=extra_feature_columns or None)
    if len(df) == 0:
        raise RuntimeError("Training table is empty.")

    Path(table_out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_out_path, index=False)
    print(f"[rf_downscale] Saved training table -> {table_out_path}")

    full_feature_columns = FEATURE_COLUMNS + extra_feature_columns
    model = train_rf(
        df,
        feature_columns=full_feature_columns,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    if extra_feature_columns and model_out_path == "data/processed/rf_downscale_model.joblib":
        print("[rf_downscale] WARNING: you are about to overwrite the PRODUCTION model path "
              "with a non-baseline experimental model. Pass --model_out to save this "
              "experiment somewhere else, e.g. data/processed/rf_experiment_<name>.joblib, "
              "unless you specifically intend to replace the production checkpoint.")

    Path(model_out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out_path)
    print(f"[rf_downscale] Saved trained model -> {model_out_path} "
          f"({len(full_feature_columns)} features, n_features_in_={model.n_features_in_})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RF downscaling model against Kriged training-well surfaces. "
                     "Default (no flags) trains the true production baseline (11 features)."
    )
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--model_out", default="data/processed/rf_downscale_model.joblib")
    parser.add_argument("--table_out", default="data/processed/rf_training_table.csv")
    parser.add_argument("--max_gap_days", type=int, default=90)
    parser.add_argument("--lag_max_gap_days", type=int, default=120)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=15,
                         help="Pass -1 for unbounded trees (max_depth=None). Default (15) "
                              "matches cv_tune.py's selected configuration -- "
                              "validation_methodology.md Section 11.")
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--with_rainfall_deficit", action="store_true",
                         help="Reproduce the REJECTED Phase 2.5 rainfall-deficit experiment "
                              "(R²=0.190 vs 0.198 baseline). Do not use for a normal run.")
    parser.add_argument("--with_soil_lulc", action="store_true",
                         help="Reproduce the REJECTED Phase 2.6 soil/LULC experiment "
                              "(CV delta +0.000, NO-GO). Do not use for a normal run.")
    args = parser.parse_args()

    resolved_max_depth = None if args.max_depth == -1 else args.max_depth

    run(args.config, args.kriged_dir, args.interim_dir, args.model_out, args.table_out,
        args.max_gap_days, args.lag_max_gap_days,
        n_estimators=args.n_estimators, max_depth=resolved_max_depth,
        min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
        with_rainfall_deficit=args.with_rainfall_deficit,
        with_soil_lulc=args.with_soil_lulc)