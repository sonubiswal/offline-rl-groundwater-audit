"""
residual_kriging.py

Residual Kriging (regression-kriging) hybrid correction for the Phase 2
RF downscaling model. Standard geostatistics technique: after fitting a
regression model (RF), compute its errors (residuals) at known point
locations, spatially interpolate those errors via Kriging, and add the
interpolated correction back to the regression prediction. This corrects
smooth, spatially-varying bias the regression model itself doesn't
capture.

CRITICAL anti-circularity design: residuals are computed ONLY at TRAINING
wells (comparing RF's prediction at each training well's location/date
against that well's own RAW reading -- not the Kriged proxy surface it
was fit to). The correction surface built from those residuals never
uses held-out wells. This module's built-in validation
(leave_some_wells_out_cv) tests the mechanism using ONLY training wells,
splitting them into "correction-source" and "evaluation" subsets across
folds -- so the true held-out set is never touched by anything in this
module. Only a separate, deliberate call to validate against held-out
data (done elsewhere, exactly once) should ever use this module's output
against real held-out wells.

CHANGELOG:
  - leave_some_wells_out_cv now splits folds using spatial_block_splits
    (src/utils/spatial_block_cv.py) instead of sklearn's GroupKFold.
    GroupKFold assigns wells to folds at random with no spatial
    constraint, so a "correction" well from another fold could sit right
    next to an "evaluation" well in this fold -- letting Kriging borrow
    nearby signal that the TRUE held-out set (built via held_out_split.py's
    spatially-stratified 10x10 grid) never provides. That mismatch made
    the old CV number (R2 +0.041) optimistic relative to the true held-out
    result (R2 -0.007). The spatial-block split below assigns whole
    10x10-grid cells to folds, so each fold's evaluation wells are
    spatially separated from its correction wells -- the same geometry
    the true held-out check poses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downscaling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from feature_utils import build_feature_stack_for_date, latlon_to_grid_cell, seasonal_features
from kriging import krige_wells_to_grid
from spatial_block_cv import spatial_block_splits


def compute_rf_predictions_at_wells(
    rf_model,
    feature_columns: list[str],
    wells_df: pd.DataFrame,
    interim_dir: str | Path,
    kriged_dir: str | Path,
    bbox: tuple[float, float, float, float],
    climatology: dict | None = None,
    max_gap_days: int = 90,
) -> pd.DataFrame:
    """Apply the RF model at every well reading's exact location/date.

    Parameters
    ----------
    wells_df : must have columns [well_id, lat, lon, date, depth_m]. Can
        be training wells OR held-out wells -- this function itself is
        neutral; the anti-circularity discipline is enforced by WHICH
        wells_df the caller passes in (see module docstring).

    Returns
    -------
    pd.DataFrame with columns [well_id, date, lat, lon, actual,
    rf_predicted, residual], skipping readings whose feature stack
    couldn't be built.

    Performance note: builds every row's feature vector first, then calls
    rf_model.predict() ONCE on the full batch. Calling .predict() row-by-row
    (tens of thousands of individual calls) is dramatically slower due to
    sklearn's per-call dispatch overhead -- easily 20-30+ minutes for a
    large well table, versus seconds when batched.
    """
    stack_cache: dict[str, dict | None] = {}
    pending_meta = []
    pending_feat_rows = []

    n_total = len(wells_df)
    n_stack_built = 0
    for i, row in enumerate(wells_df.itertuples(index=False)):
        if i > 0 and i % 10000 == 0:
            print(f"[residual_kriging]   ...built feature vectors for {i}/{n_total} rows "
                  f"({len(stack_cache)} unique dates processed so far)")

        date_str = str(row.date)
        if date_str not in stack_cache:
            stack_cache[date_str] = build_feature_stack_for_date(
                interim_dir, date_str, max_gap_days=max_gap_days,
                kriged_dir=kriged_dir, climatology=climatology,
            )
            n_stack_built += 1
        stack = stack_cache[date_str]
        if stack is None:
            continue

        r, c = latlon_to_grid_cell(row.lat, row.lon, bbox)
        season = seasonal_features(date_str)
        feat_dict = {col: stack[col][r, c] for col in stack.keys()}
        feat_dict.update(season)
        if any(col not in feat_dict or np.isnan(feat_dict[col]) for col in feature_columns):
            continue

        feat_vals = [feat_dict[col] for col in feature_columns]
        pending_feat_rows.append(feat_vals)
        pending_meta.append({
            "well_id": row.well_id, "date": date_str,
            "lat": row.lat, "lon": row.lon, "actual": row.depth_m,
        })

    if not pending_feat_rows:
        return pd.DataFrame(columns=["well_id", "date", "lat", "lon", "actual", "rf_predicted", "residual"])

    print(f"[residual_kriging]   Built {len(pending_feat_rows)} valid feature vectors "
          f"({n_stack_built} unique date-stacks computed); running one batched prediction...")

    X = np.array(pending_feat_rows)
    preds = rf_model.predict(X)  # single vectorized call, not one per row

    results = []
    for meta, pred in zip(pending_meta, preds):
        results.append({
            **meta, "rf_predicted": pred, "residual": meta["actual"] - pred,
        })

    return pd.DataFrame(results)


def apply_residual_correction(
    rf_prediction: float,
    correction_wells_df: pd.DataFrame,
    date_str: str,
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
    min_wells_for_correction: int = 5,
) -> tuple[float, bool]:
    """Correct one RF prediction using Kriged residuals from correction_wells_df.

    Returns (corrected_prediction, correction_applied). If fewer than
    min_wells_for_correction residual observations are available for this
    date, the raw RF prediction is returned unchanged (correction_applied=False)
    rather than attempting an unstable Kriging fit.
    """
    date_residuals = correction_wells_df[correction_wells_df["date"] == date_str]
    if len(date_residuals) < min_wells_for_correction:
        return rf_prediction, False

    try:
        residual_grid = krige_wells_to_grid(date_residuals, bbox, value_col="residual")
    except Exception:
        return rf_prediction, False

    r, c = latlon_to_grid_cell(lat, lon, bbox)
    correction = residual_grid[r, c]
    if np.isnan(correction):
        return rf_prediction, False

    return rf_prediction + correction, True


def leave_some_wells_out_cv(
    rf_model_path: str,
    feature_columns: list[str],
    train_wells_csv: str,
    interim_dir: str,
    kriged_dir: str,
    bbox: tuple[float, float, float, float],
    climatology: dict | None = None,
    n_splits: int = 5,
    n_grid: int = 10,
    min_wells_for_correction: int = 5,
    random_state: int = 42,
) -> dict:
    """Test whether residual-Kriging correction helps, using ONLY training wells.

    Splits training wells into n_splits groups by SPATIAL BLOCK (10x10 grid
    over bbox, matching held_out_split.py's own stratification scheme) --
    NOT by random well_id grouping. For each fold:
      - the OTHER folds' wells (spatially distant from this fold's blocks)
        supply the residual-correction surface
      - this fold's wells are evaluated, comparing raw RF vs. corrected
        RF against their own raw readings

    This never touches the true held-out set -- it is a fair, in-sample
    (but genuinely spatially out-of-fold) test of the correction mechanism,
    using the same spatial-separation geometry the true held-out check poses.

    Returns
    -------
    dict with 'raw_rf' and 'corrected' sub-dicts, each containing
    {rmse, mae, r2, n} computed by pooling all folds' evaluation
    predictions together.
    """
    rf_model = joblib.load(rf_model_path)
    train_df = pd.read_csv(train_wells_csv)

    print("[residual_kriging] Computing RF predictions at all training wells "
          "(this may take a moment)...")
    all_preds = compute_rf_predictions_at_wells(
        rf_model, feature_columns, train_df, interim_dir, kriged_dir, bbox, climatology,
    )
    print(f"[residual_kriging] Got RF predictions for {len(all_preds)} training-well readings "
          f"(of {len(train_df)} total; the rest lacked a complete feature stack).")

    if len(all_preds) < 20:
        raise RuntimeError(
            f"Only {len(all_preds)} training-well predictions available -- too few for a "
            f"meaningful cross-validation. Check feature/coverage availability."
        )

    unique_wells = all_preds["well_id"].nunique()
    n_splits = min(n_splits, unique_wells)

    raw_actual, raw_pred = [], []
    corrected_actual, corrected_pred = [], []
    n_corrections_applied, n_corrections_skipped = 0, 0

    for fold_i, (correction_idx, eval_idx) in enumerate(
        spatial_block_splits(all_preds, bbox, n_splits=n_splits, n_grid=n_grid, random_state=random_state)
    ):
        correction_wells = all_preds.iloc[correction_idx]
        eval_wells = all_preds.iloc[eval_idx]

        print(f"[residual_kriging] Fold {fold_i+1}/{n_splits}: "
              f"{len(correction_wells)} correction wells, {len(eval_wells)} eval readings "
              f"(spatially-blocked split)...")

        # CRITICAL: Krige the residual correction surface ONCE PER UNIQUE DATE
        # within this fold, not once per individual eval reading. Ordinary
        # Kriging (variogram fit + grid solve) is expensive; re-running it
        # for every one of potentially tens of thousands of eval readings
        # instead of ~100-200 unique dates is the difference between
        # seconds and hours.
        residual_grid_cache: dict[str, np.ndarray | None] = {}

        for i, (_, row) in enumerate(eval_wells.iterrows()):
            if i > 0 and i % 5000 == 0:
                print(f"[residual_kriging]   ...evaluated {i}/{len(eval_wells)} readings in fold {fold_i+1}")

            raw_actual.append(row["actual"])
            raw_pred.append(row["rf_predicted"])

            date_str = row["date"]
            if date_str not in residual_grid_cache:
                date_residuals = correction_wells[correction_wells["date"] == date_str]
                if len(date_residuals) < min_wells_for_correction:
                    residual_grid_cache[date_str] = None
                else:
                    try:
                        residual_grid_cache[date_str] = krige_wells_to_grid(
                            date_residuals, bbox, value_col="residual"
                        )
                    except Exception:
                        residual_grid_cache[date_str] = None

            residual_grid = residual_grid_cache[date_str]
            if residual_grid is None:
                corrected = row["rf_predicted"]
                applied = False
            else:
                r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
                correction = residual_grid[r, c]
                if np.isnan(correction):
                    corrected = row["rf_predicted"]
                    applied = False
                else:
                    corrected = row["rf_predicted"] + correction
                    applied = True

            corrected_actual.append(row["actual"])
            corrected_pred.append(corrected)
            if applied:
                n_corrections_applied += 1
            else:
                n_corrections_skipped += 1

    def _metrics(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return {
            "rmse": np.sqrt(mean_squared_error(actual, pred)),
            "mae": mean_absolute_error(actual, pred),
            "r2": r2_score(actual, pred),
            "n": len(actual),
        }

    raw_metrics = _metrics(raw_actual, raw_pred)
    corrected_metrics = _metrics(corrected_actual, corrected_pred)

    print()
    print(f"[residual_kriging] Correction applied to {n_corrections_applied}/"
          f"{n_corrections_applied + n_corrections_skipped} evaluation readings "
          f"(rest fell back to raw RF -- too few correction wells for that date).")

    return {"raw_rf": raw_metrics, "corrected": corrected_metrics}


def run(
    rf_model_path: str = "data/processed/rf_downscale_model.joblib",
    train_wells_csv: str = "data/interim/train_well_ids.csv",
    interim_dir: str = "data/interim",
    kriged_dir: str = "data/interim/kriged_target_monthly",
    config_path: str = "config/data_config.yaml",
    use_rainfall_deficit: bool = True,
) -> None:
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downscaling"))
    from rf_downscale import FEATURE_COLUMNS
    from feature_utils import compute_monthly_climatology

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    bbox = tuple(config["region"]["bbox"])

    climatology = compute_monthly_climatology(interim_dir, source="chirps") if use_rainfall_deficit else None

    result = leave_some_wells_out_cv(
        rf_model_path, FEATURE_COLUMNS, train_wells_csv, interim_dir, kriged_dir, bbox, climatology,
    )

    print()
    print("=" * 70)
    print("RESIDUAL KRIGING -- TRAINING-WELLS-ONLY, SPATIALLY-BLOCKED CROSS-VALIDATION")
    print("(never touches the true held-out set)")
    print("=" * 70)
    print(f"{'':20s} {'RMSE':>10s} {'MAE':>10s} {'R2':>10s} {'N':>8s}")
    r, c = result["raw_rf"], result["corrected"]
    print(f"{'Raw RF':20s} {r['rmse']:>10.3f} {r['mae']:>10.3f} {r['r2']:>10.3f} {r['n']:>8d}")
    print(f"{'RF + Residual Krig':20s} {c['rmse']:>10.3f} {c['mae']:>10.3f} {c['r2']:>10.3f} {c['n']:>8d}")
    print()
    r2_gain = c["r2"] - r["r2"]
    if r2_gain > 0.02:
        print(f"Residual Kriging shows a genuine improvement (R2 +{r2_gain:.3f}) on spatially-blocked "
              f"training-well cross-validation. Worth a final check against the true held-out set.")
    elif r2_gain > -0.02:
        print(f"Residual Kriging shows negligible change (R2 {r2_gain:+.3f}) -- within noise. "
              f"Consistent with the earlier finding that spatial bias is not the dominant error "
              f"source (see reports/limitations.md, depth-dependent bias diagnosis).")
    else:
        print(f"Residual Kriging appears to HURT performance (R2 {r2_gain:+.3f}) on this "
              f"cross-validation. Do not proceed to a held-out check with this correction.")


if __name__ == "__main__":
    run()