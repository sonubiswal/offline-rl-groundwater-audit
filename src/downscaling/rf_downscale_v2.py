"""
rf_downscale_v2.py

Improved, self-contained Phase 2 train + independent-validate script.

WHAT CHANGES vs rf_downscale.py + validate_downscale.py (and why it is
legitimate, i.e. no held-out leakage anywhere):

  1. ALIGNED WINDOWS. Training and validation both use max_gap_days=90
     (validate_downscale.py silently used 45) and no rainfall-deficit
     feature by default (validate_downscale.py silently default True,
     whose 12-month NaN-anywhere abort dropped entire dates the model
     never needed). The covariate window the model is scored under now
     matches the window it was trained under.

  2. RESIDUAL (ANOMALY) TARGET MODE. Candidate models are trained either
     on the Kriged depth itself ("depth") or on the month-over-month
     CHANGE relative to the prior Kriged surface ("residual":
     target - target_lag1), reconstructed at prediction time as
     pred = lag + delta. This forces the regressor to learn dynamics
     instead of re-learning the static spatial pattern that target_lag1
     already carries -- the main reason random-forest skill collapses at
     held-out locations. LAG IS STILL BUILT ONLY FROM THE TRAINING-WELL
     KRIGED SURFACE (feature_utils.lag_target_feature) -- zero held-out
     information touches training.

  3. SPATIAL-BLOCK CV FOR ALL SELECTION. Random KFold on (date, cell)
     rows leaks spatial autocorrelation -- that is why the old CV showed
     0.639 against a 0.198 held-out truth. Here the 64x64 grid is split
     into 4x4 = 16 spatial blocks and CV is leave-several-blocks-out:
     a test cell NEVER appears in training. Model family (RF vs HGBR)
     and target mode are selected ONLY by this spatial CV, never by
     anything computed on held-out wells.

  4. HISTGRADIENTBOOSTING CANDIDATE. HistGradientBoostingRegressor is
     compared against the RF under the identical spatial CV; the winner
     is refit on all training data. No new dependencies (sklearn only).

  5. OPTIONAL SPATIAL COORDINATE FEATURES. The original pipeline computed
     row_norm/col_norm in both trainer and validator but never included
     them in FEATURE_COLUMNS (dead code). --with_spatial_coords (default
     on for v2) adds them as candidate features. Turn off with
     --no_spatial_coords to stay strictly comparable to the 11-feature
     production contract.

  6. HONEST BASELINES IN THE VALIDATION REPORT. Alongside the model, the
     script scores two zero-skill references ON THE SAME HELD-OUT
     READINGS: (a) kriging-only = current training-well Kriged surface at
     the well's cell; (b) persistence = prior Kriged surface (the lag).
     If the model does not beat these, the reported R2 is structural,
     not a bug.

  7. SELF-CONTAINED FEATURE STACK. The stack builder below re-implements
     the covariate matching with the same feature_utils primitives but
     WITHOUT the unconditional Phase 2.6 soil/LULC block, so the script
     still runs if soil_texture_class.tif / lulc_worldcover.tif are
     absent, and does not waste 34 raster loads per date on features no
     candidate uses.

  8. V2.1 UPGRADES -- all opt-in flags, recorded in the saved meta json so
     train/validate always use the identical feature contract:

       --with_terrain        relative elevation (cell minus 9x9-neighborhood
                             mean). DEM collateral information is the most
                             consistently documented covariate for water-table
                             interpolation (Desbarats et al. 2002, J. Hydrology;
                             Ruybal et al. 2019, WRR -- regression kriging with
                             a DEM covariate improved groundwater-level
                             predictions at unmonitored locations).

       --with_distance       dist_nearest_well_km + wells_within_50km computed
                             from a TRAINING-wells-only CSV
                             (--training_wells_csv, columns well_id,lat,lon).
                             Lets the model learn WHERE the Kriged surface is
                             unreliable (far from / sparse in training wells)
                             -- the dominant error source at held-out wells.
                             NEVER pass held-out wells in this CSV.

       --lag_fallback_days N when no Kriged surface exists within the primary
                             120-day window, allow up to N days and expose a
                             lag_age_days feature so the model can learn how
                             stale the lag is. Recovers training dates and
                             held-out readings that were previously SKIPPED
                             (your run: 15/38 training dates, 11,204/16,073
                             held-out readings) instead of dropping them.

       --ablation            after selection, re-runs spatial-block CV with
                             feature groups dropped (no lag / no static /
                             no rolling / no V2.1 additions / covariates
                             only) so EVERY change is measured separately
                             on training data only. Saved to
                             reports/rf_v2_ablation.csv.

     The validation step additionally reports per-season metrics
     (reports/rf_v2_season_metrics.csv) and a paired bootstrap 95% CI for the
     RMSE improvement over each zero-skill baseline -- the significance test
     the +0.024 R2 delta needs to survive review.

  9. V2.2 UPGRADES -- evaluation coverage + honesty tooling:

       --covariate_fallback_days N  when a monthly source (typically GRACE's
                             irregular solutions) has no composite within
                             max_gap_days, allow a STALE one up to N days and
                             expose <source>_age_days channels (0.0 when fresh)
                             so the model LEARNS staleness instead of the date
                             being dropped. Recovered rows re-enter BOTH the
                             training table and the held-out evaluation.

       Prediction intervals: split-conformal 90% interval, half-width = 90th
                             percentile of |out-of-fold residuals| from the
                             SELECTED model's own spatial-block CV -- calibrated
                             on training data only. The validation report
                             includes the EMPIRICAL held-out coverage so the
                             reader can judge calibration.

       Year x season x region stratified metrics
                             (reports/rf_v2_stratified_metrics.csv) + an
                             out_of_bbox flag per reading (bbox-clamped edge
                             cells are evaluated but flagged, never silent).

       Paired diagnostic runner: src/downscaling/diagnose_heldout.py tags
                             every held-out row with ONE mutually-exclusive
                             drop_reason (counts reconcile EXACTLY to the CSV
                             row count) and tests skipped-vs-evaluated
                             representativeness (KS / chi-square / SMD).

 10. V2.3 -- DEPTH-BALANCED SAMPLE WEIGHTS (this revision):

       --depth_weighted      Upweights deep-target samples in the training
                             loss so the sparse deep tail is not effectively
                             ignored by the estimator. Diagnosis (from
                             limitations.md S1 and the v2 depth diagnostic):
                             HGBR predicts in a narrow 2-30 m band regardless
                             of true depth, so 20-50 m wells are underpredicted
                             by 14-27 m and 50+ m wells by 55-60 m. Sample
                             weighting does not add information the model
                             doesn't have, but it forces the fit to allocate
                             capacity to the tail. Applied to BOTH the
                             spatial-block CV folds AND the final refit, so
                             model selection and full-data training see the
                             same loss.

                             Binning uses the ABSOLUTE Kriged target
                             (df["target"]) in both depth and residual
                             target modes, so the weighting is comparable
                             across candidate models.

                             Default binning (edit DEPTH_BIN_CUTOFFS /
                             DEPTH_BIN_WEIGHTS below to tune):
                               0-5 m  x1,  5-10 m  x2,  10-20 m x4,
                               20-30 m x8, 30-50 m x16, 50+ m x32.

                             Expected effect: deep-bin bias shrinks materially,
                             pooled R2 may trade slightly, per-well median R2
                             should improve. The weighting flag is recorded in
                             the saved meta json.

LEAKAGE STATEMENT (for the methodology report): held-out wells are read
only in validate(); they are never present in the training table, the
spatial CV, hyperparameter choice, model-family choice, or target-mode
choice. The lag feature and both baselines derive exclusively from the
training-well Kriged surface produced by kriging.py.

USAGE
-----
  python rf_downscale_v2.py --all                 # train (CV-select) then validate
  python rf_downscale_v2.py --train               # train + save model + meta json
  python rf_downscale_v2.py --validate            # validate saved model
  python rf_downscale_v2.py --all --no_spatial_coords --extra_rolling 6

  # V2.1 recommended (each upgrade measured separately via --ablation):
  python rf_downscale_v2.py --all --with_terrain \
      --with_distance --training_wells_csv data/processed/training_wells.csv \
      --lag_fallback_days 365 --ablation

  # V2.3 depth-weighted retrain:
  python rf_downscale_v2.py --all --depth_weighted \
      --model_out data/processed/rf_v2_weighted.joblib \
      --meta_out reports/rf_v2_weighted_meta.json \
      --results_out reports/rf_v2_weighted_validation.csv \
      --metrics_out reports/rf_v2_weighted_metrics.json \
      --plot_out reports/rf_v2_weighted_predicted_vs_actual.png

Outputs:
  data/processed/rf_v2_model.joblib
  reports/rf_v2_model_meta.json          (chosen config + full CV table)
  reports/rf_v2_cv_model_comparison.csv
  reports/rf_v2_validation.csv           (per-reading: actual, predicted, baselines)
  reports/rf_v2_metrics.json
  reports/rf_v2_predicted_vs_actual.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (
    GRID_SIZE, MONTHLY_SOURCES, STATIC_SOURCES,
    find_nearest_raster, find_nearest_prior_kriged, _parse_date_from_filename,
    load_raster_array, rolling_average_features,
    lag_target_feature, latlon_to_grid_cell, seasonal_features,
)


# =====================================================================
# PRODUCTION-COMPATIBLE FEATURE CONTRACT.
# Must stay identical to rf_downscale.FEATURE_COLUMNS (11 features):
# MONTHLY(4) + STATIC(2) + seasonal(2) + roll3(2) + target_lag1(1).
# Optional v2 extensions (spatial coords, extra rolling months) are
# appended ONLY via explicit flags and recorded in the saved meta json,
# so a saved model can always be re-validated with the exact feature
# set it was trained with.
# =====================================================================
BASE_FEATURE_COLUMNS = (
    list(MONTHLY_SOURCES) + list(STATIC_SOURCES)
    + ["month_sin", "month_cos", "chirps_roll3", "gldas_roll3", "target_lag1"]
)

# Design choice A (state this sentence in the methods): the MODEL covariate
# window is 90 days (matches rf_downscale.py training), while the kriging-only
# BASELINE uses a tighter 45-day window so its reference surface is temporally
# closer to the reading date. Different justification -- baseline fairness, not
# model coverage -- so a reviewer asking "why 90 vs 45" has this answer.
KRIGED_BASELINE_MAX_GAP_DAYS = 45   # for the kriging-only baseline only
RANDOM_STATE = 42


# === DEPTH WEIGHTING (V2.3) ==========================================
# Depth-balanced sample weights. Binning is on the ABSOLUTE Kriged depth
# (df["target"]), regardless of whether the model is trained on depth or on
# the residual (target - lag). This keeps the weighting comparable across
# target modes and across candidates in the CV table.
#
# Rationale: the v2 depth diagnostic (reports/depth_diag_v2_baseline.json)
# shows HGBR predicts inside a narrow 2-30 m band regardless of true depth,
# under-predicting 20-50 m wells by 14-27 m and 50+ m wells by 55-60 m.
# Tree-ensemble leaf averaging cannot see the sparse deep tail. Sample
# weighting does not add information; it forces the fit to allocate
# capacity there. Tune by editing these two constants.
DEPTH_BIN_CUTOFFS = [5, 10, 20, 30, 50]
DEPTH_BIN_WEIGHTS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]   # len = len(cutoffs) + 1


def depth_sample_weights(absolute_depths) -> np.ndarray:
    """Per-sample weights binned by absolute Kriged depth (metres).

    absolute_depths: array-like of df['target'] values (NOT the residual
    target when target_mode == 'residual').

    Returns a float64 array with the same length as input.
    """
    arr = np.asarray(absolute_depths, dtype=np.float64)
    bins = np.digitize(arr, DEPTH_BIN_CUTOFFS)
    return np.array([DEPTH_BIN_WEIGHTS[b] for b in bins], dtype=np.float64)
# === END DEPTH WEIGHTING =============================================


# ---------------------------------------------------------------------
# V2.1 helpers: terrain + training-well distance grids + bootstrap CI
# ---------------------------------------------------------------------
def rel_elevation(elev: np.ndarray, k: int = 4) -> np.ndarray:
    """Cell elevation minus the mean elevation of its (2k+1)x(2k+1)
    neighborhood -- local topographic position. np.roll WRAPS at raster
    borders (a documented approximation; border rows/cols mix opposite
    edges). Captures valleys/discharge zones vs uplands/recharge zones,
    which absolute elevation alone does not."""
    shifted = [np.roll(np.roll(elev, dr, axis=0), dc, axis=1)
               for dr in range(-k, k + 1) for dc in range(-k, k + 1)]
    return elev - np.nanmean(np.stack(shifted), axis=0)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Broadcasting haversine: lat1/lon1 (A,1), lat2/lon2 (1,B) -> (A,B) km."""
    R = 6371.0
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_well_distance_grids(training_wells_csv: str,
                                bbox: tuple[float, float, float, float],
                                grid_size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Two (G, G) grids from a TRAINING-wells-only CSV: haversine distance to
    the nearest training well, and count of training wells within 50 km of
    each cell center. Purely a function of training-well coordinates -- zero
    held-out information. Row/lat mapping matches latlon_to_grid_cell
    (row 0 = north edge)."""
    wells = pd.read_csv(training_wells_csv)
    wlat = wells["lat"].to_numpy(float)
    wlon = wells["lon"].to_numpy(float)
    min_lon, min_lat, max_lon, max_lat = bbox
    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    cell_lat = min_lat + ((grid_size - 0.5) - rows) / grid_size * (max_lat - min_lat)
    cell_lon = min_lon + (cols + 0.5) / grid_size * (max_lon - min_lon)
    LA, LO = np.meshgrid(cell_lat, cell_lon, indexing="ij")
    la, lo = LA.ravel(), LO.ravel()
    dmin = np.empty(la.size)
    dcount = np.empty(la.size)
    for s in range(0, la.size, 256):  # chunked: avoids a (4096 x N_wells) blowup
        e = min(s + 256, la.size)
        d = _haversine_km(la[s:e, None], lo[s:e, None], wlat[None, :], wlon[None, :])
        dmin[s:e] = d.min(axis=1)
        dcount[s:e] = (d < 50.0).sum(axis=1).astype(float)
    return dmin.reshape(grid_size, grid_size), dcount.reshape(grid_size, grid_size)


def _season_label(month: int) -> str:
    if 3 <= month <= 5:
        return "pre-monsoon"
    if 6 <= month <= 9:
        return "monsoon"
    return "post-monsoon"


def paired_rmse_bootstrap(sq_model: np.ndarray, sq_base: np.ndarray,
                          n_boot: int = 5000, seed: int = RANDOM_STATE,
                          chunk: int = 500) -> tuple[float, float]:
    """95% CI for (RMSE_base - RMSE_model) on paired readings. A CI entirely
    above zero means the model's improvement over the baseline is significant
    at the 5% level despite reading-level error correlation."""
    rng = np.random.default_rng(seed)
    n = len(sq_model)
    deltas = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))
        deltas[s:e] = (np.sqrt(sq_base[idx].mean(axis=1))
                       - np.sqrt(sq_model[idx].mean(axis=1)))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ---------------------------------------------------------------------
# Feature stack (self-contained, no soil/LULC dependency, no deficit)
# ---------------------------------------------------------------------
def build_stack_v2(
    interim_dir: str | Path,
    date_str: str,
    max_gap_days: int,
    kriged_dir: str | Path,
    lag_max_gap_days: int,
    extra_rolling: tuple[int, ...] = (),
    with_terrain: bool = False,
    lag_fallback_days: int = 0,
    covariate_fallback_days: int = 0,
    dist_grids: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray] | None:
    interim_dir = Path(interim_dir)
    features: dict[str, np.ndarray] = {}

    for source in MONTHLY_SOURCES:
        matched = find_nearest_raster(interim_dir / source, date_str, max_gap_days)
        age_days = 0.0
        if matched is None and covariate_fallback_days > max_gap_days:
            # Coverage fix: allow a STALE composite (e.g. a GRACE solution gap)
            # instead of dropping the whole date; the staleness becomes an
            # explicit <source>_age_days feature the model can learn from.
            matched = find_nearest_raster(interim_dir / source, date_str,
                                          covariate_fallback_days)
            if matched is not None:
                file_date = _parse_date_from_filename(matched)
                if file_date is not None:
                    from datetime import date as _dc
                    y, mo, d = (int(x) for x in date_str.split("-")[:3])
                    age_days = float(abs((_dc(y, mo, d) - file_date).days))
        if matched is None:
            return None
        features[source] = load_raster_array(matched)
        if covariate_fallback_days > 0:
            features[f"{source}_age_days"] = np.full(
                (GRID_SIZE, GRID_SIZE), age_days, dtype=np.float32)

    for name, sub in [("srtm_elevation", ("srtm", "elevation.tif")),
                      ("srtm_slope", ("srtm", "slope.tif"))]:
        p = interim_dir.joinpath(*sub)
        if not p.exists():
            return None
        features[name] = load_raster_array(p)

    if with_terrain:
        features["rel_elevation"] = rel_elevation(features["srtm_elevation"])

    roll_window = max(max_gap_days, covariate_fallback_days) if covariate_fallback_days else max_gap_days
    roll3 = rolling_average_features(interim_dir, date_str, sources=["chirps", "gldas"],
                                     n_months=3, max_gap_days=roll_window, min_months=2)
    if roll3 is None:
        return None
    features.update(roll3)

    # Design choice B: extra_rolling values that collide with the built-in
    # 3-month roll (or with each other) are SKIPPED with a warning instead of
    # silently overwriting an existing feature key.
    for n in extra_rolling:
        if n == 3:
            print("[rf_v2] WARNING: extra_rolling includes n=3, which duplicates the "
                  "built-in chirps/gldas_roll3 -- skipping to avoid key collision.")
            continue
        roll = rolling_average_features(interim_dir, date_str, sources=["chirps", "gldas"],
                                        n_months=n, max_gap_days=max_gap_days, min_months=2)
        if roll is None:
            return None
        for key, val in roll.items():
            if key in features:
                print(f"[rf_v2] WARNING: rolling key '{key}' already present -- skipping "
                      f"to avoid silent overwrite (extra_rolling={extra_rolling}).")
                continue
            features[key] = val

    if dist_grids is not None:
        features["dist_nearest_well_km"], features["wells_within_50km"] = dist_grids

    if kriged_dir is not None:
        # V2.1: optional extended-window fallback for dates with no recent
        # Kriged surface. lag_age_days makes the (possibly long) gap an
        # explicit, learnable feature instead of silently varying staleness.
        lag_path = find_nearest_prior_kriged(kriged_dir, date_str, lag_max_gap_days)
        if lag_path is None and lag_fallback_days > lag_max_gap_days:
            lag_path = find_nearest_prior_kriged(kriged_dir, date_str, lag_fallback_days)
        if lag_path is None:
            return None
        features["target_lag1"] = np.load(lag_path)
        if lag_fallback_days > 0:
            from datetime import date as _date_cls
            file_date = _parse_date_from_filename(lag_path)
            if file_date is not None:
                y, mo, d = (int(x) for x in date_str.split("-")[:3])
                age_days = float((_date_cls(y, mo, d) - file_date).days)
                features["lag_age_days"] = np.full(
                    (GRID_SIZE, GRID_SIZE), age_days, dtype=np.float32)

    return features


def feature_columns(with_spatial: bool, extra_rolling: tuple[int, ...],
                    with_terrain: bool = False, with_distance: bool = False,
                    with_lag_age: bool = False,
                    with_cov_age: bool = False) -> list[str]:
    cols = list(BASE_FEATURE_COLUMNS)
    cols += [f"chirps_roll{n}" for n in extra_rolling]
    cols += [f"gldas_roll{n}" for n in extra_rolling]
    if with_terrain:
        cols += ["rel_elevation"]
    if with_distance:
        cols += ["dist_nearest_well_km", "wells_within_50km"]
    if with_lag_age:
        cols += ["lag_age_days"]
    if with_cov_age:
        # one deterministic staleness channel per monthly source (0.0 when the
        # composite matched inside the normal window) -- kept for ALL sources
        # so the feature contract never depends on which dates were rescued.
        cols += [f"{s}_age_days" for s in MONTHLY_SOURCES]
    if with_spatial:
        cols += ["row_norm", "col_norm"]
    return cols


# ---------------------------------------------------------------------
# Training table (spatial-block aware)
# ---------------------------------------------------------------------
def build_training_table_v2(
    kriged_dir: str | Path,
    interim_dir: str | Path,
    feat_cols: list[str],
    max_gap_days: int = 90,
    lag_max_gap_days: int = 120,
    extra_rolling: tuple[int, ...] = (),
    with_terrain: bool = False,
    lag_fallback_days: int = 0,
    covariate_fallback_days: int = 0,
    dist_grids: tuple[np.ndarray, np.ndarray] | None = None,
    n_blocks_per_axis: int = 4,
) -> pd.DataFrame:
    kriged_dir = Path(kriged_dir)
    npy_files = sorted(kriged_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No Kriged target files in {kriged_dir}. Run kriging.py first.")

    rows: list[dict] = []
    n_dates_used, n_dates_skipped = 0, 0
    b = max(1, GRID_SIZE // n_blocks_per_axis)
    stack_cache: dict[str, dict | None] = {}

    for npy_path in npy_files:
        date_str = npy_path.stem
        if date_str not in stack_cache:
            stack_cache[date_str] = build_stack_v2(
                interim_dir, date_str, max_gap_days, kriged_dir, lag_max_gap_days,
                extra_rolling, with_terrain=with_terrain,
                lag_fallback_days=lag_fallback_days,
                covariate_fallback_days=covariate_fallback_days, dist_grids=dist_grids)
        stack = stack_cache[date_str]
        if stack is None:
            n_dates_skipped += 1
            continue
        n_dates_used += 1
        target_grid = np.load(npy_path)
        season = seasonal_features(date_str)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                target = target_grid[r, c]
                if np.isnan(target):
                    continue
                lag_val = stack["target_lag1"][r, c]
                row: dict = {
                    "date": date_str, "r": r, "c": c,
                    "block": (r // b) * n_blocks_per_axis + (c // b),
                    "target": float(target), "lag": float(lag_val),
                    "row_norm": r / (GRID_SIZE - 1), "col_norm": c / (GRID_SIZE - 1),
                    **season,
                }
                for col in feat_cols:
                    # row_norm/col_norm set from (r, c); month_sin/cos from season --
                    # neither lives in the raster stack. target_lag1 IS in the stack.
                    if col in ("row_norm", "col_norm", "month_sin", "month_cos"):
                        continue
                    row[col] = float(stack[col][r, c])
                needed = feat_cols + ["lag"]
                if any(np.isnan(row[col]) for col in needed):
                    continue
                rows.append(row)

    print(f"[rf_v2] Used {n_dates_used} dates, skipped {n_dates_skipped} "
          f"(incomplete covariate/rolling/lag coverage).")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Training table is empty -- check covariate coverage.")
    print(f"[rf_v2] Training table: {len(df)} (date, cell) samples, "
          f"{len(feat_cols)} features, {df['block'].nunique()} spatial CV blocks.")
    return df


# ---------------------------------------------------------------------
# Spatial-block CV (leave-several-blocks-out). A test cell never appears
# in training. Selection uses ONLY training-well data.
# ---------------------------------------------------------------------
def spatial_block_cv(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_mode: str,
    make_model_factory,
    n_splits: int = 4,
    seed: int = RANDOM_STATE,
    collect_residuals: bool = False,
    depth_weighted: bool = False,   # === DEPTH WEIGHTING (V2.3) ===
) -> dict:
    """Review point 3: make_model_factory must be a ZERO-ARG factory returning
    a FRESH estimator per call -- each CV fold builds its own instance, so no
    estimator object is ever reused (or mutated) across folds. Also reused by
    the --ablation report (feature-group drops), so every measured change goes
    through the identical spatial-block protocol on training data only.

    When depth_weighted=True, each fold's fit is passed sample_weight derived
    from the ABSOLUTE Kriged target (df['target']), regardless of target mode.
    This is the same weighting the final refit uses, so selection and full-data
    training see the same loss.
    """
    def y_of(part: pd.DataFrame) -> np.ndarray:
        if target_mode == "depth":
            return part["target"].values
        if target_mode == "residual":
            return (part["target"] - part["lag"]).values
        raise ValueError(target_mode)

    blocks = df["block"].unique()
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(blocks)), n_splits)

    fold_rmses, fold_r2s = [], []
    oof_residuals: list[np.ndarray] = []
    for fold_idx in folds:
        test_blocks = set(blocks[i] for i in fold_idx)
        test = df[df["block"].isin(test_blocks)]
        train = df[~df["block"].isin(test_blocks)]
        model = make_model_factory()
        # === DEPTH WEIGHTING (V2.3) ===
        sw = depth_sample_weights(train["target"].values) if depth_weighted else None
        model.fit(train[feat_cols].values, y_of(train), sample_weight=sw)
        # === END DEPTH WEIGHTING ===
        pred = model.predict(test[feat_cols].values)
        if target_mode == "residual":
            pred = pred + test["lag"].values
        fold_rmses.append(float(np.sqrt(mean_squared_error(test["target"].values, pred))))
        fold_r2s.append(float(r2_score(test["target"].values, pred)))
        if collect_residuals:
            oof_residuals.append(test["target"].values - pred)
    out = {"cv_rmse_mean": float(np.mean(fold_rmses)), "cv_rmse_folds": fold_rmses,
           "cv_r2_mean": float(np.mean(fold_r2s)), "cv_r2_folds": fold_r2s}
    if collect_residuals:
        out["oof_residuals"] = np.concatenate(oof_residuals)
    return out


# ---------------------------------------------------------------------
# Ablation: re-run spatial-block CV with feature groups dropped, so each
# V2.1 upgrade is measured SEPARATELY on training data (never held-out).
# ---------------------------------------------------------------------
def run_ablation(df: pd.DataFrame, feat_cols: list[str], best_factory,
                 best_mode: str, n_splits: int, out_path: str,
                 depth_weighted: bool = False) -> pd.DataFrame:   # === DEPTH WEIGHTING (V2.3) ===
    groups: dict[str, list[str]] = {
        "full": list(feat_cols),
        "no_lag": [c for c in feat_cols if c not in ("target_lag1", "lag_age_days")],
        "no_static": [c for c in feat_cols
                      if c not in ("srtm_elevation", "srtm_slope", "rel_elevation")],
        "no_rolling": [c for c in feat_cols
                       if not (c.startswith("chirps_roll") or c.startswith("gldas_roll"))],
        "no_v21_additions": [c for c in feat_cols
                             if c not in ("rel_elevation", "dist_nearest_well_km",
                                          "wells_within_50km", "lag_age_days")
                             and not c.endswith("_age_days")],
        "covariates_only": [c for c in feat_cols
                            if c not in ("target_lag1", "lag_age_days", "row_norm",
                                         "col_norm", "month_sin", "month_cos")],
    }
    rows = []
    for name, cols in groups.items():
        if not cols:
            continue
        res = spatial_block_cv(df, cols, best_mode, best_factory,
                               n_splits=n_splits, depth_weighted=depth_weighted)
        rows.append({"ablation": name, "n_features": len(cols), **res})
        print(f"[rf_v2] ablation {name:>18} ({len(cols):>2} feats): "
              f"CV RMSE={res['cv_rmse_mean']:.3f} m, CV R2={res['cv_r2_mean']:.3f}")
    abl = pd.DataFrame(rows)[["ablation", "n_features", "cv_rmse_mean", "cv_r2_mean"]]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    abl.to_csv(out_path, index=False)
    print(f"[rf_v2] Saved ablation -> {out_path}")
    return abl


def make_rf(n_estimators: int, max_depth: int | None):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1)


def make_hgbr():
    # Review point 2: HGBR controls tree size via max_leaf_nodes (31), not
    # max_depth -- the previous signature accepted max_depth but never used
    # it (a silent lie), so the parameter was removed entirely.
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------
# Metrics / validation
# ---------------------------------------------------------------------
def metrics_block(y_true, y_pred) -> dict:
    return {"n": int(len(y_true)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def within_well_r2(results: pd.DataFrame) -> float:
    # CAVEAT (report alongside this number): wells with a single reading
    # contribute zero anomaly to BOTH numerator and denominator, so this
    # metric is dominated by multi-reading wells and is undefined if every
    # well has exactly one reading (returns NaN in that case).
    a = results["actual"] - results.groupby("well_id")["actual"].transform("mean")
    p = results["predicted"] - results.groupby("well_id")["predicted"].transform("mean")
    denom = float((a ** 2).sum())
    return float("nan") if denom <= 0 else float(1 - ((a - p) ** 2).sum() / denom)


# Per-directory cache of the kriged .npy file list (parsed dates included),
# so validate_v2's per-row calls don't re-glob N_readings x N_files times.
_KRIGED_FILE_CACHE: dict[str, list] = {}


def find_nearest_kriged_any(kriged_dir: str | Path, date_str: str,
                            max_gap_days: int = KRIGED_BASELINE_MAX_GAP_DAYS) -> Path | None:
    from datetime import date as date_cls
    kriged_dir = Path(kriged_dir)
    key = str(kriged_dir)
    if key not in _KRIGED_FILE_CACHE:  # glob + date-parse ONCE per directory
        entries = []
        if kriged_dir.exists():
            import re
            date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
            for npy_path in kriged_dir.glob("*.npy"):
                m = date_re.search(npy_path.stem)
                if m:
                    yy, mm, dd = (int(x) for x in m.group(1).split("-"))
                    entries.append((date_cls(yy, mm, dd), npy_path))
        _KRIGED_FILE_CACHE[key] = entries
    y, mo, d = (int(x) for x in date_str.split("-")[:3])
    target = date_cls(y, mo, d)
    best, best_diff = None, None
    for file_date, npy_path in _KRIGED_FILE_CACHE[key]:
        diff = abs((file_date - target).days)
        if diff <= max_gap_days and (best_diff is None or diff < best_diff):
            best, best_diff = npy_path, diff
    return best


def validate_v2(
    model, meta: dict,
    held_out_csv: str, interim_dir: str, kriged_dir: str,
    bbox: tuple[float, float, float, float],
    dist_grids: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    held_df = pd.read_csv(held_out_csv)
    feat_cols = meta["feature_columns"]
    target_mode = meta["target_mode"]
    max_gap_days = meta["max_gap_days"]
    lag_max_gap_days = meta["lag_max_gap_days"]
    extra_rolling = tuple(meta.get("extra_rolling", ()))
    with_terrain = bool(meta.get("with_terrain", False))
    lag_fallback_days = int(meta.get("lag_fallback_days", 0))
    covariate_fallback_days = int(meta.get("covariate_fallback_days", 0))
    interval_hw = meta.get("interval_halfwidth_m")
    mid_lat = (bbox[1] + bbox[3]) / 2.0
    mid_lon = (bbox[0] + bbox[2]) / 2.0

    cache: dict[str, dict | None] = {}
    krig_array_cache: dict = {}  # avoid re-np.load()-ing the same kriged surface per row
    rows_out: list[dict] = []
    n_skipped = 0

    for _, row in held_df.iterrows():
        date_str = str(row["date"])
        if date_str not in cache:
            cache[date_str] = build_stack_v2(
                interim_dir, date_str, max_gap_days, kriged_dir,
                lag_max_gap_days, extra_rolling, with_terrain=with_terrain,
                lag_fallback_days=lag_fallback_days,
                covariate_fallback_days=covariate_fallback_days, dist_grids=dist_grids)
        stack = cache[date_str]
        if stack is None:
            n_skipped += 1
            continue

        r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
        season = seasonal_features(date_str)
        vec: dict[str, float] = {}
        ok = True
        for col in feat_cols:
            if col == "row_norm":
                vec[col] = r / (GRID_SIZE - 1)
            elif col == "col_norm":
                vec[col] = c / (GRID_SIZE - 1)
            elif col in ("month_sin", "month_cos"):
                vec[col] = float(season[col])
            else:
                if col not in stack:
                    ok = False
                    break
                vec[col] = float(stack[col][r, c])
            if np.isnan(vec[col]):
                ok = False
                break
        if not ok:
            n_skipped += 1
            continue

        lag_val = float(stack["target_lag1"][r, c])
        pred = float(model.predict(np.array([vec[col] for col in feat_cols]).reshape(1, -1))[0])
        if target_mode == "residual":
            pred = pred + lag_val

        # kriging-only baseline: CURRENT training-well kriged surface at the cell
        krig_path = find_nearest_kriged_any(kriged_dir, date_str)
        if krig_path is not None:
            if krig_path not in krig_array_cache:
                krig_array_cache[krig_path] = np.load(krig_path)
            krig_val = float(krig_array_cache[krig_path][r, c])
        else:
            krig_val = np.nan

        out_of_bbox = not (bbox[0] <= row["lon"] <= bbox[2] and bbox[1] <= row["lat"] <= bbox[3])
        rec = {
            "well_id": row["well_id"], "date": date_str,
            "lat": row["lat"], "lon": row["lon"],
            "season": _season_label(int(date_str.split("-")[1])),
            "region": ("N" if row["lat"] >= mid_lat else "S") + ("W" if row["lon"] < mid_lon else "E"),
            "out_of_bbox": bool(out_of_bbox),
            "actual": float(row["depth_m"]),
            "predicted": pred,
            "baseline_kriging_only": krig_val,
            "baseline_persistence_lag": lag_val,
        }
        if interval_hw is not None:
            rec["pred_lo"] = pred - float(interval_hw)
            rec["pred_hi"] = pred + float(interval_hw)
        rows_out.append(rec)

    print(f"[rf_v2] Evaluated {len(rows_out)} held-out readings, skipped {n_skipped} "
          f"(incomplete coverage; window aligned to training: {max_gap_days}d).")
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------
def do_train(args) -> tuple[object, dict]:
    with open(args.config) as f:
        config = yaml.safe_load(f)
    bbox = tuple(config["region"]["bbox"])

    extra_rolling = (args.extra_rolling,) if args.extra_rolling else ()
    dist_grids = None
    if args.with_distance:
        if not Path(args.training_wells_csv).exists():
            raise FileNotFoundError(
                f"--with_distance needs a TRAINING-wells-only CSV at "
                f"{args.training_wells_csv} (columns: well_id,lat,lon). "
                f"NEVER pass held-out wells in this file.")
        dist_grids = compute_well_distance_grids(args.training_wells_csv, bbox)
        print("[rf_v2] Computed training-well distance grids "
              "(dist_nearest_well_km, wells_within_50km).")
    feat_cols = feature_columns(args.with_spatial_coords, extra_rolling,
                                with_terrain=args.with_terrain,
                                with_distance=args.with_distance,
                                with_lag_age=args.lag_fallback_days > 0,
                                with_cov_age=args.covariate_fallback_days > 0)
    print(f"[rf_v2] Feature set ({len(feat_cols)}): {feat_cols}")

    if args.depth_weighted:
        # === DEPTH WEIGHTING (V2.3) ===
        print(f"[rf_v2] Depth-weighted training ENABLED. "
              f"Cutoffs={DEPTH_BIN_CUTOFFS}, weights={DEPTH_BIN_WEIGHTS}")
        # === END DEPTH WEIGHTING ===

    df = build_training_table_v2(
        args.kriged_dir, args.interim_dir, feat_cols,
        max_gap_days=args.max_gap_days, lag_max_gap_days=args.lag_max_gap_days,
        extra_rolling=extra_rolling, with_terrain=args.with_terrain,
        lag_fallback_days=args.lag_fallback_days,
        covariate_fallback_days=args.covariate_fallback_days, dist_grids=dist_grids)

    # Review point 3: each candidate is a ZERO-ARG FACTORY so spatial CV
    # builds a fresh estimator per fold -- no object reuse across folds and
    # no lambda-captured CV instance leaking into the final full-data fit.
    candidate_factories = {
        "rf_depth": lambda: make_rf(args.n_estimators, args.max_depth),
        "rf_residual": lambda: make_rf(args.n_estimators, args.max_depth),
        "hgbr_depth": make_hgbr,
        "hgbr_residual": make_hgbr,
    }
    candidates = [("rf_depth", "depth"), ("rf_residual", "residual"),
                  ("hgbr_depth", "depth"), ("hgbr_residual", "residual")]
    cv_rows, fitted = [], {}
    for name, mode in candidates:
        res = spatial_block_cv(df, feat_cols, mode, candidate_factories[name],
                               n_splits=args.cv_splits,
                               depth_weighted=args.depth_weighted)   # === DEPTH WEIGHTING (V2.3) ===
        res.update({"candidate": name, "target_mode": mode})
        cv_rows.append(res)
        fitted[name] = (candidate_factories[name](), mode)  # FRESH instance for final fit
        print(f"[rf_v2] {name:>14}: spatial-CV RMSE={res['cv_rmse_mean']:.3f} m "
              f"(folds={['%.3f' % f for f in res['cv_rmse_folds']]}), "
              f"CV R2={res['cv_r2_mean']:.3f}")

    cv_df = pd.DataFrame(cv_rows)[["candidate", "target_mode", "cv_rmse_mean", "cv_r2_mean"]]
    Path(args.cv_out).parent.mkdir(parents=True, exist_ok=True)   # bug fix: parent may not exist
    cv_df.to_csv(args.cv_out, index=False)
    best_name = min(cv_rows, key=lambda x: x["cv_rmse_mean"])["candidate"]
    best_model, best_mode = fitted[best_name]
    print(f"[rf_v2] Selected by spatial-block CV: {best_name} "
          f"(RMSE={min(r['cv_rmse_mean'] for r in cv_rows):.3f} m). Refitting on all training data...")

    y_full = df["target"].values if best_mode == "depth" else (df["target"] - df["lag"]).values
    # === DEPTH WEIGHTING (V2.3) ===
    full_sw = depth_sample_weights(df["target"].values) if args.depth_weighted else None
    best_model.fit(df[feat_cols].values, y_full, sample_weight=full_sw)
    # === END DEPTH WEIGHTING ===

    # Prediction intervals (V2.2, split-conformal): half-width = 90th pct of
    # |out-of-fold residuals| from the SELECTED model's own spatial-block CV --
    # calibrated on training data only, never tuned on held-out. Caveat: OOF
    # blocks still sit near training wells, so this half-width is optimistic
    # for far-from-well locations; the EMPIRICAL held-out coverage is reported
    # at validation time so the reader can judge calibration themselves.
    interval_halfwidth = None
    if args.prediction_intervals:
        print("[rf_v2] Calibrating 90% conformal interval from spatial-CV "
              "out-of-fold residuals of the selected model...")
        oof = spatial_block_cv(df, feat_cols, best_mode, candidate_factories[best_name],
                               n_splits=args.cv_splits, collect_residuals=True,
                               depth_weighted=args.depth_weighted)   # === DEPTH WEIGHTING (V2.3) ===
        interval_halfwidth = float(np.quantile(np.abs(oof["oof_residuals"]), 0.90))
        print(f"[rf_v2] Interval: pred +/- {interval_halfwidth:.3f} m "
              f"(90th pct of |OOF residuals|, n={len(oof['oof_residuals'])}).")

    if args.ablation:
        print("[rf_v2] Running ablation (spatial-block CV with feature groups "
              "dropped -- training data only)...")
        run_ablation(df, feat_cols, candidate_factories[best_name], best_mode,
                     args.cv_splits, args.ablation_out,
                     depth_weighted=args.depth_weighted)   # === DEPTH WEIGHTING (V2.3) ===

    meta = {
        "candidate": best_name, "target_mode": best_mode,
        "feature_columns": feat_cols,
        "max_gap_days": args.max_gap_days, "lag_max_gap_days": args.lag_max_gap_days,
        "extra_rolling": list(extra_rolling),
        "with_spatial_coords": bool(args.with_spatial_coords),
        "with_terrain": bool(args.with_terrain),
        "with_distance": bool(args.with_distance),
        "training_wells_csv": args.training_wells_csv,
        "lag_fallback_days": int(args.lag_fallback_days),
        "covariate_fallback_days": int(args.covariate_fallback_days),
        "interval_halfwidth_m": interval_halfwidth,
        # === DEPTH WEIGHTING (V2.3) ===
        "depth_weighted": bool(args.depth_weighted),
        "depth_bin_cutoffs": list(DEPTH_BIN_CUTOFFS) if args.depth_weighted else None,
        "depth_bin_weights": list(DEPTH_BIN_WEIGHTS) if args.depth_weighted else None,
        # === END DEPTH WEIGHTING ===
        "model_params": best_model.get_params(),
        "cv_table": cv_rows,
        "n_training_rows": int(len(df)),
        "leakage_note": "Held-out wells used nowhere in training, CV, or selection; "
                        "lag/baselines derive only from training-well kriging.",
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_out)
    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[rf_v2] Saved model -> {args.model_out}\n"
          f"[rf_v2] Saved meta  -> {args.meta_out}\n"
          f"[rf_v2] Saved CV comparison -> {args.cv_out}")
    return best_model, meta


def do_validate(args) -> None:
    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])
    model = joblib.load(args.model_out)
    with open(args.meta_out) as f:
        meta = json.load(f)

    dist_grids = None
    if meta.get("with_distance"):
        dist_grids = compute_well_distance_grids(meta["training_wells_csv"], bbox)

    results = validate_v2(model, meta, args.held_out_csv, args.interim_dir,
                          args.kriged_dir, bbox, dist_grids=dist_grids)
    if results.empty:
        raise RuntimeError("No held-out readings could be evaluated -- check coverage/paths.")

    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_out, index=False)

    m_model = metrics_block(results["actual"].values, results["predicted"].values)
    m_krig = metrics_block(results["actual"].values, results["baseline_kriging_only"].values)
    m_pers = metrics_block(results["actual"].values, results["baseline_persistence_lag"].values)
    m_model["within_well_r2"] = within_well_r2(results)

    # Per-season metrics: CGWB readings are quarterly, and monsoon/post-monsoon
    # skill typically differs -- the within-well R2 of 0.247 suggests real
    # seasonal structure worth reporting separately.
    season_rows = []
    for season, grp in results.groupby("season"):
        season_rows.append({
            "season": season, "n": len(grp),
            **{f"model_{k}": v for k, v in metrics_block(grp["actual"].values, grp["predicted"].values).items()},
            "kriging_rmse": metrics_block(grp["actual"].values, grp["baseline_kriging_only"].values)["rmse"],
            "persistence_rmse": metrics_block(grp["actual"].values, grp["baseline_persistence_lag"].values)["rmse"],
        })
    season_df = pd.DataFrame(season_rows).sort_values("season")
    Path(args.season_out).parent.mkdir(parents=True, exist_ok=True)
    season_df.to_csv(args.season_out, index=False)

    # Paired bootstrap 95% CI for RMSE improvement over each baseline (same
    # readings, paired errors): a CI entirely above 0 = significant at 5%.
    sq_m = ((results["actual"] - results["predicted"]) ** 2).values
    sq_k = ((results["actual"] - results["baseline_kriging_only"]) ** 2).values
    sq_p = ((results["actual"] - results["baseline_persistence_lag"]) ** 2).values
    ci_vs_krig = paired_rmse_bootstrap(sq_m, sq_k)
    ci_vs_pers = paired_rmse_bootstrap(sq_m, sq_p)

    # V2.2: year x season x region stratified metrics (robustness table).
    results["year"] = results["date"].str[:4]
    strat_rows = []
    strata = ["year", "season", "region", "out_of_bbox",
              "year+season", "year+region", "season+region", "year+season+region"]
    for stratum in strata:
        cols = stratum.split("+")
        for key, grp in results.groupby(cols, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            label = " | ".join(f"{c}={k}" for c, k in zip(cols, key))
            row = {"stratum": stratum, "value": label, "n": len(grp)}
            row.update(metrics_block(grp["actual"].values, grp["predicted"].values))
            row["kriging_rmse"] = metrics_block(
                grp["actual"].values, grp["baseline_kriging_only"].values)["rmse"]
            strat_rows.append(row)
    strat_df = pd.DataFrame(strat_rows)
    Path(args.stratified_out).parent.mkdir(parents=True, exist_ok=True)
    strat_df.to_csv(args.stratified_out, index=False)

    # V2.2: empirical coverage of the conformal 90% interval on HELD-OUT
    # readings (nominal 90%; a large deviation means the OOF-calibrated
    # half-width is optimistic for real held-out locations, as warned).
    interval_coverage = None
    if "pred_lo" in results.columns:
        inside = ((results["actual"] >= results["pred_lo"])
                  & (results["actual"] <= results["pred_hi"]))
        interval_coverage = float(inside.mean())
    n_out_of_bbox = int(results["out_of_bbox"].sum())

    report = {
        "model_rf_v2": m_model,
        "baseline_kriging_only": m_krig,
        "baseline_persistence": m_pers,
        "bootstrap_ci_rmse_improvement": {
            "vs_kriging_only": {"lower_95": ci_vs_krig[0], "upper_95": ci_vs_krig[1],
                                "significant_at_5pct": ci_vs_krig[0] > 0},
            "vs_persistence": {"lower_95": ci_vs_pers[0], "upper_95": ci_vs_pers[1],
                               "significant_at_5pct": ci_vs_pers[0] > 0},
        },
        "season_metrics": season_rows,
        "interval_halfwidth_m": (float(meta["interval_halfwidth_m"])
                                 if meta.get("interval_halfwidth_m") else None),
        "interval_empirical_coverage_heldout": interval_coverage,
        "n_out_of_bbox_evaluated": n_out_of_bbox,
        "production_baseline_reference": {"r2": 0.198, "rmse": 5.226, "mae": 3.371,
                                          "source": "validation_methodology.md Section 12"},
    }
    with open(args.metrics_out, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("INDEPENDENT VALIDATION (held-out CGWB wells, aligned windows)")
    print("=" * 72)
    for label, m in [("RF v2 (%s)" % meta["candidate"], m_model),
                     ("Kriging-only baseline", m_krig),
                     ("Persistence (lag) baseline", m_pers)]:
        print(f"  {label:<28} N={m['n']:>5}  RMSE={m['rmse']:>7.3f}  "
              f"MAE={m['mae']:>7.3f}  R2={m['r2']:>7.3f}")
    print(f"  RF v2 within-well R2 (anomaly skill): {m_model['within_well_r2']:.3f}")
    print()
    print("  Paired bootstrap 95% CI for RMSE improvement (m):")
    print(f"    vs kriging-only: [{ci_vs_krig[0]:+.3f}, {ci_vs_krig[1]:+.3f}]"
          f"  {'SIGNIFICANT' if ci_vs_krig[0] > 0 else 'not significant at 5%'}")
    print(f"    vs persistence:  [{ci_vs_pers[0]:+.3f}, {ci_vs_pers[1]:+.3f}]"
          f"  {'SIGNIFICANT' if ci_vs_pers[0] > 0 else 'not significant at 5%'}")
    print()
    print("  Per-season RMSE (m): " + ", ".join(
        f"{r['season']}={r['model_rmse']:.2f} (krig {r['kriging_rmse']:.2f})"
        for _, r in season_df.iterrows()))
    print(f"  [rf_v2] Season metrics -> {args.season_out}")
    if interval_coverage is not None:
        print(f"  Conformal 90% interval: half-width {float(meta['interval_halfwidth_m']):.2f} m, "
              f"EMPIRICAL held-out coverage {interval_coverage:.1%} (nominal 90%)")
    print(f"  Out-of-bbox readings (edge-clamped cells, flagged not dropped): {n_out_of_bbox}")
    print(f"  [rf_v2] Stratified (year x season x region) -> {args.stratified_out}")
    print()
    print("Reference -- production baseline (Section 12): R2=0.198, RMSE=5.226, MAE=3.371")
    if m_model["rmse"] >= min(m_krig["rmse"], m_pers["rmse"]):
        print("NOTE: RF v2 does NOT beat the best zero-skill baseline -- the held-out R2 is")
        print("      structurally limited by kriging quality + persistence at well locations,")
        print("      and that should be stated in the methodology report rather than tuned away.")
    print("=" * 72)

    # scatter plot: model + kriging-only baseline
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(results["actual"], results["predicted"], alpha=0.45, s=16,
               label=f"RF v2 ({meta['candidate']}) R2={m_model['r2']:.3f}")
    ax.scatter(results["actual"], results["baseline_kriging_only"], alpha=0.25, s=12,
               marker="x", label=f"Kriging-only R2={m_krig['r2']:.3f}")
    # include the kriging baseline in axis limits so no baseline point is clipped
    all_preds = pd.concat([results["predicted"], results["baseline_kriging_only"].dropna()])
    lims = [min(results["actual"].min(), all_preds.min()),
            max(results["actual"].max(), all_preds.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual depth (m) -- held-out CGWB wells")
    ax.set_ylabel("Prediction (m)")
    ax.set_title("RF v2 vs Kriging-only on independent held-out wells")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_out, dpi=150)
    plt.close(fig)
    print(f"[rf_v2] Saved results -> {args.results_out}")
    print(f"[rf_v2] Saved metrics -> {args.metrics_out}")
    print(f"[rf_v2] Saved plot    -> {args.plot_out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="RF downscaling v2: spatial-CV model selection + aligned independent validation.")
    p.add_argument("--train", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--all", action="store_true", help="train then validate (default if neither given)")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    p.add_argument("--model_out", default="data/processed/rf_v2_model.joblib")
    p.add_argument("--meta_out", default="reports/rf_v2_model_meta.json")
    p.add_argument("--cv_out", default="reports/rf_v2_cv_model_comparison.csv")
    p.add_argument("--results_out", default="reports/rf_v2_validation.csv")
    p.add_argument("--metrics_out", default="reports/rf_v2_metrics.json")
    p.add_argument("--plot_out", default="reports/rf_v2_predicted_vs_actual.png")
    p.add_argument("--max_gap_days", type=int, default=90,
                   help="Covariate matching window. 90 matches rf_downscale.py training. "
                        "The old validator's 45 was a silent mismatch.")
    p.add_argument("--lag_max_gap_days", type=int, default=120)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=15, help="-1 for unbounded (None).")
    p.add_argument("--cv_splits", type=int, default=4, help="Number of spatial-block CV folds.")
    p.add_argument("--extra_rolling", type=int, default=0,
                   help="If >0, add chirps_rollN/gldas_rollN features (e.g. 6).")
    p.add_argument("--with_spatial_coords", dest="with_spatial_coords", action="store_true",
                   default=True, help="Include row_norm/col_norm (computed-but-unused in v1).")
    p.add_argument("--no_spatial_coords", dest="with_spatial_coords", action="store_false")
    # ---- V2.1 upgrade flags (all recorded in meta json; all measured by --ablation) ----
    p.add_argument("--with_terrain", action="store_true",
                   help="Add rel_elevation (cell minus 9x9-neighborhood DEM mean) -- "
                        "topographic position, the best-documented covariate for "
                        "water-table interpolation (Desbarats 2002; Ruybal 2019).")
    p.add_argument("--with_distance", action="store_true",
                   help="Add dist_nearest_well_km + wells_within_50km from the "
                        "TRAINING-wells-only CSV (--training_wells_csv).")
    p.add_argument("--training_wells_csv", default="data/processed/training_wells.csv",
                   help="CSV of TRAINING wells only (well_id,lat,lon) for --with_distance. "
                        "NEVER include held-out wells.")
    p.add_argument("--lag_fallback_days", type=int, default=0,
                   help="If >lag_max_gap_days (e.g. 365), allow older Kriged surfaces as "
                        "lag fallback and add a lag_age_days feature. Recovers previously "
                        "skipped dates/readings (yours: 15/38 train, 11204/16073 held-out).")
    p.add_argument("--covariate_fallback_days", type=int, default=0,
                   help="If >max_gap_days (e.g. 400), allow STALE monthly composites "
                        "(typical culprit: GRACE's irregular solution dates) and add "
                        "<source>_age_days channels so the model learns staleness "
                        "instead of the date being dropped.")
    p.add_argument("--prediction_intervals", action="store_true",
                   help="Add a split-conformal 90% interval (half-width = 90th pct of the "
                        "selected model's spatial-CV out-of-fold |residuals|). Adds "
                        "pred_lo/pred_hi to the validation CSV and reports the EMPIRICAL "
                        "held-out coverage so calibration can be judged.")
    p.add_argument("--ablation", action="store_true",
                   help="After selection, re-run spatial-block CV with feature groups "
                        "dropped (no lag / no static / no rolling / no V2.1 additions / "
                        "covariates only) -- training data only.")
    p.add_argument("--ablation_out", default="reports/rf_v2_ablation.csv")
    p.add_argument("--season_out", default="reports/rf_v2_season_metrics.csv")
    p.add_argument("--stratified_out", default="reports/rf_v2_stratified_metrics.csv")
    # === DEPTH WEIGHTING (V2.3) ===
    p.add_argument("--depth_weighted", action="store_true",
                   help="Upweight deep-target samples in the loss (depth-balanced sample "
                        "weights, binned by absolute Kriged depth). Applied to BOTH the "
                        "spatial-block CV folds and the final refit. See DEPTH_BIN_CUTOFFS "
                        "/ DEPTH_BIN_WEIGHTS in this file to tune the bins.")
    # === END DEPTH WEIGHTING ===
    args = p.parse_args()

    # Review point 1: -1 means unbounded -- resolve to None BEFORE any
    # make_rf() call, mirroring the old rf_downscale.py behaviour
    # (sklearn rejects max_depth=-1 with ValueError).
    args.max_depth = None if args.max_depth == -1 else args.max_depth

    if not (args.train or args.validate or args.all):
        args.all = True
    if args.all:
        args.train = args.validate = True

    if args.train:
        do_train(args)
    if args.validate:
        do_validate(args)


if __name__ == "__main__":
    main()