"""
fast_spatial_holdout.py

Fast version of the spatial-generalization holdout test (checklist item 3).

WHAT THIS TESTS: can the already-selected model (whatever
reports/rf_v2_model_meta.json records as "candidate" / "target_mode" /
"feature_columns" -- normally hgbr_depth) extrapolate to a region where it
has NO TRAINING-TABLE rows, even though the kriged target surface in that
region was still built using training wells there.

WHAT THIS DOES NOT TEST (the caveat -- state this in the methods note if
this is the version you report): the kriged .npy surfaces are NOT rebuilt.
A cell inside the held-out quadrant can still carry information from
training wells via kriging's spatial smoothing, so this is not a clean
test of whether the *pipeline* (kriging + RF) extrapolates -- only whether
the RF adds anything beyond what the (unchanged) kriged surface already
encodes there. The strict version (re-krige with held-out wells removed,
rebuild the training table, then retrain) is the version with no caveat.

FOLD DEFINITION: 4 quadrants (NW / NE / SW / SE), split at the bbox
midpoint, using the SAME row/col -> lat/lon convention as
feature_utils.latlon_to_grid_cell (row 0 = north edge, col 0 = west edge)
and the SAME quadrant labels validate_v2 already uses for its "region"
column (N/S x W/E split at bbox midpoint). This keeps per-quadrant numbers
directly comparable to the existing per-region validation tables.

FOR EACH QUADRANT:
  1. Build the full training table once (unchanged kriged surface,
     unchanged feature set/window from the saved meta json).
  2. Exclude training-table rows whose grid cell (r, c) center falls in
     that quadrant.
  3. Refit a FRESH model instance with the exact (candidate, target_mode,
     hyperparameters) recorded in the saved meta -- apples-to-apples with
     the production model, no re-tuning.
  4. Evaluate on held-out CGWB wells whose lat/lon falls in that same
     quadrant, using rf_downscale_v2.validate_v2 (same aligned covariate
     window, same kriging-only / persistence baselines, unchanged).

OUTPUT:
  reports/fast_spatial_holdout.csv          (per-quadrant + pooled metrics)
  reports/fast_spatial_holdout_summary.json (same, plus config used)

USAGE
-----
  python fast_spatial_holdout.py
  python fast_spatial_holdout.py --config config/data_config.yaml \
      --meta_out reports/rf_v2_model_meta.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rf_downscale_v2 import (
    GRID_SIZE, build_training_table_v2, compute_well_distance_grids,
    make_hgbr, make_rf, metrics_block, validate_v2,
)

QUADRANTS = ["NW", "NE", "SW", "SE"]


# ---------------------------------------------------------------------
# Quadrant assignment -- same convention as validate_v2's "region" column
# (mid_lat/mid_lon split), applied to cell centers for the training table
# and to actual lat/lon for held-out wells.
# ---------------------------------------------------------------------
def quadrant_label(lat: float, lon: float, mid_lat: float, mid_lon: float) -> str:
    return ("N" if lat >= mid_lat else "S") + ("W" if lon < mid_lon else "E")


def cell_center_latlon(r: np.ndarray, c: np.ndarray,
                       bbox: tuple[float, float, float, float],
                       grid_size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Matches compute_well_distance_grids' cell-center formula exactly
    (row 0 = north edge, col 0 = west edge)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    cell_lat = min_lat + ((grid_size - 0.5) - r) / grid_size * (max_lat - min_lat)
    cell_lon = min_lon + (c + 0.5) / grid_size * (max_lon - min_lon)
    return cell_lat, cell_lon


def add_quadrant_column(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    mid_lat = (bbox[1] + bbox[3]) / 2.0
    mid_lon = (bbox[0] + bbox[2]) / 2.0
    clat, clon = cell_center_latlon(df["r"].to_numpy(), df["c"].to_numpy(), bbox)
    df = df.copy()
    df["quadrant"] = [quadrant_label(la, lo, mid_lat, mid_lon) for la, lo in zip(clat, clon)]
    return df


def make_fold_model_factory(meta: dict):
    """Fresh-instance factory matching the saved production model's exact
    hyperparameters, dispatched by meta['candidate']."""
    name = meta["candidate"]
    params = meta.get("model_params", {})
    if name.startswith("rf"):
        n_estimators = params.get("n_estimators", 300)
        max_depth = params.get("max_depth", 15)
        return lambda: make_rf(n_estimators, max_depth)
    return make_hgbr


def run(args) -> None:
    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])
    with open(args.meta_out) as f:
        meta = json.load(f)

    feat_cols = meta["feature_columns"]
    target_mode = meta["target_mode"]
    extra_rolling = tuple(meta.get("extra_rolling", ()))
    with_terrain = bool(meta.get("with_terrain", False))
    lag_fallback_days = int(meta.get("lag_fallback_days", 0))
    covariate_fallback_days = int(meta.get("covariate_fallback_days", 0))

    dist_grids = None
    if meta.get("with_distance"):
        dist_grids = compute_well_distance_grids(meta["training_wells_csv"], bbox)

    print(f"[fast_holdout] Building full training table "
          f"(candidate={meta['candidate']}, target_mode={target_mode}, "
          f"{len(feat_cols)} features)...")
    df = build_training_table_v2(
        args.kriged_dir, args.interim_dir, feat_cols,
        max_gap_days=meta["max_gap_days"], lag_max_gap_days=meta["lag_max_gap_days"],
        extra_rolling=extra_rolling, with_terrain=with_terrain,
        lag_fallback_days=lag_fallback_days,
        covariate_fallback_days=covariate_fallback_days, dist_grids=dist_grids)
    df = add_quadrant_column(df, bbox)
    print("[fast_holdout] Training rows per quadrant:\n" +
          df["quadrant"].value_counts().to_string())

    mid_lat = (bbox[1] + bbox[3]) / 2.0
    mid_lon = (bbox[0] + bbox[2]) / 2.0
    held_df_all = pd.read_csv(args.held_out_csv)
    held_df_all["quadrant"] = [
        quadrant_label(la, lo, mid_lat, mid_lon)
        for la, lo in zip(held_df_all["lat"], held_df_all["lon"])
    ]

    factory = make_fold_model_factory(meta)
    fold_rows = []
    pooled_results = []

    for quadrant in QUADRANTS:
        held_fold = held_df_all[held_df_all["quadrant"] == quadrant].drop(columns=["quadrant"])
        n_held = len(held_fold)
        if n_held == 0:
            print(f"[fast_holdout] {quadrant}: no held-out wells in this quadrant, skipping.")
            continue

        train_fold = df[df["quadrant"] != quadrant]
        print(f"[fast_holdout] {quadrant}: training on {len(train_fold)} rows "
              f"(excluded {len(df) - len(train_fold)} in-quadrant rows), "
              f"evaluating on {n_held} held-out wells...")

        model = factory()
        y = (train_fold["target"].values if target_mode == "depth"
             else (train_fold["target"] - train_fold["lag"]).values)
        model.fit(train_fold[feat_cols].values, y)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            held_fold.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        try:
            results = validate_v2(model, meta, tmp_path, args.interim_dir,
                                  args.kriged_dir, bbox, dist_grids=dist_grids)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if results.empty:
            print(f"[fast_holdout] {quadrant}: 0 held-out readings evaluated "
                  f"after coverage filtering, skipping.")
            continue

        results["quadrant"] = quadrant
        pooled_results.append(results)

        m_model = metrics_block(results["actual"].values, results["predicted"].values)
        m_krig = metrics_block(results["actual"].values, results["baseline_kriging_only"].values)
        m_pers = metrics_block(results["actual"].values, results["baseline_persistence_lag"].values)
        fold_rows.append({
            "quadrant": quadrant,
            "n_train_rows": int(len(train_fold)),
            "n_excluded_rows": int(len(df) - len(train_fold)),
            "n_held_out_readings": int(len(results)),
            "model_rmse": m_model["rmse"], "model_mae": m_model["mae"], "model_r2": m_model["r2"],
            "kriging_rmse": m_krig["rmse"], "kriging_r2": m_krig["r2"],
            "persistence_rmse": m_pers["rmse"], "persistence_r2": m_pers["r2"],
        })
        print(f"[fast_holdout] {quadrant}: model R2={m_model['r2']:.3f} RMSE={m_model['rmse']:.3f}  "
              f"(kriging-only R2={m_krig['r2']:.3f}, persistence R2={m_pers['r2']:.3f})")

    if not fold_rows:
        raise RuntimeError("No quadrant produced evaluable held-out readings -- check paths/coverage.")

    fold_df = pd.DataFrame(fold_rows)

    all_results = pd.concat(pooled_results, ignore_index=True)
    m_model = metrics_block(all_results["actual"].values, all_results["predicted"].values)
    m_krig = metrics_block(all_results["actual"].values, all_results["baseline_kriging_only"].values)
    m_pers = metrics_block(all_results["actual"].values, all_results["baseline_persistence_lag"].values)
    pooled_row = {
        "quadrant": "POOLED", "n_train_rows": None, "n_excluded_rows": None,
        "n_held_out_readings": int(len(all_results)),
        "model_rmse": m_model["rmse"], "model_mae": m_model["mae"], "model_r2": m_model["r2"],
        "kriging_rmse": m_krig["rmse"], "kriging_r2": m_krig["r2"],
        "persistence_rmse": m_pers["rmse"], "persistence_r2": m_pers["r2"],
    }
    fold_df = pd.concat([fold_df, pd.DataFrame([pooled_row])], ignore_index=True)

    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(args.results_out, index=False)
    all_results.to_csv(args.per_reading_out, index=False)

    summary = {
        "test": "fast_spatial_holdout",
        "candidate": meta["candidate"], "target_mode": target_mode,
        "caveat": "Kriged target surface held fixed across folds; only the RF "
                  "training set was region-restricted. Cells inside each held-out "
                  "quadrant may still carry information from training wells via "
                  "kriging's spatial smoothing.",
        "pooled": {"model": m_model, "kriging_only": m_krig, "persistence": m_pers},
        "per_quadrant": fold_rows,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 72)
    print("FAST SPATIAL-GENERALIZATION HOLDOUT (checklist item 3, fast version)")
    print("=" * 72)
    print(fold_df.to_string(index=False))
    print()
    print(f"POOLED model R2={m_model['r2']:.3f} RMSE={m_model['rmse']:.3f}  "
          f"(kriging-only R2={m_krig['r2']:.3f}, persistence R2={m_pers['r2']:.3f})")
    print()
    print(f"[fast_holdout] Saved per-quadrant metrics -> {args.results_out}")
    print(f"[fast_holdout] Saved per-reading results   -> {args.per_reading_out}")
    print(f"[fast_holdout] Saved summary               -> {args.summary_out}")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Fast-version spatial-generalization holdout test.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    p.add_argument("--meta_out", default="reports/rf_v2_model_meta.json",
                   help="Saved meta json from rf_downscale_v2.py --train (defines "
                        "candidate/target_mode/feature_columns/hyperparameters to reuse).")
    p.add_argument("--results_out", default="reports/fast_spatial_holdout.csv")
    p.add_argument("--per_reading_out", default="reports/fast_spatial_holdout_readings.csv")
    p.add_argument("--summary_out", default="reports/fast_spatial_holdout_summary.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()