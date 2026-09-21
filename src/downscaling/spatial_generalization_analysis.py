#!/usr/bin/env python3
"""
spatial_generalization_analysis.py
==================================
Two analyses to close Phase 2 item #3:

1. Distance-to-nearest-training-well analysis:
   For each held-out reading, compute haversine distance from its well to the
   nearest training well, bucket by distance, report RMSE/MAE/R2 per bucket.

2. 5-fold partition of the held-out wells:
   Partition the unique held-out well IDs into 5 groups (seeded), assign each
   reading to its well's group, report RMSE/MAE/R2 per fold + mean±SD.

Both use the existing trained model's predictions. No retraining.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EARTH_R = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "rmse": float("nan"),
                "mae": float("nan"), "r2": float("nan")}
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", default="reports/rf_v2_noS2_validation.csv")
    ap.add_argument("--training_wells", default="data/processed/training_wells.csv")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    val = pd.read_csv(args.val_csv)
    tw = pd.read_csv(args.training_wells)
    print(f"[load] validation readings: {len(val)}")
    print(f"[load] training wells:      {len(tw)}")

    for col in ["well_id", "lat", "lon", "actual", "predicted"]:
        if col not in val.columns:
            raise SystemExit(f"validation csv missing '{col}'. has: {val.columns.tolist()}")
    for col in ["lat", "lon"]:
        if col not in tw.columns:
            raise SystemExit(f"training wells csv missing '{col}'. has: {tw.columns.tolist()}")

    # ---- Analysis 1: distance to nearest training well ----
    tw_lat = tw["lat"].to_numpy()
    tw_lon = tw["lon"].to_numpy()
    uniq = val[["well_id", "lat", "lon"]].drop_duplicates("well_id").reset_index(drop=True)
    print(f"[distance] unique held-out wells: {len(uniq)}")

    min_dist = np.empty(len(uniq), dtype=float)
    for i, row in uniq.iterrows():
        d = haversine_km(row["lat"], row["lon"], tw_lat, tw_lon)
        min_dist[i] = d.min()
    uniq["dist_to_nearest_train_km"] = min_dist

    print(f"[distance] nearest-train distance stats (km):")
    print(f"  min    = {uniq['dist_to_nearest_train_km'].min():.2f}")
    print(f"  median = {uniq['dist_to_nearest_train_km'].median():.2f}")
    print(f"  mean   = {uniq['dist_to_nearest_train_km'].mean():.2f}")
    print(f"  max    = {uniq['dist_to_nearest_train_km'].max():.2f}")

    val = val.merge(uniq[["well_id", "dist_to_nearest_train_km"]], on="well_id", how="left")

    bins = [0, 25, 50, 100, np.inf]
    labels = ["0-25 km", "25-50 km", "50-100 km", "100+ km"]
    val["dist_bucket"] = pd.cut(val["dist_to_nearest_train_km"],
                                bins=bins, labels=labels, right=False)

    print("\n" + "=" * 70)
    print("DISTANCE-TO-TRAINING ANALYSIS (does error grow with remoteness?)")
    print("=" * 70)
    dist_rows = []
    for bucket in labels:
        g = val[val["dist_bucket"] == bucket]
        if len(g) == 0:
            continue
        m = metrics(g["actual"], g["predicted"])
        mk = metrics(g["actual"], g["baseline_kriging_only"])
        dist_rows.append({
            "bucket": str(bucket),
            "n_readings": m["n"],
            "rmse_model": m["rmse"], "mae_model": m["mae"], "r2_model": m["r2"],
            "rmse_kriging": mk["rmse"], "r2_kriging": mk["r2"],
        })
        print(f"  {bucket:<12s}  N={m['n']:>5}  "
              f"model RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}   |   "
              f"kriging RMSE={mk['rmse']:.3f}  R2={mk['r2']:.3f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dist_rows).to_csv(
        Path(args.out_dir) / "spatial_distance_buckets.csv", index=False)
    print(f"\nwrote {args.out_dir}/spatial_distance_buckets.csv")

    # ---- Analysis 2: 5-fold partition of held-out wells ----
    print("\n" + "=" * 70)
    print(f"{args.n_folds}-FOLD PARTITION OF HELD-OUT WELLS (is R2 dominated by some wells?)")
    print("=" * 70)

    well_ids = np.sort(uniq["well_id"].unique())
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(well_ids))
    fold_of_well = {well_ids[idx]: i % args.n_folds for i, idx in enumerate(perm)}
    val["fold"] = val["well_id"].map(fold_of_well)

    fold_rows = []
    for k in range(args.n_folds):
        g = val[val["fold"] == k]
        if len(g) == 0:
            continue
        m = metrics(g["actual"], g["predicted"])
        mk = metrics(g["actual"], g["baseline_kriging_only"])
        n_wells = int(g["well_id"].nunique())
        fold_rows.append({
            "fold": k, "n_wells": n_wells, "n_readings": m["n"],
            "rmse_model": m["rmse"], "mae_model": m["mae"], "r2_model": m["r2"],
            "rmse_kriging": mk["rmse"], "r2_kriging": mk["r2"],
        })
        print(f"  fold {k}:  wells={n_wells:>3}  N={m['n']:>5}  "
              f"model RMSE={m['rmse']:.3f} R2={m['r2']:.3f}   |   "
              f"kriging RMSE={mk['rmse']:.3f} R2={mk['r2']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    print(f"\n  model RMSE: mean={fold_df['rmse_model'].mean():.3f}  "
          f"std={fold_df['rmse_model'].std():.3f}  "
          f"min={fold_df['rmse_model'].min():.3f}  "
          f"max={fold_df['rmse_model'].max():.3f}")
    print(f"  model R2:   mean={fold_df['r2_model'].mean():.3f}  "
          f"std={fold_df['r2_model'].std():.3f}  "
          f"min={fold_df['r2_model'].min():.3f}  "
          f"max={fold_df['r2_model'].max():.3f}")

    fold_df.to_csv(Path(args.out_dir) / "spatial_holdout_5fold.csv", index=False)
    print(f"\nwrote {args.out_dir}/spatial_holdout_5fold.csv")

    overall = metrics(val["actual"], val["predicted"])
    print(f"\noverall held-out (N={overall['n']}): "
          f"RMSE={overall['rmse']:.3f}  MAE={overall['mae']:.3f}  R2={overall['r2']:.3f}")


if __name__ == "__main__":
    main()