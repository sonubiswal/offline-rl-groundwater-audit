#!/usr/bin/env python3
"""
spatial_cv_repeated.py
=======================
Repeat spatial-block CV across multiple block granularities and seeds to test
whether the single-partition result was a fold-luck artifact.

Self-contained. Loads whichever training table exists, adapts to v1 or v2
schema, and runs the CV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

GRID_SIZE = 64
RANDOM_STATE = 42

# Feature list uses the NORMALIZED column names (after normalize_schema renames
# target_lag1 -> lag).
V1_FEATURES = [
    "chirps", "gldas", "sentinel2_ndvi", "grace",
    "srtm_elevation", "srtm_slope",
    "month_sin", "month_cos",
    "chirps_roll3", "gldas_roll3",
    "lag",
]


def make_hgbr():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=RANDOM_STATE)


def make_rf():
    return RandomForestRegressor(
        n_estimators=300, max_depth=None,
        min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1)


def make_factory(name):
    return {"hgbr": make_hgbr, "rf": make_rf}[name]


def normalize_schema(df):
    """Adapt v1 column names to v2 equivalents."""
    renames = {}
    if "row" in df.columns and "r" not in df.columns:
        renames["row"] = "r"
    if "col" in df.columns and "c" not in df.columns:
        renames["col"] = "c"
    if "target_lag1" in df.columns and "lag" not in df.columns:
        renames["target_lag1"] = "lag"
    if renames:
        df = df.rename(columns=renames)
        print(f"[normalize] renamed: {renames}")
    if "row_norm" not in df.columns and "r" in df.columns:
        df["row_norm"] = df["r"] / (GRID_SIZE - 1)
    if "col_norm" not in df.columns and "c" in df.columns:
        df["col_norm"] = df["c"] / (GRID_SIZE - 1)
    return df


def assign_blocks(df, nbpa):
    b = max(1, GRID_SIZE // nbpa)
    return (df["r"] // b) * nbpa + (df["c"] // b)


def spatial_cv(df, feat_cols, target_mode, factory, n_splits, seed):
    def y_of(part):
        if target_mode == "depth":
            return part["target"].values
        if target_mode == "residual":
            return (part["target"] - part["lag"]).values
        raise ValueError(target_mode)

    blocks = df["block"].unique()
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(blocks)), n_splits)

    rmses, r2s = [], []
    for fold_idx in folds:
        test_blocks = set(blocks[i] for i in fold_idx)
        test = df[df["block"].isin(test_blocks)]
        train = df[~df["block"].isin(test_blocks)]
        m = factory()
        m.fit(train[feat_cols].values, y_of(train))
        pred = m.predict(test[feat_cols].values)
        if target_mode == "residual":
            pred = pred + test["lag"].values
        rmses.append(float(np.sqrt(mean_squared_error(test["target"].values, pred))))
        r2s.append(float(r2_score(test["target"].values, pred)))

    return {"cv_rmse_mean": float(np.mean(rmses)),
            "cv_rmse_folds": rmses,
            "cv_r2_mean": float(np.mean(r2s)),
            "cv_r2_folds": r2s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/processed/rf_training_table.csv")
    ap.add_argument("--meta", default=None,
                    help="optional meta json; if its feature_columns all exist "
                         "in the table, they are used instead of V1_FEATURES")
    ap.add_argument("--model_family", choices=["hgbr", "rf"], default="hgbr")
    ap.add_argument("--target_mode", choices=["depth", "residual"], default="depth")
    ap.add_argument("--block_grids", default="2,4,8",
                    help="comma-separated n_blocks_per_axis values")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seeds per granularity")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--out", default="reports/rf_v2_spatial_cv_repeated.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    print(f"[load] {args.table}  rows={len(df)}  cols={len(df.columns)}")
    df = normalize_schema(df)

    # Feature selection: prefer meta's list if all its columns survive normalize.
    # Meta lists raw names like "target_lag1"; map through the same rename map.
    rename_map = {"target_lag1": "lag", "row": "r", "col": "c"}
    feat_cols = V1_FEATURES
    if args.meta and Path(args.meta).exists():
        meta = json.load(open(args.meta))
        meta_feats_raw = meta.get("feature_columns", [])
        meta_feats = [rename_map.get(c, c) for c in meta_feats_raw]
        missing = [c for c in meta_feats if c not in df.columns]
        if not missing:
            feat_cols = meta_feats
            print(f"[features] using meta's {len(feat_cols)} feature columns")
        else:
            print(f"[features] meta has {len(missing)} cols not in table "
                  f"({missing}); falling back to V1_FEATURES")

    if args.target_mode == "residual" and "lag" not in df.columns:
        raise SystemExit("target_mode=residual requires 'lag' column")

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"training table missing feature columns: {missing}\n"
            f"table has: {df.columns.tolist()}"
        )

    print(f"[features] using {len(feat_cols)}: {feat_cols}")
    print(f"[target_mode] {args.target_mode}")
    print(f"[model_family] {args.model_family}")

    grids = [int(x) for x in args.block_grids.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    factory = make_factory(args.model_family)

    rows = []
    for nbpa in grids:
        df["block"] = assign_blocks(df, nbpa)
        n_blocks = df["block"].nunique()
        if n_blocks < args.n_splits:
            print(f"[skip] nbpa={nbpa}: only {n_blocks} blocks, need >= {args.n_splits}")
            continue
        for seed in seeds:
            res = spatial_cv(df, feat_cols, args.target_mode, factory,
                             args.n_splits, seed)
            rows.append({
                "n_blocks_per_axis": nbpa,
                "n_blocks_present": n_blocks,
                "seed": seed,
                "n_splits": args.n_splits,
                "cv_rmse_mean": res["cv_rmse_mean"],
                "cv_r2_mean": res["cv_r2_mean"],
                "cv_rmse_folds": res["cv_rmse_folds"],
                "cv_r2_folds": res["cv_r2_folds"],
            })
            print(f"  nbpa={nbpa:>2} seed={seed:>2} "
                  f"RMSE={res['cv_rmse_mean']:.3f}  R2={res['cv_r2_mean']:.3f}")

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print()
    print("=" * 60)
    overall = out_df["cv_r2_mean"]
    print(f"OVERALL R2:   mean={overall.mean():.3f}  std={overall.std():.3f}  "
          f"min={overall.min():.3f}  max={overall.max():.3f}")
    print(f"OVERALL RMSE: mean={out_df['cv_rmse_mean'].mean():.3f}  "
          f"std={out_df['cv_rmse_mean'].std():.3f}")
    print()
    print("BY GRANULARITY:")
    for nbpa, g in out_df.groupby("n_blocks_per_axis"):
        print(f"  nbpa={nbpa:>2}: R2={g['cv_r2_mean'].mean():.3f} "
              f"+/- {g['cv_r2_mean'].std():.3f}   "
              f"RMSE={g['cv_rmse_mean'].mean():.3f} "
              f"+/- {g['cv_rmse_mean'].std():.3f}")
    print()
    print("Interpretation:")
    print("  std < 0.05 -> CV is stable; the single-partition result was reliable")
    print("  std > 0.10 -> winner selection was partition-dependent")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()