#!/usr/bin/env python3
"""
heldout_model_comparison.py
===========================
Train all 4 candidate models (rf_depth, rf_residual, hgbr_depth, hgbr_residual)
on the same v2 training table, evaluate each on the same held-out readings,
and report RMSE/MAE/R2 per candidate.

Answers: does the spatial-CV-selected winner also win on held-out?

FIX vs the original draft: (r, c) for held-out readings is now computed with
feature_utils.latlon_to_grid_cell -- the SAME cell-center convention used to
build the training table (build_training_table_v2), validate_v2, and the
distance grids. The original draft reimplemented its own edge-based mapping
((GRID_SIZE - 1) divisor, no half-cell offset), which assigns a different
(r, c) than the training table for most lat/lon values away from the grid
center. That mismatch would silently join a held-out reading's row to a
different training-table cell's covariates -- exactly the kind of error
that invalidates a "last scientific unknown" comparison without producing
an obvious symptom (the join still succeeds, just against the wrong cell).

Approach:
  - Load training table.
  - Map each held-out reading's lat/lon to (r, c) via the canonical function.
  - Join held-out covariates from the training table via (date, r, c).
  - For each candidate, fit on the full training table, predict on held-out.
  - Compare against kriging / persistence baselines already in the held-out
    validation CSV (unchanged, not rejoined).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import latlon_to_grid_cell  # canonical mapping -- do not reimplement
import yaml

RANDOM_STATE = 42


def make_rf():
    return RandomForestRegressor(n_estimators=300, max_depth=15,
                                 min_samples_split=10, min_samples_leaf=5,
                                 random_state=RANDOM_STATE, n_jobs=-1)


def make_hgbr():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=RANDOM_STATE)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/data_config.yaml",
                    help="Supplies the bbox used for latlon_to_grid_cell -- "
                         "must be the same bbox the training table was built with.")
    ap.add_argument("--table", default="data/processed/rf_training_table.csv",
                    help="NOTE: rf_downscale_v2.py's do_train() does not save this "
                         "CSV by default -- confirm you have a step that writes it "
                         "(or point this at wherever your training table lives) "
                         "before running.")
    ap.add_argument("--val_csv", default="reports/rf_v2_validation.csv",
                    help="Held-out validation CSV with columns actual, lat, lon, date, "
                         "baseline_kriging_only, baseline_persistence_lag -- point this "
                         "at whichever held-out run you want to compare candidates on.")
    ap.add_argument("--out", default="reports/heldout_model_comparison.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])

    df = pd.read_csv(args.table)
    print(f"[load] training table: {len(df)} rows, {df.columns.tolist()[:8]}...")

    # normalize v1 -> canonical
    rename = {}
    if "row" in df.columns and "r" not in df.columns: rename["row"] = "r"
    if "col" in df.columns and "c" not in df.columns: rename["col"] = "c"
    if "target_lag1" in df.columns and "lag" not in df.columns: rename["target_lag1"] = "lag"
    if rename:
        df = df.rename(columns=rename)
        print(f"[normalize] {rename}")

    feat_cols = ["chirps", "gldas", "sentinel2_ndvi", "grace",
                 "srtm_elevation", "srtm_slope",
                 "month_sin", "month_cos",
                 "chirps_roll3", "gldas_roll3", "lag"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"training table missing: {missing}")

    val = pd.read_csv(args.val_csv)
    print(f"[load] held-out readings: {len(val)}")

    # canonical (r, c) -- same convention the training table itself uses
    rc = [latlon_to_grid_cell(la, lo, bbox) for la, lo in zip(val["lat"], val["lon"])]
    val["r"], val["c"] = zip(*rc)

    lookup = df[["date", "r", "c"] + feat_cols].drop_duplicates(["date", "r", "c"])
    merged = val.merge(lookup, on=["date", "r", "c"], how="left")
    n_before = len(merged)
    merged = merged.dropna(subset=feat_cols)
    print(f"[join] {len(merged)} / {n_before} held-out readings have full covariates")

    if len(merged) == 0:
        raise SystemExit("no held-out readings could be joined -- check r/c mapping / bbox")

    X_test = merged[feat_cols].values
    y_test = merged["actual"].values
    lag_test = merged["lag"].values if "lag" in merged.columns else None

    candidates = [
        ("rf_depth",       make_rf(),  "depth"),
        ("rf_residual",    make_rf(),  "residual"),
        ("hgbr_depth",     make_hgbr(), "depth"),
        ("hgbr_residual",  make_hgbr(), "residual"),
    ]

    rows = []
    for name, model, mode in candidates:
        print(f"\n[{name}] fitting on {len(df)} rows, mode={mode}...")
        if mode == "depth":
            y_train = df["target"].values
        else:
            y_train = (df["target"] - df["lag"]).values

        model.fit(df[feat_cols].values, y_train)
        pred = model.predict(X_test)
        if mode == "residual":
            pred = pred + lag_test

        m = metrics(y_test, pred)
        rows.append({"candidate": name, "target_mode": mode, **m})
        print(f"  {name:>16s}  N={m['n']:>5}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

    # baselines on same readings (unchanged from the held-out CSV, not rejoined)
    m_kr = metrics(y_test, merged["baseline_kriging_only"].values)
    m_pe = metrics(y_test, merged["baseline_persistence_lag"].values)
    rows.append({"candidate": "kriging_only", "target_mode": "n/a", **m_kr})
    rows.append({"candidate": "persistence", "target_mode": "n/a", **m_pe})

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print()
    print("=" * 78)
    print("HELD-OUT MODEL COMPARISON")
    print("=" * 78)
    print(out.to_string(index=False))
    print()
    best = out[~out["candidate"].isin(["kriging_only", "persistence"])]
    best = best.sort_values("rmse").iloc[0]
    print(f"best held-out: {best['candidate']}  RMSE={best['rmse']:.4f}  R2={best['r2']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()