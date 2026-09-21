#!/usr/bin/env python3
"""
compare_v1_v2.py
================
Compare v1 (Section 12 production model, 11 features, requires Sentinel-2)
against v2-noS2 (16 features, no Sentinel-2) on the readings where v1 can
actually predict.

v1 requires sentinel2_ndvi at the cell within 90 days. v2-noS2 doesn't.
The comparison subset is therefore the intersection: readings where BOTH
models can produce a prediction.

Reports:
  - N of comparison set vs N of full v2 evaluation
  - RMSE/MAE/R2 for v1, v2, kriging on the SAME comparison set
  - Paired bootstrap CI for (v2 - v1) and (v1 - kriging)
  - Representativeness check: which readings were dropped, and how their
    depth/lat/lon/season distribution differs from the full set
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (  # noqa: E402
    find_nearest_raster,
    load_raster_array,
    rolling_average_features,
    lag_target_feature,
    latlon_to_grid_cell,
    seasonal_features,
)

V1_FEATURES = [
    "chirps", "gldas", "sentinel2_ndvi", "grace",
    "srtm_elevation", "srtm_slope",
    "month_sin", "month_cos",
    "chirps_roll3", "gldas_roll3", "target_lag1",
]

INTERIM = Path("data/interim")
KRIGED = Path("data/interim/kriged_target_monthly")
V1_MAX_GAP = 90
V1_LAG_MAX_GAP = 120


def build_v1_stack(date_str):
    """Build the v1 covariate stack for one date. Returns dict or None."""
    feats = {}
    for src in ["chirps", "gldas", "sentinel2_ndvi", "grace"]:
        p = find_nearest_raster(INTERIM / src, date_str, V1_MAX_GAP)
        if p is None:
            return None
        feats[src] = load_raster_array(p)
    for name, sub in [("srtm_elevation", ("srtm", "elevation.tif")),
                      ("srtm_slope", ("srtm", "slope.tif"))]:
        p = INTERIM.joinpath(*sub)
        if not p.exists():
            return None
        feats[name] = load_raster_array(p)
    roll3 = rolling_average_features(
        INTERIM, date_str, sources=["chirps", "gldas"],
        n_months=3, max_gap_days=V1_MAX_GAP, min_months=2)
    if roll3 is None:
        return None
    feats.update(roll3)
    lag = lag_target_feature(KRIGED, date_str, V1_LAG_MAX_GAP)
    if lag is None:
        return None
    feats["target_lag1"] = lag
    return feats


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "rmse": float("nan"),
                "mae": float("nan"), "r2": float("nan")}
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(1 - np.sum((y_true - y_pred) ** 2) /
                    np.sum((y_true - y_true.mean()) ** 2)),
    }


def main():
    cfg = yaml.safe_load(open("config/data_config.yaml"))
    bbox = tuple(cfg["region"]["bbox"])

    v1 = joblib.load("data/processed/rf_downscale_model.joblib")
    print(f"[load] v1 model  n_features_in_={v1.n_features_in_}  "
          f"(expect 11)")
    if v1.n_features_in_ != 11:
        raise SystemExit(
            f"v1 model has {v1.n_features_in_} features, expected 11. "
            f"It may have been overwritten by an experimental run."
        )

    ev = pd.read_csv("reports/rf_v2_noS2_validation.csv")
    print(f"[load] v2 evaluation readings: {len(ev)}")

    # Cache v1 stacks per unique date -- only ~34 dates, so this is fast.
    unique_dates = sorted(ev["date"].astype(str).unique())
    print(f"[v1] building covariate stacks for {len(unique_dates)} dates...")
    stack_cache = {}
    for i, d in enumerate(unique_dates):
        stack_cache[d] = build_v1_stack(d)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(unique_dates)} dates done")

    v1_preds = []
    v2_preds = []
    actual = []
    kriging = []
    well_ids = []
    dates = []
    lat_list = []
    lon_list = []
    n_skipped = 0

    for _, r in ev.iterrows():
        date_str = str(r["date"])
        stack = stack_cache.get(date_str)
        if stack is None:
            n_skipped += 1
            continue
        rr, cc = latlon_to_grid_cell(r["lat"], r["lon"], bbox)
        season = seasonal_features(date_str)
        vec = {}
        ok = True
        for col in V1_FEATURES:
            if col in ("month_sin", "month_cos"):
                vec[col] = float(season[col])
            else:
                v = float(stack[col][rr, cc])
                if np.isnan(v):
                    ok = False
                    break
                vec[col] = v
        if not ok:
            n_skipped += 1
            continue
        x = np.array([vec[c] for c in V1_FEATURES]).reshape(1, -1)
        v1_preds.append(float(v1.predict(x)[0]))
        v2_preds.append(float(r["predicted"]))
        actual.append(float(r["actual"]))
        kriging.append(float(r["baseline_kriging_only"]))
        well_ids.append(r["well_id"])
        dates.append(date_str)
        lat_list.append(float(r["lat"]))
        lon_list.append(float(r["lon"]))

    print()
    print("=" * 70)
    print("COMPARISON SET")
    print("=" * 70)
    print(f"  v2 full evaluation:  {len(ev)} readings")
    print(f"  v1-comparable:       {len(actual)} readings "
          f"({100 * len(actual) / len(ev):.1f}% of v2)")
    print(f"  v1 cannot predict:   {n_skipped} readings "
          f"({100 * n_skipped / len(ev):.1f}% of v2)")

    if len(actual) < 100:
        raise SystemExit("Comparison set too small to report.")

    df = pd.DataFrame({
        "well_id": well_ids, "date": dates,
        "lat": lat_list, "lon": lon_list,
        "actual": actual, "pred_v1": v1_preds,
        "pred_v2": v2_preds, "kriging": kriging,
    })

    m1 = metrics(df["actual"], df["pred_v1"])
    m2 = metrics(df["actual"], df["pred_v2"])
    mk = metrics(df["actual"], df["kriging"])

    print()
    print("=" * 70)
    print(f"METRICS ON THE {len(df)}-READING COMPARISON SET")
    print("=" * 70)
    print(f"  {'model':<12s} {'N':>6} {'RMSE':>9} {'MAE':>9} {'R2':>9}")
    for name, m in [("v1", m1), ("v2-noS2", m2), ("kriging", mk)]:
        print(f"  {name:<12s} {m['n']:>6} {m['rmse']:>9.4f} "
              f"{m['mae']:>9.4f} {m['r2']:>9.4f}")

    e1 = (df["actual"] - df["pred_v1"]).values ** 2
    e2 = (df["actual"] - df["pred_v2"]).values ** 2
    ek = (df["actual"] - df["kriging"]).values ** 2

    rng = np.random.default_rng(42)
    n = len(df)
    idx_all = np.arange(n)
    boot_v2_v1 = []
    boot_v1_k = []
    for _ in range(5000):
        idx = rng.choice(idx_all, n, replace=True)
        r_v1 = np.sqrt(e1[idx].mean())
        r_v2 = np.sqrt(e2[idx].mean())
        r_k = np.sqrt(ek[idx].mean())
        boot_v2_v1.append(r_v2 - r_v1)
        boot_v1_k.append(r_v1 - r_k)

    d21 = np.array(boot_v2_v1)
    d1k = np.array(boot_v1_k)
    rmse1 = float(np.sqrt(e1.mean()))
    rmse2 = float(np.sqrt(e2.mean()))
    rmsek = float(np.sqrt(ek.mean()))

    print()
    print("=" * 70)
    print("PAIRED BOOTSTRAP (95% CI, 5000 replicates)")
    print("=" * 70)
    print(f"  v2 - v1 RMSE:    {rmse2 - rmse1:+.4f}   "
          f"CI [{np.percentile(d21, 2.5):+.4f}, "
          f"{np.percentile(d21, 97.5):+.4f}]")
    print(f"  v1 - krig RMSE:  {rmse1 - rmsek:+.4f}   "
          f"CI [{np.percentile(d1k, 2.5):+.4f}, "
          f"{np.percentile(d1k, 97.5):+.4f}]")

    # Representativeness: compare dropped vs kept
    print()
    print("=" * 70)
    print("REPRESENTATIVENESS OF THE COMPARISON SUBSET")
    print("=" * 70)
    ev["date_str"] = ev["date"].astype(str)
    kept_keys = set(zip(df["well_id"], df["date"]))
    ev["in_v1_set"] = [k in kept_keys for k in
                       zip(ev["well_id"], ev["date_str"])]
    dropped = ev[~ev["in_v1_set"]]

    for col in ["actual", "lat", "lon"]:
        a = ev[ev["in_v1_set"]][col]
        b = dropped[col]
        pooled = np.sqrt((a.var() + b.var()) / 2)
        smd = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
        flag = "  <-- |SMD|>0.25" if abs(smd) > 0.25 else ""
        print(f"  {col:<8s}  kept={a.mean():.3f}  dropped={b.mean():.3f}  "
              f"SMD={smd:+.3f}{flag}")

    ev["month"] = pd.to_datetime(ev["date"]).dt.month
    ev["season"] = ev["month"].apply(
        lambda m: "monsoon" if m in [6, 7, 8, 9]
        else "post-monsoon" if m in [10, 11] else "pre-monsoon")
    print()
    print("  Season coverage in comparison set:")
    for s in ["monsoon", "pre-monsoon", "post-monsoon"]:
        n_kept = int(((ev["in_v1_set"]) & (ev["season"] == s)).sum())
        n_total = int((ev["season"] == s).sum())
        print(f"    {s:<14s}  kept={n_kept:>5}/{n_total:>5} "
              f"({100 * n_kept / max(n_total, 1):.1f}%)")

    df.to_csv("reports/v1_vs_v2_comparison.csv", index=False)
    print()
    print("wrote reports/v1_vs_v2_comparison.csv")


if __name__ == "__main__":
    main()