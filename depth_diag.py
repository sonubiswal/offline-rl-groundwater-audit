"""
depth_diag.py -- depth-stratified error diagnostic for RF downscaling predictions.

Reads a CSV with actual + predicted columns, prints per-bin metrics
(RMSE, MAE, bias, within-bin R^2), plus pooled R^2, median per-well R^2,
and the predicted vs actual maximum. Read-only: never touches any
existing artifact.

Usage:
  python depth_diag.py --csv reports/rf_v2_noS2_validation.csv \
                       --actual actual --pred predicted \
                       --label v2_baseline \
                       --out reports/depth_diag_v2_baseline.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Depth bins used for stratification (matches limitations.md S1)
BINS = [0, 5, 10, 20, 30, 50, 1000]
LABELS = ["0-5", "5-10", "10-20", "20-30", "30-50", "50+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--actual", default="actual")
    ap.add_argument("--pred", default="predicted")
    ap.add_argument("--well-id", default="well_id")
    ap.add_argument("--label", default="model")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[depth_diag] {args.csv}: {len(df)} rows")
    print(f"[depth_diag] columns: {list(df.columns)}")

    # ---- resolve actual column ----
    if args.actual not in df.columns:
        for c in ("depth_m", "y_true", "actual_depth", "truth"):
            if c in df.columns:
                args.actual = c
                break
        else:
            raise SystemExit(
                f"no actual column found; tried {args.actual}, depth_m, y_true, actual_depth, truth"
            )

    # ---- resolve predicted column ----
    if args.pred not in df.columns:
        for c in ("pred", "prediction", "y_pred", "rf_pred", "pred_rf", "model_pred"):
            if c in df.columns:
                args.pred = c
                break
        else:
            raise SystemExit(
                f"no predicted column found; tried {args.pred}, pred, prediction, y_pred, rf_pred, pred_rf, model_pred"
            )

    print(f"[depth_diag] using actual='{args.actual}'  predicted='{args.pred}'")

    df = df.dropna(subset=[args.actual, args.pred]).copy()
    df["residual"] = df[args.pred].astype(float) - df[args.actual].astype(float)

    # ---- pooled R^2 ----
    resid_all = df["residual"].values
    actual_all = df[args.actual].values.astype(float)
    var_all = float(np.var(actual_all))
    if var_all > 0:
        pooled_r2 = float(1.0 - np.sum(resid_all ** 2) / (len(df) * var_all))
    else:
        pooled_r2 = float("nan")

    # ---- per-depth-bin metrics ----
    df["bin"] = pd.cut(df[args.actual], bins=BINS, labels=LABELS)
    bin_rows = []
    for b in LABELS:
        g = df[df["bin"] == b]
        if len(g) == 0:
            bin_rows.append({
                "bin": b, "n": 0,
                "rmse": None, "mae": None, "bias": None, "r2_within": None,
            })
            continue
        resid = g["residual"].values
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        bias = float(np.mean(resid))
        gv = float(np.var(g[args.actual].values.astype(float)))
        if gv > 1e-9:
            r2_within = float(1.0 - np.sum(resid ** 2) / (len(g) * gv))
        else:
            r2_within = float("nan")
        bin_rows.append({
            "bin": b, "n": int(len(g)),
            "rmse": rmse, "mae": mae, "bias": bias, "r2_within": r2_within,
        })

    # ---- per-well R^2 ----
    median_well_r2 = float("nan")
    n_wells_used = 0
    if args.well_id in df.columns:
        well_r2s = []
        for wid, g in df.groupby(args.well_id):
            if len(g) < 2:
                continue
            gv = float(np.var(g[args.actual].values.astype(float)))
            if gv < 1e-9:
                continue
            r = 1.0 - np.sum((g[args.pred] - g[args.actual]).values ** 2) / (len(g) * gv)
            well_r2s.append(float(r))
        if well_r2s:
            median_well_r2 = float(np.median(well_r2s))
            n_wells_used = len(well_r2s)

    # ---- summary ----
    out = {
        "label": args.label,
        "csv": str(args.csv),
        "n": int(len(df)),
        "pooled_r2": pooled_r2,
        "median_per_well_r2": median_well_r2,
        "n_wells_used_for_per_well_r2": n_wells_used,
        "predicted_min": float(df[args.pred].min()),
        "predicted_max": float(df[args.pred].max()),
        "actual_min": float(df[args.actual].min()),
        "actual_max": float(df[args.actual].max()),
        "bins": bin_rows,
    }

    print()
    print(f"[depth_diag] {args.label}")
    print(f"  pooled R^2           : {pooled_r2:.4f}")
    print(f"  median per-well R^2  : {median_well_r2:.4f}  (over {n_wells_used} wells)")
    print(f"  predicted range      : {out['predicted_min']:.2f} .. {out['predicted_max']:.2f} m")
    print(f"  actual range         : {out['actual_min']:.2f} .. {out['actual_max']:.2f} m")
    print()
    header = f"  {'bin':<8} {'n':>7} {'rmse':>8} {'mae':>8} {'bias':>10} {'r2_within':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in bin_rows:
        if r["n"] == 0:
            print(f"  {r['bin']:<8} {r['n']:>7} {'-':>8} {'-':>8} {'-':>10} {'-':>10}")
            continue
        print(f"  {r['bin']:<8} {r['n']:>7} {r['rmse']:>8.2f} {r['mae']:>8.2f} "
              f"{r['bias']:>+10.2f} {r['r2_within']:>10.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\n[depth_diag] wrote {args.out}")


if __name__ == "__main__":
    main()