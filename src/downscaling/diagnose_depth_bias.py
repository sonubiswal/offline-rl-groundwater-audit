"""
diagnose_depth_bias.py

Follow-up diagnostic after generate_phase2_report_plots.py revealed a
strong depth-dependent bias (overprediction at shallow depths,
underprediction at deep depths). This script does two things, using
ONLY already-computed files -- no retraining, no re-touching the held-out
set:

  1. Stratified metrics by depth bin -- quantifies exactly how much worse
     the model gets at depth, for report-ready numbers.

  2. Root-cause check: compares the TRAINING TARGET's value range
     (data/processed/rf_training_table.csv, the Kriged surface RF was
     fit to) against the HELD-OUT ACTUAL depth range (reports/
     rf_downscale_validation.csv). If the Kriged target rarely/never
     reaches the depths where held-out wells go deep, that points to
     Kriging itself smoothing away extremes (sparse deep wells get
     averaged with shallower neighbors) as the root cause -- meaning no
     amount of RF retuning would fix it, since the model was never shown
     deep targets to learn from in the first place. If the target range
     DOES cover deep values but RF still underpredicts them, that points
     to RF's averaging/leaf-based prediction mechanism (which cannot
     output values outside, and tends to shrink toward the center of,
     its training target distribution) as the primary cause instead.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def stratified_metrics_by_depth(results_csv: str, bins: list[float] | None = None) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    bins = bins or [0, 10, 20, 30, 50, np.inf]
    labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}m" if bins[i+1] != np.inf else f"{bins[i]:.0f}m+"
              for i in range(len(bins) - 1)]
    df["depth_bin"] = pd.cut(df["actual"], bins=bins, labels=labels, right=False)

    rows = []
    for label in labels:
        sub = df[df["depth_bin"] == label]
        if len(sub) < 2:
            rows.append({"depth_bin": label, "n": len(sub), "rmse": np.nan, "mae": np.nan,
                         "r2": np.nan, "mean_bias": np.nan})
            continue
        rmse = np.sqrt(mean_squared_error(sub["actual"], sub["predicted"]))
        mae = mean_absolute_error(sub["actual"], sub["predicted"])
        r2 = r2_score(sub["actual"], sub["predicted"]) if len(sub) > 1 else np.nan
        bias = (sub["predicted"] - sub["actual"]).mean()
        rows.append({"depth_bin": label, "n": len(sub), "rmse": rmse, "mae": mae,
                     "r2": r2, "mean_bias": bias})

    return pd.DataFrame(rows)


def compare_target_vs_actual_range(training_table_csv: str, results_csv: str) -> dict:
    train_df = pd.read_csv(training_table_csv)
    held_df = pd.read_csv(results_csv)

    target_stats = {
        "min": train_df["target"].min(), "max": train_df["target"].max(),
        "p95": train_df["target"].quantile(0.95), "p99": train_df["target"].quantile(0.99),
        "mean": train_df["target"].mean(),
    }
    actual_stats = {
        "min": held_df["actual"].min(), "max": held_df["actual"].max(),
        "p95": held_df["actual"].quantile(0.95), "p99": held_df["actual"].quantile(0.99),
        "mean": held_df["actual"].mean(),
    }
    pred_stats = {
        "min": held_df["predicted"].min(), "max": held_df["predicted"].max(),
    }

    return {"target": target_stats, "actual": actual_stats, "predicted": pred_stats}


def run(
    results_csv: str = "reports/rf_downscale_validation.csv",
    training_table_csv: str = "data/processed/rf_training_table.csv",
) -> None:
    print("=" * 70)
    print("1. STRATIFIED METRICS BY DEPTH BIN")
    print("=" * 70)
    strat_df = stratified_metrics_by_depth(results_csv)
    print(strat_df.to_string(index=False))
    print()

    print("=" * 70)
    print("2. ROOT CAUSE CHECK: training target range vs. held-out actual range")
    print("=" * 70)
    stats = compare_target_vs_actual_range(training_table_csv, results_csv)
    print(f"{'':25s} {'min':>10s} {'mean':>10s} {'p95':>10s} {'p99':>10s} {'max':>10s}")
    t, a, p = stats["target"], stats["actual"], stats["predicted"]
    print(f"{'Target (Kriged, training)':25s} {t['min']:>10.2f} {t['mean']:>10.2f} {t['p95']:>10.2f} {t['p99']:>10.2f} {t['max']:>10.2f}")
    print(f"{'Actual (held-out wells)':25s} {a['min']:>10.2f} {a['mean']:>10.2f} {a['p95']:>10.2f} {a['p99']:>10.2f} {a['max']:>10.2f}")
    print(f"{'Predicted (RF output)':25s} {p['min']:>10.2f} {'':>10s} {'':>10s} {'':>10s} {p['max']:>10.2f}")
    print()

    if t["max"] < a["max"] * 0.7:
        print(">>> DIAGNOSIS: the Kriged TRAINING TARGET rarely/never reaches the depths seen")
        print(">>> in held-out wells. RF was never shown deep targets to learn from -- this points")
        print(">>> to Kriging itself smoothing away deep extremes (sparse deep wells get averaged")
        print(">>> with shallower neighbors during interpolation) as a root cause, not just RF's")
        print(">>> own extrapolation limits. No amount of RF hyperparameter tuning can fix this;")
        print(">>> it would require revisiting the Kriging step itself (e.g. checking variogram")
        print(">>> choice, or accepting this as a stated data/methodology limitation).")
    else:
        print(">>> DIAGNOSIS: the Kriged target DOES cover the deep range seen in held-out wells,")
        print(">>> but RF's predictions still fall short of it. This points to RF's own averaging/")
        print(">>> leaf-based prediction mechanism -- which structurally shrinks predictions toward")
        print(">>> the center of the training distribution -- as the primary cause, separate from")
        print(">>> any Kriging-side data limitation.")
    print()
    print("Either way: this is a genuine, well-diagnosed model limitation worth stating plainly")
    print("in the report/paper, not something to chase away via further held-out iteration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose the depth-dependent bias found in Phase 2 report plots.")
    parser.add_argument("--results_csv", default="reports/rf_downscale_validation.csv")
    parser.add_argument("--training_table_csv", default="data/processed/rf_training_table.csv")
    args = parser.parse_args()

    run(args.results_csv, args.training_table_csv)