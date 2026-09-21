"""
regen_plot.py -- regenerate the RF v2 predicted-vs-actual scatter plot
from an existing validation CSV. Does not re-run the model.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/rf_v2_noS2_validation.csv", type=Path)
    ap.add_argument("--out", default="reports/rf_v2_noS2_predicted_vs_actual.png", type=Path)
    ap.add_argument("--title", default="RF v2 vs Kriging-only on independent held-out wells")
    ap.add_argument("--label", default="RF v2 (hgbr_residual)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["actual", "predicted"])

    m_model_r2 = r2_score(df["actual"], df["predicted"])
    m_model_rmse = float(np.sqrt(mean_squared_error(df["actual"], df["predicted"])))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(df["actual"], df["predicted"], alpha=0.45, s=16,
               label=f"{args.label} R2={m_model_r2:.3f} (RMSE={m_model_rmse:.2f} m)")

    if "baseline_kriging_only" in df.columns:
        kdf = df.dropna(subset=["baseline_kriging_only"])
        m_krig_r2 = r2_score(kdf["actual"], kdf["baseline_kriging_only"])
        ax.scatter(kdf["actual"], kdf["baseline_kriging_only"], alpha=0.25, s=12,
                   marker="x", label=f"Kriging-only R2={m_krig_r2:.3f}")
        all_preds = pd.concat([df["predicted"], kdf["baseline_kriging_only"]])
    else:
        all_preds = df["predicted"]

    lims = [min(df["actual"].min(), all_preds.min()),
            max(df["actual"].max(), all_preds.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual depth (m) -- held-out CGWB wells")
    ax.set_ylabel("Prediction (m)")
    ax.set_title(args.title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()