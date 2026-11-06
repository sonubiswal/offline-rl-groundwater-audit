#!/usr/bin/env python3
"""
make_all_figures.py — generate all TRISHNA-OPAL Paper 1 figures.

Outputs (PDF + PNG, 300 dpi) into ./figures/:
  fig1_wellsplit    — training vs held-out well map
  fig2_pipeline     — data-flow schematic
  fig3_scatter      — predicted vs observed, three methods
  fig4_distance     — ML vs Kriging RMSE by distance bucket
  fig5_depthbias    — RMSE and bias by depth
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---- configuration -----------------------------------------------------
FROZEN = Path(r"C:\trishna-qc\REAL_V2\ORIGINAL_FROZEN\data")
PRED   = Path(r"C:\trishna-qc\REAL_V2\strict_kriging_v2\predictions.parquet")
OUT    = Path("figures"); OUT.mkdir(exist_ok=True)

DPI = 300


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    print(f"[fig] wrote {OUT / (stem + '.png')}")


# ========================================================================
# Figure 1 — Well split map
# ========================================================================
def fig1_wellsplit():
    feat  = pd.read_parquet(FROZEN / "feature_table.parquet")
    split = pd.read_csv(FROZEN / "split_assignments.csv")
    wells = (feat.drop_duplicates("well_id")[["well_id", "lat", "lon"]]
             .merge(split, on="well_id", how="left"))
    tr = wells[wells.fold_id != 0]
    ho = wells[wells.fold_id == 0]

    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=DPI)
    ax.scatter(tr.lon, tr.lat, s=6, c="#3b6fa8", alpha=0.65,
               label=f"training wells (n = {len(tr)})", edgecolors="none")
    ax.scatter(ho.lon, ho.lat, s=18, c="#d94f4f", alpha=0.95,
               label=f"held-out wells (n = {len(ho)})",
               edgecolors="black", linewidths=0.3)
    ax.set_xlim(72.6, 80.9); ax.set_ylim(15.6, 22.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title("Maharashtra study area — well split", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig1_wellsplit"); plt.close(fig)


# ========================================================================
# Figure 2 — Pipeline schematic
# ========================================================================
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=DPI)
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.06,rounding_size=0.12",
                     linewidth=1.2, edgecolor="black", facecolor=fc))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=12,
                     linewidth=1.2, linestyle=ls, color="black"))

    box(0.2, 4.2, 2.2, 0.7, "CGWB\nwell observations\n(92,227 / 4,063 wells)", "#dce6f2")
    box(2.9, 4.2, 2.2, 0.7, "Ordinary kriging\n(training wells only)", "#dce6f2")
    box(5.6, 4.2, 2.2, 0.7, "Random-forest\ndownscaling\n(GRACE, CHIRPS, …)", "#dce6f2")
    box(8.3, 4.2, 2.4, 0.7, "30,000-transition\noffline dataset\n(1,500 ep × 20 steps)", "#dce6f2")
    box(3.3, 2.7, 2.2, 0.7, "Behavioural\ncloning (BC)", "#a8c4e0")
    box(6.4, 2.7, 2.2, 0.7, "Conservative\nQ-learning (CQL)", "#e5a8a8")
    box(0.3, 0.9, 2.1, 0.8, "FQE\n(seed bootstrap)", "#f0f0d0")
    box(2.7, 0.9, 2.1, 0.8, "Direct rollout\n(episode bootstrap)", "#f0f0d0")
    box(5.1, 0.9, 2.3, 0.8, "Banded OOD\n(4 Mahalanobis bands)", "#f0f0d0")
    box(7.7, 0.9, 2.3, 0.8, "Strict per-fold\nKriging baseline", "#f0f0d0")

    arrow(2.4, 4.55, 2.9, 4.55)
    arrow(5.1, 4.55, 5.6, 4.55)
    arrow(7.8, 4.55, 8.3, 4.55)
    arrow(6.7, 4.2, 4.4, 3.4)
    arrow(7.5, 4.2, 7.5, 3.4)
    arrow(4.4, 2.7, 1.35, 1.7, ls="--")
    arrow(4.4, 2.7, 3.75, 1.7, ls="--")
    arrow(7.5, 2.7, 6.25, 1.7, ls="--")
    arrow(7.5, 2.7, 8.85, 1.7, ls="--")

    ax.text(5.5, 3.55, "training", ha="center", fontsize=8,
            style="italic", color="#555")
    fig.tight_layout()
    save(fig, "fig2_pipeline"); plt.close(fig)


# ========================================================================
# Figure 3 — Predicted vs observed
# ========================================================================
def fig3_scatter():
    df = pd.read_parquet(PRED)
    mask = (~np.isnan(df.krig_pred) & ~np.isnan(df.ml_pred)
            & ~np.isnan(df.persistence_pred))
    df = df[mask].reset_index(drop=True)
    y = df.gwl_depth_m.values

    panels = [("ML (HGBR)", df.ml_pred.values, "#3b6fa8"),
              ("Strict Kriging", df.krig_pred.values, "#2e8b57"),
              ("Persistence", df.persistence_pred.values, "#b8860b")]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=DPI, sharey=True)
    for ax, (name, pred, c) in zip(axes, panels):
        ax.scatter(y, pred, s=4, alpha=0.35, c=c, edgecolors="none")
        lo, hi = 0, max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.text(0.05, 0.95, f"RMSE = {rmse:.3f} m\nR² = {r2:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9))
        ax.set_xlabel("Observed depth (m)", fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.3)

    axes[0].set_ylabel("Predicted depth (m)", fontsize=10)
    fig.suptitle("Held-out predictions vs observed groundwater depth (N = 8,107)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_scatter"); plt.close(fig)


# ========================================================================
# Figure 4 — Distance-stratified comparison
# ========================================================================
def fig4_distance():
    labels  = ["0–2", "2–5", "5–10", "10–20", "20–30", "30–50", "50+"]
    n       = [863, 1975, 3918, 1310, 37, 0, 4]
    rmse_ml = [5.8697, 5.5729, 5.0738, 5.0060, 4.4633, np.nan, np.nan]
    rmse_kr = [5.7196, 5.6492, 4.8102, 5.2953, 4.2457, np.nan, np.nan]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=DPI)
    ax.bar(x - w/2, rmse_ml, w, label="ML (HGBR)", color="#3b6fa8",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, rmse_kr, w, label="Strict Kriging", color="#2e8b57",
           edgecolor="black", linewidth=0.6)

    for xi, ni in zip(x, n):
        ymax = max([v for v in [rmse_ml[xi], rmse_kr[xi]] if not np.isnan(v)],
                   default=0)
        ax.text(xi, ymax + 0.1, f"n={ni}", ha="center", fontsize=8, color="#555")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Distance to nearest training well (km)", fontsize=11)
    ax.set_ylabel("Held-out RMSE (m)", fontsize=11)
    ax.set_title("Distance-stratified ML vs Kriging (N = 8,107 matched)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.set_ylim(0, 6.8)
    fig.tight_layout()
    save(fig, "fig4_distance"); plt.close(fig)


# ========================================================================
# Figure 5 — Depth-stratified RMSE and bias
# ========================================================================
def fig5_depthbias():
    labels = ["0–5", "5–10", "10–20", "20–40", "40+"]
    n      = [4103, 2627, 1155, 209, 17]
    rmse   = [3.82, 3.55, 6.28, 15.71, 37.43]
    bias   = [+2.53, -0.20, -4.81, -13.80, -36.60]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=DPI)

    ax1.bar(x, rmse, color="#c96a6a", edgecolor="black", linewidth=0.6)
    for xi, (r, ni) in enumerate(zip(rmse, n)):
        ax1.text(xi, r + 0.8, f"n={ni}", ha="center", fontsize=8, color="#555")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Observed depth (m)", fontsize=10)
    ax1.set_ylabel("RMSE (m)", fontsize=10)
    ax1.set_title("(a) RMSE by depth", fontsize=11)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.35)

    ax2.axhline(0, color="black", lw=0.8)
    ax2.plot(x, bias, "o-", color="#3b6fa8", lw=1.6, markersize=7)
    for xi, (b, ni) in enumerate(zip(bias, n)):
        ax2.annotate(f"n={ni}", (xi, b), textcoords="offset points",
                     xytext=(0, 8 if b > 0 else -12), ha="center",
                     fontsize=8, color="#555")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel("Observed depth (m)", fontsize=10)
    ax2.set_ylabel("Bias (predicted − observed, m)", fontsize=10)
    ax2.set_title("(b) Bias by depth", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle("Depth-stratified prediction error (N = 8,111, canonical QC v2)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_depthbias"); plt.close(fig)


# ========================================================================
# Run all
# ========================================================================
if __name__ == "__main__":
    fig1_wellsplit()
    fig2_pipeline()
    fig3_scatter()
    fig4_distance()
    fig5_depthbias()
    print(f"\n[done] all figures written to {OUT.resolve()}")