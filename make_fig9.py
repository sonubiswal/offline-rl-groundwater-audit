#!/usr/bin/env python3
"""Fig 9: rollout noise replicates under the repaired seeding scheme.

Twelve independent seed bases, with the colliding value and the reported
value marked for comparison. Legend placed below the axes so it does
not overlap any data point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(r"C:\trishna-opal")
REV = REPO / "revision_experiments"
OUT = REPO / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

d = json.loads((REV / "rng_noise_study.json").read_text())
gaps = np.array(d["gaps"], float)
mean = float(d["mean"])
sd = float(d["sd"])
colliding = float(d["colliding_value"])
reported = float(d["reported_value"])

# ---------------------------------------------------------------------
# Figure with extra bottom space for a legend row below the axes
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.1))

x = np.arange(1, len(gaps) + 1)

# Shaded mean +/- 1 SD band
ax.axhspan(mean - sd, mean + sd,
           color="#1F5A96", alpha=0.10, zorder=1,
           label=f"mean $\\pm$ 1 SD = {mean:+.3f} $\\pm$ {sd:.3f}")

# Twelve data points
ax.scatter(x, gaps, s=45, color="#1F5A96", zorder=4,
           edgecolor="white", linewidth=0.6)

# Mean line
ax.axhline(mean, color="#1F5A96", lw=1.0, ls="--", zorder=3)

# Colliding and reported reference lines
ax.axhline(colliding, color="#C1272D", lw=1.3, zorder=2)
ax.axhline(reported, color="#2E8B57", lw=1.3, zorder=2)

ax.axhline(0, color="black", lw=0.5, alpha=0.4, zorder=1)

# Axes
ax.set_xlim(0.5, len(gaps) + 0.5)
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_xlabel("Independent seed-base replicate")
ax.set_ylabel("Rollout gap (ten-seed mean)")
ax.set_title("RNG repair: colliding value lies within seed-base noise")

# Room above for the reference lines' text and below for the legend
ax.set_ylim(min(gaps.min(), colliding) - 0.02,
            max(gaps.max(), reported) + 0.04)

# Annotate the two reference lines directly on the plot
ax.text(len(gaps) + 0.35, colliding,
        f"colliding\n{colliding:+.4f}",
        ha="right", va="center",
        fontsize=7, color="#C1272D",
        bbox=dict(boxstyle="round,pad=0.18",
                  facecolor="white", edgecolor="#C1272D",
                  linewidth=0.5, alpha=0.9))

ax.text(len(gaps) + 0.35, reported,
        f"reported\n{reported:+.4f}",
        ha="right", va="center",
        fontsize=7, color="#2E8B57",
        bbox=dict(boxstyle="round,pad=0.18",
                  facecolor="white", edgecolor="#2E8B57",
                  linewidth=0.5, alpha=0.9))

# Legend placed BELOW the axes, in three columns, no frame
handles = [
    plt.Line2D([], [], marker="o", ls="", color="#1F5A96",
               markersize=6, label="12 seed bases"),
    plt.Line2D([], [], color="#1F5A96", lw=1.0, ls="--",
               label=f"mean = {mean:+.4f}"),
    plt.Line2D([], [], color="#C1272D", lw=1.3,
               label=f"colliding = {colliding:+.4f}"),
    plt.Line2D([], [], color="#2E8B57", lw=1.3,
               label=f"reported = {reported:+.4f}"),
]
ax.legend(handles=handles,
          loc="upper center",
          bbox_to_anchor=(0.5, -0.22),
          ncol=2, frameon=False,
          handletextpad=0.5,
          columnspacing=1.2)

fig.savefig(OUT / "fig09_rng_noise.pdf")
fig.savefig(OUT / "fig09_rng_noise.png", dpi=400)
plt.close(fig)

print(f"wrote {OUT / 'fig09_rng_noise.pdf'}")
print(f"wrote {OUT / 'fig09_rng_noise.png'}")
print(f"  mean {mean:+.4f}  SD {sd:.4f}  "
      f"range [{gaps.min():+.4f}, {gaps.max():+.4f}]")
print(f"  colliding {colliding:+.4f}  reported {reported:+.4f}")