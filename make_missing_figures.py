#!/usr/bin/env python3
"""Generate the three figures that are cited in the manuscript but were
never saved as files:

    Fig. 1  matched-start recipe schematic
    Fig. 3  per-seed paired contrast at the recorded FQE interval
    Fig. 5  FQE-rollout offset by policy and target-update interval

All output goes to C:\\trishna-opal\\paper_figures\\.

The script is self-contained, uses only numpy, scipy and matplotlib, and
reads the empirical values from the archived JSON files. It does not
recompute any experiment; it only plots numbers that are already in the
paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ------------------------------------------------------------------ paths
REPO = Path(r"C:\trishna-opal")
REV = REPO / "revision_experiments"
OUT = REPO / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Ten training seeds in the order used throughout the paper
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
})

# Colour palette matching the rest of the paper
C_BLUE   = "#1F5A96"
C_ORANGE = "#D97B22"
C_RED    = "#C1272D"
C_GREEN  = "#2E8B57"
C_GREY   = "#555555"
C_LIGHT  = "#EAF0F7"


# ================================================================== Fig. 1
def fig1_recipe_schematic() -> None:
    """Four-step matched-start audit recipe as a horizontal flow.

    No LaTeX maths in the boxes; the schematic is a diagram, not an
    equation. Formula A_s lives in the paper text, not here.
    """
    fig, ax = plt.subplots(figsize=(7.1, 1.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        (0.10,
         "Fix inputs",
         "policies, starts,\ntarget, reward,\nhorizon"),
        (2.60,
         "Evaluate\nboth estimators",
         "FQE value\nand\nrollout return"),
        (5.10,
         "Pair by seed",
         "per-seed\ncontrast\n$A_s$"),
        (7.60,
         "Report contrast",
         "seed-level $t$,\nbootstrap,\npermutation"),
    ]

    for x, title, body in steps:
        box = FancyBboxPatch(
            (x, 0.14), 2.10, 0.72,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            linewidth=0.8,
            edgecolor=C_BLUE,
            facecolor=C_LIGHT,
        )
        ax.add_patch(box)
        ax.text(x + 1.05, 0.66, title,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=C_BLUE,
                linespacing=1.15)
        ax.text(x + 1.05, 0.35, body,
                ha="center", va="center",
                fontsize=7.5, color="#333333",
                linespacing=1.35)

    for x in (2.24, 4.74, 7.24):
        ax.add_patch(FancyArrowPatch(
            (x, 0.50), (x + 0.32, 0.50),
            arrowstyle="->,head_length=4,head_width=3",
            color=C_BLUE, lw=1.0,
        ))

    fig.savefig(OUT / "fig01_recipe.pdf")
    plt.close(fig)
    print("wrote fig01_recipe.pdf")


# ================================================================== Fig. 3
def fig3_paired_seed_contrast() -> None:
    """Per-seed paired FQE-rollout contrast at the recorded interval.

    Bar chart, one bar per training seed, positive bars in blue and
    negative bars in red, with the ten-seed mean marked.
    """
    e2_path = REV / "E2" / "e2_fqe.json"
    if not e2_path.exists():
        e2_path = REV / "e2_fqe.json"
    fqe = np.array(json.loads(e2_path.read_text())
                   ["primary_episode_starts"]["per_seed_diffs"])

    roll = np.array([
        r["mean_gap"]
        for r in json.loads(
            (REV / "e3c_test225_constructions_rerun.json").read_text()
        )["per_seed"]
    ])

    if fqe.shape != (10,) or roll.shape != (10,):
        raise SystemExit("Expected ten per-seed values; check the JSONs.")

    contrast = fqe - roll
    mean_val = float(contrast.mean())

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    x = np.arange(len(contrast))

    colors = [C_BLUE if c > 0 else C_RED for c in contrast]
    ax.bar(x, contrast, width=0.72, color=colors, edgecolor="none",
           zorder=3)

    ax.axhline(0, color="black", lw=0.7, zorder=4)
    ax.axhline(mean_val, color=C_GREEN, lw=1.2, ls="--", zorder=5,
               label=f"mean = {mean_val:+.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS],
                       rotation=45, ha="right")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Paired contrast (FQE $-$ rollout)")
    ax.set_title("Per-seed paired contrast at the recorded interval")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(top=max(contrast.max() * 1.18, 1.0))

    fig.savefig(OUT / "fig03_paired_seed_contrast.pdf")
    plt.close(fig)
    print(f"wrote fig03_paired_seed_contrast.pdf  "
          f"(mean {mean_val:+.4f}, "
          f"{(contrast > 0).sum()} of 10 positive)")


# ================================================================== Fig. 5
def fig5_offset_bars() -> None:
    """FQE-rollout offset per policy and target-update interval.

    Grouped bar chart: at each of the three FQE target-update intervals
    (100, 200, 500) the offsets for BC and CQL are shown side by side,
    and a separate single bar shows the derived BCQ offset at the
    recorded interval.
    """
    # Values from Table 8 and Table 9 of the paper
    bc_fqe  = {100: 11.600, 200: 12.918, 500: 9.666}
    cql_fqe = {100: 12.498, 200: 13.266, 500: 10.489}
    bc_roll  = 9.857
    cql_roll = 9.871
    intervals = [100, 200, 500]

    bc_off  = [bc_fqe[t]  - bc_roll  for t in intervals]
    cql_off = [cql_fqe[t] - cql_roll for t in intervals]

    # Derived BCQ offset at interval 200, from the E4 extension
    bcq_off = 17.92

    fig, ax = plt.subplots(figsize=(5.6, 2.9))

    x = np.arange(len(intervals) + 1)   # three intervals plus BCQ slot
    w = 0.34

    bc_x  = x[:-1] - w / 2
    cql_x = x[:-1] + w / 2
    bcq_x = x[-1] + 0.10

    ax.bar(bc_x,  bc_off,  w, color=C_BLUE,   label="BC",
           zorder=3)
    ax.bar(cql_x, cql_off, w, color=C_ORANGE, label="CQL",
           zorder=3)
    ax.bar(bcq_x, bcq_off, w, color=C_GREEN,
           label="BCQ (derived, $\\tau = 200$)", zorder=3)

    # Value labels on top of the BCQ bar
    ax.text(bcq_x, bcq_off + 0.4, f"{bcq_off:.1f}",
            ha="center", va="bottom",
            fontsize=8, color=C_GREEN)

    ax.axhline(0, color="black", lw=0.6, zorder=4)
    ax.set_xticks(list(x[:-1]) + [bcq_x])
    ax.set_xticklabels(["100", "200", "500", "200\n(BCQ)"])
    ax.set_xlabel("FQE target-update interval $\\tau$")
    ax.set_ylabel("FQE $-$ rollout offset")
    ax.set_title("FQE overstates each policy by an interval-dependent amount")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(top=max(bcq_off * 1.15, 1.0))

    fig.savefig(OUT / "fig05_offset_bars.pdf")
    plt.close(fig)
    print(f"wrote fig05_offset_bars.pdf  "
          f"(BC offsets {bc_off}, CQL offsets {cql_off}, BCQ {bcq_off})")


# ================================================================== main
def main() -> None:
    fig1_recipe_schematic()
    fig3_paired_seed_contrast()
    fig5_offset_bars()
    print(f"\nall three figures written to {OUT}")


if __name__ == "__main__":
    main()