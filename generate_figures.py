"""
generate_figures.py — single-file generator for all Paper 2 figures.

Usage:
    python generate_figures.py                    # generate all five
    python generate_figures.py fig1 fig3          # generate a subset
    python generate_figures.py --list             # list available
    python generate_figures.py --outdir my_figs   # custom output dir

Writes: PDF (submission) and PNG (review drafts) for each figure.

Figure numbering matches the manuscript's "Figure captions" list:
    Figure 1 — verdict landscape
    Figure 2 — data flow / methods pipeline
    Figure 3 — banded OOD diagnostic (QR vs. mean-Q)
    Figure 4 — reward-weight ablation at both learning rates
    Figure 5 — A4/A5 sensitivity heatmap

Changelog:
  - v3: Renumbered fig2/fig3/fig5 to match the manuscript's final
        "Figure captions" list. The banded-OOD curve was previously
        emitted as fig2 (should be fig3); the pipeline diagram was
        previously emitted as fig5 (should be fig2); the A4/A5 heatmap
        was previously emitted as fig3 (should be fig5). Content,
        data, and rendering are unchanged — only the figure numbers,
        function names, output stems, and constant names were
        corrected to remove the body/caption-list mismatch.
  - v2: Figure 5 (now Figure 3) cell labels use per-cell precision
        matching Table 11 (canonical cell reads +19.72, not +19.7).
        Figure 4 legend uses mathtext so the superscript minus renders
        without missing-glyph boxes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# =========================================================================
# Style
# =========================================================================

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    # Mathtext: use the same sans-serif family so $...$ labels match the
    # surrounding text and superscript minus renders correctly.
    "mathtext.fontset":   "dejavusans",
    "mathtext.default":   "regular",
})

W_SINGLE = 90  / 25.4
W_15COL  = 140 / 25.4
W_DOUBLE = 190 / 25.4

C_BC   = "#4477AA"
C_CQL  = "#CC6677"
C_QR   = "#CC6677"
C_MEAN = "#4477AA"
C_GREY = "#888888"


def _save(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outdir / f"{stem}.pdf"))
    fig.savefig(str(outdir / f"{stem}.png"))
    plt.close(fig)
    print(f"  wrote {outdir / stem}.pdf and .png")


# =========================================================================
# Figure 1 — verdict landscape
# =========================================================================

FIG1_GROUPS = [
    ("Estimator\n(ε config)", [
        ("rollout",        +0.3402, +0.2433, +0.4407, True),
        ("FQE",            -0.0018, -0.0416, +0.0385, False),
    ]),
    ("Dataset config\n(rollout)", [
        ("no_oracle α=1",  +0.9504, +0.6976, +1.2181, True),
        ("no_oracle",      +0.5843, +0.3236, +0.8534, True),
        ("no_oracle α=10", +0.5144, +0.2676, +0.7783, True),
        ("ε",              +0.3402, +0.2433, +0.4407, True),
        ("base",           -1.1959, -1.6651, -0.7465, True),
    ]),
    ("State band\n(expanded programme)", [
        ("in-distribution", -16.0057, -17.80, -14.20, True),
        ("mild",            -23.6220, -25.00, -22.10, True),
        ("moderate",        -29.7458, -31.20, -28.30, True),
        ("severe OOD",      -38.3978, -40.10, -36.70, True),
    ]),
    ("Reward params\n(A4/A5 sweep)", [
        ("A4=10, A5=3.0",   +79.30, +75.80, +82.60, True),
        ("A4=10, A5=2.0",   +54.90, +51.60, +58.40, True),
        ("A4=15, A5=3.0",   +29.80, +27.30, +32.40, True),
        ("A4=10, A5=1.0",   +24.20, +21.90, +26.60, True),
        ("A4=15, A5=2.0",   +19.72, +17.25, +22.27, True),
        ("A4=15, A5=1.0",   +7.60,  +6.20,  +9.10,  True),
        ("A4=25, A5=3.0",   +7.00,  +5.80,  +8.20,  True),
        ("A4=25, A5=2.0",   +3.80,  +3.10,  +4.50,  True),
        ("A4=25, A5=1.0",   +0.35,  +0.21,  +0.49,  True),
    ]),
]


def fig1_verdict_landscape(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 4.4))

    rows, row_labels = [], []
    group_centres, group_labels = [], []
    y = 0

    for gname, entries in FIG1_GROUPS:
        y0 = y
        for label, m, lo, hi, sig in entries:
            ax.errorbar(
                m, y,
                xerr=[[m - lo], [hi - m]],
                fmt="o", markersize=4.5,
                markerfacecolor=(C_QR if sig else "white"),
                markeredgecolor="black", markeredgewidth=0.6,
                ecolor="black", elinewidth=0.6, capsize=2,
                zorder=3,
            )
            rows.append(y)
            row_labels.append(label)
            y += 1
        group_centres.append((y0 + y - 1) / 2)
        group_labels.append(gname)
        y += 1.2

    ax.axvline(0, color="black", linewidth=0.8, zorder=1)
    ax.axvspan(-50, -1, alpha=0.04, color="grey")
    ax.axvspan(-1,  1, alpha=0.06, color="grey")

    ax.set_yticks(rows)
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_ylim(y - 0.5, -1.5)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(group_centres)
    ax2.set_yticklabels(group_labels, fontsize=7, fontweight="bold")
    for s in ("right", "left", "top", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(-50, 100)
    ax.set_xticks([-50, -10, -1, 0, 1, 10, 50, 100])
    ax.set_xticklabels(["-50", "-10", "-1", "0", "1", "10", "50", "100"])
    ax.set_xlabel("CQL − BC (return units, symlog scale; linear within ±1)")

    ax.text(-45, -1.2, "sign flips\n(estimator, base, OOD)",
            fontsize=6.5, style="italic", va="top", color="#333333")
    ax.text(1.2, -1.2, "magnitude stretches\n(reward params)",
            fontsize=6.5, style="italic", va="top", color="#333333")

    fig.subplots_adjust(right=0.78)
    _save(fig, outdir, "fig1_verdict_landscape")


# =========================================================================
# Figure 2 — data flow / methods pipeline
# =========================================================================

def _box(ax, x, y, w, h, label, fc="white", ec="black", fontsize=6.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=0.6))
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=6,
        linewidth=0.5, color="black", linestyle=ls,
        shrinkA=0, shrinkB=0))


def fig2_pipeline(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # Row 1 — data sources
    _box(ax, 0.1, 3.80, 2.1, 0.55, "CGWB quarterly\nwell observations")
    _box(ax, 2.6, 3.80, 2.1, 0.55, "Training-well\nordinary kriging")
    _box(ax, 5.1, 3.80, 2.1, 0.55, "RF / HGBR\ndownscaling")
    _box(ax, 7.6, 3.80, 2.3, 0.55, "Downscaled surface\n(companion paper v1)")

    # Row 2 — RL environment
    _box(ax, 0.1, 2.70, 3.0, 0.60,
         "Behaviour policies\nrandom · greedy · ε-greedy", fc="#f5f5f5")
    _box(ax, 3.6, 2.70, 3.0, 0.60,
         "Synthetic simulator\n8×8 blocks · quarterly", fc="#f5f5f5")
    _box(ax, 7.1, 2.70, 2.8, 0.60,
         "30,000-transition\noffline dataset", fc="#f5f5f5")

    # Row 3 — training
    _box(ax, 1.4, 1.55, 2.4, 0.60, "Behavioural\ncloning (BC)", fc="#4477AA")
    _box(ax, 6.2, 1.55, 2.4, 0.60, "Conservative\nQ-learning (CQL)", fc="#CC6677")

    # Row 4 — evaluation
    _box(ax, 0.1, 0.40, 2.3, 0.60, "FQE\nseed bootstrap")
    _box(ax, 2.7, 0.40, 2.3, 0.60, "Direct rollout\nepisode bootstrap")
    _box(ax, 5.3, 0.40, 2.3, 0.60, "Banded OOD\n4 Mahalanobis bands")
    _box(ax, 7.9, 0.40, 2.0, 0.60, "§5.2\nFQE ≠ rollout", fc="#ffe8e8")

    # Row 1 internal arrows
    for x1, x2 in [(2.2, 2.6), (4.7, 5.1), (7.2, 7.6)]:
        _arrow(ax, x1, 4.075, x2, 4.075)

    # Row 1 → Row 2
    _arrow(ax, 1.6, 3.80, 1.6, 3.30)
    _arrow(ax, 1.6, 3.30, 3.6, 3.30)
    _arrow(ax, 6.6, 3.30, 7.1, 3.30)
    _arrow(ax, 8.55, 3.80, 8.55, 3.30)

    # Row 2 → Row 3
    _arrow(ax, 4.9, 2.70, 3.0, 2.15)
    _arrow(ax, 7.0, 2.70, 7.4, 2.15)
    _arrow(ax, 8.5, 2.70, 8.0, 2.15)

    # Row 3 → Row 4
    _arrow(ax, 2.8, 1.55, 1.25, 1.00)
    _arrow(ax, 3.4, 1.55, 3.85, 1.00)
    _arrow(ax, 7.4, 1.55, 6.45, 1.00)
    _arrow(ax, 7.4, 1.55, 8.50, 1.00, ls="--")

    # Row 4 cross-link
    _arrow(ax, 5.0, 0.70, 7.9, 0.70, ls="--")

    # Group labels on the left
    ax.text(0.05, 4.55, "Data",           fontsize=7, fontweight="bold")
    ax.text(0.05, 3.50, "RL environment", fontsize=7, fontweight="bold")
    ax.text(0.05, 2.35, "Training",       fontsize=7, fontweight="bold")
    ax.text(0.05, 1.20, "Evaluation",     fontsize=7, fontweight="bold")

    _save(fig, outdir, "fig2_pipeline")


# =========================================================================
# Figure 3 — banded OOD curve
# =========================================================================

FIG3_QR   = [-16.01, -23.62, -29.75, -38.40]
FIG3_MEAN = [ -8.64, -13.63, -15.76, -18.17]
FIG3_BANDS = ["in-\ndistrib.", "mild", "moderate", "severe\nOOD"]


def fig3_banded_ood(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.8))
    x = np.arange(4)

    ax.plot(x, FIG3_QR,   "-o", color=C_QR,   markersize=4.5, linewidth=0.9,
            label="QR-32 critic")
    ax.plot(x, FIG3_MEAN, "-s", color=C_MEAN, markersize=4.5, linewidth=0.9,
            label="Mean-Q critic")

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(FIG3_BANDS, fontsize=6.5)
    ax.set_xlabel("Mahalanobis band (near → far from training distribution)")
    ax.set_ylabel("CQL − BC (return)")
    ax.set_ylim(-42, 3)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6)

    ax.annotate("monotone widening",
                xy=(3, FIG3_QR[-1]), xytext=(1.55, -34),
                fontsize=6.5, style="italic",
                arrowprops=dict(arrowstyle="->", linewidth=0.5))

    ax.annotate("critic-independent sign",
                xy=(0, FIG3_QR[0]), xytext=(0.05, -3),
                fontsize=6.5, style="italic",
                arrowprops=dict(arrowstyle="->", linewidth=0.5))

    _save(fig, outdir, "fig3_banded_ood")


# =========================================================================
# Figure 4 — reward ablation at two rates
# =========================================================================

FIG4_CONFIGS = ["(1,0,0)", "(1,2,0)", "(0,2,0.5)", "(1,0,0.5)",
                "(1,1,0.5)", "(1,4,0.5)", "(1,2,0.5)"]

FIG4_CANONICAL = {
    "m":  [+0.073, +0.221, +0.579, +0.068, +0.099, +1.141, +0.207],
    "lo": [+0.059, +0.142, +0.417, +0.056, +0.058, +0.857, +0.121],
    "hi": [+0.089, +0.297, +0.767, +0.081, +0.140, +1.475, +0.289],
}

FIG4_DEFAULT = {
    "m":  [+0.069, +0.302, +0.886, +0.031, +0.062, +1.470, +0.345],
    "lo": [+0.051, +0.157, +0.694, +0.015, -0.010, +1.128, +0.185],
    "hi": [+0.087, +0.450, +1.088, +0.047, +0.133, +1.828, +0.507],
}


def _yerr(stat: dict):
    m = np.array(stat["m"]); lo = np.array(stat["lo"]); hi = np.array(stat["hi"])
    return np.vstack([m - lo, hi - m])


def fig4_reward_ablation_two_rates(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(W_15COL, 3.0))
    x = np.arange(len(FIG4_CONFIGS))
    w = 0.38

    # Mathtext labels: superscript minus renders via the dejavusans fontset,
    # which is configured in the style block above.
    ax.bar(x - w/2, FIG4_DEFAULT["m"], width=w, color=C_BC, alpha=0.55,
           edgecolor="black", linewidth=0.4,
           label=r"default LR  ($6.25\times10^{-5}$)")
    ax.errorbar(x - w/2, FIG4_DEFAULT["m"], yerr=_yerr(FIG4_DEFAULT),
                fmt="none", ecolor="black", elinewidth=0.5, capsize=2)

    ax.bar(x + w/2, FIG4_CANONICAL["m"], width=w, color=C_CQL,
           edgecolor="black", linewidth=0.4,
           label=r"canonical LR  ($3\times10^{-4}$)")
    ax.errorbar(x + w/2, FIG4_CANONICAL["m"], yerr=_yerr(FIG4_CANONICAL),
                fmt="none", ecolor="black", elinewidth=0.5, capsize=2)

    i = FIG4_CONFIGS.index("(1,1,0.5)")
    ax.annotate("crosses zero\nat default LR",
                xy=(i - w/2, 0.062),
                xytext=(i - 1.7, 1.35),
                fontsize=6.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", linewidth=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(FIG4_CONFIGS, rotation=25, ha="right", fontsize=6.5)
    ax.set_xlabel("(w1, w2, w3) reward weights")
    ax.set_ylabel("CQL − BC (return)")
    ax.set_ylim(-0.15, 2.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)

    _save(fig, outdir, "fig4_reward_ablation_two_rates")


# =========================================================================
# Figure 5 — A4/A5 heatmap
# =========================================================================

FIG5_A4_VALUES = [10.0, 15.0, 25.0]
FIG5_A5_VALUES = [1.0, 2.0, 3.0]
FIG5_CANONICAL_IJ = (1, 1)
FIG5_GRID = np.array([
    [24.2,  54.9,  79.3],
    [ 7.6,  19.72, 29.8],
    [ 0.35,  3.8,   7.0],
])

# Per-cell label strings, matching the manuscript's Table 11 precision.
# The canonical cell needs two decimals (+19.72); the rest match Table 11.
FIG5_CELL_LABELS = [
    ["+24.2",  "+54.9",  "+79.3"],
    [ "+7.6", "+19.72",  "+29.8"],
    [ "+0.35",  "+3.8",   "+7.0"],
]


def fig5_a4a5_heatmap(outdir: Path) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "a4a5", ["#f7f7f7", "#fee0d2", "#fc9272", "#de2d26", "#67000d"])

    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.9))
    im = ax.imshow(FIG5_GRID, cmap=cmap, aspect="auto", origin="upper")

    ax.set_xticks(range(len(FIG5_A5_VALUES)))
    ax.set_xticklabels([f"{v:.1f}" for v in FIG5_A5_VALUES])
    ax.set_yticks(range(len(FIG5_A4_VALUES)))
    ax.set_yticklabels([f"{v:.0f}" for v in FIG5_A4_VALUES])
    ax.set_xlabel("A5 — drawdown coefficient")
    ax.set_ylabel("A4 — sustainability threshold (m)")

    for i in range(FIG5_GRID.shape[0]):
        for j in range(FIG5_GRID.shape[1]):
            v = FIG5_GRID[i, j]
            txt = FIG5_CELL_LABELS[i][j]
            is_canonical = ((i, j) == FIG5_CANONICAL_IJ)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5,
                    fontweight="bold" if is_canonical else "normal",
                    color="white" if v > 45 else "black")

    ci, cj = FIG5_CANONICAL_IJ
    ax.add_patch(Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False,
                           edgecolor="black", linewidth=1.4))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CQL − BC (return)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.text(-0.55, -0.75, "A4 ↓ → penalty activates deeper → effect shrinks",
            fontsize=6.5, style="italic", ha="left")
    ax.text(-0.55, 3.35, "range +0.35 to +79.3  (226×)",
            fontsize=6.5, fontweight="bold", ha="left")

    _save(fig, outdir, "fig5_a4a5_heatmap")


# =========================================================================
# Dispatcher
# =========================================================================

FIGURES = {
    "fig1": ("fig1_verdict_landscape",          fig1_verdict_landscape),
    "fig2": ("fig2_pipeline",                   fig2_pipeline),
    "fig3": ("fig3_banded_ood",                 fig3_banded_ood),
    "fig4": ("fig4_reward_ablation_two_rates",  fig4_reward_ablation_two_rates),
    "fig5": ("fig5_a4a5_heatmap",               fig5_a4a5_heatmap),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate all Paper 2 figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "targets", nargs="*",
        help="figure keys to generate (default: all). "
             "Choices: " + ", ".join(FIGURES))
    parser.add_argument(
        "--outdir", type=Path, default=Path("figures"),
        help="output directory (default: ./figures)")
    parser.add_argument(
        "--list", action="store_true",
        help="list available figures and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available figure keys:")
        for key, (stem, _) in FIGURES.items():
            print(f"  {key}  ->  {stem}.pdf / {stem}.png")
        return 0

    if args.targets:
        unknown = [t for t in args.targets if t not in FIGURES]
        if unknown:
            print(f"Unknown figure(s): {unknown}", file=sys.stderr)
            print(f"Available: {list(FIGURES)}", file=sys.stderr)
            return 2
        selected = [(key, FIGURES[key]) for key in args.targets]
    else:
        selected = list(FIGURES.items())

    outdir = args.outdir
    print(f"Output directory: {outdir.resolve()}")
    print(f"Generating {len(selected)} figure(s)...")

    for key, (stem, fn) in selected:
        print(f"\n[{key}] {stem}")
        try:
            fn(outdir)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())