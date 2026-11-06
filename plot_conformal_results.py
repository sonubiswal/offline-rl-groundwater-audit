"""
Publication-quality multi-panel figures for the conformal calibration
experiment. Produces two figures, each with three panels:

  Fig 6: summary of calibration strategies (coverage, width, trade-off)
  Fig 7: diagnostics (coverage by distance, residual distribution, |resid| vs pred)
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ---------- Global styling ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "legend.fancybox": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "grid.linestyle": ":",
    # Embed TrueType (Type 42) fonts; many publishers reject Type 3
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------- Config ----------
RESULTS_DIR = Path("conformal_results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

NOMINAL = 0.90
SEED = 2026
N_BOOT = 2000
MIN_WELLS_FOR_CI = 5
DIST_ORDER = ["0-5km", "5-10km", "10-20km", "20km+"]  # must match CSV labels

# Original (OOF-calibrated) reference values. If you have them in a file,
# put them in conformal_results/oof_reference.json as
# {"coverage": 0.537, "halfwidth": 2.711} and they will be loaded instead.
OOF_COVERAGE = 0.537
OOF_HALFWIDTH = 2.711
_oof_path = RESULTS_DIR / "oof_reference.json"
if _oof_path.exists():
    with open(_oof_path) as f:
        _oof = json.load(f)
    OOF_COVERAGE = _oof["coverage"]
    OOF_HALFWIDTH = _oof["halfwidth"]
OOF_WIDTH = 2 * OOF_HALFWIDTH

# Muted journal palette
C_RED = "#B03A2E"     # invalid / OOF
C_BLUE = "#2E5C8A"    # corrected global
C_GREEN = "#3D7A47"   # distance-stratified
C_GREY = "#5D6D7E"    # neutral
C_GOLD = "#B7950B"    # annotations
C_LIGHT = "#EAECEE"   # background shading

# ---------- Load results ----------
summary = pd.read_csv(RESULTS_DIR / "conformal_summary.csv")
distance = pd.read_csv(RESULTS_DIR / "distance_coverage.csv")
test_pred = pd.read_csv(RESULTS_DIR / "test_predictions_with_intervals.csv")
with open(RESULTS_DIR / "distance_quantiles.json") as f:
    quantiles = json.load(f)


def summary_value(method, col):
    return summary.loc[summary["method"] == method, col].iloc[0]


c0_cov = summary_value("C0_global", "observation_coverage")
c1_cov = summary_value("C1_distance_stratified", "observation_coverage")
c0_width = summary_value("C0_global", "mean_width")
c1_width = summary_value("C1_distance_stratified", "mean_width")
q_global = quantiles["global_quantile"]


def bootstrap_coverage_ci(hits, well_ids, n_boot=N_BOOT, seed=SEED):
    """Well-clustered bootstrap 95% CI for coverage."""
    rng = np.random.default_rng(seed)
    well_ids = np.asarray(well_ids)
    wells = np.unique(well_ids)
    well_to_idx = {w: np.where(well_ids == w)[0] for w in wells}
    covs = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(wells, size=len(wells), replace=True)
        idx = np.concatenate([well_to_idx[w] for w in sample])
        covs[b] = hits[idx].mean()
    return np.percentile(covs, [2.5, 97.5])


def style_bar_axes(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


# ==========================================================================
# FIGURE 6 --- Summary of calibration strategies
# ==========================================================================
fig = plt.figure(figsize=(13.5, 4.4))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                       left=0.06, right=0.985, top=0.86, bottom=0.17)

methods = ["OOF-calibrated\n(original)", "Global C0\n(corrected)",
           "Distance C1\n(corrected)"]
colors = [C_RED, C_BLUE, C_GREEN]

# ---- Panel A: coverage bars ----
axA = fig.add_subplot(gs[0, 0])
covs_pct = np.array([OOF_COVERAGE, c0_cov, c1_cov]) * 100

bars = axA.bar(methods, covs_pct, color=colors, edgecolor="black",
               linewidth=0.9, width=0.62, zorder=3)
axA.axhspan(85, 95, color=C_GREEN, alpha=0.10, zorder=1)
axA.axhline(NOMINAL * 100, color="black", linestyle="--", linewidth=1.3,
            zorder=4)

for bar, cov, col in zip(bars, covs_pct, colors):
    axA.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{cov:.1f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=col)

# Delta arrow OOF -> C0, lifted clear of the value labels
arrow_y0 = covs_pct[0] + 10
arrow_y1 = covs_pct[1] + 10
axA.annotate("", xy=(1, arrow_y1), xytext=(0, arrow_y0),
             arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.4,
                             connectionstyle="arc3,rad=-0.25"))
axA.text(0.5, 0.5 * (arrow_y0 + arrow_y1) - 22,
         f"+{covs_pct[1] - covs_pct[0]:.1f} pp",
         fontsize=9, color=C_GOLD, fontweight="bold", ha="center",
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_GOLD, lw=0.8))

axA.set_ylabel("Empirical coverage (%)")
axA.set_ylim(0, 112)
axA.set_title("(a) Coverage", pad=8)
axA.legend(handles=[
    plt.Line2D([0], [0], color="black", ls="--", lw=1.3,
               label=f"Nominal {NOMINAL:.0%}"),
    Patch(facecolor=C_GREEN, alpha=0.25, label="±5 pp target band"),
], loc="upper left", fontsize=8.5)
style_bar_axes(axA)

# ---- Panel B: interval width bars ----
axB = fig.add_subplot(gs[0, 1])
widths = np.array([OOF_WIDTH, c0_width, c1_width])

bars = axB.bar(methods, widths, color=colors, edgecolor="black",
               linewidth=0.9, width=0.62, zorder=3)
for bar, w, col in zip(bars, widths, colors):
    axB.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.28,
             f"{w:.2f} m", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=col)

ratio = c0_width / OOF_WIDTH
axB.annotate("", xy=(1, c0_width + 1.4), xytext=(0, OOF_WIDTH + 1.4),
             arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.4,
                             connectionstyle="arc3,rad=-0.25"))
axB.text(0.5, 0.5 * (c0_width + OOF_WIDTH) + 4.5, f"{ratio:.2f}× wider",
         fontsize=9, color=C_GOLD, fontweight="bold", ha="center",
         bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C_GOLD, lw=0.8))

axB.set_ylabel("Mean interval width (m)")
axB.set_ylim(0, max(widths) * 1.35)
axB.set_title("(b) Interval width", pad=8)
style_bar_axes(axB)

# ---- Panel C: coverage-width trade-off ----
axC = fig.add_subplot(gs[0, 2])
axC.axhline(NOMINAL * 100, color="black", linestyle="--", linewidth=1.2,
            zorder=2)
# Shade the region narrower than the (invalid) OOF interval
axC.axvspan(0, OOF_WIDTH, color=C_LIGHT, alpha=0.55, zorder=0)

# (x, y, colour, label, label offset in points)
pts = [
    (OOF_WIDTH, OOF_COVERAGE * 100, C_RED, "OOF\n(invalid)", (10, -4)),
    (c0_width, c0_cov * 100, C_BLUE, "C0\n(corrected)", (10, 8)),
    (c1_width, c1_cov * 100, C_GREEN, "C1\n(corrected)", (10, -22)),
]
for x, y, c, lab, off in pts:
    axC.scatter(x, y, s=340, color=c, edgecolor="black", linewidth=1.1,
                zorder=4)
    axC.annotate(lab, xy=(x, y), xytext=off, textcoords="offset points",
                 fontsize=8.5, fontweight="bold", color=c)

axC.text(16.8, NOMINAL * 100 + 1.2, f"Nominal {NOMINAL:.0%}", fontsize=8,
         style="italic", ha="right", va="bottom")
axC.text(OOF_WIDTH / 2, 96.5, "IDEAL:\nnarrow at ~90%",
         fontsize=8, style="italic", color=C_GREEN, ha="center",
         va="center", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_GREEN,
                   lw=0.9, alpha=0.9))

axC.set_xlabel("Mean interval width (m)")
axC.set_ylabel("Empirical coverage (%)")
axC.set_title("(c) Coverage–width trade-off", pad=8)
axC.set_xlim(0, 17)
axC.set_ylim(45, 102)
axC.grid(True, zorder=1)
axC.set_axisbelow(True)

fig.savefig(FIG_DIR / "fig6_conformal_summary.pdf")
fig.savefig(FIG_DIR / "fig6_conformal_summary.png")
plt.close(fig)
print("Saved fig6_conformal_summary.{pdf,png}")

# ==========================================================================
# FIGURE 7 --- Diagnostics
# ==========================================================================
fig = plt.figure(figsize=(13.5, 4.4))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                       left=0.06, right=0.985, top=0.86, bottom=0.17)

# ---- Panel A: coverage by distance stratum ----
axA = fig.add_subplot(gs[0, 0])

distance = distance.assign(distance_bin=distance["distance_bin"].astype(str))
test_pred["distance_bin"] = test_pred["distance_bin"].astype(str)

missing = [b for b in DIST_ORDER if b not in set(distance["distance_bin"])]
if missing:
    raise ValueError(
        f"Distance bins {missing} not found in distance_coverage.csv. "
        f"Available: {sorted(set(distance['distance_bin']))}"
    )
dist = distance.set_index("distance_bin").loc[DIST_ORDER]

covs_ord = dist["coverage"].values * 100
ns_ord = dist["n_test"].values
qs_ord = dist["quantile"].values

lo_err, hi_err, hi_edge = [], [], []
for b, cov in zip(DIST_ORDER, covs_ord):
    sub = test_pred[test_pred["distance_bin"] == b]
    hits = sub["C1_covered"].values.astype(float)
    if sub["well_id"].nunique() >= MIN_WELLS_FOR_CI:
        lo, hi = bootstrap_coverage_ci(hits, sub["well_id"].values)
        lo, hi = lo * 100, hi * 100
        lo_err.append(max(0.0, cov - lo))   # clip: point estimate can sit
        hi_err.append(max(0.0, hi - cov))   # outside its own percentile CI
        hi_edge.append(max(cov, hi))
    else:
        lo_err.append(0.0)
        hi_err.append(0.0)
        hi_edge.append(cov)

x = np.arange(len(DIST_ORDER))
axA.bar(x, covs_ord, color=C_GREEN, edgecolor="black", linewidth=0.9,
        width=0.62, zorder=3, alpha=0.85)
axA.errorbar(x, covs_ord, yerr=[lo_err, hi_err], fmt="none", ecolor="black",
             capsize=4, capthick=1.1, lw=1.0, zorder=4)
axA.axhspan(85, 95, color=C_GREEN, alpha=0.10, zorder=1)
axA.axhline(NOMINAL * 100, color="black", linestyle="--", linewidth=1.2,
            zorder=4)

for xi, (cov, top, n, q) in enumerate(zip(covs_ord, hi_edge, ns_ord, qs_ord)):
    # place the value label above the error bar, not on it
    axA.text(xi, top + 1.5, f"{cov:.1f}%", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color="#1E5631")
    axA.text(xi, 3.5, f"n={n}\nq={q:.2f} m", ha="center", va="bottom",
             fontsize=7.5, color="white", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", fc=C_GREEN, alpha=0.85,
                       ec="none"))

axA.set_xticks(x)
axA.set_xticklabels(DIST_ORDER)
axA.set_ylabel("Empirical coverage (%)")
axA.set_xlabel("Distance to nearest training well")
axA.set_ylim(0, 112)
axA.set_title("(a) Coverage by distance stratum (C1)", pad=8)
style_bar_axes(axA)

# ---- Panel B: residual distribution vs calibration half-widths ----
resid_all = (test_pred["true_y"] - test_pred["pred_y"]).abs().values
resid = resid_all[np.isfinite(resid_all)]
x_max = np.percentile(resid, 99.5)

axB = fig.add_subplot(gs[0, 1])
axB.hist(resid, bins=np.linspace(0, x_max, 60), color=C_GREY, alpha=0.55,
         edgecolor="white", linewidth=0.3, zorder=2,
         label=f"Held-out |residual| (n={len(resid):,})")
axB.axvline(OOF_HALFWIDTH, color=C_RED, linewidth=2.0, zorder=4,
            label=f"OOF half-width = {OOF_HALFWIDTH:.2f} m")
axB.axvline(q_global, color=C_BLUE, linewidth=2.0, zorder=4,
            label=f"Corrected (global) = {q_global:.2f} m")
axB.axvspan(OOF_HALFWIDTH, x_max, color=C_RED, alpha=0.08, zorder=1)

overshoot = (resid > OOF_HALFWIDTH).mean()
axB.text(0.78, 0.50, f"{overshoot * 100:.0f}% of\nresiduals\nexceed OOF",
         transform=axB.transAxes, fontsize=9, color=C_RED,
         fontweight="bold", ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_RED, lw=0.9))

axB.set_xlim(0, x_max)
axB.set_xlabel("Absolute held-out residual (m)")
axB.set_ylabel("Frequency")
axB.set_title("(b) Residual distribution vs calibration widths", pad=8)
axB.legend(loc="upper right", fontsize=8)
axB.grid(axis="y", zorder=0)
axB.set_axisbelow(True)

# ---- Panel C: |residual| vs prediction ----
axC = fig.add_subplot(gs[0, 2])
pred = test_pred["pred_y"].values
mask = np.isfinite(pred) & np.isfinite(resid_all)
pred_m, resid_m = pred[mask], resid_all[mask]

axC.scatter(pred_m, resid_m, s=7, alpha=0.30, color=C_GREY, edgecolor="none",
            label="Held-out reading", zorder=2)
axC.axhline(OOF_HALFWIDTH, color=C_RED, linewidth=2.0, zorder=4,
            label=f"OOF half-width = {OOF_HALFWIDTH:.2f} m")
axC.axhline(q_global, color=C_BLUE, linewidth=2.0, zorder=4,
            label=f"Corrected global (C0) = {q_global:.2f} m")

exc_oof = (resid_m > OOF_HALFWIDTH).mean()
exc_c0 = (resid_m > q_global).mean()
exc_c1 = 1.0 - test_pred["C1_covered"].astype(float).mean()
target = (1 - NOMINAL) * 100
axC.text(0.98, 0.04,
         "Exceedance (target {:.0f}%):\n"
         "OOF:            {:5.1f}%\n"
         "Global C0:      {:5.1f}%\n"
         "Distance C1:    {:5.1f}%".format(target, exc_oof * 100,
                                           exc_c0 * 100, exc_c1 * 100),
         transform=axC.transAxes, fontsize=8, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.9,
                   alpha=0.95),
         family="monospace")

axC.set_xlabel("Predicted depth (m)")
axC.set_ylabel("|Residual| (m)")
axC.set_title("(c) |Residual| vs prediction", pad=8)
axC.set_ylim(0, np.percentile(resid_m, 99.7))
axC.legend(loc="upper left", fontsize=8)
axC.grid(True, zorder=1)
axC.set_axisbelow(True)

fig.savefig(FIG_DIR / "fig7_conformal_diagnostics.pdf")
fig.savefig(FIG_DIR / "fig7_conformal_diagnostics.png")
plt.close(fig)
print("Saved fig7_conformal_diagnostics.{pdf,png}")

print("\nAll figures written to:", FIG_DIR.resolve())