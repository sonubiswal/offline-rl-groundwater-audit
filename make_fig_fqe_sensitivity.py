#!/usr/bin/env python3
"""Fig 2: FQE target-update interval sensitivity. Per-seed paired contrasts
at tau = 100, 200, 500 with seed-t intervals."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path(r"C:\trishna-opal")
REV = REPO / "revision_experiments"
FIG = REPO / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


# Build per-seed FQE gaps at each interval
fqe = {}
fqe[200] = np.array(json.loads(
    (REV / "E2" / "e2_fqe.json").read_text()
)["primary_episode_starts"]["per_seed_diffs"])
for tui in (100, 500):
    out = []
    for s in SEEDS:
        z = np.load(REV / "E2" / "seeds" / f"tui{tui}" / f"seed_{s}.npz")
        out.append(float((z["cql_q_starts"] - z["bc_q_starts"]).mean()))
    fqe[tui] = np.array(out)

roll = np.array([
    r["mean_gap"]
    for r in json.loads(
        (REV / "e3c_test225_constructions_rerun.json").read_text()
    )["per_seed"]
])

# Paired contrast per interval
contrast = {tui: fqe[tui] - roll for tui in (100, 200, 500)}

# Plot
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(4.2, 3.0))
intervals = [100, 200, 500]
positions = np.arange(len(intervals))

# Per-seed grey points
for k, tui in enumerate(intervals):
    jitter = np.random.default_rng(2026).uniform(-0.06, 0.06, 10)
    ax.scatter(np.full(10, k) + jitter, contrast[tui],
               s=22, color="#888888", zorder=2, alpha=0.7)

# Mean + t-interval
for k, tui in enumerate(intervals):
    m, lo, hi = t_ci(contrast[tui])
    color = "#C1272D" if tui == 200 else "#1F5A96"
    ax.errorbar(k, m, yerr=[[m - lo], [hi - m]], fmt="D",
                color=color, capsize=4, markersize=8, zorder=4)

ax.axhline(0, color="black", lw=0.7, ls="--")
ax.set_xticks(positions)
ax.set_xticklabels([f"{t}\n({'recorded' if t == 200 else 'library default' if t == 100 else 'sensitivity'})"
                    for t in intervals])
ax.set_xlabel("FQE target-update interval")
ax.set_ylabel("Paired contrast (FQE $-$ rollout)")
ax.set_title("FQE sensitivity to the target-update interval")

fig.savefig(FIG / "fig_fqe_sensitivity.pdf")
print(f"wrote {FIG / 'fig_fqe_sensitivity.pdf'}")
for tui in intervals:
    m, lo, hi = t_ci(contrast[tui])
    print(f"  tau={tui:>3}: contrast {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")