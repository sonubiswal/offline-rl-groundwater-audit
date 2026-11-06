#!/usr/bin/env python3
"""Fig 8: RNG noise study — rollout gap under twelve independent seed bases
in the repaired scheme, with the colliding value and the reported value
marked for comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(r"C:\trishna-opal")
REV = REPO / "revision_experiments"
FIG = REPO / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.bbox": "tight",
})

d = json.loads((REV / "rng_noise_study.json").read_text())
gaps = np.array(d["gaps"])
mean = d["mean"]
sd = d["sd"]
colliding = d["colliding_value"]     # -0.0895
reported = d["reported_value"]       # +0.0141

fig, ax = plt.subplots(figsize=(4.2, 2.8))

x = np.arange(1, len(gaps) + 1)
ax.scatter(x, gaps, s=40, color="#1F5A96", zorder=3,
           label="12 seed bases (repaired scheme)")
ax.axhline(mean, color="#1F5A96", lw=1.2, ls="--", alpha=0.7,
           label=f"mean = {mean:+.3f}")
ax.fill_between([0.5, len(gaps) + 0.5], mean - sd, mean + sd,
                color="#1F5A96", alpha=0.10,
                label=f"±1 SD = {sd:.3f}")

ax.axhline(colliding, color="#C1272D", lw=1.5, zorder=2,
           label=f"colliding = {colliding:+.4f}")
ax.axhline(reported, color="#2E8B57", lw=1.5, zorder=2,
           label=f"reported = {reported:+.4f}")

ax.axhline(0, color="black", lw=0.6, alpha=0.5)

ax.set_xlim(0.5, len(gaps) + 0.5)
ax.set_xlabel("Independent seed-base replicate")
ax.set_ylabel("Rollout gap (ten-seed mean)")
ax.set_title("RNG repair: collision value lies within seed-base noise")
ax.legend(frameon=False, loc="lower right", ncol=1, fontsize=7)

fig.savefig(FIG / "fig8_rng_noise.pdf")
print(f"wrote {FIG / 'fig8_rng_noise.pdf'}")
print(f"  mean {mean:+.4f}  SD {sd:.4f}  range [{gaps.min():+.4f}, {gaps.max():+.4f}]")
print(f"  colliding {colliding:+.4f} inside range: {gaps.min() <= colliding <= gaps.max()}")