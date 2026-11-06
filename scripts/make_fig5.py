#!/usr/bin/env python3
"""
make_fig5.py — regenerate Fig. 5 (A4 x A5 sensitivity surface) with the
corrected "227x" range label (was 226x).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# --- data from Table XI -------------------------------------------------
A4_values = [10.0, 15.0, 25.0]      # rows (sustainability threshold, m)
A5_values = [1.0, 2.0, 3.0]         # cols (drawdown coefficient)

grid = np.array([
    [24.2, 54.9, 79.3],
    [7.6, 19.72, 29.8],
    [0.35, 3.8, 7.0],
])

# --- plot ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)

im = ax.imshow(grid, cmap="Reds", aspect="auto",
               vmin=grid.min(), vmax=grid.max())

# annotate each cell
for i in range(len(A4_values)):
    for j in range(len(A5_values)):
        v = grid[i, j]
        # two-decimal for canonical cell, one-decimal otherwise
        if i == 1 and j == 1:
            txt = f"+{v:.2f}"
            color = "black"
            weight = "bold"
        else:
            txt = f"+{v:.1f}" if v >= 1 else f"+{v:.2f}"
            color = "white" if v > 40 else "black"
            weight = "normal"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=12, color=color, fontweight=weight)

# box the canonical cell
ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False,
                       edgecolor="black", linewidth=2.5))

# ticks
ax.set_xticks(range(len(A5_values)))
ax.set_xticklabels([f"{v:.1f}" for v in A5_values])
ax.set_yticks(range(len(A4_values)))
ax.set_yticklabels([f"{v:.0f}" for v in A4_values])
ax.set_xlabel("A5 — drawdown coefficient", fontsize=12)
ax.set_ylabel("A4 — sustainability threshold (m)", fontsize=12)

# top annotation and range label — CORRECTED to 227x
ax.set_title("A4 ↓ → penalty activates deeper → effect shrinks",
             fontsize=11, pad=12)

# caption-within-plot for the range (this was the 226x label)
fig.text(0.12, 0.015, "range +0.35 to +79.3  (227×)",
         fontsize=11, fontweight="bold")

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("CQL − BC (return)", fontsize=11)

fig.tight_layout(rect=[0, 0.03, 1, 1])

# --- save ---------------------------------------------------------------
out_path = Path("figures/fig5.png")   # adjust if your figure dir differs
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"[fig5] wrote {out_path.resolve()}")