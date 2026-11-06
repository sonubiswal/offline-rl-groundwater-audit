#!/usr/bin/env python3
"""Fig 1: matched-start estimator audit workflow.

Publication-quality four-stage schematic. Pure plain text. Short titles
that fit inside the box width at the chosen font size.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
OUT = Path(r"C:\trishna-opal\paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------
NAVY   = "#1F3B5A"
BLUE   = "#2D6FA3"
LIGHT  = "#EAF2F8"
BORDER = "#B8C4CF"
ARROW  = "#55758F"
DARK   = "#2F3336"
MID    = "#6A737B"
GREEN  = "#2E7D5B"
LGREEN = "#EAF4EE"
DIV    = "#DCE3E9"

# ---------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 3.10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.2)
ax.axis("off")

# Titles
ax.text(7.0, 4.02, "Matched-start estimator audit",
        ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=NAVY)
ax.text(7.0, 3.74,
        "A controlled comparison of fitted evaluation and direct simulator rollout",
        ha="center", va="center",
        fontsize=8.5, color=MID)

# ---------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------
stages = [
    dict(x=0.20, num="01",
         title="Fix inputs",
         body=["same trained policies",
               "same 225 test starts",
               "same reward and horizon",
               "same undiscounted target"],
         footer="Controlled setup"),
    dict(x=3.60, num="02",
         title="Evaluate estimators",
         body=["FQE value estimate",
               "direct rollout return",
               "identical policy pairs",
               "collision-free RNG streams"],
         footer="Estimator outputs"),
    dict(x=7.00, num="03",
         title="Pair by seed",
         body=["same seed both policies",
               "compute per-seed contrast",
               "retain paired observations",
               "preserve seed dependence"],
         footer="Paired contrast"),
    dict(x=10.40, num="04",
         title="Quantify uncertainty",
         body=["seed-level t interval",
               "paired bootstrap",
               "permutation sensitivity",
               "leave-one-out diagnostics"],
         footer="Robust inference"),
]

BOX_W = 3.40
BOX_H = 2.32
BOX_Y = 1.05

for s in stages:
    x = s["x"]

    # Card
    card = FancyBboxPatch(
        (x, BOX_Y), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=BORDER, facecolor="white", zorder=2)
    ax.add_patch(card)

    # Number badge
    badge = FancyBboxPatch(
        (x + 0.14, BOX_Y + BOX_H - 0.42), 0.44, 0.26,
        boxstyle="round,pad=0.006,rounding_size=0.04",
        linewidth=0, facecolor=BLUE, zorder=3)
    ax.add_patch(badge)
    ax.text(x + 0.36, BOX_Y + BOX_H - 0.29, s["num"],
            ha="center", va="center",
            fontsize=7.6, fontweight="bold", color="white", zorder=4)

    # Stage title (font 8.4, starts at x+0.68)
    ax.text(x + 0.68, BOX_Y + BOX_H - 0.29, s["title"],
            ha="left", va="center",
            fontsize=8.4, fontweight="bold", color=NAVY, zorder=4)

    # Divider under title
    ax.plot([x + 0.14, x + BOX_W - 0.14],
            [BOX_Y + BOX_H - 0.58, BOX_Y + BOX_H - 0.58],
            color=DIV, lw=0.7, zorder=3)

    # Body bullets
    body_text = "\n".join(f"- {item}" for item in s["body"])
    ax.text(x + 0.22, BOX_Y + BOX_H - 0.75, body_text,
            ha="left", va="top",
            fontsize=7.9, color=DARK, linespacing=1.55, zorder=4)

    # Footer strip
    footer = FancyBboxPatch(
        (x + 0.14, BOX_Y + 0.13), BOX_W - 0.28, 0.32,
        boxstyle="round,pad=0.006,rounding_size=0.03",
        linewidth=0, facecolor=LIGHT, zorder=3)
    ax.add_patch(footer)
    ax.text(x + BOX_W / 2, BOX_Y + 0.29, s["footer"],
            ha="center", va="center",
            fontsize=7.4, fontweight="bold", color=BLUE, zorder=4)

# ---------------------------------------------------------------------
# Arrows between stages
# ---------------------------------------------------------------------
for x1, x2 in [(3.64, 3.96), (7.04, 7.36), (10.44, 10.76)]:
    arrow = FancyArrowPatch(
        (x1, BOX_Y + BOX_H / 2),
        (x2, BOX_Y + BOX_H / 2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color=ARROW, zorder=1)
    ax.add_patch(arrow)

# ---------------------------------------------------------------------
# Bottom interpretation panel
# ---------------------------------------------------------------------
PX, PY, PW, PH = 0.20, 0.20, 13.60, 0.62

panel = FancyBboxPatch(
    (PX, PY), PW, PH,
    boxstyle="round,pad=0.015,rounding_size=0.06",
    linewidth=0.8, edgecolor="#B9D2C2", facecolor=LGREEN, zorder=2)
ax.add_patch(panel)

# Label on the left
ax.text(PX + 0.22, PY + PH / 2, "AUDIT OUTPUT",
        ha="left", va="center",
        fontsize=7.2, fontweight="bold", color=GREEN)

# Separator moved right to make room
ax.plot([PX + 1.75, PX + 1.75],
        [PY + 0.11, PY + PH - 0.11],
        color="#B9D2C2", lw=0.8, zorder=3)

# Description starts after the separator
ax.text(PX + 1.95, PY + PH / 2,
        "Compare estimator gaps, quantify the paired contrast, "
        "and test the robustness of the conclusion.",
        ha="left", va="center",
        fontsize=7.9, color=DARK)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
pdf_path = OUT / "fig01_matched_start_audit.pdf"
png_path = OUT / "fig01_matched_start_audit.png"

fig.savefig(pdf_path)
fig.savefig(png_path, dpi=400)
plt.close(fig)

print(f"Wrote {pdf_path}")
print(f"Wrote {png_path}")