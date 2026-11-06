#!/usr/bin/env python3
"""Copy the nine figures the manuscript cites into a single folder
`paper_figures/`, renaming them to match the paper's figure numbering."""
from __future__ import annotations
import shutil
from pathlib import Path

REPO = Path(r"C:\trishna-opal")
OUT = REPO / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# (paper slot, source path, new name)
MAPPING = [
    ("Fig. 1", OUT / "fig01_recipe.pdf",
               "fig01_recipe.pdf"),
    ("Fig. 2", REPO / "figs" / "fig1_estimators.pdf",
               "fig02_per_seed_estimator_bars.pdf"),
    ("Fig. 3", OUT / "fig03_paired_seed_contrast.pdf",
               "fig03_paired_seed_contrast.pdf"),
    ("Fig. 4", REPO / "figures" / "fig2_interval_sensitivity.pdf",
               "fig04_fqe_interval_sensitivity.pdf"),
    ("Fig. 5", OUT / "fig05_offset_bars.pdf",
               "fig05_offset_bars.pdf"),
    ("Fig. 6", REPO / "figs" / "fig2_pairings.pdf",
               "fig06_family_start_forest.pdf"),
    ("Fig. 7", REPO / "figs" / "fig3_bc_sweep.pdf",
               "fig07_bc_sweep.pdf"),
    ("Fig. 8", REPO / "figs" / "fig4_bands.pdf",
               "fig08_band_gaps.pdf"),
    ("Fig. 9", REPO / "figures" / "fig8_rng_noise.pdf",
               "fig09_rng_noise.pdf"),
]

report = []
for slot, src, name in MAPPING:
    if not src.exists():
        report.append((slot, str(src), "MISSING"))
        continue
    dst = OUT / name
    if src.resolve() == dst.resolve():
        report.append((slot, str(dst), "already in folder"))
        continue
    shutil.copy2(src, dst)
    report.append((slot, str(dst), "copied"))

print(f"{'slot':>6}  {'status':>18}  path")
print("-" * 70)
for slot, path, status in report:
    print(f"{slot:>6}  {status:>18}  {path}")

print(f"\nfinal folder: {OUT}")
print(f"files in folder: {len(list(OUT.glob('*.pdf')))}")