"""
check_phase1_data.py

Diagnostic script for closing out Phase 1's remaining Definition-of-Done
items:
  1. Reports which monthly composites gee_fetch.py actually pulled per
     source, and which are missing (silent failures during the fetch loop
     get logged but don't stop the run, so this is the way to catch them
     after the fact).
  2. Produces a visual spot-check: plots one aligned layer per source
     (from data/interim/, i.e. post align_grid.py) side by side, so you
     can eyeball whether they're correctly co-registered over the same
     study region.
  3. Reports unique well counts from the held-out split, to fill in
     Section 2 of reports/validation_methodology.md.

Run from the repo root:
    python src/data/check_phase1_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def report_gee_pull_completeness(config_path: str, raw_dir: str) -> None:
    """Compare expected months (from config date_range) against what's
    actually cached in data/raw/<source>/*.tif, per source."""
    config = load_config(config_path)
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    expected_months = pd.period_range(start=start, end=end, freq="M")
    expected_month_strs = {p.start_time.strftime("%Y-%m-%d") for p in expected_months}

    raw_root = Path(raw_dir)
    if not raw_root.exists():
        print(f"[check] {raw_root} does not exist -- gee_fetch.py hasn't been run yet.")
        return

    print("=" * 70)
    print("GEE PULL COMPLETENESS")
    print("=" * 70)
    print(f"Expected months: {len(expected_month_strs)} ({start} to {end})\n")

    for source_dir in sorted(raw_root.iterdir()):
        if not source_dir.is_dir():
            continue
        found_files = {f.stem for f in source_dir.glob("*.tif")}
        missing = expected_month_strs - found_files
        pct = 100 * len(found_files) / len(expected_month_strs) if expected_month_strs else 0

        status = "OK" if not missing else "INCOMPLETE"
        print(f"[{status}] {source_dir.name}: {len(found_files)}/{len(expected_month_strs)} months ({pct:.0f}%)")
        if missing:
            missing_sorted = sorted(missing)
            preview = missing_sorted[:5]
            more = f" (+{len(missing_sorted) - 5} more)" if len(missing_sorted) > 5 else ""
            print(f"         Missing: {preview}{more}")
    print()


def report_well_counts(held_out_csv: str, train_csv: str) -> None:
    print("=" * 70)
    print("HELD-OUT SPLIT WELL COUNTS")
    print("=" * 70)

    held_path = Path(held_out_csv)
    train_path = Path(train_csv)

    if not held_path.exists() or not train_path.exists():
        print("[check] Split files not found -- run held_out_split.py first.")
        return

    held_df = pd.read_csv(held_path)
    train_df = pd.read_csv(train_path)

    held_unique = held_df["well_id"].nunique()
    train_unique = train_df["well_id"].nunique()
    total_unique = held_unique + train_unique

    print(f"Train:     {len(train_df):,} readings across {train_unique:,} unique wells")
    print(f"Held-out:  {len(held_df):,} readings across {held_unique:,} unique wells")
    print(f"Total:     {len(train_df) + len(held_df):,} readings across {total_unique:,} unique wells")
    print(f"Held-out fraction (by unique well): {held_unique / total_unique:.1%}")
    print()
    print(">>> Copy these numbers into reports/validation_methodology.md, Section 2.")
    print()


def plot_alignment_spotcheck(
    interim_dir: str,
    out_path: str,
    sources: list[str] | None = None,
) -> None:
    """Plot one representative aligned raster per source, side by side.

    Picks the first available month found for each source directory under
    data/interim/. Saves a PNG grid for visual inspection -- confirm all
    panels show data over the same spatial extent with no obvious
    misregistration (e.g. features shifted, flipped, or cropped
    differently between sources).
    """
    interim_root = Path(interim_dir)
    if not interim_root.exists():
        print(f"[check] {interim_root} does not exist -- align_grid.py hasn't been run yet.")
        return

    source_dirs = sorted([d for d in interim_root.iterdir() if d.is_dir()])
    if sources:
        source_dirs = [d for d in source_dirs if d.name in sources]

    if not source_dirs:
        print(f"[check] No aligned source directories found under {interim_root}.")
        return

    n = len(source_dirs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    plotted = 0
    for i, source_dir in enumerate(source_dirs):
        tif_files = sorted(source_dir.glob("*.tif"))
        ax = axes[i // ncols][i % ncols]
        if not tif_files:
            ax.set_title(f"{source_dir.name} (no files)")
            ax.axis("off")
            continue

        sample_path = tif_files[0]
        with rasterio.open(sample_path) as f:
            arr = f.read(1)
            bounds = f.bounds
            nodata = f.nodata

        # Mask nodata explicitly so it renders as blank/transparent, not a
        # misleading in-range color (this is what caught the earlier bug:
        # unmasked nodata pixels rendered as solid blocks at the color
        # scale's extreme, indistinguishable from real high/low values).
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr) if not np.isnan(nodata) else arr
        valid_frac = np.mean(~np.isnan(arr)) if np.isnan(arr).any() or nodata is not None else 1.0

        im = ax.imshow(arr, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                        cmap="viridis", origin="upper")
        ax.set_title(f"{source_dir.name}\n{sample_path.stem} ({valid_frac:.0%} coverage)", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plotted += 1

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Phase 1 Grid Alignment Spot-Check — one aligned layer per source", fontsize=13)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=" * 70)
    print("GRID ALIGNMENT SPOT-CHECK")
    print("=" * 70)
    print(f"Plotted {plotted}/{n} sources.")
    print(f"Saved -> {out_path}")
    print("Inspect this image: all panels should cover the same lat/lon extent,")
    print("with no obvious shifting, flipping, or cropping differences between sources.")
    print()


def run_all(
    config_path: str = "config/data_config.yaml",
    raw_dir: str = "data/raw",
    interim_dir: str = "data/interim",
    held_out_csv: str = "data/held_out_wells/held_out_ids.csv",
    train_csv: str = "data/interim/train_well_ids.csv",
    plot_out: str = "reports/grid_alignment_spotcheck.png",
) -> None:
    report_gee_pull_completeness(config_path, raw_dir)
    report_well_counts(held_out_csv, train_csv)
    plot_alignment_spotcheck(interim_dir, plot_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 diagnostic: GEE pull completeness, well counts, alignment spot-check.")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    parser.add_argument("--train_csv", default="data/interim/train_well_ids.csv")
    parser.add_argument("--plot_out", default="reports/grid_alignment_spotcheck.png")
    args = parser.parse_args()

    run_all(
        config_path=args.config,
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        held_out_csv=args.held_out_csv,
        train_csv=args.train_csv,
        plot_out=args.plot_out,
    )