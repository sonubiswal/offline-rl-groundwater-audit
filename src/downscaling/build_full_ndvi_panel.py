#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_full_ndvi_panel.py
=========================
Builds the genuine pixel x date panel for the missing-NDVI recovery
experiment, per the review: constructed from the ALIGNED RASTER SOURCE
(data/interim/<var>/*.tif), not by expanding feature_table.csv - the table
has already been filtered to NDVI-present rows and cannot recover what was
dropped.

Grid: 64x64, EPSG:4326, bounds (72.6, 15.6) - (80.9, 22.0), confirmed
identical across chirps/gldas/grace/srtm/sentinel2_ndvi (same shape,
transform, bounds - checked directly against each raster's own georeferencing
rather than assumed).

Expected dates: read directly from feature_table.csv's own unique `date`
values (23 dates, 2015-08-01 .. 2021-01-01) - the actual campaign months the
groundwater feature pipeline uses - rather than the raw kriged_target_monthly
folder, which contains 38 files including 8 pre-2015 campaigns that predate
the CHIRPS/GLDAS/GRACE archive (2015-01 onward) and were already correctly
excluded upstream.

For each of the 23 dates x 4096 pixels:
    chirps, gldas_sm, grace   -> read from that date's raster; NaN if the
                                 raster file for that date doesn't exist at
                                 all, or if the pixel is NaN inside it
    elevation, slope          -> read once from the static SRTM rasters
    sentinel2_ndvi            -> read from that date's raster; NaN pixels
                                 preserved AS NaN (this is the whole point -
                                 do not drop them)
    month_sin / month_cos     -> derived from the date

sentinel1_vv is intentionally NOT included in this build (see chat) - it
exists in data/interim/sentinel1_vv/ but is left out here to avoid changing
the panel construction and the covariate set in the same step.

This script does not touch the 710 held-out wells, groundwater targets, or
the V3 model/experiment selection. It only builds the panel and prints the
coverage summary.

Usage:
    python build_full_ndvi_panel.py
    (edit ALIGNED_DIR / FEATURE_TABLE_CSV / OUTPUT_CSV below if paths differ)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import rasterio

# --------------------------------------------------------------------------
# Paths - edit if your layout differs
# --------------------------------------------------------------------------
ALIGNED_DIR = os.environ.get("NDVI_ALIGNED_DIR", "data/interim")
FEATURE_TABLE_CSV = os.environ.get("NDVI_FEATURE_TABLE_CSV", "data/feature_table.csv")
OUTPUT_CSV = os.environ.get("NDVI_FULL_PANEL_CSV",
                            "data/processed/full_panel_with_gaps.csv")

GRID_SHAPE = (64, 64)  # (n_rows, n_cols)

# Raster variable -> output column name (matches feature_table.csv's existing
# names so this panel is a drop-in replacement for the V3 config)
MONTHLY_VARS = {
    "chirps": "chirps",
    "gldas": "gldas_sm",
    "grace": "grace",
}
TARGET_VAR_DIR = "sentinel2_ndvi"
TARGET_COL = "sentinel2_ndvi"
STATIC_VARS = {
    "elevation": ("srtm", "elevation.tif"),
    "slope": ("srtm", "slope.tif"),
}


# --------------------------------------------------------------------------
# Grid geometry - derived from the raster's own affine transform, not
# hardcoded, so it's guaranteed to match the real georeferencing.
# --------------------------------------------------------------------------
def load_grid_geometry(aligned_dir, target_var_dir):
    """Use one real Sentinel-2 raster to get the authoritative transform,
    then verify chirps/gldas/grace/srtm share the exact same grid."""
    sample_dir = os.path.join(aligned_dir, target_var_dir)
    sample_file = sorted(os.listdir(sample_dir))[0]
    with rasterio.open(os.path.join(sample_dir, sample_file)) as src:
        transform = src.transform
        shape = src.shape
        bounds = src.bounds
        crs = src.crs

    checks = {
        "chirps": os.path.join(aligned_dir, "chirps"),
        "gldas": os.path.join(aligned_dir, "gldas"),
        "grace": os.path.join(aligned_dir, "grace"),
    }
    for name, d in checks.items():
        f = sorted(os.listdir(d))[0]
        with rasterio.open(os.path.join(d, f)) as src2:
            if src2.shape != shape or src2.bounds != bounds:
                raise ValueError(
                    f"{name} grid does not match {target_var_dir} grid: "
                    f"{src2.shape}/{src2.bounds} vs {shape}/{bounds}")
    for name, (d, f) in STATIC_VARS.items():
        with rasterio.open(os.path.join(aligned_dir, d, f)) as src2:
            if src2.shape != shape or src2.bounds != bounds:
                raise ValueError(
                    f"{name} grid does not match {target_var_dir} grid: "
                    f"{src2.shape}/{src2.bounds} vs {shape}/{bounds}")
    print(f"[grid] shape={shape} crs={crs} bounds={bounds} "
          f"- confirmed identical across chirps/gldas/grace/srtm/{target_var_dir}")
    return transform, shape


def pixel_centers(transform, shape):
    """row, col -> lon, lat at PIXEL CENTER (not corner), via the raster's
    own affine transform. Returns arrays of length n_rows*n_cols in row-major
    (row, then col) order, matching how the raster arrays are flattened."""
    n_rows, n_cols = shape
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    # rasterio.transform.xy defaults to pixel center
    lons, lats = rasterio.transform.xy(transform, rows, cols)
    return rows, cols, np.asarray(lons), np.asarray(lats)


# --------------------------------------------------------------------------
# Raster loading
# --------------------------------------------------------------------------
def read_raster_flat(path, shape):
    """Read band 1, flatten in the same row-major order as pixel_centers()."""
    with rasterio.open(path) as src:
        if src.shape != shape:
            raise ValueError(f"{path} has shape {src.shape}, expected {shape}")
        arr = src.read(1).astype("float64")
    return arr.ravel()  # row-major: matches meshgrid(indexing="ij").ravel()


def date_filename(date):
    return f"{pd.Timestamp(date).strftime('%Y-%m-%d')}.tif"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # ---- expected dates: straight from feature_table.csv, not the raw
    #      kriged_target_monthly folder (see docstring) ----
    ft = pd.read_csv(FEATURE_TABLE_CSV)
    expected_dates = sorted(pd.to_datetime(ft["date"]).unique())
    print(f"[dates] {len(expected_dates)} expected campaign dates "
          f"(from {FEATURE_TABLE_CSV}): "
          f"{pd.Timestamp(expected_dates[0]).date()} .. "
          f"{pd.Timestamp(expected_dates[-1]).date()}")

    # ---- grid geometry, confirmed consistent across all rasters ----
    transform, shape = load_grid_geometry(ALIGNED_DIR, TARGET_VAR_DIR)
    n_rows, n_cols = shape
    n_pixels = n_rows * n_cols
    rows, cols, lons, lats = pixel_centers(transform, shape)
    pixel_id = rows * n_cols + cols  # row-major id, 0..4095

    # ---- static SRTM layers, read once ----
    static_flat = {}
    for out_col, (subdir, fname) in STATIC_VARS.items():
        static_flat[out_col] = read_raster_flat(
            os.path.join(ALIGNED_DIR, subdir, fname), shape)

    # ---- build the full panel, one block of 4096 rows per date ----
    blocks = []
    missing_file_log = {v: [] for v in list(MONTHLY_VARS) + [TARGET_VAR_DIR]}
    for date in expected_dates:
        ts = pd.Timestamp(date)
        block = pd.DataFrame({
            "pixel_id": pixel_id,
            "row": rows,
            "col": cols,
            "lon": lons,
            "lat": lats,
            "date": ts,
        })
        for out_col in static_flat:
            block[out_col] = static_flat[out_col]

        for var_dir, out_col in MONTHLY_VARS.items():
            path = os.path.join(ALIGNED_DIR, var_dir, date_filename(ts))
            if os.path.exists(path):
                block[out_col] = read_raster_flat(path, shape)
            else:
                block[out_col] = np.nan
                missing_file_log[var_dir].append(str(ts.date()))

        target_path = os.path.join(ALIGNED_DIR, TARGET_VAR_DIR, date_filename(ts))
        if os.path.exists(target_path):
            block[TARGET_COL] = read_raster_flat(target_path, shape)
        else:
            block[TARGET_COL] = np.nan
            missing_file_log[TARGET_VAR_DIR].append(str(ts.date()))

        block["month"] = ts.month
        block["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        block["month_cos"] = np.cos(2 * np.pi * ts.month / 12)
        blocks.append(block)

    panel = pd.concat(blocks, ignore_index=True)
    panel.to_csv(OUTPUT_CSV, index=False)

    # ---- coverage summary ----
    n_rows_total = len(panel)
    n_missing = int(panel[TARGET_COL].isna().sum())
    n_observed = n_rows_total - n_missing
    print("\nFULL PANEL")
    print("----------")
    print(f"rows: {n_rows_total}")
    print(f"unique pixels: {panel['pixel_id'].nunique()}")
    print(f"unique dates: {panel['date'].nunique()}")
    print("\nNDVI:")
    print(f"  observed: {n_observed}")
    print(f"  missing:  {n_missing}")
    print(f"  coverage: {100 * n_observed / n_rows_total:.1f}%")
    for var_dir, out_col in MONTHLY_VARS.items():
        n_miss = int(panel[out_col].isna().sum())
        print(f"{out_col.upper()} missing: {n_miss} "
              f"({100 * n_miss / n_rows_total:.1f}%)"
              + (f"  [no raster file for dates: {missing_file_log[var_dir]}]"
                 if missing_file_log[var_dir] else ""))
    if missing_file_log[TARGET_VAR_DIR]:
        print(f"NOTE: no {TARGET_VAR_DIR} raster file at all (not even a "
              f"NaN-filled one) for dates: {missing_file_log[TARGET_VAR_DIR]}")
    for out_col in static_flat:
        n_miss = int(panel[out_col].isna().sum())
        print(f"{out_col.upper()} missing: {n_miss} "
              f"({100 * n_miss / n_rows_total:.1f}%)")

    print(f"\nwritten to {OUTPUT_CSV}")
    print("\nThis panel is unmodified w.r.t. wells/groundwater targets/V3 "
          "model selection - it only adds the missing-NDVI rows the "
          "recovery experiment needs. Feed it to V3 via:")
    print(f"  $env:NDVI_INPUT_CSV=\"{os.path.abspath(OUTPUT_CSV)}\"")
    


if __name__ == "__main__":
    main()