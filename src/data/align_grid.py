"""
align_grid.py

Reprojects and resamples every raw data source (CHIRPS, GRACE, GLDAS,
Sentinel-1/2 composites from gee_fetch.py, plus rasterized CGWB well
points) onto a single common 64x64 grid over the study region bbox.

This is what makes ConvLSTM training possible in Phase 3 -- every input
channel and the target must share identical spatial dimensions and
pixel-to-coordinate mapping.

Design notes:
  - The target grid is defined ONCE from config/data_config.yaml's bbox,
    at a fixed 64x64 resolution, and every source is resampled onto it
    via rasterio's reproject/resample -- never re-derived per source.
  - Bilinear resampling is used for continuous fields (precipitation,
    soil moisture, NDVI, backscatter); nearest-neighbor is used for
    categorical sources (soil texture class, LULC) -- see
    NEAREST_NEIGHBOR_SOURCES below (Phase 2.6 addition).
  - Held-out wells (data/held_out_wells/held_out_ids.csv) are tracked
    separately and are NEVER rasterized into the training grid stack --
    only used later for point extraction during evaluation. This module
    only rasterizes TRAINING wells, consistent with the anti-circularity
    requirement enforced in held_out_split.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


GRID_SIZE = 64  # fixed per the roadmap: common 64x64 grid

# Phase 2.6 addition: source directory names that hold categorical class
# codes rather than continuous physical quantities. These MUST use
# nearest-neighbor resampling -- bilinear-interpolating a class code
# (e.g. averaging cropland=40 and built_up=50) produces a meaningless
# fractional value between two unrelated categories.
NEAREST_NEIGHBOR_SOURCES = {"soil_texture_class", "lulc_worldcover"}


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def target_transform_and_crs(bbox: tuple[float, float, float, float]):
    """Build the affine transform for the common 64x64 target grid.

    Returns (transform, crs, width, height) where width=height=GRID_SIZE.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, GRID_SIZE, GRID_SIZE)
    crs = "EPSG:4326"
    return transform, crs, GRID_SIZE, GRID_SIZE


def reproject_raster_to_grid(
    src_path: str | Path,
    dst_path: str | Path,
    bbox: tuple[float, float, float, float],
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Reproject/resample a single raster onto the common 64x64 grid and save.

    CRITICAL: propagates nodata correctly. Source rasters from gee_fetch.py
    use -9999 to mark pixels with no satellite observation for that month
    (e.g. outside that month's swath coverage, or masked by clouds). If
    nodata isn't declared here, rasterio's reproject() will blend -9999
    into neighboring real values during bilinear resampling, silently
    corrupting pixels near any data gap. This function reads the source's
    declared nodata value, passes it through reproject() as both src_nodata
    and dst_nodata (output as NaN for clean downstream handling), and sets
    the same nodata declaration on the output file.

    Parameters
    ----------
    src_path : path to the source GeoTIFF (e.g. a monthly composite from gee_fetch.py)
    dst_path : path to write the aligned 64x64 GeoTIFF
    bbox : study region bbox, must match config/data_config.yaml
    resampling : rasterio Resampling method (bilinear for continuous fields,
        nearest for categorical fields -- see NEAREST_NEIGHBOR_SOURCES)
    """
    dst_transform, dst_crs, width, height = target_transform_and_crs(bbox)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata  # None if source never declared one (assume no gaps)
        dst_array = np.full((src.count, height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, list(range(1, src.count + 1))),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )

        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            dst_path, "w",
            driver="GTiff",
            height=height, width=width,
            count=src.count, dtype="float32",
            crs=dst_crs, transform=dst_transform,
            nodata=np.nan,
        ) as dst:
            dst.write(dst_array)


def rasterize_training_wells(
    train_wells_csv: str | Path,
    bbox: tuple[float, float, float, float],
    date_filter: str | None = None,
) -> np.ndarray:
    """Rasterize TRAINING wells only onto the 64x64 grid via simple binning.

    For grid cells with multiple wells, values are averaged. Cells with no
    wells are set to NaN (to be filled by Kriging in a later phase --
    this function only does the raw binning step).

    Parameters
    ----------
    train_wells_csv : path to data/interim/train_well_ids.csv-derived readings
        (must have columns [lat, lon, depth_m], optionally [date])
    bbox : study region bbox
    date_filter : optional exact date string to filter readings to a single
        time slice before rasterizing (useful for building one time-step
        of the ConvLSTM input sequence)

    Returns
    -------
    np.ndarray of shape (GRID_SIZE, GRID_SIZE), NaN where no training well falls.
    """
    df = pd.read_csv(train_wells_csv)
    if date_filter is not None and "date" in df.columns:
        df = df[df["date"] == date_filter]

    min_lon, min_lat, max_lon, max_lat = bbox
    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
    counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    lon_idx = ((df["lon"] - min_lon) / (max_lon - min_lon) * GRID_SIZE).astype(int).clip(0, GRID_SIZE - 1)
    lat_idx = ((df["lat"] - min_lat) / (max_lat - min_lat) * GRID_SIZE).astype(int).clip(0, GRID_SIZE - 1)
    # Row 0 = top of grid = max_lat, so flip the lat index
    row_idx = (GRID_SIZE - 1) - lat_idx
    col_idx = lon_idx

    sums = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for r, c, val in zip(row_idx, col_idx, df["depth_m"]):
        if np.isnan(val):
            continue
        sums[r, c] += val
        counts[r, c] += 1

    nonzero = counts > 0
    grid[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)

    return grid


def align_all_sources(
    config_path: str = "config/data_config.yaml",
    raw_dir: str = "data/raw",
    interim_dir: str = "data/interim",
) -> None:
    """Batch-align every cached raster in data/raw/<source>/*.tif onto the common grid.

    Output structure mirrors input: data/interim/<source>/<month>.tif,
    each now guaranteed to be GRID_SIZE x GRID_SIZE and co-registered.
    """
    config = load_config(config_path)
    bbox = tuple(config["region"]["bbox"])

    raw_root = Path(raw_dir)
    if not raw_root.exists():
        print(f"[align_grid] {raw_root} does not exist yet -- run gee_fetch.py first.")
        return

    n_aligned = 0
    for source_dir in sorted(raw_root.iterdir()):
        if not source_dir.is_dir():
            continue
        resampling_method = (
            Resampling.nearest
            if source_dir.name in NEAREST_NEIGHBOR_SOURCES
            else Resampling.bilinear
        )
        for tif_path in sorted(source_dir.glob("*.tif")):
            dst_path = Path(interim_dir) / source_dir.name / tif_path.name
            if dst_path.exists():
                continue
            try:
                print(f"[align_grid] {source_dir.name}: using {resampling_method.name} resampling")
                reproject_raster_to_grid(
                    tif_path, dst_path, bbox, resampling=resampling_method
                )
                n_aligned += 1
            except Exception as e:
                print(f"[align_grid] FAILED aligning {tif_path}: {e}")

    print(f"[align_grid] Aligned {n_aligned} rasters onto the common {GRID_SIZE}x{GRID_SIZE} grid.")
    print(f"[align_grid] Output: {interim_dir}/<source>/<month>.tif")
    print("[align_grid] Reminder: verify alignment visually (plot each layer) before Phase 2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align all raw raster sources onto the common 64x64 grid.")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--interim_dir", default="data/interim")
    args = parser.parse_args()

    align_all_sources(args.config, args.raw_dir, args.interim_dir)