"""
srtm_fetch.py

Fetches SRTM elevation and derived slope over the study region. Unlike
CHIRPS/GRACE/GLDAS/Sentinel, SRTM is a single static DEM (no time
dimension), so this is a one-time pull rather than a monthly loop.

Output: data/raw/srtm/elevation.tif and data/raw/srtm/slope.tif, at the
same export scale as other sources so align_grid.py can reproject them
onto the common 64x64 grid identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geemap
import rasterio
import yaml

from gee_fetch import init_ee, load_config, _region_geometry  # reuse Phase 1 helpers

SRTM_COLLECTION = "USGS/SRTMGL1_003"


def fetch_and_export_srtm(config_path: str = "config/data_config.yaml", raw_dir: str = "data/raw") -> None:
    config = load_config(config_path)
    project_id = config.get("gee_project_id")
    init_ee(project_id)

    bbox = tuple(config["region"]["bbox"])
    scale_m = config.get("export_scale_m", 5000)
    region = _region_geometry(bbox)

    dem = ee.Image(SRTM_COLLECTION).select("elevation").clip(region)
    slope = ee.Terrain.slope(dem)

    out_dir = Path(raw_dir) / "srtm"
    out_dir.mkdir(parents=True, exist_ok=True)

    elevation_path = out_dir / "elevation.tif"
    slope_path = out_dir / "slope.tif"

    if not elevation_path.exists():
        print("[srtm_fetch] Exporting elevation...")
        geemap.ee_export_image(dem, filename=str(elevation_path), scale=scale_m, region=region, file_per_band=False)
        with rasterio.open(elevation_path, "r+") as f:
            f.nodata = -9999.0
        print(f"[srtm_fetch] Saved -> {elevation_path}")
    else:
        print(f"[srtm_fetch] {elevation_path} already cached, skipping.")

    if not slope_path.exists():
        print("[srtm_fetch] Exporting slope...")
        geemap.ee_export_image(slope, filename=str(slope_path), scale=scale_m, region=region, file_per_band=False)
        with rasterio.open(slope_path, "r+") as f:
            f.nodata = -9999.0
        print(f"[srtm_fetch] Saved -> {slope_path}")
    else:
        print(f"[srtm_fetch] {slope_path} already cached, skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch static SRTM elevation/slope.")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()

    fetch_and_export_srtm(args.config, args.raw_dir)