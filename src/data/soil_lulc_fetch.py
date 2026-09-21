"""
soil_lulc_fetch.py

Fetches two static covariates for Phase 2.6, mirroring srtm_fetch.py's
pattern (Section 7 of validation_methodology.md): pulled once, no time
dimension, reused identically for every date.

  1. Soil texture class  — OpenLandMap / SoilGrids-derived USDA texture
     class, 0-10cm depth (categorical -> will be one-hot encoded downstream)
  2. Land use / land cover — ESA WorldCover v200 (categorical, 10m native,
     resampled to the study grid)

Config schema confirmed against the real config/data_config.yaml (2026-09):
  region.bbox, gee_project_id, export_scale_m are read directly from
  there — no more guessed key names.

OUTPUT / DOWNLOAD LOCATION (confirmed against your real data/raw/
layout, which uses one top-level folder per source — chirps/, gldas/,
grace/, sentinel1_vv/, sentinel2_ndvi/, srtm/ — not a shared "static/"
folder):

  This script exports to Google Drive (folder "trishna_opal_static"),
  not to a local path — GEE batch exports always land in Drive first.
  After both tasks show "Completed" at
  https://code.earthengine.google.com/tasks, download the two GeoTIFFs
  and place them at:

      data/raw/soil_texture_class/soil_texture_class.tif
      data/raw/lulc_worldcover/lulc_worldcover.tif

  (create both folders manually if they don't exist — mirrors how
  srtm/ is its own top-level folder despite also being a static,
  no-time-dimension source).

  align_grid.py's align_all_sources() walks data/raw/<source>/*.tif by
  folder name, so these folder names must match exactly for it to find
  and process them. See align_grid_patch.txt for the required change to
  align_grid.py itself (NEAREST_NEIGHBOR_SOURCES) before running it on
  these two — bilinear is currently the hardcoded default for every
  source and would corrupt these categorical rasters if left unpatched.

USDA texture class codes (OpenLandMap 0cm, band 'b0'), for reference when
building the one-hot encoder downstream:
  1 Clay, 2 Silty Clay, 3 Sandy Clay, 4 Clay Loam, 5 Silty Clay Loam,
  6 Sandy Clay Loam, 7 Loam, 8 Silty Loam, 9 Sandy Loam, 10 Silt,
  11 Loamy Sand, 12 Sand
"""

import ee
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/data_config.yaml")
# NOTE: this script exports to Google Drive, not to a local path -- OUT_DIR
# below is not actually written to by this script. See the docstring above
# for the real destination folders you need to create and place the
# downloaded GeoTIFFs into by hand.

# GEE collection IDs
SOIL_TEXTURE_COLLECTION = "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02"
SOIL_TEXTURE_BAND = "b0"  # 0 cm depth slice

LULC_COLLECTION = "ESA/WorldCover/v200"
LULC_BAND = "Map"

# ESA WorldCover class codes (for downstream one-hot encoding)
LULC_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_veg",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_bbox(cfg):
    # Real schema (confirmed from config/data_config.yaml):
    #   region.bbox: [min_lon, min_lat, max_lon, max_lat]
    bbox = cfg["region"]["bbox"]
    return ee.Geometry.Rectangle(bbox)


def fetch_soil_texture(region: ee.Geometry, export_resolution_m: int):
    img = ee.Image(SOIL_TEXTURE_COLLECTION).select(SOIL_TEXTURE_BAND).clip(region)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description="soil_texture_class",
        folder="trishna_opal_static",
        fileNamePrefix="soil_texture_class",
        region=region,
        scale=export_resolution_m,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    return task


def fetch_lulc(region: ee.Geometry, export_resolution_m: int):
    # WorldCover is a single-image collection (one global mosaic per version);
    # mosaic() is defensive in case of future multi-tile releases.
    img = ee.ImageCollection(LULC_COLLECTION).mosaic().select(LULC_BAND).clip(region)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description="lulc_worldcover",
        folder="trishna_opal_static",
        fileNamePrefix="lulc_worldcover",
        region=region,
        scale=export_resolution_m,
        crs="EPSG:4326",
        maxPixels=1e9,
    )
    task.start()
    return task


def main():
    cfg = load_config()

    # Real schema (confirmed): gee_project_id, not a hardcoded literal.
    ee.Initialize(project=cfg["gee_project_id"])
    region = load_bbox(cfg)

    # Real schema (confirmed): export_scale_m, not export_resolution_m.
    # Section 4 of validation_methodology.md flags this 5000m value as a
    # placeholder still pending review — using the config's actual value
    # keeps this pull consistent with every other GEE source rather than
    # hardcoding a second copy of the same placeholder here.
    export_resolution_m = cfg["export_scale_m"]

    t1 = fetch_soil_texture(region, export_resolution_m)
    t2 = fetch_lulc(region, export_resolution_m)

    print("Submitted GEE export tasks:")
    print(f"  soil_texture_class -> task id {t1.id}")
    print(f"  lulc_worldcover    -> task id {t2.id}")
    print("Monitor at https://code.earthengine.google.com/tasks")
    print("Once both show Completed, download from Drive folder")
    print("'trishna_opal_static' and place at:")
    print("  data/raw/soil_texture_class/soil_texture_class.tif")
    print("  data/raw/lulc_worldcover/lulc_worldcover.tif")
    print("Then apply align_grid_patch.txt to align_grid.py BEFORE running")
    print("it on these two sources (nearest-neighbor resampling required).")


if __name__ == "__main__":
    main()