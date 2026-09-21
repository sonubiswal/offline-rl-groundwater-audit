"""
gee_fetch.py

Pulls the four core satellite/reanalysis data sources for Trishna-OPAL via
Google Earth Engine:
  - Sentinel-2 (NDVI, surface reflectance)
  - Sentinel-1 (SAR backscatter, useful as a soil-moisture proxy)
  - CHIRPS (precipitation)
  - GRACE mascon (groundwater storage anomaly, coarse ~300km)
  - GLDAS (soil moisture / land surface)

Design notes:
  - Every pull is cached to data/raw/ immediately as a GeoTIFF (or CSV for
    time series) so repeated runs during development don't re-hit GEE and
    risk rate limiting (see Risk Register, row 3, in the master roadmap).
  - All region/date parameters are read from config/data_config.yaml --
    never hardcode a bbox or date range in this file.
  - This module assumes `earthengine authenticate` has already been run
    once on this machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geemap
import rasterio
import yaml


# GEE collection IDs, centralized so they're easy to audit / update.
COLLECTIONS = {
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel1": "COPERNICUS/S1_GRD",
    "chirps": "UCSB-CHG/CHIRPS/DAILY",
    "grace": "NASA/GRACE/MASS_GRIDS_V04/MASCON",
    "gldas": "NASA/GLDAS/V021/NOAH/G025/T3H",
}


def init_ee(project_id: str | None = None) -> None:
    """Initialize the Earth Engine API, raising a clear error if auth/project setup is missing.

    Since GEE now requires every session to specify a Cloud project (not
    just an authenticated account), project_id must be provided -- either
    directly, or read from config/data_config.yaml's `gee_project_id` field
    by the caller before this is invoked.
    """
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception as e:
        raise RuntimeError(
            "Earth Engine initialization failed. Make sure:\n"
            "  1. You've run `earthengine authenticate` in your terminal.\n"
            "  2. config/data_config.yaml has a valid `gee_project_id` set to "
            "your Google Cloud project ID (find/create one at "
            "https://code.earthengine.google.com/ -- check the project "
            "dropdown near the top of the page)."
        ) from e


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _region_geometry(bbox: tuple[float, float, float, float]) -> ee.Geometry:
    """bbox is (min_lon, min_lat, max_lon, max_lat)."""
    return ee.Geometry.Rectangle(list(bbox))


def fetch_chirps(bbox, start_date: str, end_date: str) -> ee.ImageCollection:
    """Monthly-summed CHIRPS precipitation over the region and date range."""
    region = _region_geometry(bbox)
    ic = (
        ee.ImageCollection(COLLECTIONS["chirps"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select("precipitation")
    )
    return ic


def fetch_grace(bbox, start_date: str, end_date: str) -> ee.ImageCollection:
    """GRACE mascon groundwater/total water storage anomaly.

    NOTE: kept for compatibility, but see fetch_grace_nearest() below --
    strict calendar-month filterDate() badly undercounts GRACE coverage
    because GRACE mascon solutions are irregular ~30-day windows that
    rarely align to calendar month boundaries (e.g. a solution's
    system:time_start might be 2015-01-15, so it's invisible to both a
    strict January AND a strict February calendar-month filter). Use
    fetch_grace_nearest() for actual pulls.
    """
    region = _region_geometry(bbox)
    ic = (
        ee.ImageCollection(COLLECTIONS["grace"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select("lwe_thickness")
    )
    return ic


def fetch_grace_nearest(bbox, target_month_start: str, window_days: int = 45) -> ee.ImageCollection:
    """Find the single GRACE mascon image closest to the target month.

    GRACE mascon "monthly" solutions are irregular in period length and
    don't align to calendar month boundaries, so a strict
    filterDate(month_start, month_end) query frequently returns nothing
    even when a perfectly good nearby GRACE solution exists just outside
    that window. This instead searches a wider window around the target
    month and picks whichever available image's timestamp is closest to
    the target month's midpoint -- the standard approach for working with
    irregular-cadence satellite products.

    Parameters
    ----------
    bbox : study region bbox
    target_month_start : str, e.g. "2015-02-01" -- the calendar month being requested
    window_days : how many days on either side of the target month's
        midpoint to search for a candidate image. 45 days comfortably
        covers GRACE's irregular ~30-35 day solution spacing.

    Returns
    -------
    ee.ImageCollection
        Either a single-image collection (the nearest match), or an empty
        collection if nothing exists within window_days (a genuine gap,
        e.g. the 2017-2018 GRACE/GRACE-FO mission transition).
    """
    region = _region_geometry(bbox)
    target = ee.Date(target_month_start).advance(15, "day")  # approx month midpoint

    ic = (
        ee.ImageCollection(COLLECTIONS["grace"])
        .filterBounds(region)
        .filterDate(target.advance(-window_days, "day"), target.advance(window_days, "day"))
        .select("lwe_thickness")
    )

    def _with_diff(img):
        diff = ee.Number(img.date().difference(target, "day")).abs()
        return img.set("date_diff", diff)

    ic_sorted = ic.map(_with_diff).sort("date_diff")
    return ic_sorted.limit(1)  # empty if no images were found in the window


def fetch_gldas(bbox, start_date: str, end_date: str) -> ee.ImageCollection:
    """GLDAS soil moisture (top layer) as a covariate for downscaling."""
    region = _region_geometry(bbox)
    ic = (
        ee.ImageCollection(COLLECTIONS["gldas"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select("SoilMoi0_10cm_inst")
    )
    return ic


def fetch_sentinel2_ndvi(bbox, start_date: str, end_date: str, cloud_cover_max: int = 20) -> ee.ImageCollection:
    """Cloud-filtered Sentinel-2 NDVI composite."""
    region = _region_geometry(bbox)

    def add_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)

    ic = (
        ee.ImageCollection(COLLECTIONS["sentinel2"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
        .map(add_ndvi)
        .select("NDVI")
    )
    return ic


def fetch_sentinel1_backscatter(bbox, start_date: str, end_date: str) -> ee.ImageCollection:
    """Sentinel-1 VV backscatter, a soil-moisture-sensitive SAR proxy."""
    region = _region_geometry(bbox)
    ic = (
        ee.ImageCollection(COLLECTIONS["sentinel1"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    return ic


def export_monthly_composite(
    ic: ee.ImageCollection,
    band_name: str,
    bbox,
    scale_m: int,
    out_path: Path,
    reducer: ee.Reducer = None,
    nodata_value: float = -9999.0,
) -> str:
    """Reduce an ImageCollection to a single composite and export to local GeoTIFF.

    CRITICAL nodata handling: composites over a large study region are often
    built from only 1-2 satellite passes per month (Sentinel-1/2 swaths are
    ~250-290km, far smaller than an 800km+ study region). Pixels outside
    actual coverage for that month are MASKED by GEE, not zero. If left
    unmasked before export, geemap/GEE's download path silently fills masked
    pixels with 0 -- which then gets treated as a real observation downstream
    (e.g. "backscatter = 0 dB" or "NDVI = 0", both physically wrong and
    silently corrupting anything trained on this data).

    This function explicitly unmasks to `nodata_value` (-9999, distinct from
    any real physical value in these bands) before export, then re-opens the
    written file to set proper GeoTIFF nodata metadata so downstream readers
    (align_grid.py) can correctly treat it as missing rather than signal.

    Returns
    -------
    str
        One of "ok", "no_images" (collection was empty for this period --
        a genuine data gap, not a failure), or raises on export failure.
    """
    reducer = reducer or ee.Reducer.mean()
    region = _region_geometry(bbox)

    if ic.size().getInfo() == 0:
        return "no_images"

    composite = ic.reduce(reducer).clip(region)
    composite_unmasked = composite.unmask(nodata_value)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    geemap.ee_export_image(
        composite_unmasked,
        filename=str(out_path),
        scale=scale_m,
        region=region,
        file_per_band=False,
    )

    # Post-process: declare the nodata value in the GeoTIFF's own metadata so
    # any tool reading this file (rasterio, QGIS, align_grid.py) knows -9999
    # means "no observation", not "observed value of -9999".
    with rasterio.open(out_path, "r+") as f:
        f.nodata = nodata_value

    return "ok"


def fetch_all_monthly(config_path: str = "config/data_config.yaml", raw_dir: str = "data/raw") -> None:
    """Driver: pull every source as monthly composites across the configured date range.

    Caches each monthly composite to data/raw/<source>/<YYYY-MM>.tif so a
    crashed or interrupted run can resume without re-fetching completed
    months. Also writes data/raw/fetch_log.csv recording the outcome for
    every (source, month) pair -- "ok", "no_images" (a genuine data gap,
    e.g. satellite not yet launched or a known GRACE mission gap), or
    "failed" (an actual error, e.g. quota/network) -- so gaps in the final
    completeness report can be diagnosed rather than guessed at.
    """
    config = load_config(config_path)
    project_id = config.get("gee_project_id")
    init_ee(project_id)
    bbox = tuple(config["region"]["bbox"])
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    scale_m =   config.get("export_scale_m", 5000)  # coarse default; refine per-source if needed

    months = pd_date_range_months(start, end)

    fetchers = {
        "chirps": (fetch_chirps, "precipitation"),
        "gldas": (fetch_gldas, "SoilMoi0_10cm_inst"),
        "sentinel2_ndvi": (fetch_sentinel2_ndvi, "NDVI"),
        "sentinel1_vv": (fetch_sentinel1_backscatter, "VV"),
    }
    # GRACE is handled separately -- it needs nearest-match date logic
    # (see fetch_grace_nearest) instead of strict calendar-month filtering.
    grace_source_name = "grace"
    grace_band_name = "lwe_thickness"

    log_rows = []
    log_path = Path(raw_dir) / "fetch_log.csv"

    for source_name, (fetch_fn, band_name) in fetchers.items():
        print(f"[gee_fetch] Fetching {source_name}...")
        for month_start, month_end in months:
            out_path = Path(raw_dir) / source_name / f"{month_start}.tif"
            if out_path.exists():
                print(f"[gee_fetch]   {source_name} {month_start} already cached, skipping.")
                log_rows.append({"source": source_name, "month": month_start, "status": "cached"})
                continue
            try:
                ic = fetch_fn(bbox, month_start, month_end)
                status = export_monthly_composite(ic, band_name, bbox, scale_m, out_path)
                log_rows.append({"source": source_name, "month": month_start, "status": status})
                if status == "no_images":
                    print(f"[gee_fetch]   {source_name} {month_start}: no images available "
                          f"(genuine data gap, e.g. satellite not yet launched or mission gap).")
                else:
                    print(f"[gee_fetch]   Saved {out_path}")
            except Exception as e:
                log_rows.append({"source": source_name, "month": month_start, "status": f"failed: {e}"})
                print(f"[gee_fetch]   FAILED for {source_name} {month_start}: {e}")
                print("[gee_fetch]   Continuing to next month (already-cached months are preserved).")

    print(f"[gee_fetch] Fetching {grace_source_name} (nearest-match, calendar-month filtering "
          f"doesn't fit GRACE's irregular solution cadence)...")
    for month_start, _ in months:
        out_path = Path(raw_dir) / grace_source_name / f"{month_start}.tif"
        if out_path.exists():
            print(f"[gee_fetch]   {grace_source_name} {month_start} already cached, skipping.")
            log_rows.append({"source": grace_source_name, "month": month_start, "status": "cached"})
            continue
        try:
            ic = fetch_grace_nearest(bbox, month_start)
            status = export_monthly_composite(ic, grace_band_name, bbox, scale_m, out_path)
            log_rows.append({"source": grace_source_name, "month": month_start, "status": status})
            if status == "no_images":
                print(f"[gee_fetch]   {grace_source_name} {month_start}: no images within search "
                      f"window (genuine mission gap).")
            else:
                print(f"[gee_fetch]   Saved {out_path}")
        except Exception as e:
            log_rows.append({"source": grace_source_name, "month": month_start, "status": f"failed: {e}"})
            print(f"[gee_fetch]   FAILED for {grace_source_name} {month_start}: {e}")
            print("[gee_fetch]   Continuing to next month (already-cached months are preserved).")

    import pandas as pd
    log_df = pd.DataFrame(log_rows)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"[gee_fetch] Wrote per-month fetch log -> {log_path}")
    print("[gee_fetch] Check this log to distinguish genuine data gaps (no_images) from real failures (failed).")


def pd_date_range_months(start: str, end: str) -> list[tuple[str, str]]:
    """Return list of (month_start, month_end) date string pairs covering [start, end]."""
    import pandas as pd

    periods = pd.period_range(start=start, end=end, freq="M")
    out = []
    for p in periods:
        month_start = p.start_time.strftime("%Y-%m-%d")
        month_end = p.end_time.strftime("%Y-%m-%d")
        out.append((month_start, month_end))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GEE data sources as monthly composites.")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()

    fetch_all_monthly(args.config, args.raw_dir)