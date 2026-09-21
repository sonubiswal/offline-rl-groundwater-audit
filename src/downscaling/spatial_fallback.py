r"""
spatial_fallback.py

Direct follow-up to coverage_report.py's Part 1 (temporal window sweep).

IMPORTANT SCOPING NOTE (state this in the methods section verbatim): Phase A's
temporal sweep (coverage_report.py, WINDOW_SWEEP_DAYS 60..800) established the
maximum coverage achievable by TEMPORAL fallback alone -- i.e. finding an
older/newer composite when NONE is available near the target date. It did
NOT test spatial recovery, and did not touch the "composite matched, but the
exact well pixel is masked/nodata" failure mode -- that mode is NOT fixed by
widening the temporal window (the same composite, at any distance in time,
still has no valid pixel there). This script tests spatial recovery as an
INDEPENDENT, second lever. Do not describe coverage_report.py's numbers as a
"maximum achievable" ceiling in the paper; they are the temporal-only ceiling.
This script measures whether spatial fallback pushes coverage higher, and by
how much, before concluding whether NDVI imputation (the harder, model-based
option) is actually necessary.

WHAT THIS DOES
--------------
For every held-out row where:
  (a) a composite for --source was matched within --temporal_window_days
      (the candidate window established in Phase A, default 400d), AND
  (b) that composite's pixel at the well's exact grid cell is NaN
      (nodata / cloud-masked / etc -- this is what "nan_at_cell:<source>"
      meant in diagnose_heldout.py's audit),

search that SAME composite's valid pixels for the nearest one by real
haversine distance (not grid-index rings, which distort cell-to-km
conversion away from the equator / at coarse grids), and report:
  - distance_km to the nearest valid pixel
  - the value at that pixel (what would be imputed under a nearest-valid
    spatial fallback)
  - whether it falls within each of a fixed radius ladder (RADIUS_SWEEP_KM),
    to see where the recovery curve plateaus

Rows with NO composite matched at all within the temporal window are
reported separately (n_no_temporal_match) -- spatial fallback cannot help
them; that is Phase A's problem, not this script's.

LEAKAGE STATEMENT: only ever reads covariate rasters (never held-out
depth_m) to decide what value would be imputed at a well's cell. No
held-out groundwater information is used anywhere in this script.

Outputs
-------
  reports/spatial_fallback_<source>_per_row.csv   per-row: was NaN, was
                                                  recoverable, distance_km,
                                                  fallback value, radius
                                                  buckets it clears
  reports/spatial_fallback_<source>_curve.csv     cumulative recovery curve
                                                  across RADIUS_SWEEP_KM
  console summary: before/after coverage counts at the reported radii,
  and the combined (temporal-fallback-only) vs (temporal+spatial-fallback)
  coverage percentage, so this plugs directly into the same denominator
  coverage_report.py used (16,073 held-out rows).

USAGE
-----
  python src\downscaling\spatial_fallback.py ^
      --config config\data_config.yaml ^
      --held_out_csv data\held_out_wells\held_out_ids.csv ^
      --interim_dir data\interim ^
      --source sentinel2_ndvi ^
      --temporal_window_days 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import GRID_SIZE, find_nearest_raster, load_raster_array, latlon_to_grid_cell

RADIUS_SWEEP_KM = (5, 10, 15, 20, 30, 50, 75, 100, 150)
MAX_RADIUS_KM = max(RADIUS_SWEEP_KM)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Same formula as rf_downscale_v2._haversine_km, duplicated here to keep
    this script standalone/importable without pulling in rf_downscale_v2's
    heavier dependency chain (sklearn, matplotlib) for what is a pure
    raster-diagnostic tool."""
    R = 6371.0
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _cell_center_grids(bbox: tuple[float, float, float, float],
                       grid_size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """(grid_size, grid_size) lat/lon of every cell CENTER. Same mapping as
    rf_downscale_v2.compute_well_distance_grids (row 0 = north edge) so
    distances computed here are consistent with the rest of the pipeline."""
    min_lon, min_lat, max_lon, max_lat = bbox
    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    cell_lat = min_lat + ((grid_size - 0.5) - rows) / grid_size * (max_lat - min_lat)
    cell_lon = min_lon + (cols + 0.5) / grid_size * (max_lon - min_lon)
    return np.meshgrid(cell_lat, cell_lon, indexing="ij")  # LAT, LON


def nearest_valid_pixel(arr: np.ndarray, well_lat: float, well_lon: float,
                        lat_grid: np.ndarray, lon_grid: np.ndarray,
                        max_radius_km: float = MAX_RADIUS_KM) -> tuple[float, float, int, int]:
    """Returns (distance_km, value, r, c) of the nearest NON-NaN pixel in arr
    to (well_lat, well_lon), restricted to max_radius_km. (np.nan, np.nan,
    -1, -1) if none found within that radius. Brute-force over all valid
    cells -- fine at GRID_SIZE x GRID_SIZE = a few thousand cells at most;
    results are cached per-composite by the caller so this runs once per
    unique (source, matched_file), not once per held-out row."""
    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return np.nan, np.nan, -1, -1
    vlat = lat_grid[valid_mask]
    vlon = lon_grid[valid_mask]
    vidx_r, vidx_c = np.where(valid_mask)
    dists = _haversine_km(well_lat, well_lon, vlat, vlon)
    i = int(np.argmin(dists))
    d = float(dists[i])
    if d > max_radius_km:
        return np.nan, np.nan, -1, -1
    return d, float(arr[vidx_r[i], vidx_c[i]]), int(vidx_r[i]), int(vidx_c[i])


def run(held_df: pd.DataFrame, interim_dir: Path, bbox, source: str,
       temporal_window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lat_grid, lon_grid = _cell_center_grids(bbox)
    matched_cache: dict[str, object] = {}          # date_str -> matched path or None
    array_cache: dict[object, np.ndarray] = {}      # matched path -> raster array
    nearest_cache: dict[tuple, tuple] = {}          # (matched path, r, c) -> nearest-pixel result

    rows_out = []
    n_no_temporal_match = 0
    n_pixel_already_valid = 0
    n_nan_at_cell = 0

    for _, row in held_df.iterrows():
        date_str = str(row["date"])
        if date_str not in matched_cache:
            matched_cache[date_str] = find_nearest_raster(
                interim_dir / source, date_str, temporal_window_days)
        matched = matched_cache[date_str]
        if matched is None:
            n_no_temporal_match += 1
            rows_out.append({
                "well_id": row["well_id"], "date": date_str,
                "status": "no_temporal_match", "distance_km": np.nan,
                "fallback_value": np.nan,
            })
            continue

        if matched not in array_cache:
            array_cache[matched] = load_raster_array(matched)
        arr = array_cache[matched]
        r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)

        if not np.isnan(arr[r, c]):
            n_pixel_already_valid += 1
            rows_out.append({
                "well_id": row["well_id"], "date": date_str,
                "status": "valid_at_exact_cell", "distance_km": 0.0,
                "fallback_value": float(arr[r, c]),
            })
            continue

        n_nan_at_cell += 1
        key = (matched, r, c)
        if key not in nearest_cache:
            nearest_cache[key] = nearest_valid_pixel(
                arr, float(row["lat"]), float(row["lon"]), lat_grid, lon_grid)
        dist_km, value, _, _ = nearest_cache[key]
        rows_out.append({
            "well_id": row["well_id"], "date": date_str,
            "status": "recovered_by_spatial_fallback" if not np.isnan(dist_km) else "unrecoverable_beyond_max_radius",
            "distance_km": dist_km, "fallback_value": value,
        })

    per_row = pd.DataFrame(rows_out)
    for radius in RADIUS_SWEEP_KM:
        per_row[f"within_{radius}km"] = per_row["distance_km"] <= radius

    curve_rows = []
    n_total = len(held_df)
    n_baseline_valid = n_pixel_already_valid  # coverage with temporal fallback only, no spatial help
    for radius in RADIUS_SWEEP_KM:
        n_recovered = int(per_row[f"within_{radius}km"].sum())  # excludes already-valid (distance_km=0 counts here too, harmless)
        n_covered_total = n_baseline_valid + int(
            per_row.loc[per_row["status"] == "recovered_by_spatial_fallback", f"within_{radius}km"].sum())
        curve_rows.append({
            "radius_km": radius,
            "n_nan_at_cell_recovered": int(
                per_row.loc[per_row["status"] == "recovered_by_spatial_fallback", f"within_{radius}km"].sum()),
            "n_total_covered": n_covered_total,
            "pct_total_covered": n_covered_total / n_total if n_total else np.nan,
        })
    curve = pd.DataFrame(curve_rows)

    print(f"[spatial_fallback] source={source}  temporal_window={temporal_window_days}d")
    print(f"  n_held_out                 = {n_total}")
    print(f"  n_no_temporal_match         = {n_no_temporal_match}  (not fixable by spatial fallback -- see coverage_report.py)")
    print(f"  n_valid_at_exact_cell       = {n_pixel_already_valid}  ({n_pixel_already_valid/n_total:.1%} -- temporal-fallback-only coverage)")
    print(f"  n_nan_at_cell               = {n_nan_at_cell}  (candidates for spatial fallback)")
    print()
    print("  Recovery curve (cumulative coverage of the FULL 16,073-row denominator):")
    for _, r in curve.iterrows():
        print(f"    radius={r['radius_km']:>4} km   +{r['n_nan_at_cell_recovered']:>5} recovered   "
              f"total_covered={r['n_total_covered']:>6} ({r['pct_total_covered']:.1%})")
    n_unrecoverable = int((per_row["status"] == "unrecoverable_beyond_max_radius").sum())
    print(f"  n_unrecoverable_beyond_{MAX_RADIUS_KM}km = {n_unrecoverable}  "
          f"(candidates for an NDVI imputation model instead)")
    return per_row, curve


def main() -> None:
    p = argparse.ArgumentParser(
        description="Spatial-neighborhood fallback experiment for pixel-level "
                    "(nan-at-cell) covariate missingness. No held-out target used.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--source", default="sentinel2_ndvi",
                   help="Monthly covariate subfolder under --interim_dir to test "
                        "(e.g. sentinel2_ndvi, chirps, gldas, grace).")
    p.add_argument("--temporal_window_days", type=int, default=400,
                   help="Temporal matching window -- use the candidate window "
                        "already established by coverage_report.py's sweep, "
                        "so this experiment isolates the SPATIAL lever only.")
    p.add_argument("--per_row_out", default=None,
                   help="Defaults to reports/spatial_fallback_<source>_per_row.csv")
    p.add_argument("--curve_out", default=None,
                   help="Defaults to reports/spatial_fallback_<source>_curve.csv")
    args = p.parse_args()

    per_row_out = args.per_row_out or f"reports/spatial_fallback_{args.source}_per_row.csv"
    curve_out = args.curve_out or f"reports/spatial_fallback_{args.source}_curve.csv"

    import yaml
    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])
    held_df = pd.read_csv(args.held_out_csv)
    print(f"[spatial_fallback] Held-out CSV rows: {len(held_df)}")

    per_row, curve = run(held_df, Path(args.interim_dir), bbox,
                         args.source, args.temporal_window_days)

    Path(per_row_out).parent.mkdir(parents=True, exist_ok=True)
    per_row.to_csv(per_row_out, index=False)
    Path(curve_out).parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_out, index=False)
    print(f"[spatial_fallback] Saved -> {per_row_out}")
    print(f"[spatial_fallback] Saved -> {curve_out}")
    print()
    print("[spatial_fallback] NOTE FOR THE PAPER: the coverage_report.py sweep established")
    print("           the maximum achievable by TEMPORAL fallback alone. This experiment")
    print("           adds SPATIAL fallback as a second, independent lever -- report both")
    print("           numbers, and only reach for an NDVI imputation model if the spatial")
    print("           recovery curve above plateaus well short of your coverage target.")


if __name__ == "__main__":
    main()