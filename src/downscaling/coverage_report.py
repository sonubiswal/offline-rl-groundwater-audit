r"""
coverage_report.py

Phase A of the Phase-2 repair plan: WHY is coverage only ~42%, and how much
of that is actually recoverable with a scientifically defensible fallback
window (as opposed to pushing --covariate_fallback_days up indefinitely)?

This is a DIFFERENT cut than diagnose_heldout.py's drop-reason audit.
diagnose_heldout.py tags each held-out row with exactly ONE reason, in the
validator's short-circuit order (first missing source wins). That's the
right tool for "how many rows does the CURRENT config evaluate, and why are
the rest dropped" -- but it can't answer:

  - "Of the rows missing Sentinel-2, how many are ALSO missing GRACE?"
    (short-circuit order hides this -- if CHIRPS fails first, you never see
    whether Sentinel-2 or GRACE would also have failed.)
  - "If I push the fallback window from 90 to 400 to 600 days, how much
    coverage do I actually gain, per source, before it plateaus?"
  - "Is the failure a MISSING COMPOSITE (no raster within the window) or a
    NODATA PIXEL (composite exists, but this well's cell is masked -- e.g.
    cloud-contaminated Sentinel-2)?" These need different fixes: the first
    is fixed by a wider temporal window; the second needs either a bigger
    spatial neighborhood fallback or an imputation model (Phase A step C in
    the plan) -- widening the window alone will NOT recover NaN-at-cell rows
    from the SAME composite, since the composite itself has no valid pixel
    there. A different (temporally nearby) composite might, which is why
    both checks below re-probe the STACK, not just file existence.

This script therefore checks each of the four monthly sources INDEPENDENTLY
(no short-circuit) at a sweep of candidate fallback windows, and separately
flags "file matched but pixel is NaN" vs "no file within window at all".

Part 1 -- per-source, per-window availability sweep
----------------------------------------------------
For WINDOW_SWEEP_DAYS = (60, 90, 180, 270, 400, 600, 800):
  for each monthly source (chirps, gldas, sentinel2_ndvi, grace):
    n_matched_file   = rows where find_nearest_raster succeeds within window
    n_valid_pixel    = of those, rows where the matched raster is non-NaN at
                       the well's cell
Reported as a cumulative recovery curve so you can see where it plateaus
(diminishing returns = the point past which widening the window is not
buying you real coverage, just staleness risk).

Part 2 -- combined coverage at a CANDIDATE config
---------------------------------------------------
Using one fallback window (--candidate_fallback_days, default 400 to match
what you already ran) AND independently testing every source (not
short-circuited), reports:
  n_all_sources_ok       every monthly source has a valid (non-NaN) pixel
  n_srtm_ok              static terrain present
  n_roll3_ok             3-month rolling composite buildable
  n_lag_ok               prior Kriged surface within lag window/fallback
  n_fully_evaluable      all of the above simultaneously true
This is the "recovered/imputed" denominator Phase A step D asks you to
report alongside the complete-case count.

Part 3 -- stratified coverage (year x season x region x well)
----------------------------------------------------------------
For the CURRENT config and the CANDIDATE config side by side:
  N held-out, N evaluable, coverage % per stratum
This is exactly the table Phase B's inverse-probability weighting needs as
input (P(evaluated=1 | X) is estimated from these same strata), so its
column names/strata are chosen to match what that script will consume.

LEAKAGE STATEMENT: this script never reads held-out `depth_m` for anything
except grouping/reporting (never for availability checks, never for
choosing a window). It is a pure covariate-availability audit -- the same
guarantee diagnose_heldout.py makes.

Outputs
-------
  reports/coverage_window_sweep.csv     per-source recovery curve
  reports/coverage_combined.csv         combined evaluability at the
                                        candidate config
  reports/coverage_by_stratum.csv       year/season/region/well coverage,
                                        current vs candidate config
  console summary

USAGE (from the repo root)
---------------------------
  python src\downscaling\coverage_report.py ^
      --config config\data_config.yaml ^
      --held_out_csv data\held_out_wells\held_out_ids.csv ^
      --interim_dir data\interim ^
      --kriged_dir data\interim\kriged_target_monthly ^
      --meta reports\rf_v2_nocoords_meta.json ^
      --candidate_fallback_days 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (
    MONTHLY_SOURCES, find_nearest_raster, find_nearest_prior_kriged,
    load_raster_array, rolling_average_features, latlon_to_grid_cell,
)
from rf_downscale_v2 import _season_label

WINDOW_SWEEP_DAYS = (60, 90, 180, 270, 400, 600, 800)


def _mid_region(lat: float, lon: float, mid_lat: float, mid_lon: float) -> str:
    return ("N" if lat >= mid_lat else "S") + ("W" if lon < mid_lon else "E")


# ---------------------------------------------------------------------
# Part 1: per-source, per-window availability sweep (independent checks --
# NOT short-circuited, so all four sources are probed for every row at
# every window, unlike the validator's early-return order).
# ---------------------------------------------------------------------
def source_window_sweep(held_df: pd.DataFrame, interim_dir: Path,
                        bbox, windows=WINDOW_SWEEP_DAYS) -> pd.DataFrame:
    rows = []
    # cache: (source, date_str, window) -> matched path or None, to avoid
    # re-globbing the same (source, date) pair once per window increase --
    # find_nearest_raster is called fresh per window since a wider window
    # can surface a different / more distant match.
    file_cache: dict[tuple[str, str, int], object] = {}
    array_cache: dict[object, np.ndarray] = {}

    for source in MONTHLY_SOURCES:
        for window in windows:
            n_matched = 0
            n_valid_pixel = 0
            for _, row in held_df.iterrows():
                date_str = str(row["date"])
                key = (source, date_str, window)
                if key not in file_cache:
                    file_cache[key] = find_nearest_raster(
                        interim_dir / source, date_str, window)
                matched = file_cache[key]
                if matched is None:
                    continue
                n_matched += 1
                if matched not in array_cache:
                    array_cache[matched] = load_raster_array(matched)
                arr = array_cache[matched]
                r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
                if not np.isnan(arr[r, c]):
                    n_valid_pixel += 1
            rows.append({
                "source": source, "window_days": window,
                "n_held_out": len(held_df),
                "n_file_matched": n_matched,
                "n_valid_pixel": n_valid_pixel,
                "pct_valid_pixel": n_valid_pixel / len(held_df) if len(held_df) else np.nan,
            })
            print(f"[coverage] {source:>16} window={window:>4}d  "
                  f"file_matched={n_matched:>6}  valid_pixel={n_valid_pixel:>6} "
                  f"({n_valid_pixel/len(held_df):.1%})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Part 2: combined evaluability at one candidate config (every source
# checked independently -- a row can fail more than one check here, unlike
# the validator's single drop_reason).
# ---------------------------------------------------------------------
def combined_coverage(held_df: pd.DataFrame, interim_dir: Path, kriged_dir: Path,
                      bbox, candidate_fallback_days: int,
                      lag_max_gap_days: int, lag_fallback_days: int) -> pd.DataFrame:
    static_ok = all((interim_dir / "srtm" / f).exists()
                     for f in ("elevation.tif", "slope.tif"))
    file_cache: dict[tuple, object] = {}
    array_cache: dict[object, np.ndarray] = {}
    lag_cache: dict[str, object] = {}

    per_row = []
    for _, row in held_df.iterrows():
        date_str = str(row["date"])
        r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)

        source_ok = {}
        for source in MONTHLY_SOURCES:
            key = (source, date_str)
            if key not in file_cache:
                file_cache[key] = find_nearest_raster(
                    interim_dir / source, date_str, candidate_fallback_days)
            matched = file_cache[key]
            ok = False
            if matched is not None:
                if matched not in array_cache:
                    array_cache[matched] = load_raster_array(matched)
                ok = not np.isnan(array_cache[matched][r, c])
            source_ok[source] = ok

        roll_ok = rolling_average_features(
            interim_dir, date_str, sources=["chirps", "gldas"], n_months=3,
            max_gap_days=candidate_fallback_days, min_months=2) is not None

        if date_str not in lag_cache:
            lag_path = find_nearest_prior_kriged(kriged_dir, date_str, lag_max_gap_days)
            if lag_path is None and lag_fallback_days > lag_max_gap_days:
                lag_path = find_nearest_prior_kriged(kriged_dir, date_str, lag_fallback_days)
            lag_cache[date_str] = lag_path
        lag_ok = lag_cache[date_str] is not None

        all_sources_ok = all(source_ok.values())
        fully_ok = all_sources_ok and static_ok and roll_ok and lag_ok
        per_row.append({
            "well_id": row["well_id"], "date": date_str,
            **{f"{s}_ok": source_ok[s] for s in MONTHLY_SOURCES},
            "srtm_ok": static_ok, "roll3_ok": roll_ok, "lag_ok": lag_ok,
            "all_sources_ok": all_sources_ok, "fully_evaluable": fully_ok,
        })
    return pd.DataFrame(per_row)


# ---------------------------------------------------------------------
# Part 3: stratified coverage table -- current config vs candidate config,
# same strata Phase B's IPW step will consume.
# ---------------------------------------------------------------------
def stratified_coverage(held_df: pd.DataFrame, current_eval_mask: np.ndarray,
                        candidate_eval_mask: np.ndarray, bbox) -> pd.DataFrame:
    mid_lat = (bbox[1] + bbox[3]) / 2.0
    mid_lon = (bbox[0] + bbox[2]) / 2.0
    df = held_df.copy()
    df["year"] = df["date"].astype(str).str[:4]
    df["season"] = df["date"].astype(str).str[5:7].astype(int).map(
        lambda m: _season_label(m))
    df["region"] = [
        _mid_region(lat, lon, mid_lat, mid_lon)
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    df["current_evaluable"] = current_eval_mask
    df["candidate_evaluable"] = candidate_eval_mask

    rows = []
    for stratum in ["year", "season", "region", "well_id",
                    "year+season", "season+region"]:
        cols = stratum.split("+")
        for key, grp in df.groupby(cols, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            label = " | ".join(f"{c}={k}" for c, k in zip(cols, key))
            rows.append({
                "stratum": stratum, "value": label, "n_held_out": len(grp),
                "n_current_evaluable": int(grp["current_evaluable"].sum()),
                "current_coverage_pct": float(grp["current_evaluable"].mean()),
                "n_candidate_evaluable": int(grp["candidate_evaluable"].sum()),
                "candidate_coverage_pct": float(grp["candidate_evaluable"].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase A: independent per-source coverage sweep + "
                    "stratified coverage table (missingness diagnosis, no model).")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    p.add_argument("--meta", default="reports/rf_v2_nocoords_meta.json",
                   help="Used only to read the CURRENT config's max_gap_days / "
                        "lag_max_gap_days / lag_fallback_days for the 'current' "
                        "column in the stratified table. Falls back to 90/120/0 "
                        "if absent.")
    p.add_argument("--candidate_fallback_days", type=int, default=400,
                   help="Fallback window swept in Part 2/3 as the 'candidate' "
                        "config to compare against the current one.")
    p.add_argument("--sweep_out", default="reports/coverage_window_sweep.csv")
    p.add_argument("--combined_out", default="reports/coverage_combined.csv")
    p.add_argument("--stratum_out", default="reports/coverage_by_stratum.csv")
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])

    meta = {}
    if Path(args.meta).exists():
        with open(args.meta) as f:
            meta = json.load(f)
    else:
        print(f"[coverage] WARNING: meta not found at {args.meta} -- using "
              f"defaults (max_gap_days=90, lag_max_gap_days=120, "
              f"lag_fallback_days=0) for the 'current' column.")
    current_max_gap = int(meta.get("max_gap_days", 90))
    current_cov_fallback = int(meta.get("covariate_fallback_days", 0))
    current_window = max(current_max_gap, current_cov_fallback) or current_max_gap
    lag_max_gap_days = int(meta.get("lag_max_gap_days", 120))
    lag_fallback_days = int(meta.get("lag_fallback_days", 0))

    held_df = pd.read_csv(args.held_out_csv)
    interim_dir = Path(args.interim_dir)
    kriged_dir = Path(args.kriged_dir)
    print(f"[coverage] Held-out CSV rows: {len(held_df)}")
    print(f"[coverage] Current config window: {current_window}d "
          f"(max_gap_days={current_max_gap}, covariate_fallback_days={current_cov_fallback})")
    print(f"[coverage] Candidate window for Parts 2/3: {args.candidate_fallback_days}d")

    # ---- Part 1 ----
    print()
    print("=" * 72)
    print("PART 1 -- per-source recovery curve across fallback windows")
    print("=" * 72)
    sweep_df = source_window_sweep(held_df, interim_dir, bbox)
    Path(args.sweep_out).parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(args.sweep_out, index=False)
    print(f"[coverage] Saved -> {args.sweep_out}")

    # ---- Part 2: combined coverage at CURRENT and CANDIDATE windows ----
    print()
    print("=" * 72)
    print("PART 2 -- combined evaluability (every source checked independently)")
    print("=" * 72)
    print(f"[coverage] Computing at CURRENT window ({current_window}d)...")
    current_combined = combined_coverage(
        held_df, interim_dir, kriged_dir, bbox, current_window,
        lag_max_gap_days, lag_fallback_days)
    print(f"[coverage] Computing at CANDIDATE window ({args.candidate_fallback_days}d)...")
    candidate_combined = combined_coverage(
        held_df, interim_dir, kriged_dir, bbox, args.candidate_fallback_days,
        lag_max_gap_days, max(lag_fallback_days, args.candidate_fallback_days))

    n_current_eval = int(current_combined["fully_evaluable"].sum())
    n_candidate_eval = int(candidate_combined["fully_evaluable"].sum())
    n = len(held_df)
    print(f"  Current  ({current_window:>4}d): fully_evaluable = {n_current_eval:>6} "
          f"/ {n} = {n_current_eval/n:.1%}")
    print(f"  Candidate({args.candidate_fallback_days:>4}d): fully_evaluable = {n_candidate_eval:>6} "
          f"/ {n} = {n_candidate_eval/n:.1%}")
    for s in MONTHLY_SOURCES:
        col = f"{s}_ok"
        print(f"    {s:<16} current_ok={current_combined[col].mean():.1%}  "
              f"candidate_ok={candidate_combined[col].mean():.1%}")

    combined_report = current_combined[["well_id", "date", "fully_evaluable"]].rename(
        columns={"fully_evaluable": "current_fully_evaluable"})
    combined_report["candidate_fully_evaluable"] = candidate_combined["fully_evaluable"].values
    for s in MONTHLY_SOURCES:
        combined_report[f"current_{s}_ok"] = current_combined[f"{s}_ok"].values
        combined_report[f"candidate_{s}_ok"] = candidate_combined[f"{s}_ok"].values
    Path(args.combined_out).parent.mkdir(parents=True, exist_ok=True)
    combined_report.to_csv(args.combined_out, index=False)
    print(f"[coverage] Saved -> {args.combined_out}")

    # ---- Part 3: stratified coverage, current vs candidate ----
    print()
    print("=" * 72)
    print("PART 3 -- stratified coverage (current vs candidate config)")
    print("=" * 72)
    strat_df = stratified_coverage(
        held_df, current_combined["fully_evaluable"].to_numpy(),
        candidate_combined["fully_evaluable"].to_numpy(), bbox)
    Path(args.stratum_out).parent.mkdir(parents=True, exist_ok=True)
    strat_df.to_csv(args.stratum_out, index=False)
    worst = strat_df[strat_df["stratum"] == "season"].sort_values("current_coverage_pct")
    print("  Season coverage (current -> candidate):")
    for _, r in worst.iterrows():
        print(f"    {r['value']:<20} {r['current_coverage_pct']:.1%} -> "
              f"{r['candidate_coverage_pct']:.1%}  (n_held_out={r['n_held_out']})")
    print(f"[coverage] Saved -> {args.stratum_out}")
    print("=" * 72)
    print()
    print("[coverage] NOTE: 'fully_evaluable' here is independent-check coverage, not")
    print("           the validator's actual evaluated count (which also requires the")
    print("           model's exact feature_columns to be NaN-free). Cross-check against")
    print("           rf_downscale_v2's reported 'Evaluated N' for the same window; a gap")
    print("           between them means a feature outside these checks (e.g. a v2.1/v2.2")
    print("           addition like dist_nearest_well_km) is independently causing drops.")


if __name__ == "__main__":
    main()