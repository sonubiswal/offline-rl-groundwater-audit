r"""
diagnose_heldout.py

Answers, with code rather than theory: WHY are N of the independent held-out
readings skipped, and are the evaluated ones representative?

Part 1 -- drop-reason audit (reconciliation is the correctness test)
--------------------------------------------------------------------
Every one of the 16,073 held-out rows is tagged with EXACTLY ONE
mutually-exclusive `drop_reason`, assigned in the SAME order the validator
(rf_downscale_v2.validate_v2 -> build_stack_v2) checks and early-returns:

  1. no_<source>_within_<window>d   for the first of chirps / gldas /
                                    sentinel2_ndvi / grace with no monthly
                                    composite within max_gap_days (or within
                                    covariate_fallback_days when the meta
                                    enables it). The validator returns None at
                                    the FIRST missing source (build_stack_v2
                                    lines ~296-320), so only one reason is
                                    attributed even if several are missing.
  2. no_srtm_static                 interim_dir/srtm/elevation.tif or
                                    slope.tif absent.
  3. no_roll3_chirps_gldas          rolling_average_features returned None
                                    (<2 of the trailing 3 months matchable
                                    within the roll window).
  4. no_kriged_lag_within_<W>d      no PRIOR Kriged surface within
                                    lag_max_gap_days, and either no
                                    lag_fallback_days configured or the
                                    fallback also failed. (Prior-only: the
                                    validator uses find_nearest_prior_kriged.)
  5. nan_at_cell:<feature>          a feature column the model consumes is NaN
                                    at the well's grid cell (nodata pixels in
                                    a matched raster, NaN in the Kriged
                                    surface, etc.). The FIRST such feature in
                                    FEATURE_COLUMNS order is named.
  6. evaluated                      complete feature vector -> the validator
                                    would score this row.

Because reasons are checked in the validator's exact early-return order and
every row gets exactly one tag, the counts MUST sum to the CSV row count and
the `evaluated` count MUST equal the validator's "Evaluated N". Run this
BEFORE and AFTER any coverage change: the diff is which exclusions you
actually removed.

Part 2 -- representativeness of evaluated vs skipped (statistics, not eyeballs)
------------------------------------------------------------------------------
For each comparison variable the script reports, per group:
  - KS statistic + p-value        (continuous: depth_m, lat, lon, rainfall_mm)
  - chi-square + p-value          (categorical: season, year, well_id, and
                                   `district` IF that column exists in the
                                   held-out CSV)
  - standardized mean difference  SMD = (mean_eval - mean_skip) / pooled SD,
                                   flagging |SMD| > 0.25 as a selection-bias
                                   concern (standard threshold).
Plus the evaluated FRACTION per well / district / year / season, so you can
see WHICH wells or years are being silently excluded, not just that the
groups differ.

`rainfall_mm` is the nearest CHIRPS composite value at the well's cell within
a generous 400-day window (NaN if none) -- a proxy for the hydro-climatic
regime of dropped vs kept readings.

Outputs
-------
  reports/heldout_drop_reasons.csv        one row per held-out reading:
                                          [well_id, date, lat, lon, depth_m,
                                           drop_reason, drop_detail]
  reports/heldout_representativeness.csv  per-variable comparison table
  console summary                         reconciled counts + flagged biases

USAGE (from the repo root, after running rf_downscale_v2.py --train)
--------------------------------------------------------------------
  python src\downscaling\diagnose_heldout.py ^
      --config config\data_config.yaml ^
      --held_out_csv data\held_out_wells\held_out_ids.csv ^
      --interim_dir data\interim ^
      --kriged_dir data\interim\kriged_target_monthly ^
      --meta reports\rf_v2_model_meta.json

If --meta is given, windows/fallbacks/feature columns are read from it (so
the audit audits the EXACT configuration that produced your reported N);
otherwise defaults (90/120/0/0, baseline feature set) are used.

No model is loaded and no held-out row is scored -- this is a pure
data-coverage audit; it cannot leak anything into training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (
    GRID_SIZE, MONTHLY_SOURCES,
    find_nearest_raster, find_nearest_prior_kriged, load_raster_array,
    rolling_average_features, latlon_to_grid_cell, seasonal_features,
)
from rf_downscale_v2 import (
    BASE_FEATURE_COLUMNS, _season_label, compute_well_distance_grids,
    feature_columns,
)

RAINFALL_PROXY_WINDOW_DAYS = 400
SMD_FLAG_THRESHOLD = 0.25


# ---------------------------------------------------------------------
# Part 1: mutually-exclusive drop-reason tagging, validator-order
# ---------------------------------------------------------------------
def tag_drop_reasons(held_df: pd.DataFrame, interim_dir: Path, kriged_dir: Path,
                     bbox, meta: dict) -> pd.DataFrame:
    max_gap_days = int(meta.get("max_gap_days", 90))
    lag_max_gap_days = int(meta.get("lag_max_gap_days", 120))
    lag_fallback_days = int(meta.get("lag_fallback_days", 0))
    cov_fallback_days = int(meta.get("covariate_fallback_days", 0))
    with_terrain = bool(meta.get("with_terrain", False))
    with_distance = bool(meta.get("with_distance", False))
    with_lag_age = lag_fallback_days > 0
    with_cov_age = cov_fallback_days > 0
    with_spatial = bool(meta.get("with_spatial_coords", True))
    extra_rolling = tuple(meta.get("extra_rolling", ()))
    feat_cols = meta.get("feature_columns") or feature_columns(
        with_spatial, extra_rolling, with_terrain=with_terrain,
        with_distance=with_distance, with_lag_age=with_lag_age,
        with_cov_age=with_cov_age)

    roll_window = max(max_gap_days, cov_fallback_days) if cov_fallback_days else max_gap_days
    dist_grids = None
    if with_distance and Path(meta.get("training_wells_csv", "")).exists():
        dist_grids = compute_well_distance_grids(meta["training_wells_csv"], bbox)

    stack_cache: dict[str, tuple[str | None, str] | None] = {}

    def audit_stack(date_str: str) -> tuple[str | None, str]:
        """Replicates build_stack_v2's early-return ORDER. Returns
        (drop_reason, detail); (None, '') if the stack is complete."""
        # 1. monthly sources, in MONTHLY_SOURCES order -- first miss wins
        for source in MONTHLY_SOURCES:
            matched = find_nearest_raster(interim_dir / source, date_str, max_gap_days)
            if matched is None and cov_fallback_days > max_gap_days:
                matched = find_nearest_raster(interim_dir / source, date_str, cov_fallback_days)
            if matched is None:
                return (f"no_{source}_within_{cov_fallback_days or max_gap_days}d",
                        f"nearest composite exceeds window ({source})")
        # 2. static SRTM
        for sub in [("srtm", "elevation.tif"), ("srtm", "slope.tif")]:
            if not interim_dir.joinpath(*sub).exists():
                return "no_srtm_static", f"missing {interim_dir.joinpath(*sub)}"
        # 3. 3-month rolling chirps/gldas
        if rolling_average_features(interim_dir, date_str, sources=["chirps", "gldas"],
                                    n_months=3, max_gap_days=roll_window, min_months=2) is None:
            return "no_roll3_chirps_gldas", f"<2 of trailing 3 months within {roll_window}d"
        # 4. prior Kriged lag
        lag_path = find_nearest_prior_kriged(kriged_dir, date_str, lag_max_gap_days)
        if lag_path is None and lag_fallback_days > lag_max_gap_days:
            lag_path = find_nearest_prior_kriged(kriged_dir, date_str, lag_fallback_days)
        if lag_path is None:
            w = lag_fallback_days or lag_max_gap_days
            return f"no_kriged_lag_within_{w}d", "no PRIOR kriged surface in window"
        return None, ""

    rows = []
    n_eval = 0
    for _, row in held_df.iterrows():
        date_str = str(row["date"])
        reason, detail = None, ""
        if date_str not in stack_cache:
            stack_cache[date_str] = audit_stack(date_str)
        res = stack_cache[date_str]
        if res[0] is not None:
            reason, detail = res
        else:
            # date-level coverage OK -> check the per-cell feature vector
            # EXACTLY as validate_v2 does, in FEATURE_COLUMNS order.
            stack = build_stack_like_validator(
                interim_dir, date_str, max_gap_days, kriged_dir,
                lag_max_gap_days, lag_fallback_days, cov_fallback_days,
                with_terrain, dist_grids, extra_rolling)
            r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
            season = seasonal_features(date_str)
            for col in feat_cols:
                if col == "row_norm":
                    val = r / (GRID_SIZE - 1)
                elif col == "col_norm":
                    val = c / (GRID_SIZE - 1)
                elif col in ("month_sin", "month_cos"):
                    val = float(season[col])
                else:
                    if col not in stack:
                        reason, detail = f"missing_channel:{col}", "not in stack"
                        break
                    val = float(stack[col][r, c])
                if np.isnan(val):
                    reason, detail = f"nan_at_cell:{col}", f"NaN at cell ({r},{c})"
                    break
            if reason is None:
                reason = "evaluated"
                n_eval += 1
        rows.append({
            "well_id": row["well_id"], "date": date_str,
            "lat": row["lat"], "lon": row["lon"],
            "depth_m": row.get("depth_m", np.nan),
            "season": _season_label(int(date_str.split("-")[1])),
            "year": date_str[:4],
            "drop_reason": reason, "drop_detail": detail,
        })
    out = pd.DataFrame(rows)
    # RECONCILIATION -- the correctness test for this audit:
    assert len(out) == len(held_df), "audit lost rows -- reconciliation failed"
    return out


def build_stack_like_validator(interim_dir, date_str, max_gap_days, kriged_dir,
                               lag_max_gap_days, lag_fallback_days,
                               cov_fallback_days, with_terrain, dist_grids,
                               extra_rolling) -> dict | None:
    """Rebuilds the full stack (only called for dates that passed the audit's
    early-return chain, so None here is impossible in practice)."""
    from rf_downscale_v2 import build_stack_v2
    return build_stack_v2(
        interim_dir, date_str, max_gap_days, kriged_dir, lag_max_gap_days,
        extra_rolling, with_terrain=with_terrain,
        lag_fallback_days=lag_fallback_days,
        covariate_fallback_days=cov_fallback_days, dist_grids=dist_grids)


# ---------------------------------------------------------------------
# Part 2: representativeness -- KS / chi-square / SMD
# ---------------------------------------------------------------------
def _ks_compare(eval_vals: np.ndarray, skip_vals: np.ndarray) -> dict:
    eval_vals = eval_vals[~np.isnan(eval_vals)]
    skip_vals = skip_vals[~np.isnan(skip_vals)]
    if len(eval_vals) < 2 or len(skip_vals) < 2:
        return {"ks_stat": np.nan, "ks_p": np.nan,
                "mean_eval": np.nan, "mean_skip": np.nan, "smd": np.nan,
                "n_eval": len(eval_vals), "n_skip": len(skip_vals)}
    ks = stats.ks_2samp(eval_vals, skip_vals)
    pooled_sd = np.sqrt(((len(eval_vals) - 1) * eval_vals.var(ddof=1)
                         + (len(skip_vals) - 1) * skip_vals.var(ddof=1))
                        / (len(eval_vals) + len(skip_vals) - 2))
    smd = ((eval_vals.mean() - skip_vals.mean()) / pooled_sd) if pooled_sd > 0 else np.nan
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mean_eval": float(eval_vals.mean()), "mean_skip": float(skip_vals.mean()),
            "smd": float(smd), "n_eval": len(eval_vals), "n_skip": len(skip_vals)}


def _chi2_compare(tagged: pd.DataFrame, col: str) -> dict:
    ct = pd.crosstab(tagged[col], tagged["is_evaluated"])
    if ct.shape[1] < 2 or ct.values.sum() == 0:
        return {"chi2_stat": np.nan, "chi2_p": np.nan}
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return {"chi2_stat": float(chi2), "chi2_p": float(p)}


def representativeness(tagged: pd.DataFrame, interim_dir: Path, bbox) -> pd.DataFrame:
    tagged = tagged.copy()
    tagged["is_evaluated"] = (tagged["drop_reason"] == "evaluated").astype(int)
    ev = tagged[tagged["is_evaluated"] == 1]
    sk = tagged[tagged["is_evaluated"] == 0]
    out_rows = []

    # continuous variables
    rainfall = rainfall_proxy(tagged, interim_dir, bbox)
    tagged["rainfall_mm"] = rainfall
    ev = tagged[tagged["is_evaluated"] == 1]
    sk = tagged[tagged["is_evaluated"] == 0]
    for col, note in [("depth_m", "groundwater depth (m)"),
                      ("lat", "latitude"), ("lon", "longitude"),
                      ("rainfall_mm", "nearest CHIRPS composite at cell (<=400d)")]:
        cmp_res = _ks_compare(ev[col].to_numpy(float), sk[col].to_numpy(float))
        out_rows.append({"variable": col, "type": "continuous", "note": note,
                         **cmp_res,
                         "smd_flag": ("|SMD|>0.25"
                                      if abs(cmp_res.get("smd", np.nan) or np.nan) > SMD_FLAG_THRESHOLD
                                      else "")})

    # categorical variables (chi-square) + per-level evaluated fraction
    for col in ["season", "year", "well_id"] + (["district"] if "district" in tagged.columns else []):
        cmp_res = _chi2_compare(tagged, col)
        frac = (tagged.groupby(col, observed=True)["is_evaluated"].mean()
                .sort_values())
        worst = frac.head(3)
        out_rows.append({
            "variable": col, "type": "categorical",
            "note": f"evaluated fraction range {frac.min():.0%}..{frac.max():.0%}; "
                    f"worst levels: " + ", ".join(f"{k}={v:.0%}" for k, v in worst.items()),
            **cmp_res, "smd": np.nan, "smd_flag": ""})
    return pd.DataFrame(out_rows)


def rainfall_proxy(tagged: pd.DataFrame, interim_dir: Path, bbox) -> np.ndarray:
    cache: dict[str, np.ndarray | None] = {}
    vals = np.full(len(tagged), np.nan)
    for i, (_, row) in enumerate(tagged.iterrows()):
        date_str = str(row["date"])
        if date_str not in cache:
            matched = find_nearest_raster(interim_dir / "chirps", date_str,
                                          RAINFALL_PROXY_WINDOW_DAYS)
            cache[date_str] = load_raster_array(matched) if matched is not None else None
        arr = cache[date_str]
        if arr is None:
            continue
        r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
        v = float(arr[r, c])
        vals[i] = v if not np.isnan(v) else np.nan
    return vals


# ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Audit WHY held-out readings were skipped and whether the "
                    "evaluated subset is representative. Pure data audit -- no model.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    p.add_argument("--meta", default="reports/rf_v2_model_meta.json",
                   help="meta json written by rf_downscale_v2.py --train; windows/"
                        "fallbacks/feature columns are read from it so the audit "
                        "matches the exact configuration that produced your N.")
    p.add_argument("--reasons_out", default="reports/heldout_drop_reasons.csv")
    p.add_argument("--repr_out", default="reports/heldout_representativeness.csv")
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])
    meta = {}
    if Path(args.meta).exists():
        with open(args.meta) as f:
            meta = json.load(f)
    else:
        print(f"[diagnose] WARNING: meta not found at {args.meta} -- using defaults "
              f"(90/120/0/0 windows, baseline features). Point --meta at the file "
              f"written by rf_downscale_v2.py --train for an exact audit.")

    held_df = pd.read_csv(args.held_out_csv)
    print(f"[diagnose] Held-out CSV rows: {len(held_df)}")
    tagged = tag_drop_reasons(held_df, Path(args.interim_dir),
                              Path(args.kriged_dir), bbox, meta)

    # Part 1 outputs + reconciliation (the correctness test)
    counts = tagged["drop_reason"].value_counts()
    n_eval = int(counts.get("evaluated", 0))
    assert counts.sum() == len(tagged) == len(held_df), "reconciliation failed"
    Path(args.reasons_out).parent.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(args.reasons_out, index=False)

    print()
    print("=" * 72)
    print(f"DROP-REASON AUDIT (mutually exclusive; sums to {len(tagged)} rows)")
    print("=" * 72)
    for reason, n in counts.items():
        print(f"  {reason:<42} {n:>7}")
    print(f"  {'TOTAL':<42} {counts.sum():>7}")
    print(f"  reconciliation: counts.sum()==CSV rows=={len(held_df)}  OK")
    print(f"  cross-check: 'evaluated'={n_eval} must equal the validator's "
          f"reported 'Evaluated N' for the same meta -- if it differs, the "
          f"validator ran with different windows/flags than this meta.")
    print("=" * 72)

    # Part 2: representativeness
    print()
    print("[diagnose] Testing skipped-vs-evaluated representativeness "
          "(KS / chi-square / SMD)...")
    repr_df = representativeness(tagged, Path(args.interim_dir), bbox)
    Path(args.repr_out).parent.mkdir(parents=True, exist_ok=True)
    repr_df.to_csv(args.repr_out, index=False)

    print()
    print("=" * 72)
    print("REPRESENTATIVENESS: evaluated vs skipped")
    print("=" * 72)
    for _, r in repr_df.iterrows():
        if r["type"] == "continuous":
            print(f"  {r['variable']:<14} KS={r['ks_stat']:.3f} (p={r['ks_p']:.2e})  "
                  f"mean_eval={r['mean_eval']:.2f} mean_skip={r['mean_skip']:.2f}  "
                  f"SMD={r['smd']:+.3f} {r['smd_flag']}")
        else:
            print(f"  {r['variable']:<14} chi2={r['chi2_stat']:.1f} (p={r['chi2_p']:.2e})  "
                  f"{r['note']}")
    flagged = repr_df[repr_df["smd_flag"].astype(str).str.len() > 0]
    print()
    if len(flagged):
        print("  SELECTION-BIAS FLAGS (|SMD| > 0.25): "
              + ", ".join(flagged["variable"].tolist()))
        print("  -> the evaluated subset is NOT representative on these variables;")
        print("     state the restriction explicitly in the methods/report.")
    else:
        print("  No |SMD| > 0.25 flags -- no strong evidence of selection bias "
              "on the continuous variables tested (still report the drop-reason "
              "table verbatim).")
    print("=" * 72)
    print(f"[diagnose] Saved per-row reasons -> {args.reasons_out}")
    print(f"[diagnose] Saved comparison table -> {args.repr_out}")


if __name__ == "__main__":
    main()
