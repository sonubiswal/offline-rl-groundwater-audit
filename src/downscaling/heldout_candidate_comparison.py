"""
heldout_candidate_comparison.py

Item 8: does the spatial-CV-selected candidate also win on held-out?

FIX vs both earlier drafts: no manual (date, r, c) join. The training-table
dates are kriged-surface campaign dates (~23 unique months); held-out
reading dates are actual CGWB well measurement dates (~145 unique dates).
An exact-date join between the two only matches by coincidence (4/14,062
in the last run). This script never joins on date at all -- it reuses
rf_downscale_v2.validate_v2, the same nearest-date covariate matching
already used (and already verified working) in fast_spatial_holdout.py.

FOR EACH CANDIDATE (rf_depth, rf_residual, hgbr_depth, hgbr_residual):
  1. Fit on the FULL training table (same table, same feature set for
     all 4 -- only target_mode differs, matching how spatial_block_cv
     already compares them).
  2. Evaluate with validate_v2 on the FULL held-out CSV (no quadrant
     restriction) -- same aligned covariate window, same kriging-only /
     persistence baselines as every other validation run in this repo.

OUTPUT: reports/heldout_candidate_comparison.csv -- one row per candidate
plus kriging-only and persistence, directly comparable to the CV table
already quoted for the paper.

USAGE
-----
  python heldout_candidate_comparison.py
  python heldout_candidate_comparison.py --meta_out reports/rf_v2_model_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from rf_downscale_v2 import (
    build_stack_v2, build_training_table_v2, compute_well_distance_grids,
    make_hgbr, make_rf, metrics_block, validate_v2,
)


def check_feature_schema(meta: dict, kriged_dir, interim_dir, dist_grids) -> None:
    """Fail fast, with a readable per-date message, if meta['feature_columns']
    does not match what build_stack_v2 actually produces for ANY kriged date
    under the CURRENTLY-imported feature_utils.py. Checking every date (not
    just the first) matters here: build_stack_v2 has no code path that
    returns a partial dict -- it either returns None or a dict with every
    MONTHLY_SOURCES key set -- so a missing key on some date is either a rare,
    date-specific bug (e.g. a raster file that loads but produces a
    differently-named key) or a caching issue, and naming the exact date(s)
    turns a 20-30 min blind re-run into a few seconds of diagnosis."""
    from pathlib import Path as _P
    npy_files = sorted(_P(kriged_dir).glob("*.npy"))
    if not npy_files:
        return  # let build_training_table_v2 raise its own "no files" error
    skip = {"row_norm", "col_norm", "month_sin", "month_cos"}
    bad: dict[str, list[str]] = {}
    checked, skipped_none = 0, 0
    for npy_path in npy_files:
        date_str = npy_path.stem
        stack = build_stack_v2(
            interim_dir, date_str, meta["max_gap_days"], kriged_dir, meta["lag_max_gap_days"],
            tuple(meta.get("extra_rolling", ())), with_terrain=bool(meta.get("with_terrain", False)),
            lag_fallback_days=int(meta.get("lag_fallback_days", 0)),
            covariate_fallback_days=int(meta.get("covariate_fallback_days", 0)), dist_grids=dist_grids)
        if stack is None:
            skipped_none += 1
            continue
        checked += 1
        missing = [c for c in meta["feature_columns"] if c not in skip and c not in stack]
        if missing:
            bad[date_str] = missing
    print(f"[item8] Schema check: {checked} dates produced a stack, "
          f"{skipped_none} returned None (coverage gap, not a schema issue).")
    if bad:
        sample = list(bad.items())[:5]
        detail = "; ".join(f"{d}: missing {m}" for d, m in sample)
        more = f" (+{len(bad) - 5} more dates)" if len(bad) > 5 else ""
        raise SystemExit(
            f"[item8] SCHEMA MISMATCH on {len(bad)}/{checked} dates: {detail}{more}. "
            f"build_stack_v2 has no code path that returns a dict missing a "
            f"MONTHLY_SOURCES key, so this points to either (a) meta['feature_columns'] "
            f"not matching the installed feature_utils.py after all, or (b) a real "
            f"date-specific bug in build_stack_v2/find_nearest_raster worth looking at "
            f"directly for these dates before re-running the full comparison.")

CANDIDATES = [
    ("rf_depth", "depth"),
    ("rf_residual", "residual"),
    ("hgbr_depth", "depth"),
    ("hgbr_residual", "residual"),
]


def make_factory(name: str, meta: dict):
    if name.startswith("rf"):
        n_estimators = meta.get("model_params", {}).get("n_estimators", 300)
        max_depth = meta.get("model_params", {}).get("max_depth", 15)
        return lambda: make_rf(n_estimators, max_depth)
    return make_hgbr


def run(args) -> None:
    with open(args.config) as f:
        bbox = tuple(yaml.safe_load(f)["region"]["bbox"])
    with open(args.meta_out) as f:
        meta = json.load(f)

    feat_cols = meta["feature_columns"]
    extra_rolling = tuple(meta.get("extra_rolling", ()))
    with_terrain = bool(meta.get("with_terrain", False))
    lag_fallback_days = int(meta.get("lag_fallback_days", 0))
    covariate_fallback_days = int(meta.get("covariate_fallback_days", 0))

    dist_grids = None
    if meta.get("with_distance"):
        dist_grids = compute_well_distance_grids(meta["training_wells_csv"], bbox)

    check_feature_schema(meta, args.kriged_dir, args.interim_dir, dist_grids)

    print(f"[item8] Building the training table once ({len(feat_cols)} features)...")
    df = build_training_table_v2(
        args.kriged_dir, args.interim_dir, feat_cols,
        max_gap_days=meta["max_gap_days"], lag_max_gap_days=meta["lag_max_gap_days"],
        extra_rolling=extra_rolling, with_terrain=with_terrain,
        lag_fallback_days=lag_fallback_days,
        covariate_fallback_days=covariate_fallback_days, dist_grids=dist_grids)
    print(f"[item8] Training table: {len(df)} rows.")

    rows = []
    baselines_done = False
    for name, mode in CANDIDATES:
        print(f"\n[item8] Fitting {name} (target_mode={mode}) on {len(df)} rows...")
        model = make_factory(name, meta)()
        y = df["target"].values if mode == "depth" else (df["target"] - df["lag"]).values
        model.fit(df[feat_cols].values, y)

        cand_meta = dict(meta)
        cand_meta["target_mode"] = mode
        cand_meta["feature_columns"] = feat_cols

        results = validate_v2(model, cand_meta, args.held_out_csv, args.interim_dir,
                              args.kriged_dir, bbox, dist_grids=dist_grids)
        if results.empty:
            raise RuntimeError(f"{name}: 0 held-out readings evaluated -- check paths/coverage.")

        m = metrics_block(results["actual"].values, results["predicted"].values)
        rows.append({"candidate": name, "target_mode": mode, **m})
        print(f"[item8] {name:>14}  N={m['n']:>5}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

        if not baselines_done:
            m_kr = metrics_block(results["actual"].values, results["baseline_kriging_only"].values)
            m_pe = metrics_block(results["actual"].values, results["baseline_persistence_lag"].values)
            rows.append({"candidate": "kriging_only", "target_mode": "n/a", **m_kr})
            rows.append({"candidate": "persistence", "target_mode": "n/a", **m_pe})
            baselines_done = True

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print()
    print("=" * 78)
    print("HELD-OUT MODEL COMPARISON (item 8)")
    print("=" * 78)
    print(out.to_string(index=False))
    best = out[~out["candidate"].isin(["kriging_only", "persistence"])].sort_values("rmse").iloc[0]
    print(f"\nbest held-out: {best['candidate']}  RMSE={best['rmse']:.4f}  R2={best['r2']:.4f}")
    print(f"CV-selected candidate: {meta['candidate']}  "
          f"({'MATCHES' if best['candidate'] == meta['candidate'] else 'DOES NOT MATCH'} held-out best)")
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Item 8: 4-candidate held-out comparison via validate_v2.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    p.add_argument("--interim_dir", default="data/interim")
    p.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    p.add_argument("--meta_out", default="reports/rf_v2_model_meta.json",
                   help="Meta json defining the feature set / windows shared by all "
                        "4 candidates -- use the same run whose CV table you're quoting.")
    p.add_argument("--out", default="reports/heldout_candidate_comparison.csv")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()