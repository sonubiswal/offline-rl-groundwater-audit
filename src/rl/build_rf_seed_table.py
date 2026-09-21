"""
src/rl/build_rf_seed_table.py

Phase 4 prerequisite: generate_offline_dataset.py needs real RF-predicted
groundwater levels tagged with lat/lon to seed RL initial states from
(see its docstring). Phase 2's validate_downscale.py only saves
predictions at the ~4,869 held-out well POINTS -- too sparse and the
wrong purpose (that's a validation artifact, not a spatial seed pool).

This script instead runs the trained RF model across every grid cell,
for every available covariate date, producing a full (grid_cell, date)
prediction table with real lat/lon -- the same spatial coverage RF is
actually meant to provide (Section 16.2(a) framing: RF predicts
"anywhere," not just at wells).

Output columns: [lat, lon, month, gw_level_pred, chirps_roll3]
Saved to: reports/rf_grid_predictions.csv (default)

Usage:
    python src/rl/build_rf_seed_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downscaling"))
from feature_utils import build_feature_stack_for_date, seasonal_features  # noqa: E402


def grid_cell_to_latlon(row: int, col: int, bbox: tuple, n_grid: int = 64) -> tuple[float, float]:
    """Inverse of latlon_to_grid_cell -- maps a grid index back to the
    cell's center lat/lon, assuming the same uniform bbox partition used
    throughout this pipeline."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (col + 0.5) / n_grid * (max_lon - min_lon)
    lat = min_lat + (row + 0.5) / n_grid * (max_lat - min_lat)
    return lat, lon


def list_available_dates(kriged_dir: str) -> list[str]:
    return sorted({p.stem for p in Path(kriged_dir).glob("*.npy")})


def main(
    rf_model_path: str = "data/processed/rf_downscale_model.joblib",
    interim_dir: str = "data/interim",
    kriged_dir: str = "data/interim/kriged_target_monthly",
    config_path: str = "config/data_config.yaml",
    out_path: str = "reports/rf_grid_predictions.csv",
    max_dates: int | None = None,
    subsample_cells: int | None = None,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    bbox = tuple(config["region"]["bbox"])

    from rf_downscale import FEATURE_COLUMNS

    rf_model = joblib.load(rf_model_path)
    dates = list_available_dates(kriged_dir)
    if max_dates:
        dates = dates[:max_dates]
    print(f"[build_rf_seed_table] Running RF inference across the full grid "
          f"for {len(dates)} dates...")

    all_rows = []
    for d in dates:
        stack = build_feature_stack_for_date(interim_dir, d, kriged_dir=kriged_dir, lag_max_gap_days=120)
        if stack is None:
            continue

        season = seasonal_features(d)
        month = int(d.split("-")[1])

        h, w = next(iter(stack.values())).shape
        cell_indices = [(r, c) for r in range(h) for c in range(w)]
        if subsample_cells:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(cell_indices), size=min(subsample_cells, len(cell_indices)), replace=False)
            cell_indices = [cell_indices[i] for i in idx]

        feat_rows, meta_rows = [], []
        for r, c in cell_indices:
            feat_dict = {col: stack[col][r, c] for col in stack.keys() if col in FEATURE_COLUMNS}
            feat_dict.update(season)
            if any(col not in feat_dict or np.isnan(feat_dict[col]) for col in FEATURE_COLUMNS):
                continue
            feat_rows.append([feat_dict[col] for col in FEATURE_COLUMNS])
            lat, lon = grid_cell_to_latlon(r, c, bbox, n_grid=h)
            meta_rows.append({
                "lat": lat, "lon": lon, "month": month,
                "chirps_roll3": feat_dict.get("chirps_roll3", np.nan),
            })

        if not feat_rows:
            continue

        preds = rf_model.predict(np.array(feat_rows))
        for meta, pred in zip(meta_rows, preds):
            all_rows.append({**meta, "gw_level_pred": float(pred)})

    result = pd.DataFrame(all_rows)
    print(f"[build_rf_seed_table] Built {len(result)} (grid_cell, date) prediction rows "
          f"across {len(dates)} candidate dates.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"[build_rf_seed_table] Saved -> {out_path}")


if __name__ == "__main__":
    main()