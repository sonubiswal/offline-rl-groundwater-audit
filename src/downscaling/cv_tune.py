"""
cv_tune.py

Hyperparameter exploration for the RF downscaling model, using ONLY
training-derived data (the Kriged surfaces built from training wells).
This NEVER reads data/held_out_wells/held_out_ids.csv -- that file should
be touched exactly once, at the very end, by validate_downscale.py, after
a final model has been chosen here.

Why this exists: repeatedly checking held-out R² and adjusting the model
in response quietly turns the held-out set into a de facto second
training set, which invalidates it as an honest generalization estimate --
a serious problem if this work is going into a paper. Proper practice is
to do all exploration/tuning via cross-validation on the training data
only, and touch the true held-out set a single time at the end.

Folding strategy: GroupKFold grouped by DATE, not a plain random K-fold
over individual (date, cell) rows. Grid cells within the same date are
spatially autocorrelated (neighboring cells look similar), so a random
split could put nearly-identical cells from the same date's surface into
both the train and validation fold of a split -- leaking information and
producing an optimistic CV score. Grouping by date keeps each date's
cells entirely in one fold or the other, which is a fair (if slightly
more conservative) estimate of held-out performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_downscale import build_training_table, FEATURE_COLUMNS


DEFAULT_PARAM_GRID = [
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 500, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 8,  "min_samples_split": 20, "min_samples_leaf": 10},
    {"n_estimators": 300, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5,  "min_samples_leaf": 2},
]


def assign_spatial_block(row: int, col: int, n_blocks_x: int = 8, n_blocks_y: int = 8, grid_size: int = 64) -> str:
    """Assign a grid cell to a spatial block for spatial-holdout CV.

    Divides the 64x64 grid into an n_blocks_x by n_blocks_y tiling (default
    8x8 blocks of 8x8 cells each) and returns a block id string. Used to
    group CV folds by LOCATION rather than by date -- see module docstring
    on why this better matches what validate_downscale.py actually
    measures (spatial generalization within known time periods, since
    held-out wells were chosen by a spatial split, not a temporal one).
    """
    block_x = min(int(col / grid_size * n_blocks_x), n_blocks_x - 1)
    block_y = min(int(row / grid_size * n_blocks_y), n_blocks_y - 1)
    return f"{block_x}_{block_y}"


def cross_validate_config(
    df: pd.DataFrame,
    params: dict,
    n_splits: int = 5,
    random_state: int = 42,
    group_by: str = "spatial_block",
) -> dict:
    """Run GroupKFold CV for one hyperparameter config.

    group_by controls what defines a fold group:
      - "spatial_block": holds out contiguous spatial regions, mixing
        dates freely within each fold. This is the RECOMMENDED default --
        it mirrors what validate_downscale.py actually measures, since
        held-out wells come from a spatial (not temporal) split, so most
        held-out readings fall on dates the model already saw via other
        training wells. Use this for hyperparameter selection.
      - "date": holds out entire time periods, testing whether the model
        can extrapolate to genuinely unseen dates. This is a HARDER, and
        for our deployment scenario, less relevant test -- but still
        worth reporting once as an honest secondary finding about the
        model's temporal generalization (or lack thereof), since it may
        matter if the model is ever applied to future/unseen dates rather
        than used to interpolate within the historical record.

    Returns per-fold and summary R²/RMSE, purely from training-derived
    Kriged data -- no held-out wells involved anywhere in this function.
    """
    X = df[FEATURE_COLUMNS].values
    y = df["target"].values

    if group_by == "spatial_block":
        groups = df.apply(lambda r: assign_spatial_block(int(r["row"]), int(r["col"])), axis=1).values
    elif group_by == "date":
        groups = df["date"].values
    else:
        raise ValueError(f"Unknown group_by: {group_by!r}. Use 'spatial_block' or 'date'.")

    n_unique_groups = len(set(groups))
    actual_splits = min(n_splits, n_unique_groups)
    if actual_splits < 2:
        raise ValueError(
            f"Only {n_unique_groups} unique group(s) ({group_by}) -- "
            f"need at least 2 for cross-validation."
        )

    gkf = GroupKFold(n_splits=actual_splits)
    fold_r2, fold_rmse = [], []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        r2 = r2_score(y[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
        fold_r2.append(r2)
        fold_rmse.append(rmse)

    return {
        "params": params,
        "group_by": group_by,
        "n_splits": actual_splits,
        "fold_r2": fold_r2,
        "fold_rmse": fold_rmse,
        "mean_r2": np.mean(fold_r2),
        "std_r2": np.std(fold_r2),
        "mean_rmse": np.mean(fold_rmse),
    }


def run(
    kriged_dir: str = "data/interim/kriged_target_monthly",
    interim_dir: str = "data/interim",
    max_gap_days: int = 90,
    lag_max_gap_days: int = 120,
    param_grid: list[dict] | None = None,
    n_splits: int = 5,
    group_by: str = "spatial_block",
    results_out: str = "reports/cv_tuning_results.csv",
) -> pd.DataFrame:
    param_grid = param_grid or DEFAULT_PARAM_GRID

    print("[cv_tune] Building training table from Kriged (training-well-derived) surfaces...")
    print("[cv_tune] NOTE: this script never reads held_out_ids.csv. Held-out wells are")
    print("[cv_tune] not touched anywhere in this process.")
    print(f"[cv_tune] Grouping strategy: {group_by!r} "
          f"({'spatial regions, dates mixed -- matches validate_downscale.py' if group_by=='spatial_block' else 'entire time periods -- tests temporal extrapolation'})")
    print()
    df = build_training_table(kriged_dir, interim_dir, max_gap_days, lag_max_gap_days)

    n_dates = df["date"].nunique()
    print(f"[cv_tune] Training table: {len(df)} samples across {n_dates} unique dates.")
    if n_dates < 5:
        print(f"[cv_tune] WARNING: only {n_dates} unique dates available. CV results with")
        print("[cv_tune] this few dates will have high variance between folds -- treat any")
        print("[cv_tune] single config's advantage over another with appropriate skepticism")
        print("[cv_tune] unless the gap is large relative to the std_r2 spread.")
    print()

    results = []
    for i, params in enumerate(param_grid):
        print(f"[cv_tune] Config {i+1}/{len(param_grid)}: {params}")
        result = cross_validate_config(df, params, n_splits=n_splits, group_by=group_by)
        print(f"    Mean CV R²: {result['mean_r2']:.3f} (± {result['std_r2']:.3f} across {result['n_splits']} folds)")
        print(f"    Mean CV RMSE: {result['mean_rmse']:.3f}")
        results.append(result)
        print()

    results_df = pd.DataFrame([
        {**r["params"], "mean_r2": r["mean_r2"], "std_r2": r["std_r2"],
         "mean_rmse": r["mean_rmse"], "n_splits": r["n_splits"]}
        for r in results
    ])
    results_df = results_df.sort_values("mean_r2", ascending=False).reset_index(drop=True)

    Path(results_out).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_out, index=False)

    print("=" * 70)
    print("CV TUNING RESULTS (sorted by mean CV R², best first)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print()
    best = results_df.iloc[0]
    print(f"Best config by mean CV R²: {dict(best[['n_estimators','max_depth','min_samples_split','min_samples_leaf']])}")
    print()
    print("Next step: retrain rf_downscale.py's train_rf() with this config (edit the")
    print("defaults or add CLI args), then run validate_downscale.py EXACTLY ONCE against")
    print("the true held-out set. That result -- not this CV number -- is what should be")
    print("reported as your final validation metric.")
    print(f"[cv_tune] Saved full results -> {results_out}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-validate RF hyperparameters using training-derived data only "
                     "(never touches held-out wells)."
    )
    parser.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--max_gap_days", type=int, default=90)
    parser.add_argument("--lag_max_gap_days", type=int, default=120)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--group_by", choices=["spatial_block", "date"], default="spatial_block",
                         help="'spatial_block' (recommended, matches validate_downscale.py's "
                              "spatial held-out test) or 'date' (tests temporal extrapolation "
                              "to entirely unseen time periods -- harder, worth running once as "
                              "a secondary honest diagnostic, not for hyperparameter selection).")
    parser.add_argument("--results_out", default="reports/cv_tuning_results.csv")
    args = parser.parse_args()

    run(
        kriged_dir=args.kriged_dir,
        interim_dir=args.interim_dir,
        max_gap_days=args.max_gap_days,
        lag_max_gap_days=args.lag_max_gap_days,
        n_splits=args.n_splits,
        group_by=args.group_by,
        results_out=args.results_out,
    )