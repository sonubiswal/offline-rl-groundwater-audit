"""
validate_downscale.py

The critical independent check for Phase 2: compares the trained RF
downscaling model's predictions against HELD-OUT wells -- wells never
used in Kriging (kriging.py) or RF training (rf_downscale.py). This is
what the roadmap calls the number that matters, as opposed to training-
set R² against the Kriged surface (which is expected to look good almost
by construction).

Reports RMSE, MAE, and R² for this independent check, and saves a
predicted-vs-actual scatter plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import (
    build_feature_stack_for_date, latlon_to_grid_cell, spatial_features, seasonal_features,
    compute_monthly_climatology,
)
from rf_downscale import FEATURE_COLUMNS


def validate(
    model_path: str,
    held_out_csv: str,
    interim_dir: str,
    bbox: tuple[float, float, float, float],
    kriged_dir: str,
    max_gap_days: int = 45,
    use_rainfall_deficit: bool = True,
) -> pd.DataFrame:
    """Run the trained RF model against every held-out well reading.

    kriged_dir is passed through to build_feature_stack_for_date() so the
    lag feature (target_lag1) is built the SAME way as during training --
    from the training-well-derived Kriged surface, never from the held-out
    well's own readings. This is what keeps the lag feature leakage-free.

    use_rainfall_deficit must match whatever the model was trained with --
    the climatology itself is computed purely from CHIRPS satellite data
    (no well information at all), so there's no leakage risk either way,
    but the FEATURE_COLUMNS the model expects must be built consistently.

    Returns
    -------
    pd.DataFrame with columns [well_id, date, lat, lon, actual, predicted]
    for every reading where a complete feature vector could be built.
    Readings where covariates couldn't be matched within max_gap_days are
    skipped and logged (not silently dropped).
    """
    model = joblib.load(model_path)
    held_df = pd.read_csv(held_out_csv)

    climatology = None
    if use_rainfall_deficit:
        climatology = compute_monthly_climatology(interim_dir, source="chirps")

    results = []
    n_skipped = 0
    # Cache feature stacks per date -- many readings share the same date,
    # no need to rebuild the stack for each one.
    feature_stack_cache: dict[str, dict | None] = {}

    for _, row in held_df.iterrows():
        date_str = str(row["date"])
        if date_str not in feature_stack_cache:
            feature_stack_cache[date_str] = build_feature_stack_for_date(
                interim_dir, date_str, max_gap_days, kriged_dir=kriged_dir,
                climatology=climatology,
            )
        stack = feature_stack_cache[date_str]

        if stack is None:
            n_skipped += 1
            continue

        r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
        feat_dict = {col: stack[col][r, c] for col in stack.keys()}
        if any(np.isnan(v) for v in feat_dict.values()):
            n_skipped += 1
            continue
        feat_dict.update(spatial_features(r, c))
        feat_dict.update(seasonal_features(date_str))
        feat_vals = [feat_dict[col] for col in FEATURE_COLUMNS]

        pred = model.predict(np.array(feat_vals).reshape(1, -1))[0]
        results.append({
            "well_id": row["well_id"], "date": date_str,
            "lat": row["lat"], "lon": row["lon"],
            "actual": row["depth_m"], "predicted": pred,
        })

    print(f"[validate_downscale] Evaluated {len(results)} held-out readings, "
          f"skipped {n_skipped} (incomplete covariate/rolling/lag coverage for that date).")

    return pd.DataFrame(results)


def compute_metrics(results_df: pd.DataFrame) -> dict:
    y_true = results_df["actual"].values
    y_pred = results_df["predicted"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(results_df)}


def plot_predicted_vs_actual(results_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(results_df["actual"], results_df["predicted"], alpha=0.4, s=15)

    lims = [
        min(results_df["actual"].min(), results_df["predicted"].min()),
        max(results_df["actual"].max(), results_df["predicted"].max()),
    ]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual depth (m) -- held-out CGWB wells")
    ax.set_ylabel("RF-downscaled prediction (m)")
    ax.set_title("RF Downscaling: Predicted vs Actual\n(independent held-out wells)")
    ax.legend()
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    config_path: str = "config/data_config.yaml",
    model_path: str = "data/processed/rf_downscale_model.joblib",
    held_out_csv: str = "data/held_out_wells/held_out_ids.csv",
    interim_dir: str = "data/interim",
    kriged_dir: str = "data/interim/kriged_target_monthly",
    results_out: str = "reports/rf_downscale_validation.csv",
    plot_out: str = "reports/rf_downscale_predicted_vs_actual.png",
    use_rainfall_deficit: bool = True,
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    bbox = tuple(config["region"]["bbox"])

    results_df = validate(model_path, held_out_csv, interim_dir, bbox, kriged_dir,
                           use_rainfall_deficit=use_rainfall_deficit)

    if len(results_df) == 0:
        raise RuntimeError(
            "No held-out readings could be evaluated -- check covariate coverage "
            "and that the model/held-out CSV paths are correct."
        )

    Path(results_out).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_out, index=False)

    metrics = compute_metrics(results_df)
    print()
    print("=" * 60)
    print("INDEPENDENT VALIDATION (held-out CGWB wells)")
    print("=" * 60)
    print(f"N readings evaluated: {metrics['n']}")
    print(f"RMSE: {metrics['rmse']:.3f} m")
    print(f"MAE:  {metrics['mae']:.3f} m")
    print(f"R²:   {metrics['r2']:.3f}")
    print()
    print("This R² is expected to be LOWER than the training-set R² reported by "
          "rf_downscale.py -- that's expected and defensible, not a failure "
          "(see roadmap Phase 2 Definition of Done).")

    plot_predicted_vs_actual(results_df, plot_out)
    print(f"[validate_downscale] Saved results -> {results_out}")
    print(f"[validate_downscale] Saved plot -> {plot_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate RF downscaling model against independent held-out wells.")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--model", default="data/processed/rf_downscale_model.joblib")
    parser.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--kriged_dir", default="data/interim/kriged_target_monthly")
    parser.add_argument("--results_out", default="reports/rf_downscale_validation.csv")
    parser.add_argument("--plot_out", default="reports/rf_downscale_predicted_vs_actual.png")
    parser.add_argument("--no_rainfall_deficit", action="store_true",
                         help="Must match whatever rf_downscale.py was trained with.")
    args = parser.parse_args()

    run(args.config, args.model, args.held_out_csv, args.interim_dir, args.kriged_dir, args.results_out, args.plot_out,
        use_rainfall_deficit=not args.no_rainfall_deficit)