"""
validate_with_residual_correction.py

The final, single held-out check for Phase 2.5: RF (with the rainfall-
deficit feature) + Residual Kriging correction, evaluated against the
TRUE held-out wells exactly once.

The residual-Kriging correction surface is built from ALL training wells
(not folded -- leave_some_wells_out_cv already did the folded, training-
only test that justified running this final check at all). This script
is the production-configuration evaluation: it should be run once, and
its result is what gets reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downscaling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from feature_utils import compute_monthly_climatology
from kriging import krige_wells_to_grid
from src.experiments.residual_kriging import compute_rf_predictions_at_wells


def run(
    config_path: str = "config/data_config.yaml",
    rf_model_path: str = "data/processed/rf_downscale_model.joblib",
    train_wells_csv: str = "data/interim/train_well_ids.csv",
    held_out_csv: str = "data/held_out_wells/held_out_ids.csv",
    interim_dir: str = "data/interim",
    kriged_dir: str = "data/interim/kriged_target_monthly",
    min_wells_for_correction: int = 5,
    use_rainfall_deficit: bool = True,
    results_out: str = "reports/rf_residual_krig_validation.csv",
    plot_out: str = "reports/rf_residual_krig_predicted_vs_actual.png",
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    bbox = tuple(config["region"]["bbox"])

    from rf_downscale import FEATURE_COLUMNS

    feature_columns = FEATURE_COLUMNS if use_rainfall_deficit else [
        c for c in FEATURE_COLUMNS if c != "chirps_cum12_anomaly"
    ]

    rf_model = joblib.load(rf_model_path)
    climatology = compute_monthly_climatology(interim_dir, source="chirps") if use_rainfall_deficit else None

    # Step 1: RF residuals at ALL training wells (production correction source)
    print("[final_validation] Computing RF predictions at all training wells "
          "(builds the residual-correction source)...")
    train_df = pd.read_csv(train_wells_csv)
    train_preds = compute_rf_predictions_at_wells(
        rf_model, feature_columns, train_df, interim_dir, kriged_dir, bbox, climatology,
    )
    print(f"[final_validation] {len(train_preds)}/{len(train_df)} training-well readings "
          f"available to build the correction surface.")

    # Step 2: RF predictions at the TRUE held-out wells (touched once, here)
    print("[final_validation] Computing RF predictions at the TRUE held-out wells "
          "(this is the single final check)...")
    held_df = pd.read_csv(held_out_csv)
    held_preds = compute_rf_predictions_at_wells(
        rf_model, feature_columns, held_df, interim_dir, kriged_dir, bbox, climatology,
    )
    print(f"[final_validation] {len(held_preds)}/{len(held_df)} held-out readings evaluated.")

    if len(held_preds) == 0:
        raise RuntimeError("No held-out readings could be evaluated -- check coverage.")

    # Step 3: apply residual-Kriging correction, caching the Kriged surface
    # per unique date (same performance-safe pattern as leave_some_wells_out_cv)
    print("[final_validation] Applying residual-Kriging correction...")
    residual_grid_cache: dict[str, np.ndarray | None] = {}
    corrected_preds = []
    n_applied, n_skipped = 0, 0

    from feature_utils import latlon_to_grid_cell

    for _, row in held_preds.iterrows():
        date_str = row["date"]
        if date_str not in residual_grid_cache:
            date_residuals = train_preds[train_preds["date"] == date_str]
            if len(date_residuals) < min_wells_for_correction:
                residual_grid_cache[date_str] = None
            else:
                try:
                    residual_grid_cache[date_str] = krige_wells_to_grid(
                        date_residuals, bbox, value_col="residual"
                    )
                except Exception:
                    residual_grid_cache[date_str] = None

        residual_grid = residual_grid_cache[date_str]
        if residual_grid is None:
            corrected_preds.append(row["rf_predicted"])
            n_skipped += 1
            continue

        r, c = latlon_to_grid_cell(row["lat"], row["lon"], bbox)
        correction = residual_grid[r, c]
        if np.isnan(correction):
            corrected_preds.append(row["rf_predicted"])
            n_skipped += 1
        else:
            corrected_preds.append(row["rf_predicted"] + correction)
            n_applied += 1

    held_preds = held_preds.copy()
    held_preds["predicted"] = corrected_preds
    held_preds = held_preds.rename(columns={"rf_predicted": "raw_rf_predicted"})

    print(f"[final_validation] Correction applied to {n_applied}/{len(held_preds)} held-out readings.")

    Path(results_out).parent.mkdir(parents=True, exist_ok=True)
    held_preds.to_csv(results_out, index=False)

    def _metrics(actual, pred):
        return {
            "rmse": np.sqrt(mean_squared_error(actual, pred)),
            "mae": mean_absolute_error(actual, pred),
            "r2": r2_score(actual, pred),
            "n": len(actual),
        }

    raw_metrics = _metrics(held_preds["actual"], held_preds["raw_rf_predicted"])
    corrected_metrics = _metrics(held_preds["actual"], held_preds["predicted"])

    print()
    print("=" * 70)
    print("FINAL HELD-OUT VALIDATION -- RF (+ rainfall deficit) vs. RF + Residual Kriging")
    print("=" * 70)
    print(f"{'':25s} {'RMSE':>10s} {'MAE':>10s} {'R2':>10s} {'N':>8s}")
    print(f"{'RF (raw, no correction)':25s} {raw_metrics['rmse']:>10.3f} {raw_metrics['mae']:>10.3f} "
          f"{raw_metrics['r2']:>10.3f} {raw_metrics['n']:>8d}")
    print(f"{'RF + Residual Kriging':25s} {corrected_metrics['rmse']:>10.3f} {corrected_metrics['mae']:>10.3f} "
          f"{corrected_metrics['r2']:>10.3f} {corrected_metrics['n']:>8d}")
    print()
    print("This is the final, single held-out check for Phase 2.5. Report this number "
          "(RF + Residual Kriging row) as the final result, alongside the raw-RF row for "
          "honest comparison.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(held_preds["actual"], held_preds["predicted"], alpha=0.4, s=15)
    lims = [min(held_preds["actual"].min(), held_preds["predicted"].min()),
            max(held_preds["actual"].max(), held_preds["predicted"].max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual depth (m) -- held-out CGWB wells")
    ax.set_ylabel("RF + Residual Kriging prediction (m)")
    ax.set_title("Final Phase 2.5 Result: Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"[final_validation] Saved results -> {results_out}")
    print(f"[final_validation] Saved plot -> {plot_out}")


if __name__ == "__main__":
    run()