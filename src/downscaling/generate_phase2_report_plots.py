"""
generate_phase2_report_plots.py

Produces report-quality diagnostic plots from the ALREADY-COMPUTED
held-out validation results (reports/rf_downscale_validation.csv). This
is pure visualization of an existing result -- it does not retrain
anything, does not touch the model, and does not re-run validate_downscale.py.
Safe to run as many times as useful with zero risk to the held-out
methodology.

Produces:
  1. Spatial error map -- where geographically does the model over/under-
     predict? (Phase 2 roadmap Definition of Done explicitly asks for this)
  2. Residual histogram -- is the error distribution centered/symmetric,
     or systematically biased?
  3. Residual vs. actual depth -- does error grow with depth (the
     extrapolation-ceiling pattern discussed earlier)?
  4. Error by year -- ties to the skip-bias finding; shows whether
     accuracy varies across the evaluated years.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_csv: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = {"well_id", "date", "lat", "lon", "actual", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{results_csv} is missing expected columns: {missing}")
    df["residual"] = df["predicted"] - df["actual"]  # positive = overprediction
    df["year"] = df["date"].astype(str).str[:4]
    return df


def plot_spatial_error_map(df: pd.DataFrame, out_path: str, bbox: tuple[float, float, float, float] | None = None) -> None:
    """Scatter of held-out wells colored by residual, to reveal geographic bias patterns."""
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = np.percentile(np.abs(df["residual"]), 95)  # robust color scale, ignore extreme outliers
    sc = ax.scatter(
        df["lon"], df["lat"], c=df["residual"], cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, s=25, alpha=0.7, edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Residual (predicted − actual, m)\nBlue = underprediction, Red = overprediction")

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Error Map — RF Downscaling\n(held-out CGWB wells, colored by prediction residual)")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report_plots] Saved spatial error map -> {out_path}")


def plot_residual_histogram(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["residual"], bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero error")
    mean_resid = df["residual"].mean()
    ax.axvline(mean_resid, color="red", linestyle="-", linewidth=1.5,
               label=f"Mean residual = {mean_resid:+.2f} m")
    ax.set_xlabel("Residual (predicted − actual, m)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution — Held-Out Wells")
    ax.legend()
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report_plots] Saved residual histogram -> {out_path}")


def plot_residual_vs_actual(df: pd.DataFrame, out_path: str) -> None:
    """Reveals systematic patterns like the extrapolation ceiling: does
    error grow (in magnitude or sign) as actual depth increases?"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["actual"], df["residual"], alpha=0.35, s=15, color="steelblue")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Rolling-window mean trend line to make any systematic pattern visible
    sorted_df = df.sort_values("actual")
    window = max(10, len(sorted_df) // 40)
    rolling_mean = sorted_df["residual"].rolling(window, center=True, min_periods=5).mean()
    ax.plot(sorted_df["actual"], rolling_mean, color="darkred", linewidth=2, label="Rolling mean residual")

    ax.set_xlabel("Actual depth (m)")
    ax.set_ylabel("Residual (predicted − actual, m)")
    ax.set_title("Residual vs. Actual Depth\n(reveals systematic bias at shallow/deep extremes)")
    ax.legend()
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report_plots] Saved residual-vs-actual plot -> {out_path}")


def plot_error_by_year(df: pd.DataFrame, out_path: str) -> None:
    """Boxplot of residuals per year -- ties to the skip-bias finding;
    shows whether accuracy is consistent across the evaluated years or
    concentrated in specific ones."""
    years = sorted(df["year"].unique())
    data_by_year = [df[df["year"] == y]["residual"].values for y in years]
    counts = [len(d) for d in data_by_year]

    fig, ax = plt.subplots(figsize=(9, 5))
    label_list = [f"{y}\n(n={n})" for y, n in zip(years, counts)]
    try:
        bp = ax.boxplot(data_by_year, tick_labels=label_list, showfliers=False)
    except TypeError:
        # Older matplotlib (<3.9) uses the 'labels' argument name instead.
        bp = ax.boxplot(data_by_year, labels=label_list, showfliers=False)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Residual (predicted − actual, m)")
    ax.set_title("Prediction Error by Year — Held-Out Wells")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report_plots] Saved error-by-year plot -> {out_path}")


def run(
    results_csv: str = "reports/rf_downscale_validation.csv",
    out_dir: str = "reports",
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    df = load_results(results_csv)
    print(f"[report_plots] Loaded {len(df)} held-out results from {results_csv}")

    plot_spatial_error_map(df, f"{out_dir}/phase2_spatial_error_map.png", bbox=bbox)
    plot_residual_histogram(df, f"{out_dir}/phase2_residual_histogram.png")
    plot_residual_vs_actual(df, f"{out_dir}/phase2_residual_vs_actual.png")
    plot_error_by_year(df, f"{out_dir}/phase2_error_by_year.png")

    print()
    print("Summary stats:")
    print(f"  Mean residual (bias): {df['residual'].mean():+.3f} m "
          f"({'overpredicting' if df['residual'].mean() > 0 else 'underpredicting'} on average)")
    print(f"  Median residual: {df['residual'].median():+.3f} m")
    print(f"  Std of residuals: {df['residual'].std():.3f} m")


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Generate Phase 2 diagnostic plots from validate_downscale.py's saved results.")
    parser.add_argument("--results_csv", default="reports/rf_downscale_validation.csv")
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--config", default="config/data_config.yaml",
                         help="Used only to pull the study bbox for the spatial map's axis limits.")
    args = parser.parse_args()

    bbox = None
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        bbox = tuple(config["region"]["bbox"])
    except Exception:
        pass  # bbox is optional -- spatial map will just auto-scale to the data

    run(args.results_csv, args.out_dir, bbox)