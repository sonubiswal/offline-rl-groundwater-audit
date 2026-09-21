"""
kriging.py

Ordinary Kriging interpolation of CGWB well readings onto the common 64x64
grid, using TRAINING WELLS ONLY. This produces the fine-resolution target
surface that rf_downscale.py learns to predict from coarse covariates.

CRITICAL: this module must never be called with held-out wells. The whole
point of the held-out split (Phase 1) is to have an independent set of
wells the Kriged surface -- and therefore the RF model trained against it
-- has never seen. validate_downscale.py is what checks RF predictions
against held-out wells; this module only touches training wells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging


GRID_SIZE = 64


def krige_wells_to_grid(
    wells_df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    date_filter: str | None = None,
    variogram_model: str = "spherical",
    grid_size: int = GRID_SIZE,
    value_col: str = "depth_m",
) -> np.ndarray:
    """Krige a set of well readings (or any per-well scalar, via value_col)
    onto the common grid.

    Parameters
    ----------
    wells_df : pd.DataFrame
        Must have columns [lat, lon, value_col], optionally [date]. Should
        contain ONLY training wells -- this function does not check well
        membership itself; that responsibility lies with the caller (see
        rf_downscale.py, which loads from data/interim/train_well_ids.csv).
    bbox : tuple
        (min_lon, min_lat, max_lon, max_lat) study region.
    date_filter : str, optional
        If provided and wells_df has a 'date' column, restrict to readings
        from this exact date before kriging (one time-slice at a time).
    variogram_model : str
        pykrige variogram model. 'spherical' is a reasonable default for
        groundwater depth, which tends to have a clear correlation range
        and then flattens out (sill).
    grid_size : int
        Output grid resolution (fixed at 64 per the project's common grid).
    value_col : str
        Which column to Krige. Defaults to 'depth_m' (raw well depth);
        pass 'residual' (or any other column) to Krige something else,
        e.g. RF prediction residuals for residual-kriging correction.

    Returns
    -------
    np.ndarray of shape (grid_size, grid_size)
        Kriged surface.
    """
    df = wells_df.copy()
    if date_filter is not None and "date" in df.columns:
        df = df[df["date"] == date_filter]

    df = df.dropna(subset=["lat", "lon", value_col])
    if len(df) < 3:
        raise ValueError(
            f"Only {len(df)} training wells available for this date/slice -- "
            f"Kriging needs at least 3 non-collinear points. Check date_filter "
            f"or well coverage for this period."
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    grid_lon = np.linspace(min_lon, max_lon, grid_size)
    grid_lat = np.linspace(max_lat, min_lat, grid_size)  # descending: row 0 = north

    ok = OrdinaryKriging(
        df["lon"].values,
        df["lat"].values,
        df[value_col].values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )
    z, ss = ok.execute("grid", grid_lon, grid_lat)  # z: (grid_size, grid_size), ss: kriging variance

    return np.asarray(z, dtype=np.float32)


def krige_wells_to_grid_with_variance(
    wells_df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    date_filter: str | None = None,
    variogram_model: str = "spherical",
    grid_size: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as krige_wells_to_grid, but also returns the Kriging variance
    surface -- useful for flagging low-confidence cells far from any
    training well (sparse coverage regions)."""
    df = wells_df.copy()
    if date_filter is not None and "date" in df.columns:
        df = df[df["date"] == date_filter]
    df = df.dropna(subset=["lat", "lon", "depth_m"])
    if len(df) < 3:
        raise ValueError(f"Only {len(df)} training wells available -- need at least 3.")

    min_lon, min_lat, max_lon, max_lat = bbox
    grid_lon = np.linspace(min_lon, max_lon, grid_size)
    grid_lat = np.linspace(max_lat, min_lat, grid_size)

    ok = OrdinaryKriging(
        df["lon"].values, df["lat"].values, df["depth_m"].values,
        variogram_model=variogram_model, verbose=False, enable_plotting=False,
    )
    z, ss = ok.execute("grid", grid_lon, grid_lat)
    return np.asarray(z, dtype=np.float32), np.asarray(ss, dtype=np.float32)


def krige_all_available_dates_by_month(
    train_wells_csv: str,
    bbox: tuple[float, float, float, float],
    out_dir: str = "data/interim/kriged_target_monthly",
    variogram_model: str = "spherical",
    min_wells: int = 5,
) -> None:
    """Batch-krige training wells aggregated by CALENDAR MONTH instead of exact date.

    CGWB field teams read different wells on different specific days within
    the same survey campaign (e.g. four different dates in May 2019 alone),
    which fragments krige_all_available_dates() into many sparse single-day
    slices -- often only 1-2 wells each, producing noisy, unstable Kriged
    surfaces (or getting skipped entirely for having <3 wells).

    This function instead groups all training well readings by the calendar
    month they fall in (matching how covariates are already matched --
    monthly composites), giving each Kriging call many more points to work
    with and a far more stable, representative surface. If a single well
    has multiple readings within the same month, its readings are averaged
    first (Kriging requires one value per coordinate; duplicate/near-
    duplicate coordinates with different values would make the variogram
    estimation ill-posed).

    Parameters
    ----------
    min_wells : int
        Minimum wells required for a month to be Kriged. Set higher than
        the bare mathematical minimum (3) since very sparse months still
        produce unstable surfaces even when Kriging technically succeeds.
    """
    df = pd.read_csv(train_wells_csv)
    if "date" not in df.columns:
        raise ValueError("train_wells_csv must have a 'date' column.")

    df = df.dropna(subset=["lat", "lon", "depth_m", "date"])
    df["month_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-01")

    # Average multiple readings from the same well within the same month
    # (a well shouldn't appear twice at the same coordinates with two
    # different depth values in one Kriging call).
    monthly_df = (
        df.groupby(["month_key", "well_id"], as_index=False)
        .agg({"lat": "first", "lon": "first", "depth_m": "mean"})
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_ok, n_skipped = 0, 0
    for month_key, group in monthly_df.groupby("month_key"):
        target_path = out_path / f"{month_key}.npy"
        if target_path.exists():
            continue
        n_wells = len(group)
        if n_wells < min_wells:
            print(f"[kriging] Skipping {month_key}: only {n_wells} training wells (<{min_wells}).")
            n_skipped += 1
            continue
        try:
            grid = krige_wells_to_grid(group, bbox, date_filter=None, variogram_model=variogram_model)
            np.save(target_path, grid)
            n_ok += 1
            print(f"[kriging] Kriged {month_key} using {n_wells} training wells.")
        except Exception as e:
            print(f"[kriging] FAILED for {month_key}: {e}")
            n_skipped += 1

    print(f"[kriging] Kriged {n_ok} months, skipped {n_skipped} (insufficient wells or error).")
    print(f"[kriging] Output -> {out_dir}/<YYYY-MM-01>.npy")


def krige_all_available_dates(
    train_wells_csv: str,
    bbox: tuple[float, float, float, float],
    out_dir: str = "data/interim/kriged_target",
    variogram_model: str = "spherical",
) -> None:
    """Batch-krige every distinct reading date found in the training wells CSV.

    Saves one .npy file per date to out_dir, named <date>.npy, each of
    shape (64, 64). Dates with fewer than 3 training wells are skipped and
    logged (Kriging is mathematically underdetermined below 3 points).
    """
    df = pd.read_csv(train_wells_csv)
    if "date" not in df.columns:
        raise ValueError("train_wells_csv must have a 'date' column to krige per time-slice.")

    dates = sorted(df["date"].dropna().unique())
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_ok, n_skipped = 0, 0
    for date in dates:
        target_path = out_path / f"{date}.npy"
        if target_path.exists():
            continue
        n_wells_this_date = (df["date"] == date).sum()
        if n_wells_this_date < 3:
            print(f"[kriging] Skipping {date}: only {n_wells_this_date} training wells (<3).")
            n_skipped += 1
            continue
        try:
            grid = krige_wells_to_grid(df, bbox, date_filter=date, variogram_model=variogram_model)
            np.save(target_path, grid)
            n_ok += 1
        except Exception as e:
            print(f"[kriging] FAILED for {date}: {e}")
            n_skipped += 1

    print(f"[kriging] Kriged {n_ok} dates, skipped {n_skipped} (insufficient wells or error).")
    print(f"[kriging] Output -> {out_dir}/<date>.npy")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Krige training wells onto the common grid, per available date.")
    parser.add_argument("--train_csv", default="data/interim/train_well_ids.csv")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--by_month", action="store_true",
                         help="Aggregate wells by calendar month instead of exact reading date "
                              "(recommended -- see krige_all_available_dates_by_month docstring: "
                              "produces far more stable surfaces since CGWB campaigns span many "
                              "days within a month).")
    parser.add_argument("--min_wells", type=int, default=5,
                         help="Minimum wells required per month (only used with --by_month).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    bbox = tuple(config["region"]["bbox"])

    if args.by_month:
        out_dir = args.out_dir or "data/interim/kriged_target_monthly"
        krige_all_available_dates_by_month(args.train_csv, bbox, out_dir, min_wells=args.min_wells)
    else:
        out_dir = args.out_dir or "data/interim/kriged_target"
        krige_all_available_dates(args.train_csv, bbox, out_dir)