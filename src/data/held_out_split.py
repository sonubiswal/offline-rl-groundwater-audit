"""
held_out_split.py

Critical module for Trishna-OPAL. Performs a SPATIALLY STRATIFIED split of
CGWB wells into a training set and a held-out validation set, BEFORE any
Kriging or Random Forest downscaling touches the data.

Why spatial stratification (not a plain random split)?
A purely random split can accidentally cluster all held-out wells in one
corner of the region, in which case they don't actually test whether the
model generalizes spatially -- they just test whether it memorized a
sub-region. By dividing the bounding box into a grid of cells and holding
out a fraction of wells from EACH cell, the held-out set is forced to be
spatially distributed across the whole study region.

This split is idempotent given a fixed random seed (stored in
config/data_config.yaml), and once committed to
data/held_out_wells/held_out_ids.csv it must NEVER be regenerated with a
different seed or logic later in the project -- that would silently
reintroduce the circular ground truth problem the whole pipeline exists to
avoid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = ["well_id", "lat", "lon", "date", "depth_m"]


def load_wells(csv_path: str | Path) -> pd.DataFrame:
    """Load and lightly validate the raw CGWB well CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV with columns [well_id, lat, lon, date, depth_m].

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with missing coordinates / implausible depths
        dropped.
    """
    df = pd.read_csv(csv_path)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CGWB well CSV is missing required columns: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=["well_id", "lat", "lon"])
    # Implausible depth guard: negative depths, or absurd outliers (>1000m
    # water table depth is essentially never real for CGWB monitoring wells).
    df = df[(df["depth_m"].isna()) | ((df["depth_m"] >= 0) & (df["depth_m"] < 1000))]
    after = len(df)
    if after < before:
        print(f"[held_out_split] Dropped {before - after} wells with missing/implausible values.")

    return df.reset_index(drop=True)


def assign_grid_cells(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    n_cells_x: int = 10,
    n_cells_y: int = 10,
) -> pd.DataFrame:
    """Assign each well to a spatial grid cell within the study bbox.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    bbox : tuple
        (min_lon, min_lat, max_lon, max_lat) for the study region.
    n_cells_x, n_cells_y : int
        Number of grid divisions along longitude and latitude.

    Returns
    -------
    pd.DataFrame
        Copy of df with an added 'grid_cell' column (string id like "3_7").
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    df = df.copy()

    # Wells outside the bbox get clipped into the nearest edge cell rather
    # than silently dropped -- flag them instead so they can be reviewed.
    out_of_bounds = (
        (df["lon"] < min_lon) | (df["lon"] > max_lon) |
        (df["lat"] < min_lat) | (df["lat"] > max_lat)
    )
    if out_of_bounds.any():
        print(f"[held_out_split] Warning: {out_of_bounds.sum()} wells fall outside "
              f"the configured bbox {bbox}. Clipping into edge cells.")

    lon_clipped = df["lon"].clip(min_lon, max_lon - 1e-9)
    lat_clipped = df["lat"].clip(min_lat, max_lat - 1e-9)

    x_idx = ((lon_clipped - min_lon) / (max_lon - min_lon) * n_cells_x).astype(int)
    y_idx = ((lat_clipped - min_lat) / (max_lat - min_lat) * n_cells_y).astype(int)
    x_idx = x_idx.clip(0, n_cells_x - 1)
    y_idx = y_idx.clip(0, n_cells_y - 1)

    df["grid_cell"] = x_idx.astype(str) + "_" + y_idx.astype(str)
    return df


def spatial_stratified_split(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    held_out_frac: float = 0.175,
    n_cells_x: int = 10,
    n_cells_y: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split wells into train/held-out sets, stratified by spatial grid cell.

    From each grid cell, `held_out_frac` of that cell's wells are randomly
    selected for the held-out set. Cells with very few wells (1-2) still
    get a chance to contribute a held-out well via random rounding, so the
    held-out set isn't systematically biased toward densely-sampled cells.

    Parameters
    ----------
    df : pd.DataFrame
        Well dataframe (output of load_wells).
    bbox : tuple
        Study region bounding box.
    held_out_frac : float
        Fraction of wells to hold out per cell (default 0.175, i.e. within
        the 15-20% range specified in the roadmap).
    n_cells_x, n_cells_y : int
        Grid resolution for stratification.
    seed : int
        Random seed for reproducibility. MUST match config/data_config.yaml.

    Returns
    -------
    (train_df, held_out_df) : tuple of pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    # CRITICAL: the input df is often one row PER READING, not per well --
    # CGWB wells are read quarterly across many years, so the same well_id
    # can appear dozens of times. If we sampled rows directly, selecting a
    # single reading from a well would pull that well's ENTIRE reading
    # history into the held-out set (since membership is checked by
    # well_id), which inflates the held-out fraction far past
    # `held_out_frac` -- in the worst case, nearly the whole dataset.
    # So: split must happen on UNIQUE WELLS, then rejoin all their readings.
    unique_wells = df.drop_duplicates(subset="well_id", keep="first").reset_index(drop=True)
    unique_wells = assign_grid_cells(unique_wells, bbox, n_cells_x, n_cells_y)

    held_out_ids = []
    for _, cell_df in unique_wells.groupby("grid_cell"):
        n_wells = len(cell_df)
        n_hold = int(round(n_wells * held_out_frac))
        if n_hold == 0 and n_wells > 0:
            # Give sparse cells a probabilistic chance to contribute one
            # held-out well, so held-out coverage isn't limited only to
            # dense cells.
            if rng.random() < held_out_frac:
                n_hold = 1
        if n_hold > 0:
            chosen = rng.choice(cell_df["well_id"].values, size=n_hold, replace=False)
            held_out_ids.extend(chosen.tolist())

    held_out_ids = set(held_out_ids)

    # Now filter the FULL reading-level df (all quarters/years) by well
    # membership, so every reading for a held-out well ends up in
    # held_out_df, and every reading for a training well ends up in
    # train_df -- consistently, with zero well-level overlap.
    held_out_df = df[df["well_id"].isin(held_out_ids)].reset_index(drop=True)
    train_df = df[~df["well_id"].isin(held_out_ids)].reset_index(drop=True)

    return train_df, held_out_df


def assert_no_overlap(train_df: pd.DataFrame, held_out_df: pd.DataFrame) -> None:
    """Guarantee zero well_id overlap between train and held-out sets.

    Raises
    ------
    AssertionError
        If any well_id appears in both sets.
    """
    train_ids = set(train_df["well_id"])
    held_ids = set(held_out_df["well_id"])
    overlap = train_ids & held_ids
    assert len(overlap) == 0, f"Overlap detected between train and held-out wells: {overlap}"


def run(
    raw_csv_path: str,
    config_path: str = "config/data_config.yaml",
    out_dir: str = "data",
) -> None:
    """End-to-end driver: load config, split wells, save outputs.

    Reads region bbox and random_seed from config/data_config.yaml so the
    split is fully reproducible and traceable to a single source of truth.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bbox = tuple(config["region"]["bbox"])
    seed = config.get("random_seed", 42)
    held_out_frac = config.get("held_out_frac", 0.175)

    df = load_wells(raw_csv_path)
    train_df, held_out_df = spatial_stratified_split(
        df, bbox=bbox, held_out_frac=held_out_frac, seed=seed
    )
    assert_no_overlap(train_df, held_out_df)

    held_out_path = Path(out_dir) / "held_out_wells" / "held_out_ids.csv"
    train_path = Path(out_dir) / "interim" / "train_well_ids.csv"
    held_out_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    held_out_df.to_csv(held_out_path, index=False)
    train_df.to_csv(train_path, index=False)

    print(f"[held_out_split] Total wells: {len(df)}")
    print(f"[held_out_split] Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"[held_out_split] Held-out: {len(held_out_df)} ({len(held_out_df)/len(df):.1%})")
    print(f"[held_out_split] Saved held-out IDs -> {held_out_path}")
    print(f"[held_out_split] Saved train IDs -> {train_path}")
    print("[held_out_split] This split is now IMMUTABLE. Do not regenerate with a "
          "different seed later in the project.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spatially stratified held-out well split.")
    parser.add_argument("--raw_csv", required=True, help="Path to raw CGWB well CSV.")
    parser.add_argument("--config", default="config/data_config.yaml", help="Path to data_config.yaml")
    parser.add_argument("--out_dir", default="data", help="Base data output directory.")
    args = parser.parse_args()

    run(args.raw_csv, args.config, args.out_dir)