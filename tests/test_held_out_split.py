"""
test_held_out_split.py

Non-negotiable regression test. This must ALWAYS pass -- it is the automated
guardrail against reintroducing the circular ground truth problem. Do not
skip, xfail, or delete this test regardless of time pressure (see Risk
Register, row 1, in the master roadmap).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.held_out_split import (
    assign_grid_cells,
    assert_no_overlap,
    load_wells,
    spatial_stratified_split,
)


@pytest.fixture
def synthetic_wells():
    """Generate a synthetic set of wells scattered across a fake bbox."""
    rng = np.random.default_rng(0)
    n = 500
    bbox = (72.0, 15.0, 81.0, 22.0)  # min_lon, min_lat, max_lon, max_lat
    lon = rng.uniform(bbox[0], bbox[2], n)
    lat = rng.uniform(bbox[1], bbox[3], n)
    df = pd.DataFrame({
        "well_id": [f"W{i:04d}" for i in range(n)],
        "lat": lat,
        "lon": lon,
        "date": "2020-01-01",
        "depth_m": rng.uniform(2, 50, n),
    })
    return df, bbox


def test_load_wells_drops_missing_and_implausible(tmp_path):
    df = pd.DataFrame({
        "well_id": ["A", "B", "C", "D"],
        "lat": [20.0, None, 21.0, 19.5],
        "lon": [75.0, 76.0, 77.0, 74.0],
        "date": ["2020-01-01"] * 4,
        "depth_m": [10.0, 15.0, 5000.0, -3.0],  # C implausible, D negative
    })
    csv_path = tmp_path / "wells.csv"
    df.to_csv(csv_path, index=False)

    cleaned = load_wells(csv_path)
    # B dropped (missing lat), C dropped (implausible depth), D dropped (negative)
    assert set(cleaned["well_id"]) == {"A"}


def test_assign_grid_cells_within_bounds(synthetic_wells):
    df, bbox = synthetic_wells
    gridded = assign_grid_cells(df, bbox, n_cells_x=10, n_cells_y=10)
    assert "grid_cell" in gridded.columns
    assert gridded["grid_cell"].notna().all()

    # Every cell id should decompose into valid x_idx, y_idx within range
    for cell in gridded["grid_cell"].unique():
        x_idx, y_idx = cell.split("_")
        assert 0 <= int(x_idx) < 10
        assert 0 <= int(y_idx) < 10


def test_split_zero_overlap(synthetic_wells):
    """THE critical test: train and held-out sets must never share a well_id."""
    df, bbox = synthetic_wells
    train_df, held_out_df = spatial_stratified_split(
        df, bbox=bbox, held_out_frac=0.175, seed=42
    )
    assert_no_overlap(train_df, held_out_df)  # should not raise

    train_ids = set(train_df["well_id"])
    held_ids = set(held_out_df["well_id"])
    assert len(train_ids & held_ids) == 0
    assert train_ids | held_ids == set(df["well_id"])  # no wells lost


def test_split_fraction_roughly_correct(synthetic_wells):
    df, bbox = synthetic_wells
    train_df, held_out_df = spatial_stratified_split(
        df, bbox=bbox, held_out_frac=0.175, seed=42
    )
    frac = len(held_out_df) / len(df)
    # Allow tolerance since sparse-cell rounding introduces some noise
    assert 0.12 <= frac <= 0.23


def test_split_is_spatially_distributed(synthetic_wells):
    """Held-out wells shouldn't all cluster in one grid cell."""
    df, bbox = synthetic_wells
    train_df, held_out_df = spatial_stratified_split(
        df, bbox=bbox, held_out_frac=0.175, seed=42
    )
    held_out_gridded = assign_grid_cells(held_out_df, bbox, n_cells_x=10, n_cells_y=10)
    n_unique_cells = held_out_gridded["grid_cell"].nunique()
    # With 500 wells spread over a 10x10 grid, held-out wells should span
    # a meaningful number of distinct cells, not just one or two.
    assert n_unique_cells > 10


def test_split_is_idempotent(synthetic_wells):
    """Same seed -> same split, every time."""
    df, bbox = synthetic_wells
    train_1, held_1 = spatial_stratified_split(df, bbox=bbox, held_out_frac=0.175, seed=42)
    train_2, held_2 = spatial_stratified_split(df, bbox=bbox, held_out_frac=0.175, seed=42)

    assert set(held_1["well_id"]) == set(held_2["well_id"])
    assert set(train_1["well_id"]) == set(train_2["well_id"])


def test_split_handles_repeated_readings_per_well():
    """Regression test for the row-vs-well bug: CGWB wells are read
    quarterly across many years, so the same well_id appears many times
    as separate reading rows. The split must operate on UNIQUE wells, not
    rows -- otherwise selecting a single reading pulls that well's entire
    history into held-out, inflating the held-out fraction far past the
    configured value (observed in production: 98.4% held out instead of
    ~17.5%).
    """
    rng = np.random.default_rng(1)
    bbox = (72.0, 15.0, 81.0, 22.0)

    n_wells = 200
    readings_per_well = 20  # simulate ~5 years of quarterly readings

    well_ids = [f"W{i:04d}" for i in range(n_wells)]
    lats = rng.uniform(bbox[1], bbox[3], n_wells)
    lons = rng.uniform(bbox[0], bbox[2], n_wells)

    rows = []
    for i in range(n_wells):
        for r in range(readings_per_well):
            rows.append({
                "well_id": well_ids[i],
                "lat": lats[i],
                "lon": lons[i],
                "date": f"2020-{(r % 12) + 1:02d}-01",
                "depth_m": rng.uniform(2, 50),
            })
    df = pd.DataFrame(rows)
    assert len(df) == n_wells * readings_per_well  # 4000 reading-rows, 200 unique wells

    train_df, held_out_df = spatial_stratified_split(df, bbox=bbox, held_out_frac=0.175, seed=42)
    assert_no_overlap(train_df, held_out_df)

    train_unique = train_df["well_id"].nunique()
    held_unique = held_out_df["well_id"].nunique()

    # The held-out FRACTION OF UNIQUE WELLS should be close to 17.5%,
    # not close to 100% (which is what the row-sampling bug produced).
    held_frac = held_unique / n_wells
    assert 0.10 <= held_frac <= 0.25, (
        f"Held-out well fraction {held_frac:.1%} is way outside the configured "
        f"17.5% target -- likely regressed to sampling rows instead of unique wells."
    )
    # Every reading for a given well must land entirely in one split or the other.
    assert train_unique + held_unique == n_wells


def test_assert_no_overlap_raises_on_bad_input():
    """assert_no_overlap should actually catch a deliberately broken split."""
    train_df = pd.DataFrame({"well_id": ["A", "B", "C"]})
    held_out_df = pd.DataFrame({"well_id": ["C", "D"]})  # C leaked into both

    with pytest.raises(AssertionError):
        assert_no_overlap(train_df, held_out_df)