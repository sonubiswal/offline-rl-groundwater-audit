"""
spatial_block_cv.py

Drop-in replacement for the well-splitting logic inside
`leave_some_wells_out_cv` (residual_kriging.py). Replaces sklearn's
GroupKFold (random group assignment) with a spatially-stratified block
split -- the SAME 10x10-grid logic used by held_out_split.py and
cv_tune.py -- so the CV test's geometry actually matches the true
held-out set's geometry.

Why this matters: GroupKFold splits by well_id with no spatial
constraint, so a "correction" well from another random fold can sit
right next to an "evaluation" well in this fold. Residual Kriging then
has a nearby point to lean on and looks like it works. The TRUE held-out
set was deliberately built so held-out wells are spatially separated
from all training wells -- so this CV was measuring a different,
easier problem than the one the final held-out check actually poses.

Usage: import `spatial_block_splits` and use it in place of
`GroupKFold(...).split(...)` inside leave_some_wells_out_cv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_spatial_block(lat: float, lon: float, bbox: tuple, n_grid: int = 10) -> int:
    """Assigns a (lat, lon) point to one of n_grid x n_grid spatial cells
    over bbox, returning a flat block id. Mirrors the logic held_out_split.py
    uses to build its 10x10 stratification grid."""
    min_lon, min_lat, max_lon, max_lat = bbox
    col = int((lon - min_lon) / (max_lon - min_lon) * n_grid)
    row = int((lat - min_lat) / (max_lat - min_lat) * n_grid)
    col = min(max(col, 0), n_grid - 1)
    row = min(max(row, 0), n_grid - 1)
    return row * n_grid + col


def spatial_block_splits(
    wells_df: pd.DataFrame,
    bbox: tuple,
    n_splits: int = 5,
    n_grid: int = 10,
    random_state: int = 42,
):
    """
    Yields (correction_idx, eval_idx) index arrays into wells_df, analogous
    to GroupKFold.split(), but assigns whole SPATIAL BLOCKS (not individual
    wells) to folds. All wells inside a block move together, so a fold's
    evaluation wells are guaranteed to be spatially clustered away from
    that fold's correction wells -- matching the true held-out set's
    geometry instead of an easier, spatially-mixed one.

    wells_df must have a 'well_id', 'lat', 'lon' column (or duplicated rows
    per well are fine, as in the residual df -- block assignment is by
    well_id, computed once per unique well).
    """
    unique_wells = wells_df[["well_id", "lat", "lon"]].drop_duplicates("well_id").copy()
    unique_wells["block"] = unique_wells.apply(
        lambda r: assign_spatial_block(r["lat"], r["lon"], bbox, n_grid), axis=1
    )

    rng = np.random.RandomState(random_state)
    blocks = unique_wells["block"].unique()
    rng.shuffle(blocks)

    # Deal blocks round-robin into n_splits groups, so each fold gets a
    # roughly equal, spatially-scattered (not spatially-clustered-in-one-
    # corner) set of blocks -- avoids one fold accidentally getting a
    # disproportionate share of the region.
    fold_of_block = {block: i % n_splits for i, block in enumerate(blocks)}
    unique_wells["fold"] = unique_wells["block"].map(fold_of_block)

    well_to_fold = dict(zip(unique_wells["well_id"], unique_wells["fold"]))
    wells_df = wells_df.copy()
    wells_df["_fold"] = wells_df["well_id"].map(well_to_fold)

    for fold_i in range(n_splits):
        eval_mask = wells_df["_fold"] == fold_i
        correction_mask = ~eval_mask
        yield np.where(correction_mask)[0], np.where(eval_mask)[0]


# ---------------------------------------------------------------------
# How to wire this into residual_kriging.py's leave_some_wells_out_cv:
#
#   from spatial_block_cv import spatial_block_splits
#
#   # REPLACE:
#   #   gkf = GroupKFold(n_splits=n_splits)
#   #   for fold_i, (correction_idx, eval_idx) in enumerate(
#   #           gkf.split(dummy_X, groups=all_preds["well_id"])):
#   #
#   # WITH:
#   for fold_i, (correction_idx, eval_idx) in enumerate(
#           spatial_block_splits(all_preds, bbox, n_splits=n_splits)):
#
# Everything downstream (correction_wells, eval_wells, the per-date
# Kriging cache, metrics) stays exactly the same -- only the fold
# assignment logic changes.
# ---------------------------------------------------------------------