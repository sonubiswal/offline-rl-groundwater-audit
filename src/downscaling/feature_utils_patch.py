"""
feature_utils_patch.py

NOT a standalone module — this is the addition to make to your existing
src/downscaling/feature_utils.py. I don't have your actual file, so I
can't str_replace it directly; copy the relevant pieces in by hand and
diff against your real implementation of `build_feature_stack_for_date`.

WHAT THIS ADDS
--------------
Two new static covariates, integrated the same way srtm_elevation/
srtm_slope already are (Section 7 of validation_methodology.md): loaded
once, aligned once via align_grid.py, reused identically for every date.

Categorical handling (the one real difference from SRTM):
  - soil_texture_class (12 USDA classes) -> one-hot, 12 channels
  - lulc_class (11 WorldCover classes)   -> one-hot, 11 channels

That's 23 new channels total. This is a meaningful change to the feature
count (current stack per Section 9 is ~11 channels), so:
  - Re-run cv_tune.py's spatially-blocked CV from scratch with this
    stack — do not assume the previously-selected hyperparameters
    (n_estimators=300, max_depth=15, ...) are still optimal; a 23-channel
    jump changes the bias/variance tradeoff enough to warrant re-tuning,
    not just re-scoring the old config on new features.
  - Watch for the row_norm/col_norm failure mode (Section 10): one-hot
    LULC in particular can act as a near-unique spatial fingerprint if
    class boundaries are sparse relative to grid cell size. Re-check the
    train-R²-vs-held-out-R²-gap sanity check after adding these, not just
    the final CV number.

INTEGRATION POINTS
-------------------
1. align_grid.py: register the two new static rasters
   (data/raw/static/soil_texture_class.tif, data/raw/static/lulc_worldcover.tif)
   in whatever registry/list align_grid.py uses for SRTM, so they get
   reprojected onto the same 64x64 grid. Categorical rasters should use
   NEAREST resampling, not bilinear — bilinear-interpolating a class code
   (e.g. averaging "cropland"=40 and "built_up"=50) produces meaningless
   fractional class values. Confirm align_grid.py lets you set resampling
   method per-source; if it currently hardcodes bilinear for all sources,
   this needs a small signature change there too.
2. feature_utils.py: add the function below, call it from
   build_feature_stack_for_date() alongside the existing srtm_elevation/
   srtm_slope load, and extend whatever `channels` list/order constant
   the LSTM and RF pipelines both key off of.
"""

import numpy as np

SOIL_TEXTURE_CLASSES = list(range(1, 13))  # 1..12, see soil_lulc_fetch.py docstring
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def load_static_soil_lulc_onehot(grid_dir):
    """
    Loads the aligned (64x64) soil_texture_class.npy and lulc_worldcover.npy
    grids (produced by align_grid.py, NEAREST-resampled) and returns a
    stacked array of one-hot channels.

    Returns
    -------
    np.ndarray, shape (23, 64, 64)
        Channels 0-11:  soil texture one-hot (class 1..12)
        Channels 12-22: LULC one-hot (class 10,20,...,100)
    channel_names : list[str]
        Parallel list of channel names, for wiring into your existing
        `channels` ordering constant.
    """
    soil = np.load(grid_dir / "soil_texture_class.npy")  # (64, 64), int codes
    lulc = np.load(grid_dir / "lulc_worldcover.npy")      # (64, 64), int codes

    channels = []
    names = []

    for cls in SOIL_TEXTURE_CLASSES:
        channels.append((soil == cls).astype(np.float32))
        names.append(f"soil_texture_{cls}")

    for cls in LULC_CLASSES:
        channels.append((lulc == cls).astype(np.float32))
        names.append(f"lulc_{cls}")

    stack = np.stack(channels, axis=0)  # (23, 64, 64)

    # Pixels with no valid soil/LULC classification (nodata) end up all-zero
    # across their one-hot block, which is a legitimate "unknown" encoding
    # rather than a false membership in class 0 — no further NaN handling
    # needed here, unlike the continuous covariates in Section 4.
    return stack, names


# --- Call site sketch for build_feature_stack_for_date() ---
#
# In your existing function, alongside the current:
#     stack.append(srtm_elevation)
#     stack.append(srtm_slope)
#
# add:
#     soil_lulc_stack, soil_lulc_names = load_static_soil_lulc_onehot(static_grid_dir)
#     for i in range(soil_lulc_stack.shape[0]):
#         stack.append(soil_lulc_stack[i])
#
# and extend whatever ordered `CHANNEL_NAMES` (or similar) constant both
# rf_downscale.py and train_point_lstm.py import from, so column order
# stays consistent across every consumer of the stack.