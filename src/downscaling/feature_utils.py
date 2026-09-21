"""
feature_utils.py

Shared logic for rf_downscale.py and validate_downscale.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio


GRID_SIZE = 64

MONTHLY_SOURCES = [
    "chirps",
    "gldas",
    "grace",
]

STATIC_SOURCES = [
    "srtm_elevation",
    "srtm_slope",
]

SOIL_TEXTURE_CLASSES = list(range(1, 13))

LULC_CLASSES = [
    10, 20, 30, 40, 50, 60,
    70, 80, 90, 95, 100,
]


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_date_from_filename(path: Path):

    m = _DATE_RE.search(path.stem)

    if not m:
        return None

    from datetime import date

    y, mo, d = (
        int(x)
        for x in m.group(1).split("-")
    )

    return date(y, mo, d)


def find_nearest_raster(
    source_dir: str | Path,
    target_date_str: str,
    max_gap_days: int = 90,
) -> Path | None:

    from datetime import date as date_cls

    source_dir = Path(source_dir)

    if not source_dir.exists():
        return None

    y, mo, d = (
        int(x)
        for x in target_date_str.split("-")[:3]
    )

    target = date_cls(y, mo, d)

    best_path = None
    best_diff = None

    for tif_path in source_dir.glob("*.tif"):

        file_date = _parse_date_from_filename(tif_path)

        if file_date is None:
            continue

        diff = abs(
            (file_date - target).days
        )

        if (
            diff <= max_gap_days
            and (
                best_diff is None
                or diff < best_diff
            )
        ):
            best_path = tif_path
            best_diff = diff

    return best_path


def load_raster_array(
    path: str | Path,
) -> np.ndarray:

    with rasterio.open(path) as f:

        arr = f.read(1).astype(np.float32)

        nodata = f.nodata

        if (
            nodata is not None
            and not np.isnan(nodata)
        ):
            arr = np.where(
                arr == nodata,
                np.nan,
                arr,
            )

    return arr


def load_categorical_onehot(
    raster_path: str | Path,
    class_code: int,
) -> np.ndarray:

    raw = load_raster_array(
        raster_path
    )

    return (
        raw == class_code
    ).astype(np.float32)


def _shift_months(
    date_str: str,
    n_months: int,
) -> str:

    import calendar

    y, mo, d = (
        int(x)
        for x in date_str.split("-")[:3]
    )

    total = (
        y * 12
        + (mo - 1)
        + n_months
    )

    new_y, new_mo0 = divmod(
        total,
        12,
    )

    new_mo = new_mo0 + 1

    last_day = calendar.monthrange(
        new_y,
        new_mo,
    )[1]

    new_d = min(
        d,
        last_day,
    )

    return (
        f"{new_y:04d}-"
        f"{new_mo:02d}-"
        f"{new_d:02d}"
    )


def rolling_average_features(
    interim_dir: str | Path,
    target_date_str: str,
    sources: list[str] = (
        "chirps",
        "gldas",
    ),
    n_months: int = 3,
    max_gap_days: int = 90,
    min_months: int = 2,
) -> dict[str, np.ndarray] | None:

    interim_dir = Path(interim_dir)

    result = {}

    for source in sources:

        month_arrays = []

        for offset in range(n_months):

            anchor_date = _shift_months(
                target_date_str,
                -offset,
            )

            matched = find_nearest_raster(
                interim_dir / source,
                anchor_date,
                max_gap_days,
            )

            if matched is not None:

                month_arrays.append(
                    load_raster_array(matched)
                )

        if len(month_arrays) < min_months:
            return None

        stacked = np.stack(
            month_arrays,
            axis=0,
        )

        result[
            f"{source}_roll{n_months}"
        ] = np.nanmean(
            stacked,
            axis=0,
        )

    return result


def compute_monthly_climatology(
    interim_dir: str | Path,
    source: str = "chirps",
) -> dict[int, np.ndarray]:

    source_dir = (
        Path(interim_dir)
        / source
    )

    if not source_dir.exists():
        return {}

    by_month = {
        m: []
        for m in range(1, 13)
    }

    for tif_path in source_dir.glob(
        "*.tif"
    ):

        file_date = (
            _parse_date_from_filename(
                tif_path
            )
        )

        if file_date is None:
            continue

        arr = load_raster_array(
            tif_path
        )

        by_month[
            file_date.month
        ].append(arr)

    climatology = {}

    for month, arrays in by_month.items():

        if arrays:

            climatology[month] = np.nanmean(
                np.stack(
                    arrays,
                    axis=0,
                ),
                axis=0,
            )

    return climatology


def cumulative_rainfall_deficit(
    interim_dir: str | Path,
    target_date_str: str,
    climatology: dict[int, np.ndarray],
    n_months: int = 12,
    max_gap_days: int = 90,
) -> np.ndarray | None:

    interim_dir = Path(interim_dir)

    actual_sum = None
    clim_sum = None

    for offset in range(n_months):

        month_str = _shift_months(
            target_date_str,
            -offset,
        )

        matched = find_nearest_raster(
            interim_dir / "chirps",
            month_str,
            max_gap_days,
        )

        if matched is None:
            return None

        actual = load_raster_array(
            matched
        )

        if np.isnan(actual).any():
            return None

        calendar_month = int(
            month_str.split("-")[1]
        )

        if calendar_month not in climatology:
            return None

        clim = climatology[
            calendar_month
        ]

        actual_sum = (
            actual
            if actual_sum is None
            else actual_sum + actual
        )

        clim_sum = (
            clim
            if clim_sum is None
            else clim_sum + clim
        )

    return actual_sum - clim_sum


def find_nearest_prior_kriged(
    kriged_dir: str | Path,
    target_date_str: str,
    max_gap_days: int = 120,
) -> Path | None:

    from datetime import date as date_cls

    kriged_dir = Path(
        kriged_dir
    )

    if not kriged_dir.exists():
        return None

    y, mo, d = (
        int(x)
        for x in target_date_str.split("-")[:3]
    )

    target_month = (
        y,
        mo,
    )

    target = date_cls(
        y,
        mo,
        d,
    )

    best_path = None
    best_diff = None

    for npy_path in kriged_dir.glob(
        "*.npy"
    ):

        file_date = (
            _parse_date_from_filename(
                npy_path
            )
        )

        if file_date is None:
            continue

        file_month = (
            file_date.year,
            file_date.month,
        )

        if file_month >= target_month:
            continue

        diff = (
            target - file_date
        ).days

        if (
            diff <= max_gap_days
            and (
                best_diff is None
                or diff < best_diff
            )
        ):
            best_path = npy_path
            best_diff = diff

    return best_path


def lag_target_feature(
    kriged_dir: str | Path,
    target_date_str: str,
    max_gap_days: int = 120,
) -> np.ndarray | None:

    matched = find_nearest_prior_kriged(
        kriged_dir,
        target_date_str,
        max_gap_days,
    )

    if matched is None:
        return None

    return np.load(matched)


def build_feature_stack_for_date(
    interim_dir: str | Path,
    target_date_str: str,
    max_gap_days: int = 90,
    kriged_dir: str | Path | None = None,
    lag_max_gap_days: int = 120,
    rolling_n_months: int = 3,
    rolling_min_months: int = 2,
    climatology: dict[int, np.ndarray] | None = None,
    deficit_n_months: int = 12,
    include_soil_lulc: bool = False,
) -> dict[str, np.ndarray] | None:

    interim_dir = Path(
        interim_dir
    )

    features = {}

    # ---------------------------------------------------------------
    # Core monthly covariates
    # ---------------------------------------------------------------

    for source in MONTHLY_SOURCES:

        matched = find_nearest_raster(
            interim_dir / source,
            target_date_str,
            max_gap_days,
        )

        if matched is None:
            return None

        features[source] = load_raster_array(
            matched
        )

    # ---------------------------------------------------------------
    # Static terrain
    # ---------------------------------------------------------------

    for static_name, subdir_file in [
        (
            "srtm_elevation",
            ("srtm", "elevation.tif"),
        ),
        (
            "srtm_slope",
            ("srtm", "slope.tif"),
        ),
    ]:

        static_path = interim_dir.joinpath(
            *subdir_file
        )

        if not static_path.exists():
            return None

        features[static_name] = load_raster_array(
            static_path
        )

    # ---------------------------------------------------------------
    # Rolling features
    # ---------------------------------------------------------------

    rolling = rolling_average_features(
        interim_dir,
        target_date_str,
        sources=[
            "chirps",
            "gldas",
        ],
        n_months=rolling_n_months,
        max_gap_days=max_gap_days,
        min_months=rolling_min_months,
    )

    if rolling is None:
        return None

    features.update(rolling)

    # ---------------------------------------------------------------
    # Previous Kriged groundwater surface
    # ---------------------------------------------------------------

    if kriged_dir is not None:

        lag = lag_target_feature(
            kriged_dir,
            target_date_str,
            lag_max_gap_days,
        )

        if lag is None:
            return None

        features["target_lag1"] = lag

    # ---------------------------------------------------------------
    # Optional rejected rainfall-deficit feature
    # ---------------------------------------------------------------

    if climatology is not None:

        deficit = cumulative_rainfall_deficit(
            interim_dir,
            target_date_str,
            climatology,
            n_months=deficit_n_months,
            max_gap_days=max_gap_days,
        )

        if deficit is None:
            return None

        features[
            "chirps_cum12_anomaly"
        ] = deficit

    # ---------------------------------------------------------------
    # Optional rejected Phase 2.6 soil/LULC features
    #
    # IMPORTANT:
    # These files are NOT required for the production baseline.
    # ---------------------------------------------------------------

    if include_soil_lulc:

        soil_path = (
            interim_dir
            / "soil_texture_class"
            / "soil_texture_class.tif"
        )

        if not soil_path.exists():
            return None

        for cls in SOIL_TEXTURE_CLASSES:

            features[
                f"soil_texture_{cls}"
            ] = load_categorical_onehot(
                soil_path,
                cls,
            )

        lulc_path = (
            interim_dir
            / "lulc_worldcover"
            / "lulc_worldcover.tif"
        )

        if not lulc_path.exists():
            return None

        for cls in LULC_CLASSES:

            features[
                f"lulc_{cls}"
            ] = load_categorical_onehot(
                lulc_path,
                cls,
            )

    return features


def latlon_to_grid_cell(
    lat: float,
    lon: float,
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    grid_size: int = GRID_SIZE,
) -> tuple[int, int]:

    min_lon, min_lat, max_lon, max_lat = bbox

    col = int(
        (
            lon - min_lon
        )
        / (
            max_lon - min_lon
        )
        * grid_size
    )

    lat_idx = int(
        (
            lat - min_lat
        )
        / (
            max_lat - min_lat
        )
        * grid_size
    )

    row = (
        grid_size
        - 1
        - lat_idx
    )

    col = min(
        max(col, 0),
        grid_size - 1,
    )

    row = min(
        max(row, 0),
        grid_size - 1,
    )

    return row, col


def spatial_features(
    row: int,
    col: int,
    grid_size: int = GRID_SIZE,
) -> dict:

    return {
        "row_norm": row / (grid_size - 1),
        "col_norm": col / (grid_size - 1),
    }


def seasonal_features(
    date_str: str,
) -> dict:

    month = int(
        date_str.split("-")[1]
    )

    angle = (
        2
        * np.pi
        * (month - 1)
        / 12
    )

    return {
        "month_sin": np.sin(angle),
        "month_cos": np.cos(angle),
    }