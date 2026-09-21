"""
cgwb_fetch.py

Ingests CGWB "Changes in Depth to Water Level" well data from the India
Data Portal (ckandev.indiadataportal.com) and reshapes it into the schema
expected by held_out_split.py: [well_id, lat, lon, date, depth_m].

Source dataset details (confirmed against the live portal, Aug 2026):
  - Resource ID: 580a8f6e-3d86-4ca7-ac7d-cd5df12b443c
  - ~22,965 observation wells across India, quarterly readings
    (Jan / Mar-May / Aug / Nov), years covered 2013-2022
  - Real columns: index, date, state_name, state_code, district_name,
    district_code, station_name, latitude, longitude, basin, sub_basin,
    source, currentlevel (= Depth to Water Level, in meters below
    ground), level_diff

Important caveat: the source data has NO single unique well identifier
column. `station_name` alone is not guaranteed unique across India (e.g.
common names repeat across districts/states). This module builds a
synthetic well_id from a rounded (lat, lon) pair, which is the standard
approach for this dataset -- readings at the same coordinates over time
belong to the same physical well.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
import yaml


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_raw_csv(config: dict, out_path: str | Path) -> Path:
    """Download the raw CGWB CSV from the datastore dump endpoint and cache locally.

    Uses the datastore dump URL (not the static file download URL) since
    the dump endpoint reflects the current datastore contents and doesn't
    require auth for this public resource.
    """
    out_path = Path(out_path)
    if out_path.exists():
        print(f"[cgwb_fetch] Raw CSV already cached at {out_path}, skipping download.")
        return out_path

    url = config["cgwb_source"]["csv_dump_url"]
    print(f"[cgwb_fetch] Downloading from {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print(f"[cgwb_fetch] Saved raw CSV -> {out_path} ({len(resp.content) / 1e6:.1f} MB)")
    return out_path


def clip_to_region(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Keep only wells within the study region bbox (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mask = (
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
        (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
    )
    return df[mask].copy()


def build_well_id(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Construct a synthetic well_id from rounded (lat, lon).

    Rounding to 4 decimal places (~11m precision) groups repeated
    quarterly readings at the same physical well together while treating
    genuinely distinct nearby wells as separate.
    """
    df = df.copy()
    lat_r = df["latitude"].round(decimals)
    lon_r = df["longitude"].round(decimals)
    df["well_id"] = "W_" + lat_r.astype(str) + "_" + lon_r.astype(str)
    return df


def reshape_to_standard_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw CGWB portal columns onto [well_id, lat, lon, date, depth_m]."""
    df = build_well_id(df)
    out = pd.DataFrame({
        "well_id": df["well_id"],
        "lat": df["latitude"],
        "lon": df["longitude"],
        "date": df["date"],
        "depth_m": df["currentlevel"],
    })
    return out


def fetch_and_prepare(
    config_path: str = "config/data_config.yaml",
    raw_out_path: str = "data/raw/cgwb_raw.csv",
    processed_out_path: str = "data/interim/cgwb_wells.csv",
    clip_to_study_region: bool = True,
) -> pd.DataFrame:
    """End-to-end driver: download, reshape, optionally clip to region, save.

    The output of this function is the input CSV that held_out_split.py's
    `run()` expects (--raw_csv processed_out_path).
    """
    config = load_config(config_path)

    raw_path = download_raw_csv(config, raw_out_path)
    df = pd.read_csv(raw_path)

    print(f"[cgwb_fetch] Loaded {len(df)} raw records from source.")

    required = ["date", "latitude", "longitude", "currentlevel"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {required} not found in downloaded CSV. "
            f"Got columns: {list(df.columns)}. The portal schema may have "
            f"changed -- check cgwb_source.columns in {config_path}."
        )

    df = df.dropna(subset=["latitude", "longitude", "currentlevel"])
    print(f"[cgwb_fetch] {len(df)} records remain after dropping missing lat/lon/depth.")

    if clip_to_study_region:
        bbox = tuple(config["region"]["bbox"])
        before = len(df)
        df = clip_to_region(df, bbox)
        print(f"[cgwb_fetch] Clipped to region bbox {bbox}: {before} -> {len(df)} records.")

    standard_df = reshape_to_standard_schema(df)

    n_unique_wells = standard_df["well_id"].nunique()
    print(f"[cgwb_fetch] {n_unique_wells} unique wells (by rounded lat/lon) "
          f"across {len(standard_df)} readings.")

    out_path = Path(processed_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    standard_df.to_csv(out_path, index=False)
    print(f"[cgwb_fetch] Saved standardized well CSV -> {out_path}")
    print("[cgwb_fetch] Next step: run held_out_split.py --raw_csv "
          f"{out_path} to generate the immutable train/held-out split.")

    return standard_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and standardize CGWB well data.")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--raw_out", default="data/raw/cgwb_raw.csv")
    parser.add_argument("--processed_out", default="data/interim/cgwb_wells.csv")
    parser.add_argument("--no_clip", action="store_true", help="Skip clipping to study region bbox.")
    args = parser.parse_args()

    fetch_and_prepare(
        config_path=args.config,
        raw_out_path=args.raw_out,
        processed_out_path=args.processed_out,
        clip_to_study_region=not args.no_clip,
    )