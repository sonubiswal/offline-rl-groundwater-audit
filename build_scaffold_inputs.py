#!/usr/bin/env python3
"""
build_scaffold_inputs.py
========================
Adapter: build the three files the QC scaffold expects from TRISHNA-OPAL's
real data layout.

Produces:
  <out>/data/cgwb_observations.csv
  <out>/data/feature_table.parquet
  <out>/data/split_assignments.csv

Usage:
  python build_scaffold_inputs.py --trishna-root C:\\trishna-opal --out C:\\trishna-qc\\REAL_INPUTS
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trishna-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.trishna_root)
    out = Path(args.out) / "data"
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 1. CGWB observations -----------------------------------
    cgwb_p = root / "data" / "interim" / "cgwb_wells.csv"
    cgwb = pd.read_csv(cgwb_p)
    print(f"[adapter] read {cgwb_p}: {len(cgwb)} rows, cols={cgwb.columns.tolist()}")

    cgwb = cgwb.rename(columns={"date": "obs_date", "depth_m": "gwl_depth_m"})
    cgwb["obs_date"] = pd.to_datetime(cgwb["obs_date"], errors="coerce")
    cgwb.to_csv(out / "cgwb_observations.csv", index=False)
    print(f"[adapter] wrote cgwb_observations.csv: "
          f"{len(cgwb)} rows, {cgwb['well_id'].nunique()} unique wells")

    # ---------- 2. Well-level feature table ----------------------------
    tr_p = root / "data" / "processed" / "well_level_features_train.csv"
    ho_p = root / "data" / "processed" / "well_level_features_holdout.csv"
    train = pd.read_csv(tr_p)
    hold = pd.read_csv(ho_p)
    print(f"[adapter] {tr_p.name}: {len(train)} rows, "
          f"{train['well_id'].nunique()} wells")
    print(f"[adapter] {ho_p.name}: {len(hold)} rows, "
          f"{hold['well_id'].nunique()} wells")

    feats = pd.concat([train, hold], ignore_index=True)
    feats = feats.rename(columns={"date": "obs_date", "depth_m": "gwl_depth_m"})
    feats["obs_date"] = pd.to_datetime(feats["obs_date"], errors="coerce")

    # Derive month_sin / month_cos so the feature set is closer to the paper's
    month = feats["obs_date"].dt.month
    feats["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Prefix every model feature with f_ (the scaffold filters by this prefix)
    base_features = [
        "chirps", "gldas", "sentinel2_ndvi", "grace",
        "srtm_elevation", "srtm_slope",
        "chirps_roll3", "gldas_roll3", "target_lag1",
        "month_sin", "month_cos",
    ]
    rename_map = {c: f"f_{c}" for c in base_features if c in feats.columns}
    feats = feats.rename(columns=rename_map)

    feats.to_parquet(out / "feature_table.parquet", index=False)
    print(f"[adapter] wrote feature_table.parquet: {len(feats)} rows, "
          f"{feats['well_id'].nunique()} wells")
    print(f"[adapter]   feature cols: {sorted(rename_map.values())}")

    # ---------- 3. Split assignments -----------------------------------
    tw_p = root / "data" / "processed" / "training_wells.csv"
    train_wells = set(pd.read_csv(tw_p)["well_id"].astype(str))

    feat_wells = set(feats["well_id"].astype(str))

    # Only keep wells that actually appear in the feature table
    split = pd.DataFrame({"well_id": sorted(feat_wells)})
    split["fold_id"] = split["well_id"].apply(
        lambda w: 1 if w in train_wells else 0
    )

    split.to_csv(out / "split_assignments.csv", index=False)
    n_train = int((split["fold_id"] == 1).sum())
    n_held = int((split["fold_id"] == 0).sum())

    print()
    print("=" * 60)
    print(f"[adapter] split_assignments.csv: train={n_train}, heldout={n_held}")
    print(f"[adapter] paper claims: train=3353, heldout=710")
    print(f"[adapter] USE THESE ARGS for the QC run:")
    print(f"[adapter]   --expected-train {n_train} --expected-heldout {n_held}")
    print("=" * 60)


if __name__ == "__main__":
    main()