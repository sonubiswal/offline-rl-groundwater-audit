"""
Backfill .scaler.json sidecars for every archived .d3 checkpoint.

Run once. Each checkpoint gets a sidecar recording the scaler it was
trained with, computed from the dataset it was trained on.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.rl.policy_io import compute_scaler_from_dataset, save_scaler_sidecar


# Map: checkpoint family -> (checkpoint dir, dataset path)
FAMILIES = {
    "models/bc":                 ("data/processed/offline_rl_dataset.h5"),
    "models/cql":                ("data/processed/offline_rl_dataset.h5"),
    "models_epsilon/bc":         ("data/processed/offline_rl_dataset_epsilon.h5"),
    "models_epsilon/cql":        ("data/processed/offline_rl_dataset_epsilon.h5"),
    "models_no_oracle/bc":       ("data/processed/offline_rl_dataset_no_oracle.h5"),
    "models_no_oracle/cql":      ("data/processed/offline_rl_dataset_no_oracle.h5"),
    "models_no_oracle_alpha1":   ("data/processed/offline_rl_dataset_no_oracle.h5"),
    "models_no_oracle_alpha10":  ("data/processed/offline_rl_dataset_no_oracle.h5"),
    "reports/paper_exp/frozen_models/bc":  ("reports/paper_exp/dataset_normal_train.h5"),
    "reports/paper_exp/frozen_models/cql": ("reports/paper_exp/dataset_normal_train.h5"),
    "reports/paper_exp/cql_mean":          ("reports/paper_exp/dataset_normal_train.h5"),
    "reports/paper_exp/cql_qr":            ("reports/paper_exp/dataset_normal_train.h5"),
}


def main():
    for dir_rel, ds_rel in FAMILIES.items():
        dir_abs = REPO / dir_rel
        ds_abs = REPO / ds_rel
        if not dir_abs.exists():
            print(f"[skip] {dir_rel} not found"); continue
        if not ds_abs.exists():
            print(f"[skip] dataset {ds_rel} not found"); continue

        print(f"\n=== {dir_rel}  (dataset: {ds_rel}) ===")
        mean, std = compute_scaler_from_dataset(ds_abs)
        print(f"  scaler mean[0]={mean[0]:.6f}  std[0]={std[0]:.6f}")

        for ckpt in sorted(dir_abs.glob("*.d3")):
            save_scaler_sidecar(ckpt, mean, std, ds_abs)
            print(f"  wrote {ckpt.name}.scaler.json")

    print("\n[done] all sidecars written.")


if __name__ == "__main__":
    main()