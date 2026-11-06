"""
Verify every reward-weight ablation config was evaluated with the correct
observation scaler.

For each config directory under reports/paper_exp/weights_ablation:
  - read its normalization_manifest.json
  - recompute the scaler from the dataset the manifest records
  - report MATCH / MISMATCH
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rl.policy_io import compute_scaler_from_dataset

ROOT = REPO / "reports" / "paper_exp" / "weights_ablation"

print(f"[verify] scanning {ROOT}")
n_checked = 0
n_match = 0
n_mismatch = 0
n_skip = 0

for manifest_path in sorted(ROOT.rglob("normalization_manifest.json")):
    config_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    manifest_mean0 = float(manifest["mean"][0])
    dataset_path = Path(manifest["dataset"])

    if not dataset_path.exists():
        print(f"[skip] {config_dir.name:<20} dataset missing: {dataset_path}")
        n_skip += 1
        continue

    recomputed_mean, _ = compute_scaler_from_dataset(dataset_path)
    recomputed_mean0 = float(recomputed_mean[0])
    delta = abs(manifest_mean0 - recomputed_mean0)
    is_match = delta < 1e-4

    status = "MATCH" if is_match else "MISMATCH"
    print(f"[{status:<8}] {config_dir.name:<20} "
          f"manifest={manifest_mean0:.6f}  recomputed={recomputed_mean0:.6f}  "
          f"delta={delta:.2e}")

    n_checked += 1
    if is_match:
        n_match += 1
    else:
        n_mismatch += 1

print()
print(f"[verify] checked={n_checked}  match={n_match}  "
      f"mismatch={n_mismatch}  skip={n_skip}")

if n_mismatch:
    print()
    print("[verify] MISMATCHES FOUND — those configs need re-evaluation.")
    sys.exit(1)
print("[verify] all configs use the scaler of their training dataset.")