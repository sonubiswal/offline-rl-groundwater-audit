"""
Verify every .d3.scaler.json sidecar in the project matches the scaler
recomputed from the dataset it records.

Walks every directory under the project that contains .d3 checkpoints,
reads sidecars, recomputes scalers, and reports MATCH / MISMATCH / SKIP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.rl.policy_io import compute_scaler_from_dataset

# Any directory that might contain .d3 checkpoints.
SEARCH_ROOTS = [
    REPO / "models",
    REPO / "models_epsilon",
    REPO / "models_epsilon_meanq_verify",
    REPO / "models_epsilon_qr32_verify",
    REPO / "models_meanq_control_test",
    REPO / "models_no_oracle",
    REPO / "models_no_oracle_alpha1",
    REPO / "models_no_oracle_alpha10",
    REPO / "models_probe_A",
    REPO / "models_probe_B",
    REPO / "models_probe_C",
    REPO / "models_probe_D",
    REPO / "models_qr32_test",
    REPO / "models_tui500",
    REPO / "models_tui8000",
    REPO / "reports" / "paper_exp",
]

print(f"[J] scanning {len(SEARCH_ROOTS)} roots")
n_match = 0
n_mismatch = 0
n_skip = 0
mismatches = []

for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    for sidecar in sorted(root.rglob("*.d3.scaler.json")):
        try:
            manifest = json.loads(sidecar.read_text())
        except Exception as e:
            print(f"[ERROR] {sidecar}: {e}")
            continue

        dataset_path = Path(manifest["dataset"])
        if not dataset_path.exists():
            n_skip += 1
            continue

        manifest_mean = np.asarray(manifest["mean"], dtype=np.float32)
        manifest_std = np.asarray(manifest["std"], dtype=np.float32)

        try:
            recomputed_mean, recomputed_std = compute_scaler_from_dataset(dataset_path)
        except Exception as e:
            print(f"[ERROR] {sidecar}: recompute failed: {e}")
            n_skip += 1
            continue

        mean_ok = np.allclose(manifest_mean, recomputed_mean, atol=1e-3)
        std_ok = np.allclose(manifest_std, recomputed_std, atol=1e-3)

        if mean_ok and std_ok:
            n_match += 1
        else:
            n_mismatch += 1
            mismatches.append((sidecar, manifest_mean[0], recomputed_mean[0]))

print()
print(f"[J] MATCH    = {n_match}")
print(f"[J] MISMATCH = {n_mismatch}")
print(f"[J] SKIP     = {n_skip}")

if mismatches:
    print()
    print("[J] Mismatches:")
    for path, m_manifest, m_recomp in mismatches:
        print(f"  {path}")
        print(f"    manifest mean[0] = {m_manifest:.6f}")
        print(f"    recomputed mean[0] = {m_recomp:.6f}")
    sys.exit(1)

print("[J] all sidecars match their recorded dataset.")