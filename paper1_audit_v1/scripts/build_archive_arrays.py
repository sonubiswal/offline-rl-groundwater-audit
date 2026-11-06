#!/usr/bin/env python3
"""build_archive_arrays.py — assemble per-seed NPZ files for the archive.

Reads from the source tree at C:\\trishna-opal (unchanged).
Writes into ../per_seed_arrays/ relative to this script.
"""
import json
from pathlib import Path
import numpy as np

# Script lives at paper1_audit_v1/scripts/build_archive_arrays.py
# Archive root is its parent's parent.
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
OUT = ARCHIVE / "per_seed_arrays"
OUT.mkdir(parents=True, exist_ok=True)

# Source tree stays where it is.
REV = Path(r"C:\trishna-opal\revision_experiments")

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]

# --- matched_start_225.npz ---
roll = json.loads((REV / "e3c_test225_constructions_rerun.json").read_text())
roll_arr = np.array([r["mean_gap"] for r in roll["per_seed"]])

fqe200 = json.loads((REV / "E2" / "e2_fqe.json").read_text())
fqe200_arr = np.array(fqe200["primary_episode_starts"]["per_seed_diffs"])

def _fqe_gap(tui: int, seed: int) -> float:
    z = np.load(REV / "E2" / "seeds" / f"tui{tui}" / f"seed_{seed}.npz")
    return float((z["cql_q_starts"] - z["bc_q_starts"]).mean())

fqe100 = np.array([_fqe_gap(100, s) for s in SEEDS])
fqe500 = np.array([_fqe_gap(500, s) for s in SEEDS])

np.savez_compressed(OUT / "matched_start_225.npz",
                    seeds=np.array(SEEDS),
                    rollout=roll_arr,
                    fqe_100=fqe100,
                    fqe_200=fqe200_arr,
                    fqe_500=fqe500)
print("wrote matched_start_225.npz")

# --- family_start_grid.npz ---
z = np.load(REV / "family_start_rerun.npz")
np.savez_compressed(OUT / "family_start_grid.npz",
                    **{k: z[k] for k in z.files})
print(f"wrote family_start_grid.npz ({len(z.files)} arrays)")

# --- band_decomposition.npz ---
rem  = json.loads((REV / "ood_w3_zero_eval_only.json").read_text())
full = json.loads((REV / "ood_w3_full_eval_only.json").read_text())

endpoint_w0  = np.array(rem["band_per_seed"]["severe"])  - np.array(rem["band_per_seed"]["in_dist"])
endpoint_w05 = np.array(full["band_per_seed"]["severe"]) - np.array(full["band_per_seed"]["in_dist"])

np.savez_compressed(OUT / "band_decomposition.npz",
                    endpoint_w0=endpoint_w0,
                    endpoint_w05=endpoint_w05,
                    in_dist_w0=np.array(rem["band_per_seed"]["in_dist"]),
                    in_dist_w05=np.array(full["band_per_seed"]["in_dist"]),
                    mild_w0=np.array(rem["band_per_seed"]["mild"]),
                    mild_w05=np.array(full["band_per_seed"]["mild"]),
                    moderate_w0=np.array(rem["band_per_seed"]["moderate"]),
                    moderate_w05=np.array(full["band_per_seed"]["moderate"]),
                    severe_w0=np.array(rem["band_per_seed"]["severe"]),
                    severe_w05=np.array(full["band_per_seed"]["severe"]))
print("wrote band_decomposition.npz")

print(f"\narchive arrays written to {OUT}")