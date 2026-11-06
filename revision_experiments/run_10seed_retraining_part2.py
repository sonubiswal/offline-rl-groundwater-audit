"""
Retrain BC and CQL for the no_oracle family with 7 additional seeds.

The no_oracle dataset has 1000 episodes (2 policies x 500), not 1500,
so the default 3-policy layout does not apply. Pass the correct layout
explicitly.

Usage:
    python revision_experiments/run_10seed_retraining_part2.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEW_SEEDS = "7,99,777,1234,5678,12345,99999"

# no_oracle family: 1000 episodes = 2 policies x 500
POLICY_ORDER = ["random", "greedy_extraction"]
N_PER_POLICY = 500

FAMILIES = [
    ("no_oracle",     "data/processed/offline_rl_dataset_no_oracle.h5",
     "models_10seed/no_oracle/bc",     "models_10seed/no_oracle/cql",     4.0),
    ("no_oracle_a1",  "data/processed/offline_rl_dataset_no_oracle.h5",
     "models_10seed/no_oracle_a1/bc",  "models_10seed/no_oracle_a1/cql",  1.0),
    ("no_oracle_a10", "data/processed/offline_rl_dataset_no_oracle.h5",
     "models_10seed/no_oracle_a10/bc", "models_10seed/no_oracle_a10/cql", 10.0),
]


def run_one(name, dataset, bc_dir, cql_dir, alpha):
    print(f"\n{'='*70}\n[10seed-2] config={name}  dataset={dataset}  alpha={alpha}\n{'='*70}")

    env = dict(os.environ)
    env["TRAINING_SEEDS_OVERRIDE"] = NEW_SEEDS

    # BC
    print(f"[10seed-2] training BC -> {bc_dir}")
    bc_cmd = [
        sys.executable, "src/rl/bc_baseline.py",
        "--dataset", dataset,
        "--model-dir", bc_dir,
        "--out", f"{bc_dir}/bc_baseline_results.json",
        "--policy-order", *POLICY_ORDER,
        "--n-per-policy", str(N_PER_POLICY),
    ]
    r = subprocess.run(bc_cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"[10seed-2] BC FAILED for {name}:\n{r.stderr}")
        return False

    # CQL
    print(f"[10seed-2] training CQL -> {cql_dir}")
    cql_cmd = [
        sys.executable, "src/rl/train_cql.py",
        "--dataset", dataset,
        "--model-dir", cql_dir,
        "--out", f"{cql_dir}/cql_results.json",
        "--alpha", str(alpha),
        "--skip-sweep",
        "--policy-order", *POLICY_ORDER,
        "--n-per-policy", str(N_PER_POLICY),
    ]
    r = subprocess.run(cql_cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"[10seed-2] CQL FAILED for {name}:\n{r.stderr}")
        return False
    return True


def main():
    for name, dataset, bc_dir, cql_dir, alpha in FAMILIES:
        if not run_one(name, dataset, bc_dir, cql_dir, alpha):
            print(f"[10seed-2] ABORT: {name} failed")
            return
    print("\n[10seed-2] all remaining configs trained")


if __name__ == "__main__":
    main()