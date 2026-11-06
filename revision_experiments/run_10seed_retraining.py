"""
Retrain BC and CQL with 7 additional seeds for each of the 5 dataset
configurations. Reads configurations from the FAMILIES table below,
calls bc_baseline.py and train_cql.py via subprocess, and writes results
into models_10seed/<config>/{bc,cql}/.

Each checkpoint gets a .d3.scaler.json sidecar (written by the training
script's policy_io integration).

Usage:
    python revision_experiments/run_10seed_retraining.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEW_SEEDS = "7,99,777,1234,5678,12345,99999"
ALL_SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]

# (config_name, dataset_path, bc_model_dir, cql_model_dir, cql_alpha)
FAMILIES = [
    ("base",          "data/processed/offline_rl_dataset.h5",          "models_10seed/base/bc",          "models_10seed/base/cql",          1.0),
    ("epsilon",       "data/processed/offline_rl_dataset_epsilon.h5",  "models_10seed/epsilon/bc",       "models_10seed/epsilon/cql",       1.0),
    ("no_oracle",     "data/processed/offline_rl_dataset_no_oracle.h5","models_10seed/no_oracle/bc",     "models_10seed/no_oracle/cql",     4.0),
    ("no_oracle_a1",  "data/processed/offline_rl_dataset_no_oracle.h5","models_10seed/no_oracle_a1/bc",  "models_10seed/no_oracle_a1/cql",  1.0),
    ("no_oracle_a10", "data/processed/offline_rl_dataset_no_oracle.h5","models_10seed/no_oracle_a10/bc", "models_10seed/no_oracle_a10/cql", 10.0),
]


def run_one(name, dataset, bc_dir, cql_dir, alpha):
    print(f"\n{'='*70}")
    print(f"[10seed] config={name}  dataset={dataset}  alpha={alpha}")
    print(f"{'='*70}")

    env = dict(os.environ)
    env["TRAINING_SEEDS_OVERRIDE"] = NEW_SEEDS

    # BC
    print(f"[10seed] training BC -> {bc_dir}")
    bc_cmd = [
        sys.executable, "src/rl/bc_baseline.py",
        "--dataset", dataset,
        "--model-dir", bc_dir,
        "--out", f"{bc_dir}/bc_baseline_results.json",
    ]
    r = subprocess.run(bc_cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"[10seed] BC FAILED for {name}:\n{r.stderr}")
        return False

    # CQL
    print(f"[10seed] training CQL -> {cql_dir}")
    cql_cmd = [
        sys.executable, "src/rl/train_cql.py",
        "--dataset", dataset,
        "--model-dir", cql_dir,
        "--out", f"{cql_dir}/cql_results.json",
        "--alpha", str(alpha),
        "--skip-sweep",
    ]
    r = subprocess.run(cql_cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"[10seed] CQL FAILED for {name}:\n{r.stderr}")
        return False

    return True


def main():
    for name, dataset, bc_dir, cql_dir, alpha in FAMILIES:
        ok = run_one(name, dataset, bc_dir, cql_dir, alpha)
        if not ok:
            print(f"[10seed] ABORT: {name} failed")
            return

    print("\n[10seed] all configs trained")
    print("[10seed] next: copy the original 3-seed checkpoints into the same dirs,")
    print("[10seed]       then run the compare_policies.py with --all-seeds.")


if __name__ == "__main__":
    main()