"""
E4: absolute rollout means at the same 225 epsilon test starts FQE uses.
Fills the empty Table IX by providing BC and CQL means on an aligned
start set.

Run from C:\\trishna-opal:
    python revision_experiments\\e4_absolute_rollout.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A


def main():
    fq_path = REPO / "revision_experiments" / "E2" / "e2_fqe.json"
    if not fq_path.exists():
        fq_path = REPO / "revision_experiments" / "e2_fqe.json"
    fq = json.loads(fq_path.read_text()) if fq_path.exists() else None

    if fq is not None:
        print("FQE json keys:", list(fq)[:20])
    else:
        print(f"FQE json not found at {fq_path}")

    starts = A._load_test_episode_starts(A.DATASETS["epsilon"], n=225)
    print(f"loaded {len(starts)} test225 starts")

    bc_means, cq_means = [], []
    for s in A.SEEDS:
        ds = A.DATASETS["epsilon"]
        bc = A.load_policy(A.checkpoint("epsilon", "bc", s), ds)
        cq = A.load_policy(A.checkpoint("epsilon", "cql", s), ds)
        bc_r = A.rollout_returns(bc, len(starts), s, start_set=starts)
        cq_r = A.rollout_returns(cq, len(starts), s, start_set=starts)
        bc_means.append(float(bc_r.mean()))
        cq_means.append(float(cq_r.mean()))

    print(f"\nrollout@test225, n=10 seeds:")
    print(f"  BC  mean = {np.mean(bc_means):+.4f}  (per-seed sd {np.std(bc_means, ddof=1):.4f})")
    print(f"  CQL mean = {np.mean(cq_means):+.4f}  (per-seed sd {np.std(cq_means, ddof=1):.4f})")
    print(f"  CQL-BC   = {np.mean(cq_means) - np.mean(bc_means):+.4f}")

    out = {
        "protocol": "rollout on 225 epsilon test starts, 20-step horizon",
        "n_seeds": len(A.SEEDS),
        "per_seed_bc_mean": bc_means,
        "per_seed_cql_mean": cq_means,
        "mean_bc": float(np.mean(bc_means)),
        "mean_cql": float(np.mean(cq_means)),
        "gap": float(np.mean(cq_means) - np.mean(bc_means)),
    }
    path = REPO / "revision_experiments" / "e4_rollout_test225.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.name}")
    print("Compare these means with the FQE per-policy means printed above.")


if __name__ == "__main__":
    main()