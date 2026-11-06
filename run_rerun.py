#!/usr/bin/env python3
"""run_rerun.py — re-run the e3c matched-start analysis with the patched
(disjoint-stream) adapters, and print the paired FQE-rollout contrast.

Reads : models_10seed/epsilon/{bc,cql}/... (unchanged)
Writes: revision_experiments/e3c_test225_constructions_rerun.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\trishna-opal")
sys.path.insert(0, str(REPO / "revision_experiments"))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A                     # noqa: E402
from stats_utils import t_interval        # noqa: E402

FAM = "epsilon"
N_STARTS = 225
SEEDS = A.SEEDS

OUT = REPO / "revision_experiments" / "e3c_test225_constructions_rerun.json"


def main() -> None:
    starts = A._load_test_episode_starts(A.DATASETS[FAM], n=N_STARTS)
    print(f"loaded {len(starts)} epsilon test starts")

    diffs_by_seed = []
    per_seed = []
    for s in SEEDS:
        ds = A.DATASETS[FAM]
        bc = A.load_policy(A.checkpoint(FAM, "bc", s), ds)
        cq = A.load_policy(A.checkpoint(FAM, "cql", s), ds)
        bc_r = A.rollout_returns(bc, len(starts), s, start_set=starts)
        cq_r = A.rollout_returns(cq, len(starts), s, start_set=starts)
        d = cq_r - bc_r
        diffs_by_seed.append(d)
        per_seed.append({"seed": int(s), "mean_gap": float(d.mean())})
        print(f"  seed {s:>6}: rollout gap = {d.mean():+.4f}")

    seed_means = np.array([d.mean() for d in diffs_by_seed])
    m, lo, hi = t_interval(seed_means)
    print(f"\nrollout gap (mean over seeds): {m:+.4f}")
    print(f"rollout seed-t interval     : [{lo:+.4f}, {hi:+.4f}]")
    print(f"original (colliding) gap    : -0.0895")

    payload = {
        "protocol": {
            "family": FAM,
            "n_starts": N_STARTS,
            "rollout_seed_scheme": "RandomState(200000 + 1000*seed + i)",
            "patched": True,
        },
        "n_seeds": len(SEEDS),
        "rollout_seed_t": {"mean": m, "lo": lo, "hi": hi},
        "per_seed": per_seed,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()