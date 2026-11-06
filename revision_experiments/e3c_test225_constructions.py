"""
E3c: recompute rollout constructions on the same 225 epsilon test starts
that FQE uses, replacing Table VIII's unrecorded-.npy protocol.

Run from C:\\trishna-opal:
    python revision_experiments\\e3c_test225_constructions.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A
from stats_utils import (episode_bootstrap, seed_bootstrap,
                         hierarchical_bootstrap, t_interval)


FAM = "epsilon"
N_STARTS = 225


def main():
    starts = A._load_test_episode_starts(A.DATASETS[FAM], n=N_STARTS)
    print(f"loaded {len(starts)} epsilon test starts")

    diffs_by_seed = []
    for s in A.SEEDS:
        ds = A.DATASETS[FAM]
        bc = A.load_policy(A.checkpoint(FAM, "bc", s), ds)
        cq = A.load_policy(A.checkpoint(FAM, "cql", s), ds)
        bc_r = A.rollout_returns(bc, len(starts), s, start_set=starts)
        cq_r = A.rollout_returns(cq, len(starts), s, start_set=starts)
        diffs_by_seed.append(cq_r - bc_r)

    pooled = np.concatenate(diffs_by_seed)
    seed_means = [float(d.mean()) for d in diffs_by_seed]

    rows = {}
    rows["rollout_pooled"] = episode_bootstrap(pooled)
    rows["rollout_seed_t"] = t_interval(seed_means)
    rows["rollout_hierarchical"] = hierarchical_bootstrap(diffs_by_seed)

    print("\nTable VIII replacement (rollout on test225):")
    for name, (m, lo, hi) in rows.items():
        flag = "excl 0" if lo > 0 or hi < 0 else "spans 0"
        print(f"  {name:25s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] {flag}")

    # Per-seed absolute means for reference
    per_seed = []
    for i, s in enumerate(A.SEEDS):
        per_seed.append({"seed": s, "mean_gap": seed_means[i]})
    out = {
        "protocol": "rollout on 225 epsilon test starts, 20-step horizon",
        "n_seeds": len(A.SEEDS),
        "constructions": {
            "pooled": {"mean": rows["rollout_pooled"][0],
                       "lo": rows["rollout_pooled"][1],
                       "hi": rows["rollout_pooled"][2]},
            "seed_t": {"mean": rows["rollout_seed_t"][0],
                       "lo": rows["rollout_seed_t"][1],
                       "hi": rows["rollout_seed_t"][2]},
            "hierarchical": {"mean": rows["rollout_hierarchical"][0],
                             "lo": rows["rollout_hierarchical"][1],
                             "hi": rows["rollout_hierarchical"][2]},
        },
        "per_seed": per_seed,
    }
    path = REPO / "revision_experiments" / "e3c_test225_constructions.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.name}")


if __name__ == "__main__":
    main()