"""
run_bc_sweep.py

Does the base-vs-no_oracle sign flip depend on BC tuning? Trains BC at
each (lr, hidden) combination and re-evaluates CQL-BC against canonical
CQL checkpoints.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A
from stats_utils import hierarchical_bootstrap


GRID = list(itertools.product([1e-3, 3e-4, 1e-4], [(256, 256), (128, 128)]))
FAMILIES = ["base", "epsilon", "no_oracle"]
N_EPISODES = 500

# Per-dataset block structure. bc_baseline.py asserts
# len(policy_order) * n_per_policy == n_episodes in the dataset, so the
# no_oracle (1000-episode) file needs its own layout.
POLICY_LAYOUTS = {
    "base":      (["random", "greedy_extraction", "conservative"], 500),
    "epsilon":   (["random", "greedy_extraction", "conservative"], 500),
    "no_oracle": (["greedy_extraction", "conservative"], 500),
}


def main():
    out = {"grid": [], "protocol": {"n_episodes": N_EPISODES}}

    for lr, hidden in GRID:
        for fam in FAMILIES:
            ds = A.DATASETS[fam]
            po, npp = POLICY_LAYOUTS[fam]
            diffs_by_seed = []
            per_seed = {}
            for s in A.SEEDS:
                bc_path = REPO / "reports_2" / "bc_sweep" / \
                    f"{fam}_lr{lr}_h{hidden[0]}_s{s}.d3"
                bc_path.parent.mkdir(parents=True, exist_ok=True)
                if not bc_path.exists():
                    A.train_bc(ds, bc_path, s, lr=lr, hidden=hidden,
                               policy_order=po, n_per_policy=npp)
                bc = A.load_policy(bc_path, ds)
                cq = A.load_policy(A.checkpoint(fam, "cql", s), ds)
                bc_r = A.rollout_returns(bc, N_EPISODES, s)
                cq_r = A.rollout_returns(cq, N_EPISODES, s)
                diffs_by_seed.append(cq_r - bc_r)
                per_seed[s] = float((cq_r - bc_r).mean())

            m, lo, hi = hierarchical_bootstrap(diffs_by_seed)
            flag = "excludes 0" if lo > 0 or hi < 0 else "spans 0"
            print(f"lr={lr} h={hidden} fam={fam:9s} "
                  f"CQL-BC {m:+.4f} [{lo:+.4f},{hi:+.4f}] {flag}")
            out["grid"].append({"lr": lr, "hidden": list(hidden), "family": fam,
                                "cql_minus_bc": m, "lo": lo, "hi": hi,
                                "per_seed": per_seed})

    (REPO / "revision_experiments" / "bc_sweep.json").write_text(
        json.dumps(out, indent=2))
    print("\n[bc_sweep] wrote bc_sweep.json")


if __name__ == "__main__":
    main()