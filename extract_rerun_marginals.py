#!/usr/bin/env python3
"""extract_rerun_marginals.py — BC and CQL rollout means at test225 starts
under the disjoint-stream scheme. Fills Table VIII rollout columns."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\trishna-opal")
sys.path.insert(0, str(REPO / "revision_experiments"))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A  # noqa: E402

FAM = "epsilon"
N = 225
SEEDS = A.SEEDS
ds = A.DATASETS[FAM]

starts = A._load_test_episode_starts(ds, n=N)
print(f"loaded {len(starts)} epsilon test starts\n")

bc_means, cql_means = [], []
for s in SEEDS:
    bc = A.load_policy(A.checkpoint(FAM, "bc", s), ds)
    cq = A.load_policy(A.checkpoint(FAM, "cql", s), ds)
    bc_r = A.rollout_returns(bc, len(starts), s, start_set=starts)
    cq_r = A.rollout_returns(cq, len(starts), s, start_set=starts)
    bc_means.append(bc_r.mean())
    cql_means.append(cq_r.mean())
    print(f"  seed {s:>6}: BC {bc_r.mean():+.4f}  "
          f"CQL {cq_r.mean():+.4f}  gap {cq_r.mean() - bc_r.mean():+.4f}")

bc_m = float(np.mean(bc_means))
cql_m = float(np.mean(cql_means))
print()
print(f"Rollout BC mean   = {bc_m:+.4f}")
print(f"Rollout CQL mean  = {cql_m:+.4f}")
print(f"Rollout gap       = {cql_m - bc_m:+.4f}")
print()
print("Reference:")
print("  colliding scheme: BC +9.722  CQL +9.632  gap -0.0895")
print(f"  rerun           : BC {bc_m:+.3f}  CQL {cql_m:+.3f}  "
      f"gap {cql_m - bc_m:+.4f}")