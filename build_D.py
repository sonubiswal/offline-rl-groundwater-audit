#!/usr/bin/env python3
"""build_D.py — assemble (3, 10, 225) paired contrast array from E2 layout."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
INTERVALS = [100, 200, 500]

E2 = Path(r"C:\trishna-opal\revision_experiments\E2")
OUT = Path(r"C:\trishna-opal\results\per_seed.npz")

# Published summaries from Table VII/VIII, for the sanity check.
PUB = {
    100: dict(fqe_gap=+0.8984, contrast=+0.9879),
    200: dict(fqe_gap=+0.3478, contrast=+0.4373),
    500: dict(fqe_gap=+0.8226, contrast=+0.9121),
}

def load_main(seed):
    p = E2 / f"seed_{seed}.npz"
    if not p.exists():
        sys.exit(f"missing {p}")
    z = np.load(p)
    return (z["bc_q_starts"], z["cql_q_starts"],
            z["bc_rollout_starts"], z["cql_rollout_starts"])

def load_tui(seed, tui):
    p = E2 / "seeds" / f"tui{tui}" / f"seed_{seed}.npz"
    if not p.exists():
        sys.exit(f"missing {p}")
    z = np.load(p)
    return z["bc_q_starts"], z["cql_q_starts"]

D = np.zeros((len(INTERVALS), len(SEEDS), 225), dtype=np.float64)
fqe_gap = np.zeros((len(INTERVALS), len(SEEDS)), dtype=np.float64)

for k, tui in enumerate(INTERVALS):
    for j, seed in enumerate(SEEDS):
        # rollout marginals come from the main file
        _, _, bc_r, cq_r = load_main(seed)
        rollout_diff = cq_r - bc_r                 # (225,)
        # FQE at this interval
        bc_q, cq_q = load_tui(seed, tui)
        fqe_diff = cq_q - bc_q                     # (225,)
        D[k, j] = fqe_diff - rollout_diff
        fqe_gap[k, j] = fqe_diff.mean()

print("Reconstructed FQE CQL-BC gaps (per interval, mean over seeds):")
for k, tui in enumerate(INTERVALS):
    m = fqe_gap[k].mean()
    print(f"  tui={tui:>3}: fqe_gap={m:+.4f}  (published {PUB[tui]['fqe_gap']:+.4f})")

print("\nReconstructed paired contrasts (mean over seeds):")
for k, tui in enumerate(INTERVALS):
    m = D[k].mean(axis=1).mean()
    print(f"  tui={tui:>3}: contrast={m:+.4f}  (published {PUB[tui]['contrast']:+.4f})")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, D=D)
print(f"\nwrote {OUT}")