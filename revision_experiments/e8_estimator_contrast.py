"""
E8 (v2): paired estimator contrast on the same 225 epsilon test starts.

Reads FQE per-seed diffs from E2/e2_fqe.json's
primary_episode_starts.per_seed_diffs, indexed by SEEDS order.
Rollout per-seed gaps come from e3c_test225_constructions.json.

Run from C:\\trishna-opal:
    python revision_experiments\\e8_estimator_contrast.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stats_utils import t_interval


SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]


def main():
    # Rollout per-seed gaps
    e3c_path = REPO / "revision_experiments" / "e3c_test225_constructions.json"
    e3c = json.loads(e3c_path.read_text())
    roll = {int(r["seed"]): float(r["mean_gap"]) for r in e3c["per_seed"]}
    print(f"rollout per-seed gaps loaded: {len(roll)} seeds")

    # FQE per-seed diffs — direct from primary_episode_starts
    fqe_path = REPO / "revision_experiments" / "E2" / "e2_fqe.json"
    if not fqe_path.exists():
        fqe_path = REPO / "revision_experiments" / "e2_fqe.json"
    fqe = json.loads(fqe_path.read_text())
    prim = fqe["primary_episode_starts"]
    fqe_diffs = prim["per_seed_diffs"]
    if len(fqe_diffs) != len(SEEDS):
        print(f"warn: {len(fqe_diffs)} FQE diffs vs {len(SEEDS)} seeds; "
              "truncating to the smaller")
    fqe_gap = {s: float(d) for s, d in zip(SEEDS, fqe_diffs)}
    print(f"FQE per-seed gaps loaded: {len(fqe_gap)} seeds")
    print(f"FQE seed_mean from json: {prim['seed_mean']:+.4f} "
          f"[{prim['seed_ci_lower']:+.4f}, {prim['seed_ci_upper']:+.4f}]")

    # Pair by seed
    seeds = [s for s in SEEDS if s in roll and s in fqe_gap]
    deltas = np.array([fqe_gap[s] - roll[s] for s in seeds])
    m, lo, hi = t_interval(deltas)
    flag = "excl 0" if (lo > 0 or hi < 0) else "spans 0"

    print(f"\npaired seeds: {seeds}")
    print(f"\nFQE gap - rollout gap, per-seed paired (n={len(seeds)}):")
    print(f"  mean  {m:+.4f}")
    print(f"  95%   [{lo:+.4f}, {hi:+.4f}]  {flag}")
    print(f"\nper-seed deltas:")
    for s, d in zip(seeds, deltas):
        print(f"  seed {s:>6d}: FQE {fqe_gap[s]:+.4f}  "
              f"rollout {roll[s]:+.4f}  delta {d:+.4f}")

    out = {
        "paired_seeds": seeds,
        "per_seed_fqe_gap": [fqe_gap[s] for s in seeds],
        "per_seed_rollout_gap": [roll[s] for s in seeds],
        "deltas": deltas.tolist(),
        "mean": float(m), "lo": float(lo), "hi": float(hi),
        "construction": "seed t-interval on FQE_gap - rollout_gap",
    }
    (REPO / "revision_experiments" / "e8_estimator_contrast.json").write_text(
        json.dumps(out, indent=2))
    print("\nwrote e8_estimator_contrast.json")


if __name__ == "__main__":
    main()