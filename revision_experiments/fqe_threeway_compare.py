"""
Three-way FQE sensitivity comparison: tui100 vs tui200 vs tui500.
Prints the FQE marginal gap, the paired contrast against rollout, the
95% CI, positive-seed count, and per-seed SD for each configuration.

Run from C:\\trishna-opal:
    python revision_experiments\\fqe_threeway_compare.py
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
    e3c = json.loads((REPO / "revision_experiments"
                      / "e3c_test225_constructions.json").read_text())
    roll = {int(r["seed"]): float(r["mean_gap"]) for r in e3c["per_seed"]}

    hdr = f"{'tag':8s} {'FQE gap':>10s} {'paired':>10s} {'95% CI':>24s} {'seeds+':>7s} {'SD':>6s}  flag"
    print(hdr)
    print("-" * len(hdr))

    for tag in ["tui100", "tui200", "tui500"]:
        path = REPO / "revision_experiments" / "E2" / f"e2_fqe_{tag}.json"
        if not path.exists():
            print(f"{tag:8s} [not found]")
            continue
        d = json.loads(path.read_text())["primary_episode_starts"]
        deltas = np.array([f - roll[s] for f, s in zip(d["per_seed_diffs"], SEEDS)])
        m, lo, hi = t_interval(deltas)
        pos = int((deltas > 0).sum())
        sd = float(deltas.std(ddof=1))
        flag = "excl 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"{tag:8s} {d['seed_mean']:>+10.4f} {m:>+10.4f} "
              f"[{lo:>+8.4f}, {hi:>+8.4f}] {pos:>4d}/10 {sd:>6.3f}  {flag}")

    print()
    print("Interpretation:")
    print("  - excl 0 means the paired seed-level CI excludes zero.")
    print("  - SD is the per-seed standard deviation of the paired deltas.")


if __name__ == "__main__":
    main()