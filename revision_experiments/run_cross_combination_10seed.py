"""
run_cross_combination_10seed.py

Re-runs the paper's Table VII cross-combination diagnostic at ten seeds
with the per-checkpoint scaler. Emits a 2x2 (BC family x CQL family) table
for each evaluation start set, plus paired CQL-BC intervals.
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


FAMILIES = ["base", "no_oracle"]
SEEDS = A.SEEDS

# Per-start-set rollout count. rf_grid draws i.i.d. from the RF table
# so any n works; test225 is the fixed 225-episode test split (1500 = 1050
# train + 225 val + 225 test), so it cannot exceed 225.
START_SET_SIZES = {"rf_grid": 500, "test225": 225}


def collect(family, algo, start_set):
    n = START_SET_SIZES[start_set]
    means = []
    for s in SEEDS:
        ck = A.checkpoint(family, algo, s)
        ds = A.DATASETS[family]
        pol = A.load_policy(ck, ds)
        rets = A.rollout_returns(pol, n, s, start_set=start_set)
        means.append(rets.mean())
    return float(np.mean(means))


def main():
    out = {"protocol": {"start_set_sizes": START_SET_SIZES, "seeds": SEEDS}}
    out_path = REPO / "revision_experiments" / "cross_combination_10seed.json"

    for start_set in ("rf_grid", "test225"):
        n = START_SET_SIZES[start_set]
        print(f"\n=== start set: {start_set} (n={n}) ===")

        # Compute all four (algo, family) means once
        means = {}
        for algo, fam in itertools.product(("bc", "cql"), FAMILIES):
            means[(algo, fam)] = collect(fam, algo, start_set)
            print(f"  {algo:3s} {fam:9s}: {means[(algo, fam)]:+.4f}")

        # 2x2 table and paired interval for each pairing
        table = {}
        for fb, fc in itertools.product(FAMILIES, FAMILIES):
            diffs_by_seed = []
            for s in SEEDS:
                bc_pol = A.load_policy(A.checkpoint(fb, "bc", s), A.DATASETS[fb])
                cq_pol = A.load_policy(A.checkpoint(fc, "cql", s), A.DATASETS[fc])
                bc_rets = A.rollout_returns(bc_pol, n, s, start_set=start_set)
                cq_rets = A.rollout_returns(cq_pol, n, s, start_set=start_set)
                diffs_by_seed.append(cq_rets - bc_rets)
            m, lo, hi = hierarchical_bootstrap(diffs_by_seed)
            table[f"BC={fb}|CQL={fc}"] = {
                "bc_mean": means[("bc", fb)],
                "cql_mean": means[("cql", fc)],
                "cql_minus_bc": m, "lo": lo, "hi": hi,
            }
            flag = "excludes 0" if lo > 0 or hi < 0 else "spans 0"
            print(f"  BC={fb:9s} CQL={fc:9s}  "
                  f"BC {means[('bc', fb)]:+.3f}  CQL {means[('cql', fc)]:+.3f}  "
                  f"CQL-BC {m:+.3f} [{lo:+.3f},{hi:+.3f}]  {flag}")

        out[f"start_set_{start_set}"] = {
            "n": n,
            "means": {f"{a}|{f}": v for (a, f), v in means.items()},
            "pairings": table,
        }
        # Write after every start set: if a later block fails, the earlier
        # results are preserved on disk.
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  [partial] wrote {out_path.name}")

    print("\n[cross_comb] wrote cross_combination_10seed.json")


if __name__ == "__main__":
    main()