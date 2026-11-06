"""
E1: does the sweep's default cell (lr 1e-3, 256x256) reproduce the
canonical epsilon BC?

Both numbers the paper reports are epsilon CQL-BC. Table VIII shows
+0.047 (source: .npy files, start set unrecorded). Table S1's default
cell shows -0.057 (rf_grid starts). E1 rolls canonical epsilon BC and
sweep-default epsilon BC on IDENTICAL starts, with canonical epsilon
CQL as the reference policy, and reports:
  - gap with canonical BC  (should match .npy if start set matches)
  - gap with sweep-default BC
  - per-seed canonical BC return minus sweep-default BC return

If bc_canonical - bc_sweep CI excludes zero, the recipes differ and the
paper must explain why. If it spans zero, the +0.047 vs -0.057 gap is
due to start-set choice.

Run from C:\\trishna-opal:
    python revision_experiments\\e1_sweep_vs_canonical.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A
from stats_utils import hierarchical_bootstrap


N = 500
FAM = "epsilon"


def main():
    out = {}
    for label, bc_dir in [("canonical", None), ("sweep_default", "reports_2/bc_sweep")]:
        d_cq_bc, bc_pair = [], []
        for s in A.SEEDS:
            ds = A.DATASETS[FAM]
            cq_ck = A.checkpoint(FAM, "cql", s)
            if bc_dir is None:
                bc_ck = A.checkpoint(FAM, "bc", s)
            else:
                bc_ck = REPO / bc_dir / f"{FAM}_lr0.001_h256_s{s}.d3"
                if not bc_ck.exists():
                    raise FileNotFoundError(bc_ck)
            cq = A.load_policy(cq_ck, ds)
            bc = A.load_policy(bc_ck, ds)
            cq_r = A.rollout_returns(cq, N, s, start_set="rf_grid")
            bc_r = A.rollout_returns(bc, N, s, start_set="rf_grid")
            d_cq_bc.append(cq_r - bc_r)
            bc_pair.append(bc_r.mean())
        m, lo, hi = hierarchical_bootstrap(d_cq_bc)
        out[label] = {
            "cql_minus_bc": {"mean": m, "lo": lo, "hi": hi,
                             "excludes_zero": bool(lo > 0 or hi < 0)},
            "per_seed_bc_mean": bc_pair,
        }
        print(f"{label:15s}  CQL-BC {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
              f"{'excl 0' if out[label]['cql_minus_bc']['excludes_zero'] else 'spans 0'}")

    # Paired: canonical BC minus sweep BC, per seed
    bc_can = np.array(out["canonical"]["per_seed_bc_mean"])
    bc_swp = np.array(out["sweep_default"]["per_seed_bc_mean"])
    diff_by_seed = [np.full(N, bc_can[i] - bc_swp[i]) for i in range(len(bc_can))]
    m, lo, hi = hierarchical_bootstrap(diff_by_seed)
    print(f"\nbc_canonical - bc_sweep (paired):  {m:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"{'excl 0' if lo > 0 or hi < 0 else 'spans 0'}")
    out["bc_canonical_minus_bc_sweep"] = {
        "mean": m, "lo": lo, "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }

    path = REPO / "revision_experiments" / "e1_canonical_vs_sweep.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.name}")
    print("\nIf bc_canonical - bc_sweep excludes 0, the two BC recipes differ.")


if __name__ == "__main__":
    main()