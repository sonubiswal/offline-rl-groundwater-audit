"""
E2: Table VII with three start sets.
  - rf_grid            (500 starts from reports/rf_grid_predictions.csv)
  - common_epsilon_test (225 starts from epsilon test split)
  - own_family_test    (15% of each BC family's own episode count)

Resolves Reading A vs B: does the sign flip persist under family-specific
starts? Note: CQL family may have seen the BC family's test episodes in
training when fb != fc — flag this in the results.

Uses a local split loader that computes n_test from the file's actual
episode count (bc_baseline.py convention: 70/15/15 split).

Run from C:\\trishna-opal:
    python revision_experiments\\e2_family_specific_starts.py
"""
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A
from stats_utils import hierarchical_bootstrap


FAMS = ["base", "epsilon", "no_oracle"]
N_RF = 500
N_EPS_TEST = 225


def load_test_starts(dataset_path, n_take=None, split_seed=1):
    """Return first observation of each episode in the 15% test split.
    Split is computed from the file's actual episode count, matching
    bc_baseline.split_episode_indices (70/15/15, RandomState(split_seed))."""
    with h5py.File(dataset_path, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        N = len(ep_ids)
        rng = np.random.RandomState(split_seed)
        idx = rng.permutation(N)
        n_train = int(N * 0.70)
        n_val = int(N * 0.15)
        test_idx = idx[n_train + n_val:]
        if n_take is not None:
            test_idx = test_idx[:n_take]
        starts = np.array(
            [f[f"observations_{ep_ids[i]}"][0] for i in test_idx],
            dtype=np.float32,
        )
    return starts


def starts_for(kind, fb, fc):
    if kind == "rf_grid":
        return None  # use rf_grid via start_set="rf_grid"
    if kind == "common_epsilon_test":
        return load_test_starts(A.DATASETS["epsilon"], n_take=N_EPS_TEST)
    if kind == "own_family_test":
        return load_test_starts(A.DATASETS[fb])
    raise ValueError(kind)


def main():
    res = {}
    kinds = ["rf_grid", "common_epsilon_test", "own_family_test"]

    for kind in kinds:
        for fb, fc in itertools.product(FAMS, FAMS):
            diffs_by_seed = []
            for s in A.SEEDS:
                ds_bc = A.DATASETS[fb]
                ds_cq = A.DATASETS[fc]
                bc_ck = A.checkpoint(fb, "bc", s)
                cq_ck = A.checkpoint(fc, "cql", s)
                bc = A.load_policy(bc_ck, ds_bc)
                cq = A.load_policy(cq_ck, ds_cq)

                starts = starts_for(kind, fb, fc)
                if starts is None:
                    bc_r = A.rollout_returns(bc, N_RF, s, start_set="rf_grid")
                    cq_r = A.rollout_returns(cq, N_RF, s, start_set="rf_grid")
                else:
                    bc_r = A.rollout_returns(bc, len(starts), s, start_set=starts)
                    cq_r = A.rollout_returns(cq, len(starts), s, start_set=starts)
                diffs_by_seed.append(cq_r - bc_r)

            m, lo, hi = hierarchical_bootstrap(diffs_by_seed)
            key = f"{kind}|BC={fb}|CQL={fc}"
            res[key] = {"mean": m, "lo": lo, "hi": hi,
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "n_starts": (N_RF if starts is None
                                     else int(len(starts)))}
            flag = "excl 0" if res[key]["excludes_zero"] else "spans 0"
            print(f"{kind:20s} BC={fb:9s} CQL={fc:9s} "
                  f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] {flag}")

    path = REPO / "revision_experiments" / "e2_family_specific_starts.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {path.name}")
    print("NOTE: for fb != fc, the CQL family may have seen the BC family's")
    print("test episodes in training. Report this in the manuscript.")


if __name__ == "__main__":
    main()