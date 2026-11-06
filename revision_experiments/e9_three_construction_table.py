"""
E9: for each reproducible pairing in E2, compute pooled, seed-t and
hierarchical intervals on the same starts. Tests whether the
"pooled excludes zero, seed-aware do not" pattern is real anywhere on a
reproducible protocol.

Re-runs rollouts. Approximately 45 min for all 27 pairings.

Run from C:\\trishna-opal:
    python revision_experiments\\e9_three_construction_table.py
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
from stats_utils import episode_bootstrap, t_interval, hierarchical_bootstrap


FAMS = ["base", "epsilon", "no_oracle"]
KINDS = ["rf_grid", "common_epsilon_test", "own_family_test"]
N_RF = 500
N_EPS = 225


def load_test_starts(dataset_path, n_take=None, split_seed=1):
    with h5py.File(dataset_path, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        N = len(ep_ids)
        rng = np.random.RandomState(split_seed)
        idx = rng.permutation(N)
        n_train, n_val = int(N * 0.70), int(N * 0.15)
        test_idx = idx[n_train + n_val:]
        if n_take is not None:
            test_idx = test_idx[:n_take]
        starts = np.array(
            [f[f"observations_{ep_ids[i]}"][0] for i in test_idx],
            dtype=np.float32)
    return starts


def starts_for(kind, fb):
    if kind == "rf_grid":
        return None
    if kind == "common_epsilon_test":
        return load_test_starts(A.DATASETS["epsilon"], n_take=N_EPS)
    if kind == "own_family_test":
        return load_test_starts(A.DATASETS[fb])
    raise ValueError(kind)


def class_of(pooled, seed_t, hier):
    p = pooled[1] > 0 or pooled[2] < 0
    s = seed_t[1] > 0 or seed_t[2] < 0
    h = hier[1] > 0 or hier[2] < 0
    if p and not s and not h:
        return "pooled only"
    if p and s and h:
        return "all"
    if not p and not s and not h:
        return "none"
    return f"pooled={p} seed={s} hier={h}"


def main():
    out = {}
    for kind in KINDS:
        for fb, fc in itertools.product(FAMS, FAMS):
            ds_bc = A.DATASETS[fb]
            ds_cq = A.DATASETS[fc]
            starts = starts_for(kind, fb)

            diffs_by_seed = []
            for s in A.SEEDS:
                bc = A.load_policy(A.checkpoint(fb, "bc", s), ds_bc)
                cq = A.load_policy(A.checkpoint(fc, "cql", s), ds_cq)
                if starts is None:
                    bc_r = A.rollout_returns(bc, N_RF, s, start_set="rf_grid")
                    cq_r = A.rollout_returns(cq, N_RF, s, start_set="rf_grid")
                else:
                    bc_r = A.rollout_returns(bc, len(starts), s,
                                             start_set=starts)
                    cq_r = A.rollout_returns(cq, len(starts), s,
                                             start_set=starts)
                diffs_by_seed.append(cq_r - bc_r)

            pooled = episode_bootstrap(np.concatenate(diffs_by_seed))
            seed_t = t_interval([d.mean() for d in diffs_by_seed])
            hier = hierarchical_bootstrap(diffs_by_seed)
            tag = class_of(pooled, seed_t, hier)

            key = f"{kind}|BC={fb}|CQL={fc}"
            out[key] = {
                "pooled": {"mean": pooled[0], "lo": pooled[1], "hi": pooled[2]},
                "seed_t": {"mean": seed_t[0], "lo": seed_t[1], "hi": seed_t[2]},
                "hierarchical": {"mean": hier[0], "lo": hier[1], "hi": hier[2]},
                "class": tag,
            }
            print(f"{key:45s}  "
                  f"pooled={'Y' if pooled[1] > 0 or pooled[2] < 0 else 'N'} "
                  f"seed_t={'Y' if seed_t[1] > 0 or seed_t[2] < 0 else 'N'} "
                  f"hier={'Y' if hier[1] > 0 or hier[2] < 0 else 'N'}  [{tag}]")

    (REPO / "revision_experiments"
     / "e9_three_construction_table.json").write_text(json.dumps(out, indent=2))

    # Summary of pattern presence
    from collections import Counter
    counts = Counter(v["class"] for v in out.values())
    print("\nclass counts across 27 cells:")
    for cls, n in counts.most_common():
        print(f"  {cls:35s} {n}")

    print("\nwrote e9_three_construction_table.json")


if __name__ == "__main__":
    main()