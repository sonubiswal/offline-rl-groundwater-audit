#!/usr/bin/env python3
"""recompute_family_start.py — 9-pairing family x start grid under the
patched (disjoint-stream) adapters, three protocols.

Protocols:
    rf_grid      : 500 starts from the RF-grid prediction surface
    common_eps   : 225 held-out epsilon test starts, applied to every family
    own_family   : each BC family's own held-out test starts
                   (225 base, 225 epsilon, 150 no_oracle)

Saves:
    revision_experiments/family_start_rerun.npz    (27 per-episode arrays)
    revision_experiments/family_start_rerun.json   (27 summary rows)

Re-run with --force to recompute (~30 min).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(r"C:\trishna-opal")
sys.path.insert(0, str(REPO / "revision_experiments"))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A  # noqa: E402

FAMILIES = ["base", "epsilon", "no_oracle"]
SEEDS = A.SEEDS
N_RF = 500
N_EPS = 225
OWN_N = {"base": 225, "epsilon": 225, "no_oracle": 150}
PROTOCOLS = ("rf_grid", "common_eps", "own_family")

OUT_NPZ = REPO / "revision_experiments" / "family_start_rerun.npz"
OUT_JSON = REPO / "revision_experiments" / "family_start_rerun.json"


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def load_family_test_starts(family: str, n: int) -> np.ndarray:
    """Load the first n held-out test starts for the named family.

    Uses the same RandomState(1) permutation and 70/15/15 split as
    adapters._load_test_episode_starts, generalised to any family size.
    For epsilon this reproduces adapters._load_test_episode_starts exactly.
    """
    dataset = A.DATASETS[family]
    with h5py.File(dataset, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        N = len(ep_ids)
        n_train = int(round(0.70 * N))
        n_val = int(round(0.15 * N))
        rng = np.random.RandomState(1)
        idx = rng.permutation(N)
        test_idx = idx[n_train + n_val : n_train + n_val + n]
        assert len(test_idx) == n, \
            f"{family}: expected {n} test starts, got {len(test_idx)}"
        starts = np.array([f[f"observations_{ep_ids[i]}"][0]
                           for i in test_idx], dtype=np.float32)
    return starts


def rollout_array(family, algo, seed, starts, n):
    ds = A.DATASETS[family]
    with _quiet():
        p = A.load_policy(A.checkpoint(family, algo, seed), ds)
        return A.rollout_returns(p, n, seed, start_set=starts)


def pairing_arrays(bc_fam, cql_fam, proto, starts_map):
    """Return (10, N) array of per-seed per-episode CQL-BC gaps."""
    rows = []
    for s in SEEDS:
        if proto == "rf_grid":
            bc_r = rollout_array(bc_fam, "bc", s, "rf_grid", N_RF)
            cq_r = rollout_array(cql_fam, "cql", s, "rf_grid", N_RF)
        elif proto == "common_eps":
            bc_r = rollout_array(bc_fam, "bc", s, starts_map["common_eps"], N_EPS)
            cq_r = rollout_array(cql_fam, "cql", s, starts_map["common_eps"], N_EPS)
        elif proto == "own_family":
            starts = starts_map[bc_fam]        # BC family's own starts
            n = len(starts)
            bc_r = rollout_array(bc_fam, "bc", s, starts, n)
            cq_r = rollout_array(cql_fam, "cql", s, starts, n)
        else:
            raise ValueError(proto)
        rows.append(cq_r - bc_r)
    return np.asarray(rows, dtype=np.float64)


def report(rows):
    print()
    print(f"{'BC':>10} {'CQL':>10} {'protocol':>12} {'mean':>10}")
    print("-" * 46)
    for r in rows:
        print(f"{r['bc']:>10} {r['cql']:>10} {r['protocol']:>12} "
              f"{r['mean']:>+10.4f}")

    # Reversal count over rf_grid vs common_eps
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["bc"], r["cql"]), {})[r["protocol"]] = r["mean"]
    rev = sum(1 for k, v in by_pair.items()
              if (v["rf_grid"] > 0) != (v["common_eps"] > 0))
    print(f"\nReversals rf_grid vs common_eps: {rev} of 9")
    print("Abstract wording: 'for most pairings' (do not report a count).")

    # Duplicate rows: epsilon-BC own_family reproduces common_eps.
    print("\nDuplicated cells (own_family = common_eps for epsilon BC):")
    for r in rows:
        if r["bc"] == "epsilon" and r["protocol"] == "own_family":
            ce = by_pair[(r["bc"], r["cql"])]["common_eps"]
            print(f"  epsilon/{r['cql']:<10} own {r['mean']:+.4f} "
                  f"eps {ce:+.4f}  (diff {r['mean'] - ce:+.6f})")
    print(f"\nTotal rows: {len(rows)}  "
          f"distinct cells: {len(rows) - 3}")


def compute(starts_map):
    arrays, rows = {}, []
    total = len(FAMILIES) ** 2
    k = 0
    for bc_fam in FAMILIES:
        for cql_fam in FAMILIES:
            k += 1
            for proto in PROTOCOLS:
                print(f"  pairing {k}/{total}: {bc_fam}/{cql_fam} — "
                      f"{proto} ...", flush=True)
                arr = pairing_arrays(bc_fam, cql_fam, proto, starts_map)
                key = f"{bc_fam}__{cql_fam}__{proto}"
                arrays[key] = arr
                rows.append(dict(bc=bc_fam, cql=cql_fam, protocol=proto,
                                 mean=float(arr.mean())))
    return arrays, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if OUT_NPZ.exists() and OUT_JSON.exists() and not a.force:
        print(f"cache present: {OUT_NPZ}")
        z = np.load(OUT_NPZ)
        for k in sorted(z.files):
            print(f"  {k:40s} {z[k].shape}")
        rows = json.loads(OUT_JSON.read_text())["rows"]
        report(rows)
        return

    eps_starts = A._load_test_episode_starts(A.DATASETS["epsilon"], n=N_EPS)
    starts_map = {
        "common_eps": eps_starts,
        "base": load_family_test_starts("base", OWN_N["base"]),
        "epsilon": eps_starts,
        "no_oracle": load_family_test_starts("no_oracle", OWN_N["no_oracle"]),
    }
    print(f"loaded starts: common_eps={len(eps_starts)} "
          f"base={len(starts_map['base'])} "
          f"no_oracle={len(starts_map['no_oracle'])}")

    arrays, rows = compute(starts_map)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **arrays)
    OUT_JSON.write_text(json.dumps(dict(rows=rows), indent=2))
    print(f"\nwrote {OUT_NPZ}")
    print(f"wrote {OUT_JSON}")
    report(rows)


if __name__ == "__main__":
    main()