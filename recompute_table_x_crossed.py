"""Crossed bootstrap for the Table XI construction-dependent cells.

Starts are crossed with seeds (each of the ten seeds evaluates the same
225 or 500 starts), so the resampling must resample seeds and starts
independently, not nest starts inside seeds.

Compares the resulting cell-by-cell classification with the hierarchical
count reported in Table XI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(r"C:\trishna-opal")
NPZ = REPO / "revision_experiments" / "family_start_rerun.npz"
OUT = REPO / "revision_experiments" / "table_x_crossed.json"

FAMS = ["base", "epsilon", "no_oracle"]
PROTO = ["rf_grid", "common_eps", "own_family"]
N_BOOT = 5000
RNG_SEED = 2026


def crossed_bootstrap(arr, n_boot=N_BOOT, rng=None):
    """arr: (S, E) seeds x starts. Resample seeds and starts independently."""
    rng = rng or np.random.default_rng(RNG_SEED)
    S, E = arr.shape
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s_idx = rng.integers(0, S, S)
        e_idx = rng.integers(0, E, E)
        boots[b] = arr[np.ix_(s_idx, e_idx)].mean()
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def excludes_zero(lo, hi):
    return lo > 0 or hi < 0


def main():
    z = np.load(NPZ)
    rows = []
    for bc in FAMS:
        for cql in FAMS:
            for proto in PROTO:
                key = f"{bc}__{cql}__{proto}"
                if key not in z.files:
                    continue
                arr = z[key]
                lo, hi = crossed_bootstrap(arr)
                rows.append({
                    "bc": bc,
                    "cql": cql,
                    "protocol": proto,
                    "mean": float(arr.mean()),
                    "lo": lo,
                    "hi": hi,
                    "excludes": excludes_zero(lo, hi),
                })

    # Collapse the three epsilon-BC own-family duplicates
    distinct = [
        r for r in rows
        if not (r["bc"] == "epsilon" and r["protocol"] == "own_family")
    ]

    print(f"total rows: {len(rows)}  distinct cells: {len(distinct)}")
    print()
    header = (f"{'BC':>10} {'CQL':>10} {'proto':>12} {'mean':>9} "
              f"{'95% CI (crossed)':>22} {'excl 0':>7}")
    print(header)
    print("-" * len(header))
    for r in distinct:
        print(f"{r['bc']:>10} {r['cql']:>10} {r['protocol']:>12} "
              f"{r['mean']:>+9.4f}   [{r['lo']:+.4f}, {r['hi']:+.4f}]   "
              f"{'yes' if r['excludes'] else 'no':>4}")

    n_excl = sum(1 for r in distinct if r["excludes"])
    print()
    print(f"cells with crossed CI excluding zero: {n_excl} of {len(distinct)}")
    print("Compare with the hierarchical count of 7 to see if the pattern holds.")

    OUT.write_text(json.dumps(distinct, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()