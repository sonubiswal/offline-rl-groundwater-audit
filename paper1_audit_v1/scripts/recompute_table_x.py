#!/usr/bin/env python3
"""recompute_table_x.py — construction-dependency audit for the 24 distinct
cells of the family/start grid under collision-free streams.

Reads  : revision_experiments/family_start_rerun.npz
Writes : revision_experiments/table_x_rerun.json

Three interval constructions are compared per cell:
    pooled    - episode_bootstrap over concatenated per-seed episodes
    seed_t    - t_interval on the ten per-seed means
    hier      - hierarchical_bootstrap (seeds first, episodes within)

A cell is "construction-dependent" if the three do not agree on whether
zero is excluded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\trishna-opal")
sys.path.insert(0, str(REPO / "scripts"))

from stats_utils import (episode_bootstrap, hierarchical_bootstrap,
                         t_interval)  # noqa: E402

NPZ = REPO / "revision_experiments" / "family_start_rerun.npz"
OUT = REPO / "revision_experiments" / "table_x_rerun.json"

FAMS = ["base", "epsilon", "no_oracle"]
PROTOCOLS = ["rf_grid", "common_eps", "own_family"]


def excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def classify(p_ex: bool, t_ex: bool, h_ex: bool) -> str:
    if p_ex and t_ex and h_ex:
        return "all_exclude"
    if (not p_ex) and (not t_ex) and (not h_ex):
        return "none_exclude"
    if p_ex and (not t_ex) and (not h_ex):
        return "pooled_only"
    if p_ex and h_ex and (not t_ex):
        return "mixed"
    if (not p_ex) and (t_ex or h_ex):
        return "seed_aware_only"
    if p_ex and t_ex and (not h_ex):
        return "missing_hier"
    if p_ex and (not t_ex) and h_ex:
        return "missing_seed_t"
    return "other"


def intervals(arr: np.ndarray) -> dict:
    """arr: (10, N) per-seed per-episode gaps."""
    per_seed = [arr[s] for s in range(arr.shape[0])]
    pooled = np.concatenate(per_seed)
    seed_means = arr.mean(axis=1)

    # Match the original e9 behaviour: no rng arg, so each call uses a
    # fresh default_rng(2026).
    m_p, lo_p, hi_p = episode_bootstrap(pooled)
    m_t, lo_t, hi_t = t_interval(seed_means)
    m_h, lo_h, hi_h = hierarchical_bootstrap(per_seed)

    return {
        "pooled": {"mean": m_p, "lo": lo_p, "hi": hi_p},
        "seed_t": {"mean": m_t, "lo": lo_t, "hi": hi_t},
        "hier":   {"mean": m_h, "lo": lo_h, "hi": hi_h},
    }


def main() -> None:
    z = np.load(NPZ)

    rows = []
    for bc in FAMS:
        for cql in FAMS:
            for proto in PROTOCOLS:
                key = f"{bc}__{cql}__{proto}"
                if key not in z.files:
                    raise SystemExit(f"missing array {key}")
                iv = intervals(z[key])
                p_ex = excludes_zero(iv["pooled"]["lo"], iv["pooled"]["hi"])
                t_ex = excludes_zero(iv["seed_t"]["lo"], iv["seed_t"]["hi"])
                h_ex = excludes_zero(iv["hier"]["lo"], iv["hier"]["hi"])
                rows.append(dict(
                    bc=bc, cql=cql, protocol=proto,
                    pooled=iv["pooled"], seed_t=iv["seed_t"], hier=iv["hier"],
                    pooled_excludes=p_ex, seed_t_excludes=t_ex,
                    hier_excludes=h_ex,
                    pattern=classify(p_ex, t_ex, h_ex),
                ))

    # Collapse the three epsilon-BC own_family duplicates of common_eps.
    distinct = [r for r in rows
                if not (r["bc"] == "epsilon" and r["protocol"] == "own_family")]

    print(f"total rows: {len(rows)}  distinct cells: {len(distinct)}")
    print()

    print("Construction-dependent cells (the three constructions "
          "do not all agree):")
    hdr = (f"{'BC':>10} {'CQL':>10} {'protocol':>12} "
           f"{'pooled':>22} {'seed-t':>22} {'hier':>22} {'pattern':>16}")
    print(hdr)
    print("-" * len(hdr))
    cd_count = 0
    for r in distinct:
        pe, te, he = (r["pooled_excludes"], r["seed_t_excludes"],
                      r["hier_excludes"])
        if pe == te == he:
            continue
        cd_count += 1
        def fmt(d):
            return f"[{d['lo']:+.4f},{d['hi']:+.4f}]"
        print(f"{r['bc']:>10} {r['cql']:>10} {r['protocol']:>12} "
              f"{fmt(r['pooled']):>22} {fmt(r['seed_t']):>22} "
              f"{fmt(r['hier']):>22} {r['pattern']:>16}")

    print()
    print(f"Construction-dependent cells: {cd_count} of {len(distinct)}")
    print(f"Fraction: {cd_count / len(distinct):.1%}")

    print("\nPattern counts across distinct cells:")
    counts: dict[str, int] = {}
    for r in distinct:
        counts[r["pattern"]] = counts.get(r["pattern"], 0) + 1
    for k in sorted(counts):
        print(f"  {k:>20}: {counts[k]}")

    # Asymmetry check: the manuscript claims no cell has a seed-aware
    # interval excluding zero while the pooled interval spans it.
    bad = [r for r in distinct
           if (not r["pooled_excludes"])
           and (r["seed_t_excludes"] or r["hier_excludes"])]
    print()
    if bad:
        print(f"!! {len(bad)} cell(s) violate the claimed asymmetry:")
        for r in bad:
            print(f"   {r['bc']}/{r['cql']} {r['protocol']}")
    else:
        print("Confirmed: no cell has a seed-aware interval excluding zero "
              "while the pooled interval spans it.")

    OUT.write_text(json.dumps(distinct, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()