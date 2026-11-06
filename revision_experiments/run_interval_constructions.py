"""
run_interval_constructions.py

Both estimators, both interval constructions, hierarchical rollout.

Reads:
    reports_10seed/epsilon/policy_comparison.json
    reports_10seed/epsilon/returns/*.npy
    revision_experiments/E2/e2_fqe.json

Writes:
    revision_experiments/interval_constructions.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from stats_utils import (t_interval, episode_bootstrap,
                         seed_bootstrap, hierarchical_bootstrap)


ROLLOUT_DIR = REPO / "reports_10seed" / "epsilon"
FQE_JSON = REPO / "revision_experiments" / "E2" / "e2_fqe.json"
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]


def load_rollout():
    j = json.loads((ROLLOUT_DIR / "policy_comparison.json").read_text())
    per_seed = {}
    for s in SEEDS:
        bc_p = ROLLOUT_DIR / "returns" / f"returns_bc_seed{s}.npy"
        cq_p = ROLLOUT_DIR / "returns" / f"returns_cql_seed{s}.npy"
        if bc_p.exists() and cq_p.exists():
            bc = np.load(bc_p)
            cq = np.load(cq_p)
            per_seed[s] = (bc, cq)
    boot = j.get("cql_minus_bc_paired_bootstrap", {})
    return per_seed, {
        "obs": float(boot.get("observed_difference", np.nan)),
        "lo":  float(boot.get("ci_lower", np.nan)),
        "hi":  float(boot.get("ci_upper", np.nan)),
    }


def load_fqe():
    for p in (REPO / "revision_experiments" / "E2" / "e2_fqe.json",
              REPO / "revision_experiments" / "e2_fqe.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("e2_fqe.json not found")


def main():
    rows = []

    # -- Rollout ---------------------------------------------------------
    per_seed, boot = load_rollout()
    diffs_by_seed = [cq - bc for (bc, cq) in per_seed.values()]
    seed_means = [float((cq - bc).mean()) for (bc, cq) in per_seed.values()]

    rows.append(("Rollout", "episode bootstrap (pooled)",
                 *episode_bootstrap(np.concatenate(diffs_by_seed))))
    rows.append(("Rollout", "seed $t$-interval", *t_interval(seed_means)))
    rows.append(("Rollout", "hierarchical (seed $\\to$ episode)",
                 *hierarchical_bootstrap(diffs_by_seed)))

    # -- FQE -------------------------------------------------------------
    fqe = load_fqe()
    prim = fqe.get("primary_episode_starts", {})
    sec = fqe.get("secondary_random_states", {})
    per_seed_q = fqe.get("per_seed", {})

    seed_deltas = []
    for s, d in per_seed_q.items():
        seed_deltas.append(float(d["cql_mean_q"] - d["bc_mean_q"]))

    if seed_deltas:
        rows.append(("FQE", "seed $t$-interval", *t_interval(seed_deltas)))
    elif prim:
        rows.append(("FQE", "seed $t$-interval",
                     float(prim.get("seed_mean", np.nan)),
                     float(prim.get("seed_ci_lower", np.nan)),
                     float(prim.get("seed_ci_upper", np.nan))))

    if sec and "state_bootstrap_ci" in sec:
        rows.append(("FQE", "state bootstrap (secondary)",
                     float(prim.get("seed_mean", np.nan)),
                     float(sec["state_bootstrap_ci"][0]),
                     float(sec["state_bootstrap_ci"][1])))

    # -- Print and write -------------------------------------------------
    print("\n% paste into Table VIII")
    for inst, unit, m, lo, hi in rows:
        flag = "excludes 0" if lo > 0 or hi < 0 else "spans 0"
        print(f"{inst:8s} & {unit:35s} & {m:+.4f} & [{lo:+.4f}, {hi:+.4f}] & {flag} \\\\")

    out = {"rows": [{"estimator": r[0], "unit": r[1],
                     "obs": r[2], "lo": r[3], "hi": r[4]} for r in rows]}
    (REPO / "revision_experiments" / "interval_constructions.json").write_text(
        json.dumps(out, indent=2))
    print("\n[interval] wrote interval_constructions.json")


if __name__ == "__main__":
    main()