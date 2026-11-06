"""E4: extend the E2 matched-start analysis to a second algorithm pair (BCQ).

Runs three pairwise comparisons: (BC, CQL), (BC, BCQ), (CQL, BCQ). Each
produces a per-seed FQE-rollout contrast tested with a seed-t interval at
the recorded FQE target-update interval (200).

Verification step: for BC and CQL, the fresh FQE fits must reproduce the
per_seed_diffs stored in E2/e2_fqe.json. If they do, the FQE pipeline is
deterministic and the BCQ values are trusted. If they do not, the script
says so and does not apply the decision rule.

Reuses run_e2_fqe.py's fit_fqe, fqe_values, build_fqe_dataset and
verify_fqe_dataset functions to avoid any reimplementation of the FQE
fitting path. The FQE config (batch 100, lr 3e-4, gamma 1.0, target
update 200, Adam with clip_grad_norm=1.0) is read from the E2 module.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"C:\trishna-opal")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "rl"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "revision_experiments"))

import adapters as A                                   # noqa: E402
from stats_utils import t_interval                     # noqa: E402
import run_e2_fqe as E2                                # noqa: E402

# Match the E2 recorded target-update interval exactly.
E2.FQE_TARGET_UPDATE = 200

REV = REPO / "revision_experiments"
OUT_DIR = REV / "E4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BCQ_DIR = REPO / "models_10seed" / "epsilon" / "bcq_control"
SEEDS = E2.ALL_SEEDS
N_STARTS = 225


def load_stored_e2_gaps() -> np.ndarray:
    """Per-seed CQL-BC FQE gap from the E2 results file."""
    p = REV / "E2" / "e2_fqe.json"
    if not p.exists():
        p = REV / "e2_fqe.json"
    d = json.loads(p.read_text())
    return np.asarray(d["primary_episode_starts"]["per_seed_diffs"], float)


def prepare_data():
    """Rebuild the exact tuple E2.run_seed receives.

    Layout, as inferred from run_e2_fqe.run_seed:
        (train_ds, starts, raw_starts, random_states, mean, std, ep_len)
    Only the fields E4 needs are populated; random_states is a placeholder.
    """
    from src.rl.bc_baseline import (
        POLICY_ORDER, N_PER_POLICY, load_split_and_scaler,
    )
    _eps, _indices, groups, mean, std = load_split_and_scaler(
        E2.DATASET, POLICY_ORDER, N_PER_POLICY)
    train_eps = groups[0]

    train_ds = E2.build_fqe_dataset(train_eps, mean, std)
    E2.verify_fqe_dataset(train_ds, len(train_eps),
                          E2.PAPER_ROLLOUT_HORIZON)

    raw_starts = A._load_test_episode_starts(E2.DATASET, n=N_STARTS)
    starts = (raw_starts - mean) / std
    random_states = np.zeros((0, len(mean)), dtype=np.float32)
    ep_len = E2.PAPER_ROLLOUT_HORIZON
    return train_ds, starts, raw_starts, random_states, mean, std, ep_len


def fqe_at_starts(policy, train_ds, seed, starts, n_steps):
    fqe, _trace = E2.fit_fqe(policy, train_ds, seed, starts, n_steps)
    return np.asarray(E2.fqe_values(fqe, policy, starts))


def rollout_at_starts(ckpt, seed, raw_starts):
    fn = A.load_policy(ckpt, E2.DATASET)
    return A.rollout_returns(fn, len(raw_starts), seed, start_set=raw_starts)


def main() -> None:
    print("[E4] loading stored E2 values ...")
    stored_gaps = load_stored_e2_gaps()
    print(f"[E4] stored per-seed CQL-BC FQE gap, mean {stored_gaps.mean():+.4f}")

    print("[E4] preparing data ...")
    train_ds, starts, raw_starts, _rand, _mean, _std, _ep_len = prepare_data()
    print(f"[E4] FQE train_ds built; {len(starts)} starts")

    per_seed = []
    for s in SEEDS:
        t0 = time.time()
        print(f"\n[E4] seed {s}")

        bc = E2.load_policy_for_eval(E2.BC_DIR / f"bc_seed{s}.d3",
                                     E2.DATASET).policy
        cql = E2.load_policy_for_eval(E2.CQL_DIR / f"cql_seed{s}.d3",
                                      E2.DATASET).policy
        bcq = E2.load_policy_for_eval(BCQ_DIR / f"bcq_seed{s}.d3",
                                      E2.DATASET).policy

        bc_q = fqe_at_starts(bc, train_ds, s, starts, E2.FQE_N_STEPS)
        cql_q = fqe_at_starts(cql, train_ds, s, starts, E2.FQE_N_STEPS)
        bcq_q = fqe_at_starts(bcq, train_ds, s, starts, E2.FQE_N_STEPS)

        bc_r = rollout_at_starts(E2.BC_DIR / f"bc_seed{s}.d3", s, raw_starts)
        cql_r = rollout_at_starts(E2.CQL_DIR / f"cql_seed{s}.d3", s, raw_starts)
        bcq_r = rollout_at_starts(BCQ_DIR / f"bcq_seed{s}.d3", s, raw_starts)

        stored_gap = float(stored_gaps[SEEDS.index(s)])
        computed_gap = float(cql_q.mean() - bc_q.mean())

        print(f"  FQE  BC  {bc_q.mean():+.4f}  CQL {cql_q.mean():+.4f}  "
              f"BCQ {bcq_q.mean():+.4f}")
        print(f"  Roll BC  {bc_r.mean():+.4f}  CQL {cql_r.mean():+.4f}  "
              f"BCQ {bcq_r.mean():+.4f}")
        print(f"  E2 stored CQL-BC FQE gap {stored_gap:+.4f}, "
              f"computed {computed_gap:+.4f}  diff {computed_gap-stored_gap:+.6f}")

        per_seed.append({
            "seed": int(s),
            "bc_q_mean": float(bc_q.mean()),
            "cql_q_mean": float(cql_q.mean()),
            "bcq_q_mean": float(bcq_q.mean()),
            "bc_r_mean": float(bc_r.mean()),
            "cql_r_mean": float(cql_r.mean()),
            "bcq_r_mean": float(bcq_r.mean()),
            "stored_e2_fqe_gap": stored_gap,
            "computed_e2_fqe_gap": computed_gap,
            "bc_q_starts": bc_q.tolist(),
            "cql_q_starts": cql_q.tolist(),
            "bcq_q_starts": bcq_q.tolist(),
        })
        print(f"  elapsed {time.time()-t0:.1f}s")

    print("\n" + "=" * 72)
    print("[E4] per-seed verification of the E2 refit")
    diffs = [abs(r["computed_e2_fqe_gap"] - r["stored_e2_fqe_gap"])
             for r in per_seed]
    max_diff = max(diffs)
    print(f"  max |computed - stored| = {max_diff:.6f}")
    verification_pass = max_diff < 1e-4
    print(f"  verification {'PASS' if verification_pass else 'FAIL'}")

    print("\n" + "=" * 72)
    print("[E4] pairwise FQE-rollout contrasts at interval 200")
    print(f"  {'pair':>12s}  {'contrast':>10s}  {'95% CI':>22s}  "
          f"{'excl 0':>7s}  {'pos seeds':>10s}")

    pairs = [
        ("BC",  "CQL", "bc",  "cql"),
        ("BC",  "BCQ", "bc",  "bcq"),
        ("CQL", "BCQ", "cql", "bcq"),
    ]
    results = {}
    for a_name, b_name, a_key, b_key in pairs:
        contrasts = []
        for r in per_seed:
            fqe_gap = r[f"{b_key}_q_mean"] - r[f"{a_key}_q_mean"]
            rollout_gap = r[f"{b_key}_r_mean"] - r[f"{a_key}_r_mean"]
            contrasts.append(fqe_gap - rollout_gap)
        contrasts = np.asarray(contrasts, float)
        m, lo, hi = t_interval(contrasts)
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        pos = int((contrasts > 0).sum())
        print(f"  {a_name+' vs '+b_name:>12s}  {m:>+10.4f}  "
              f"  [{lo:+.4f}, {hi:+.4f}]  {excl:>7s}  {pos:>4d}/{len(contrasts)}")
        results[f"{a_name}_{b_name}"] = {
            "contrast": float(m), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "positive_seeds": pos, "n_seeds": len(contrasts),
            "per_seed_contrast": contrasts.tolist(),
        }

    print("\n" + "=" * 72)
    print("[E4] pre-registered decision rule")
    bc_bcq = results.get("BC_BCQ", {})
    cql_bcq = results.get("CQL_BCQ", {})
    sig_bc_bcq = bool(bc_bcq.get("excludes_zero")
                      and bc_bcq.get("contrast", 0) > 0)
    sig_cql_bcq = bool(cql_bcq.get("excludes_zero")
                       and cql_bcq.get("contrast", 0) > 0)

    if not verification_pass:
        verdict = ("UNAVAILABLE: E2 refit did not reproduce; "
                   "FQE pipeline may have drifted.")
    elif sig_bc_bcq and sig_cql_bcq:
        verdict = ("GENERALISES: both new pairs positive and significant; "
                   "report 'for three algorithm pairs'.")
    elif sig_bc_bcq or sig_cql_bcq:
        which = "BC-BCQ" if sig_bc_bcq else "CQL-BCQ"
        verdict = (f"PAIR-SPECIFIC: only {which} generalises; "
                   f"report which pair and which does not.")
    else:
        verdict = ("DOES NOT GENERALISE: neither new pair positive and "
                   "significant; report the finding as pair-specific to "
                   "BC-CQL.")
    print(f"  {verdict}")

    out = {
        "per_seed": per_seed,
        "results": results,
        "verification_max_diff": max_diff,
        "verification_pass": verification_pass,
        "verdict": verdict,
        "fqe_target_update_interval": E2.FQE_TARGET_UPDATE,
        "n_starts": N_STARTS,
    }
    out_path = OUT_DIR / "e4_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[E4] wrote {out_path}")


if __name__ == "__main__":
    main()