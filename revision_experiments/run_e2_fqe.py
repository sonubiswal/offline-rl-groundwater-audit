"""
E2 - FQE convergence diagnostic and paired comparison (CQL vs BC), epsilon checkpoints.

Design:
  * FQE trains on a dataset where the last step of every episode is a TRUE terminal,
    so with gamma=1.0 it estimates the fixed-horizon (20-step) undiscounted return,
    the same quantity as the rollout protocol. build_dataset (used for BC/CQL
    training) is left untouched.
  * Observations are standardized through episodes_to_arrays, the same code path
    the policies' states use, so there is no second implementation of the scaler.
  * Primary evaluation = standardized episode STARTS of held-out test episodes.
    Random test states are secondary.
  * FQE value on fixed probe states is logged every epoch -> convergence check.
  * The seed (n=10) is the unit of replication for the primary CI. The state
    bootstrap is secondary. Seeds and states are never pooled as independent.
  * Sanity checks: terminals, transition count, action collapse, episode length.
  * Optional --rollout-check compares FQE values with true simulator returns.
  * Per-seed results are saved as they finish; reruns resume.

Known limitation: the observation has no time-to-go, so FQE learns one value
function across all timesteps. Values at start states can therefore be biased
relative to the true 20-step return. --rollout-check measures that gap.

Usage:
    python run_e2_fqe.py --smoke            # 2 seeds, 2000 steps: check it runs
    python run_e2_fqe.py                    # full run
    python run_e2_fqe.py --rollout-check    # adds FQE-vs-simulator calibration
    python run_e2_fqe.py --force            # ignore saved per-seed results

Sensitivity (target-update interval):
    python run_e2_fqe.py --target-update-interval 200 --tag tui200 --force
    python run_e2_fqe.py --target-update-interval 500 --tag tui500
    # Writes e2_fqe_tui200.json and e2_fqe_tui500.json alongside e2_fqe.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "rl"))

import d3rlpy
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import InfiniteBuffer, MDPDataset, ReplayBuffer
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.optimizers import AdamFactory

from src.rl.bc_baseline import SPLIT_SEED, episodes_to_arrays, split_episode_indices
from src.rl.policy_io import compute_scaler_from_dataset, load_policy_for_eval

DATASET = REPO / "data" / "processed" / "offline_rl_dataset_epsilon.h5"
BC_DIR = REPO / "models_10seed" / "epsilon" / "bc"
CQL_DIR = REPO / "models_10seed" / "epsilon" / "cql"
OUT_DIR = REPO / "revision_experiments" / "E2"
ALL_SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]

N_ACTIONS = 5
FQE_N_STEPS = 20_000
FQE_STEPS_PER_EPOCH = 1_000          # -> 20 convergence checkpoints
FQE_LR = 3e-4
FQE_BATCH = 100
FQE_TARGET_UPDATE = 200              # default; overridable via --target-update-interval
FQE_GAMMA = 1.0                      # must match the rollout return definition
N_RANDOM_STATES = 500
N_BOOTSTRAP = 5000
BOOT_SEED = 2026
PAPER_ROLLOUT_HORIZON = 20
COLLAPSE_FRAC = 0.95
CONV_TOL = 0.05


# ----------------------------------------------------------------- data ---
def build_fqe_dataset(episodes, mean, std):
    """Training set for FQE only. Same transitions as build_dataset, but the last
    step of every episode is a true terminal (no bootstrapping past the horizon).
    Observations are standardized via episodes_to_arrays (shared code path)."""
    obs_l, act_l, rew_l, term_l = [], [], [], []
    for ep in episodes:
        obs = np.asarray(episodes_to_arrays([ep], mean, std)[0], dtype=np.float32)
        T = len(ep.observations)
        if obs.shape[0] != T:
            raise RuntimeError(
                f"episodes_to_arrays returned {obs.shape[0]} rows for an episode of "
                f"length {T}; cannot align actions and rewards.")
        term = np.zeros(T, dtype=np.float32)
        term[-1] = 1.0
        obs_l.append(obs)
        act_l.append(np.asarray(ep.actions).reshape(-1).astype(np.int64))
        rew_l.append(np.asarray(ep.rewards).reshape(-1).astype(np.float32))
        term_l.append(term)
    return MDPDataset(
        observations=np.concatenate(obs_l, axis=0),
        actions=np.concatenate(act_l, axis=0),
        rewards=np.concatenate(rew_l, axis=0),
        terminals=np.concatenate(term_l, axis=0),
        action_space=ActionSpace.DISCRETE,
        action_size=N_ACTIONS,
    )


def verify_fqe_dataset(train_ds, n_eps, ep_len):
    n_term = sum(bool(ep.terminated) for ep in train_ds.episodes)
    assert n_term == n_eps, (
        f"expected {n_eps} terminated episodes, got {n_term}; the forced terminal "
        f"did not take effect, so FQE would not measure the {ep_len}-step return.")
    print(f"[E2] terminals set: {n_term} (== {n_eps} episodes)")
    tc = getattr(train_ds, "transition_count", None)
    if tc is not None:
        print(f"[E2] FQE transitions: {tc} (expected {n_eps * ep_len})")
        if tc != n_eps * ep_len:
            print("[E2][WARN] transition count differs from episodes x length; "
                  "check how d3rlpy counts transitions before trusting results.")


def load_data():
    with open(DATASET, "rb") as f:
        ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(ds.episodes)
    train_idx, _, test_idx = split_episode_indices(len(episodes), seed=SPLIT_SEED)
    mean, std = compute_scaler_from_dataset(DATASET)
    print(f"[E2] scaler mean[0]={float(np.asarray(mean).ravel()[0]):.4f} "
          f"std[0]={float(np.asarray(std).ravel()[0]):.4f}  "
          f"(compare with the scaler used when these checkpoints were trained)")

    train_eps = [episodes[i] for i in train_idx]
    test_eps = [episodes[i] for i in test_idx]

    lens = np.array([len(ep.observations) for ep in train_eps])
    ep_len = int(np.median(lens))
    raw_term = np.mean([bool(getattr(ep, "terminated", False)) for ep in train_eps])
    print(f"[E2] train episodes={len(train_eps)}  length min/median/max="
          f"{lens.min()}/{ep_len}/{lens.max()}  raw fraction terminated={raw_term:.2f}")
    if lens.min() != lens.max():
        print("[E2][WARN] episode lengths vary; the fixed-horizon interpretation "
              "of the FQE value needs a second look.")
    if ep_len != PAPER_ROLLOUT_HORIZON:
        print(f"[E2][WARN] episode length {ep_len} != rollout horizon "
              f"{PAPER_ROLLOUT_HORIZON}; FQE and rollout are different quantities.")

    train_ds = build_fqe_dataset(train_eps, mean, std)
    verify_fqe_dataset(train_ds, len(train_eps), ep_len)

    starts = np.stack([episodes_to_arrays([ep], mean, std)[0][0]
                       for ep in test_eps]).astype(np.float32)
    raw_starts = np.stack([ep.observations[0] for ep in test_eps]).astype(np.float32)

    all_test = np.asarray(episodes_to_arrays(test_eps, mean, std)[0], dtype=np.float32)
    rng = np.random.RandomState(BOOT_SEED)
    pick = rng.choice(len(all_test), size=min(N_RANDOM_STATES, len(all_test)),
                      replace=False)
    random_states = all_test[pick]
    print(f"[E2] eval sets: {len(starts)} episode starts (primary), "
          f"{len(random_states)} random test states (secondary)")
    return train_ds, starts, raw_starts, random_states, mean, std, ep_len


# ------------------------------------------------------------ utilities ---
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def fl(x):
    """Plain float from a python float or a 0-d numpy array."""
    return float(np.asarray(x))


def predict_actions(algo, states):
    return np.asarray(algo.predict(states)).reshape(-1, 1).astype(np.int64)


def action_dist(acts):
    return (np.bincount(acts.ravel(), minlength=N_ACTIONS) / len(acts)).round(3).tolist()


def tail_drift(values, k=5):
    """Relative range of the last k checkpoints of a trace."""
    v = np.asarray(values, dtype=float)[-k:]
    return float((v.max() - v.min()) / max(abs(v[-1]), 1.0))


# ------------------------------------------------------------------ FQE ---
def fit_fqe(algo, train_ds, seed, probe_states, n_steps):
    d3rlpy.seed(seed)
    cfg = FQEConfig(
        batch_size=FQE_BATCH, learning_rate=FQE_LR, gamma=FQE_GAMMA,
        target_update_interval=FQE_TARGET_UPDATE,
        optim_factory=AdamFactory(clip_grad_norm=1.0),
    )
    fqe = DiscreteFQE(algo=algo, config=cfg, device="cpu")
    probe_actions = predict_actions(algo, probe_states)
    trace = []

    def callback(fqe_algo, epoch, total_step):
        q = np.asarray(fqe_algo.predict_value(probe_states, probe_actions))
        trace.append((int(total_step), float(q.mean())))

    fqe.fit(train_ds, n_steps=n_steps,
            n_steps_per_epoch=min(FQE_STEPS_PER_EPOCH, n_steps),
            show_progress=False, callback=callback)
    return fqe, trace


def fqe_values(fqe, algo, states):
    acts = predict_actions(algo, states)
    return np.asarray(fqe.predict_value(states, acts)).reshape(-1).astype(np.float64)


def rollout_returns(algo, raw_starts, mean, std, horizon):
    """True simulator return for the policy from each raw start (calibration only).
    Uses src.rl.synthetic_reward as in the earlier E1/E3 scripts."""
    from src.rl.synthetic_reward import SimState, step, load_reward_config
    cfg = load_reward_config(str(REPO / "config" / "rl_config.yaml"))
    mean = np.asarray(mean, dtype=np.float32).ravel()
    std = np.asarray(std, dtype=np.float32).ravel()
    out = []
    for i, raw in enumerate(raw_starts):
        s = SimState(*[float(x) for x in raw])
        total, disc = 0.0, 1.0
        for _ in range(horizon):
            vec = np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                            s.month_sin, s.month_cos, s.extraction_rate],
                           dtype=np.float32)
            obs = ((vec - mean) / std).reshape(1, -1).astype(np.float32)
            a = int(np.asarray(algo.predict(obs)).reshape(-1)[0])
            s, r, done = step(s, a, config=cfg, n_actions=N_ACTIONS,
                              rng=np.random.RandomState(1000 + i))
            total += disc * float(r)
            disc *= FQE_GAMMA
            if done:
                break
        out.append(total)
    return np.array(out)


# ---------------------------------------------------------- per-seed run ---
def run_seed(seed, data, n_steps, rollout_check):
    train_ds, starts, raw_starts, random_states, mean, std, ep_len = data
    bc = load_policy_for_eval(BC_DIR / f"bc_seed{seed}.d3", DATASET).policy
    cql = load_policy_for_eval(CQL_DIR / f"cql_seed{seed}.d3", DATASET).policy

    rec = {"seed": seed}
    for name, algo in (("bc", bc), ("cql", cql)):
        dist = action_dist(predict_actions(algo, starts))
        rec[f"{name}_action_dist_starts"] = dist
        if max(dist) > COLLAPSE_FRAC:
            print(f"[E2][WARN] seed {seed} {name}: {max(dist):.0%} of start-state "
                  f"actions are one action {dist}")

    traces = {}
    for name, algo in (("bc", bc), ("cql", cql)):
        t0 = time.time()
        print(f"[E2]   fitting FQE on {name} seed {seed} ...")
        fqe, trace = fit_fqe(algo, train_ds, seed, starts, n_steps)
        traces[name] = trace
        rec[f"{name}_q_starts"] = fqe_values(fqe, algo, starts)
        rec[f"{name}_q_random"] = fqe_values(fqe, algo, random_states)
        rec[f"{name}_trace"] = np.array(trace)
        if rollout_check:
            rec[f"{name}_rollout_starts"] = rollout_returns(
                algo, raw_starts, mean, std, ep_len)
        print(f"[E2]   {name} done in {time.time() - t0:.0f}s")

    bc_t = np.array([v for _, v in traces["bc"]])
    cql_t = np.array([v for _, v in traces["cql"]])
    rec["bc_tail_drift"] = tail_drift(bc_t)
    rec["cql_tail_drift"] = tail_drift(cql_t)
    rec["diff_tail_range"] = float(np.ptp((cql_t - bc_t)[-5:]))
    rec["diff_starts_mean"] = float((rec["cql_q_starts"] - rec["bc_q_starts"]).mean())
    print(f"[E2]   seed {seed}: CQL-BC at starts = {rec['diff_starts_mean']:+.4f}  "
          f"tail drift BC/CQL = {rec['bc_tail_drift']:.3f}/{rec['cql_tail_drift']:.3f}")
    return rec


def _seed_cache_dir():
    """Per-seed caches are kept in a subfolder keyed by the sensitivity tag,
    so tui200 and tui500 do not overwrite each other's per-seed npz files."""
    return OUT_DIR / "seeds"


def save_seed(rec):
    d = _seed_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"seed_{rec['seed']}.npz",
             **{k: np.asarray(v) for k, v in rec.items()})


def load_seed(seed):
    p = _seed_cache_dir() / f"seed_{seed}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------ statistics ---
def t_interval(x, level=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    sd = float(x.std(ddof=1))
    try:
        from scipy import stats
        crit = float(stats.t.ppf(0.5 + level / 2, df=n - 1))
    except Exception:
        crit = 1.96
        print("[E2][WARN] scipy not found; using z=1.96 (too narrow for small n)")
    half = crit * sd / np.sqrt(n)
    return m, float(m - half), float(m + half)


def summarize(recs, key_bc, key_cql):
    per_seed = np.array([(r[key_cql] - r[key_bc]).mean() for r in recs])
    m, lo, hi = t_interval(per_seed)
    state_diff = np.mean([r[key_cql] - r[key_bc] for r in recs], axis=0)
    rng = np.random.RandomState(BOOT_SEED)
    n = len(state_diff)
    boots = np.array([state_diff[rng.randint(0, n, n)].mean()
                      for _ in range(N_BOOTSTRAP)])
    blo, bhi = np.percentile(boots, [2.5, 97.5])
    return {
        "per_seed_diffs": per_seed.tolist(),
        "seed_mean": m, "seed_ci_lower": lo, "seed_ci_upper": hi,
        "seed_ci_excludes_zero": bool(lo > 0 or hi < 0),
        "n_seeds": int(len(per_seed)),
        "n_seeds_positive": int((per_seed > 0).sum()),
        "state_bootstrap_ci": [float(blo), float(bhi)],
        "n_states": int(n),
    }


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 seeds, 2000 steps")
    ap.add_argument("--force", action="store_true", help="ignore saved seeds")
    ap.add_argument("--rollout-check", action="store_true",
                    help="also compute true simulator returns for calibration")
    ap.add_argument("--target-update-interval", type=int, default=200,
                    help="FQE target-network update interval (default 200)")
    ap.add_argument("--tag", default="",
                    help="output suffix; writes e2_fqe_<tag>.json")
    args = ap.parse_args()

    # Sensitivity override: mutate the module-level constant directly via
    # globals() so no `global` statement is needed (which the argparse
    # default read above would have made illegal).
    globals()["FQE_TARGET_UPDATE"] = args.target_update_interval
    print(f"[E2] FQE target-update interval: {args.target_update_interval}")

    # Per-seed cache: for a tagged run, keep the tag in the subfolder name so
    # tui200 and tui500 do not clobber each other's npz caches.
    if args.tag:
        global _seed_cache_dir
        _tag = args.tag
        _seed_cache_dir = lambda: OUT_DIR / "seeds" / _tag

    seeds = ALL_SEEDS[:2] if args.smoke else ALL_SEEDS
    n_steps = 2_000 if args.smoke else FQE_N_STEPS
    if args.smoke:
        print("[E2] SMOKE MODE - results are not meaningful")

    data = load_data()
    recs = []
    for seed in seeds:
        print(f"\n[E2] seed {seed}")
        rec = None if (args.force or args.smoke) else load_seed(seed)
        if rec is not None and args.rollout_check and "bc_rollout_starts" not in rec:
            print("[E2]   saved result has no rollout data; recomputing")
            rec = None
        if rec is not None:
            print("[E2]   loaded saved result")
        else:
            rec = run_seed(seed, data, n_steps, args.rollout_check)
            if not args.smoke:
                save_seed(rec)
        recs.append(rec)

    starts_stats = summarize(recs, "bc_q_starts", "cql_q_starts")
    random_stats = summarize(recs, "bc_q_random", "cql_q_random")

    bc_drift = [fl(r["bc_tail_drift"]) for r in recs]
    cql_drift = [fl(r["cql_tail_drift"]) for r in recs]
    converged = all(d < CONV_TOL for d in bc_drift + cql_drift)

    result = {
        "protocol": {
            "dataset": str(DATASET), "seeds": seeds, "fqe_n_steps": n_steps,
            "gamma": FQE_GAMMA, "target_update_interval": FQE_TARGET_UPDATE,
            "learning_rate": FQE_LR, "convergence_tolerance": CONV_TOL,
            "terminal_forced_at_last_step": True,
            "smoke": args.smoke, "git_commit": git_commit(),
            "d3rlpy": d3rlpy.__version__,
            "run_time": datetime.now().isoformat(timespec="seconds"),
            "tag": args.tag or None,
        },
        "convergence": {
            "all_seeds_converged": bool(converged),
            "bc_tail_drift": bc_drift,
            "cql_tail_drift": cql_drift,
            "diff_tail_range": [fl(r["diff_tail_range"]) for r in recs],
        },
        "primary_episode_starts": starts_stats,
        "secondary_random_states": random_stats,
    }

    if args.rollout_check and all("bc_rollout_starts" in r for r in recs):
        cal = {}
        for name in ("bc", "cql"):
            fq = float(np.mean([r[f"{name}_q_starts"].mean() for r in recs]))
            ro = float(np.mean([r[f"{name}_rollout_starts"].mean() for r in recs]))
            cal[name] = {"fqe_mean": fq, "rollout_mean": ro, "gap": fq - ro}
        result["calibration_vs_rollout"] = cal

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        out_name = "e2_smoke.json"
    elif args.tag:
        out_name = f"e2_fqe_{args.tag}.json"
    else:
        out_name = "e2_fqe.json"
    out = OUT_DIR / out_name
    out.write_text(json.dumps(result, indent=2))

    s = starts_stats
    print("\n" + "=" * 64)
    print(f"[E2] TARGET-UPDATE INTERVAL = {FQE_TARGET_UPDATE}")
    print("[E2] PRIMARY (episode starts, seed-level interval)")
    print(f"     CQL-BC = {s['seed_mean']:+.4f}  95% CI [{s['seed_ci_lower']:+.4f}, "
          f"{s['seed_ci_upper']:+.4f}]  positive in {s['n_seeds_positive']}/{s['n_seeds']} seeds")
    print(f"     state-bootstrap CI (secondary) "
          f"{np.round(s['state_bootstrap_ci'], 4).tolist()}")
    print(f"[E2] FQE converged on all seeds: {converged} "
          f"(tail drift < {CONV_TOL:.0%} of mean Q)")
    if "calibration_vs_rollout" in result:
        for k, v in result["calibration_vs_rollout"].items():
            print(f"[E2] calibration {k}: FQE {v['fqe_mean']:+.3f} vs rollout "
                  f"{v['rollout_mean']:+.3f} (gap {v['gap']:+.3f})")
    if not converged:
        print("[E2] NOT CONVERGED: do not report these values as evidence.")
    if args.smoke:
        print("[E2] smoke run only: this checks that the code runs, not the result.")
    print(f"[E2] wrote {out}")


if __name__ == "__main__":
    main()