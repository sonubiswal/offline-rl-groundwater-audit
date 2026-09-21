"""
ood_return_test.py
Return-based state-distribution OOD test for Phase 5 BC and CQL policies.

Complements ood_state_distribution_test.py (which compared action distributions)
by rolling each policy forward from ID / mild / moderate / severe OOD states and
comparing returns.

Also tracks per-rollout step count and termination fraction, to check whether
early `done` is biasing the return comparison across policies or levels.

Reads:
  data/processed/offline_rl_dataset.h5
  reports/paper_exp/frozen_models/bc/bc_seed{42,123,2024}.d3
  reports/paper_exp/frozen_models/cql/cql_seed{42,123,2024}.d3
  config/rl_config.yaml

Writes:
  reports/ood_return_test_meanq.json
  reports/ood_return_test_meanq.png

Read-only. Does not modify any experiment artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import d3rlpy

from src.rl.synthetic_reward import SimState, step, load_reward_config

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

DATASET = "data/processed/offline_rl_dataset.h5"
MODEL_ROOT = Path("reports/paper_exp/frozen_models")
CQL_MODEL_ROOT = Path("reports/paper_exp/cql_mean")  # Mean-Q override (see ood_return_test.py for the QR version)
SEEDS = [42, 123, 2024]

SPLIT_SEED = 1
N_TRAIN, N_VAL, N_TEST = 1050, 225, 225
N_EPISODES = N_TRAIN + N_VAL + N_TEST

N_STATES_PER_LEVEL = 200
ROLLOUT_HORIZON = 20
RNG_SEED = 2026
N_ACTIONS = 5

DIM_NAMES = ["gw_level", "recent_rainfall", "crop_water_demand",
             "month_sin", "month_cos", "extraction_rate"]

PHYSICAL_BOUNDS = {
    "gw_level":          (1.0,   60.0),
    "recent_rainfall":   (0.0,   400.0),
    "crop_water_demand": (0.0,   1.2),
    "month_sin":         (-1.0,  1.0),
    "month_cos":         (-1.0,  1.0),
    "extraction_rate":   (0.0,   1.0),
}


# ------------------------------------------------------------------ data
def load_dataset():
    with h5py.File(DATASET, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        episodes = [f[f"observations_{eid}"][:] for eid in ep_ids]
    return episodes


def split_episodes(episodes, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    return (idx[:N_TRAIN],
            idx[N_TRAIN:N_TRAIN + N_VAL],
            idx[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST])


def training_support(train_obs):
    mu = train_obs.mean(axis=0)
    cov = np.cov(train_obs, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-8
    inv_cov = np.linalg.inv(cov)
    return mu, cov, inv_cov


def band_thresholds(df):
    return {
        "id_max":       float(stats.chi2.ppf(0.68,  df)),
        "mild_max":     float(stats.chi2.ppf(0.95,  df)),
        "moderate_max": float(stats.chi2.ppf(0.997, df)),
    }


def physical_mask(states):
    ok = np.ones(len(states), dtype=bool)
    for j, name in enumerate(DIM_NAMES):
        lo, hi = PHYSICAL_BOUNDS[name]
        ok &= (states[:, j] >= lo) & (states[:, j] <= hi)
    return ok


def unit_circle(states):
    s = states[:, 3]; c = states[:, 4]
    r = np.sqrt(s ** 2 + c ** 2); r[r < 1e-12] = 1.0
    states[:, 3] = s / r; states[:, 4] = c / r
    return states


def sample_bands(mu, cov, inv_cov, th, rng, n_per_band=N_STATES_PER_LEVEL):
    want = {
        "id":       lambda m2: m2 <= th["id_max"],
        "mild":     lambda m2: (m2 >  th["id_max"]) & (m2 <= th["mild_max"]),
        "moderate": lambda m2: (m2 >  th["mild_max"]) & (m2 <= th["moderate_max"]),
        "severe":   lambda m2: m2 > th["moderate_max"],
    }
    out = {k: [] for k in want}
    proposal_cov = cov * 1.5
    max_draws = 2_000_000
    drawn = 0
    while drawn < max_draws and any(len(v) < n_per_band for v in out.values()):
        batch = rng.multivariate_normal(mu, proposal_cov, size=5000)
        drawn += len(batch)
        batch = unit_circle(batch)
        batch = batch[physical_mask(batch)]
        if len(batch) == 0:
            continue
        diff = batch - mu
        m2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        for k, f in want.items():
            if len(out[k]) >= n_per_band:
                continue
            sel = batch[f(m2)]
            room = n_per_band - len(out[k])
            out[k].extend(sel[:room].tolist())
    return {k: np.array(v[:n_per_band]) for k, v in out.items()}, drawn


# ------------------------------------------------------- simulator glue
def array_to_state(arr) -> SimState:
    return SimState(
        gw_level=float(arr[0]),
        recent_rainfall=float(arr[1]),
        crop_water_demand=float(arr[2]),
        month_sin=float(arr[3]),
        month_cos=float(arr[4]),
        extraction_rate=float(arr[5]),
    )


def state_to_array(s: SimState) -> np.ndarray:
    return np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                     s.month_sin, s.month_cos, s.extraction_rate],
                    dtype=np.float32)


def rollout_return(policy, start_arr, reward_cfg, rng, horizon):
    """
    Returns (sum_reward, n_steps_executed, terminated_early).
    terminated_early is True iff the simulator's done flag fired before horizon.
    """
    s = array_to_state(start_arr)
    total = 0.0
    n_steps = 0
    terminated_early = False
    for _ in range(horizon):
        obs = state_to_array(s).reshape(1, -1)
        action = int(np.asarray(policy.predict(obs)).reshape(-1)[0])
        s, r, done = step(s, action, config=reward_cfg,
                          n_actions=N_ACTIONS, rng=rng)
        total += float(r)
        n_steps += 1
        if done:
            terminated_early = True
            break
    return total, n_steps, terminated_early


def load_policy(path: Path):
    return d3rlpy.load_learnable(str(path))


# ----------------------------------------------------------------- main
def main():
    print("[ood_return] loading dataset ...")
    episodes = load_dataset()
    train_idx, _, _ = split_episodes(episodes)
    train_obs = np.concatenate([episodes[i] for i in train_idx], axis=0)
    print(f"[ood_return] training observations: {train_obs.shape}")

    print("[ood_return] loading reward config ...")
    reward_cfg = load_reward_config("config/rl_config.yaml")

    print("[ood_return] computing training support ...")
    mu, cov, inv_cov = training_support(train_obs)
    th = band_thresholds(df=train_obs.shape[1])
    print(f"[ood_return] thresholds: {th}")

    rng_sample = np.random.RandomState(RNG_SEED)
    bands, n_drawn = sample_bands(mu, cov, inv_cov, th, rng_sample)
    for k, v in bands.items():
        print(f"[ood_return] band {k:9s}: {len(v)} states (drew {n_drawn})")

    policies = []
    for algo in ("bc", "cql"):
        for seed in SEEDS:
            p = (CQL_MODEL_ROOT / f"{algo}_seed{seed}.d3") if algo == "cql" else (MODEL_ROOT / algo / f"{algo}_seed{seed}.d3")
            if p.exists():
                policies.append((algo, seed, load_policy(p)))
            else:
                print(f"[ood_return] WARNING missing {p}")

    results = {
        "setup": {
            "horizon": ROLLOUT_HORIZON,
            "n_states_per_band": N_STATES_PER_LEVEL,
            "mahalanobis2_thresholds": th,
            "reward_config": {k: (float(v) if isinstance(v, (int, float)) else v)
                              for k, v in reward_cfg.items()},
        },
        "per_policy_per_level": {},
        "summary": {},
    }

    for level, states in bands.items():
        print(f"[ood_return] rolling out {level} ({len(states)} states)")
        for algo, seed, pol in policies:
            rng = np.random.RandomState(RNG_SEED + seed)
            rets, steps, terminated = [], [], []
            for s_arr in states:
                r, n, term = rollout_return(pol, s_arr, reward_cfg, rng,
                                            ROLLOUT_HORIZON)
                rets.append(r)
                steps.append(n)
                terminated.append(term)
            rets = np.asarray(rets)
            steps = np.asarray(steps)
            key = f"{algo}_seed{seed}"
            results["per_policy_per_level"].setdefault(key, {})[level] = {
                "mean_return":      float(rets.mean()),
                "std_return":       float(rets.std()),
                "mean_steps":       float(steps.mean()),
                "terminated_frac":  float(np.mean(terminated)),
                "n":                int(len(rets)),
            }

    print("\n[ood_return] summary (mean return across seeds)")
    print(f"{'level':<10} {'BC':>10} {'CQL':>10} {'d(CQL-BC)':>12}")
    print("-" * 46)
    for level in ["id", "mild", "moderate", "severe"]:
        bc_rets  = [v[level]["mean_return"]
                    for k, v in results["per_policy_per_level"].items()
                    if k.startswith("bc_")]
        cql_rets = [v[level]["mean_return"]
                    for k, v in results["per_policy_per_level"].items()
                    if k.startswith("cql_")]
        bc_r  = float(np.mean(bc_rets))
        cql_r = float(np.mean(cql_rets))
        results["summary"][level] = {
            "bc_mean_return":  bc_r,
            "cql_mean_return": cql_r,
            "delta_cql_minus_bc": cql_r - bc_r,
            "bc_std_across_seeds":  float(np.std(bc_rets)),
            "cql_std_across_seeds": float(np.std(cql_rets)),
        }
        print(f"{level:<10} {bc_r:>10.4f} {cql_r:>10.4f} {cql_r - bc_r:>+12.4f}")

    # ---- termination diagnostic ----
    print("\n[ood_return] termination diagnostic (mean steps, terminated fraction)")
    print(f"{'level':<10} {'policy':<14} {'mean_steps':>11} {'term_frac':>10}")
    print("-" * 48)
    for level in ["id", "mild", "moderate", "severe"]:
        for k, v in sorted(results["per_policy_per_level"].items()):
            d = v[level]
            print(f"{level:<10} {k:<14} {d['mean_steps']:>11.2f} {d['terminated_frac']:>10.3f}")

    # ---- figure ----
    levels = ["id", "mild", "moderate", "severe"]
    x = np.arange(len(levels))
    bc_means  = [results["summary"][l]["bc_mean_return"]  for l in levels]
    cql_means = [results["summary"][l]["cql_mean_return"] for l in levels]
    bc_err    = [results["summary"][l]["bc_std_across_seeds"]  for l in levels]
    cql_err   = [results["summary"][l]["cql_std_across_seeds"] for l in levels]
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, bc_means,  width=w, yerr=bc_err,  capsize=4,
           label="BC",  color="steelblue")
    ax.bar(x + w/2, cql_means, width=w, yerr=cql_err, capsize=4,
           label="CQL", color="darkorange")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in levels])
    ax.set_ylabel(f"Mean return over up to {ROLLOUT_HORIZON}-step rollout")
    ax.set_title("Return under state-distribution shift (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "ood_return_test_meanq.png", dpi=150)

    (REPORTS / "ood_return_test_meanq.json").write_text(json.dumps(results, indent=2))
    print(f"\n[ood_return] wrote {REPORTS/'ood_return_test_meanq.json'}")
    print(f"[ood_return] wrote {REPORTS/'ood_return_test_meanq.png'}")


if __name__ == "__main__":
    main()