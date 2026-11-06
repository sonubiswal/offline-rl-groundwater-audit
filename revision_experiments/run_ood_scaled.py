"""
Corrected OOD banded diagnostic for the frozen_models checkpoints.

Applies the training-split scaler (same as compare_policies.py) before
policy.predict(). Otherwise identical protocol to ood_return_test.py:
same dataset for training support, same chi2 bands, same 200 states
per band, same 20-step horizon, same seeds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import d3rlpy
from src.rl.synthetic_reward import SimState, step, load_reward_config
from _common import training_standardization, make_action_fn

DATASET  = REPO / "data" / "processed" / "offline_rl_dataset.h5"
MODEL_ROOT = REPO / "reports" / "paper_exp" / "frozen_models"
CFG_PATH = REPO / "config" / "rl_config.yaml"

SEEDS = [42, 123, 2024]
SPLIT_SEED = 1
N_TRAIN = 1050
N_STATES_PER_BAND = 200
HORIZON = 20
RNG_SEED = 2026
N_ACTIONS = 5

DIM_NAMES = ["gw_level", "recent_rainfall", "crop_water_demand",
             "month_sin", "month_cos", "extraction_rate"]
PHYSICAL_BOUNDS = {
    "gw_level": (1.0, 60.0), "recent_rainfall": (0.0, 400.0),
    "crop_water_demand": (0.0, 1.2), "month_sin": (-1.0, 1.0),
    "month_cos": (-1.0, 1.0), "extraction_rate": (0.0, 1.0),
}


def load_episodes(path):
    with h5py.File(path, "r") as f:
        ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                     if k.startswith("observations_"))
        obs = [f[f"observations_{e}"][:] for e in ids]
    return obs


def training_support(train_obs):
    mu = train_obs.mean(axis=0)
    cov = np.cov(train_obs, rowvar=False) + np.eye(train_obs.shape[1]) * 1e-8
    return mu, np.linalg.inv(cov)


def band_thresholds(df):
    return {"id_max": float(stats.chi2.ppf(0.68, df)),
            "mild_max": float(stats.chi2.ppf(0.95, df)),
            "moderate_max": float(stats.chi2.ppf(0.997, df))}


def physical_mask(states):
    ok = np.ones(len(states), dtype=bool)
    for j, name in enumerate(DIM_NAMES):
        lo, hi = PHYSICAL_BOUNDS[name]
        ok &= (states[:, j] >= lo) & (states[:, j] <= hi)
    return ok


def unit_circle(states):
    s, c = states[:, 3], states[:, 4]
    r = np.sqrt(s ** 2 + c ** 2); r[r < 1e-12] = 1.0
    states[:, 3] = s / r; states[:, 4] = c / r
    return states


def sample_bands(mu, cov, inv_cov, th, rng):
    want = {"id": lambda m: m <= th["id_max"],
            "mild": lambda m: (m > th["id_max"]) & (m <= th["mild_max"]),
            "moderate": lambda m: (m > th["mild_max"]) & (m <= th["moderate_max"]),
            "severe": lambda m: m > th["moderate_max"]}
    out = {k: [] for k in want}
    proposal_cov = cov * 1.5
    drawn = 0
    while drawn < 2_000_000 and any(len(v) < N_STATES_PER_BAND for v in out.values()):
        batch = rng.multivariate_normal(mu, proposal_cov, size=5000)
        drawn += len(batch)
        batch = unit_circle(batch)
        batch = batch[physical_mask(batch)]
        if len(batch) == 0:
            continue
        diff = batch - mu
        m2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        for k, f in want.items():
            if len(out[k]) >= N_STATES_PER_BAND:
                continue
            sel = batch[f(m2)]
            out[k].extend(sel[:N_STATES_PER_BAND - len(out[k])].tolist())
    return {k: np.array(v[:N_STATES_PER_BAND]) for k, v in out.items()}


def array_to_state(arr):
    return SimState(*[float(x) for x in arr])


def state_to_array(s):
    return np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                     s.month_sin, s.month_cos, s.extraction_rate], dtype=np.float32)


def rollout_return(action_fn, start_arr, reward_cfg, rng, horizon):
    s = array_to_state(start_arr)
    total = 0.0
    for _ in range(horizon):
        action = action_fn(state_to_array(s))
        s, r, done = step(s, action, config=reward_cfg,
                          n_actions=N_ACTIONS, rng=rng)
        total += float(r)
        if done:
            break
    return total


def main():
    print(f"[OOD] training support from: {DATASET.name}")
    obs_list = load_episodes(DATASET)
    rng_split = np.random.RandomState(SPLIT_SEED)
    idx = np.arange(len(obs_list)); rng_split.shuffle(idx)
    train_idx = idx[:N_TRAIN]
    train_obs = np.concatenate([obs_list[i] for i in train_idx], axis=0).astype(np.float32)
    print(f"[OOD] training observations: {train_obs.shape}")

    obs_mean, obs_std = training_standardization(DATASET)
    print(f"[OOD] scaler mean[0]={obs_mean[0]:.6f} std[0]={obs_std[0]:.6f}")

    reward_cfg = load_reward_config(str(CFG_PATH))
    mu, inv_cov = training_support(train_obs)
    cov = np.linalg.inv(inv_cov)
    th = band_thresholds(df=train_obs.shape[1])
    print(f"[OOD] chi2 thresholds: {th}")

    rng = np.random.RandomState(RNG_SEED)
    bands = sample_bands(mu, cov, inv_cov, th, rng)
    for k, v in bands.items():
        print(f"[OOD] band {k:9s}: {len(v)}")

    policies = {}
    for algo in ("bc", "cql"):
        for seed in SEEDS:
            p = MODEL_ROOT / algo / f"{algo}_seed{seed}.d3"
            if not p.exists():
                print(f"[OOD] missing {p}")
                continue
            pol = d3rlpy.load_learnable(str(p))
            policies[(algo, seed)] = make_action_fn(pol, obs_mean, obs_std)

    results = {}
    for level, states in bands.items():
        for (algo, seed), fn in policies.items():
            rng_l = np.random.RandomState(RNG_SEED + seed)
            rets = np.array([rollout_return(fn, s, reward_cfg, rng_l, HORIZON)
                             for s in states])
            results.setdefault((algo, seed), {})[level] = float(rets.mean())

    print("\nband        BC          CQL        CQL-BC")
    print("-" * 50)
    summary = {}
    for level in ["id", "mild", "moderate", "severe"]:
        bc_vals = [d[level] for (a, s), d in results.items() if a == "bc"]
        cql_vals = [d[level] for (a, s), d in results.items() if a == "cql"]
        bc_m = float(np.mean(bc_vals))
        cql_m = float(np.mean(cql_vals))
        print(f"{level:<10} {bc_m:>10.4f} {cql_m:>10.4f} {cql_m - bc_m:>10.4f}")
        summary[level] = {"bc": bc_m, "cql": cql_m, "cql_minus_bc": cql_m - bc_m}

    (HERE / "ood_scaled_result.json").write_text(
        json.dumps({"summary": summary,
                    "per_policy_per_band": {f"{a}|{s}": d
                                             for (a, s), d in results.items()}},
                   indent=2))
    print(f"\n[OOD] wrote {HERE/'ood_scaled_result.json'}")
    print(f"[OOD] paper's published Table X:")
    print(f"      id: -16.0057  mild: -23.6220  moderate: -29.7458  severe: -38.3978")


if __name__ == "__main__":
    main()