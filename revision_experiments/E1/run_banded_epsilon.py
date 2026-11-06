"""
revision_experiments/E1/run_banded_epsilon.py

E1: banded state-distance diagnostic on epsilon checkpoints, with the
training scaler applied before every policy.predict().
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

import d3rlpy
from src.rl.synthetic_reward import SimState, step, load_reward_config
from _common import (training_standardization, make_action_fn, load_episodes)

EVAL_DATASET = REPO / "data" / "processed" / "offline_rl_dataset_epsilon.h5"
CFG_PATH     = REPO / "config" / "rl_config.yaml"

BC_DIR = REPO / "models_epsilon" / "bc"
CQL_DIRS = {
    "cql_main":         REPO / "models_epsilon" / "cql",
    "cql_qr32_verify":  REPO / "models_epsilon_qr32_verify",
    "cql_meanq_verify": REPO / "models_epsilon_meanq_verify",
}

SEEDS = [42, 123, 2024]
N_STATES = 200
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


def training_support(obs):
    mu = obs.mean(axis=0)
    cov = np.cov(obs, rowvar=False) + np.eye(obs.shape[1]) * 1e-8
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


def sample_bands(mu, cov, inv_cov, th, rng, n_per_band=N_STATES):
    want = {"id": lambda m: m <= th["id_max"],
            "mild": lambda m: (m > th["id_max"]) & (m <= th["mild_max"]),
            "moderate": lambda m: (m > th["mild_max"]) & (m <= th["moderate_max"]),
            "severe": lambda m: m > th["moderate_max"]}
    out = {k: [] for k in want}
    proposal_cov = cov * 1.5
    drawn = 0
    while drawn < 2_000_000 and any(len(v) < n_per_band for v in out.values()):
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
            out[k].extend(sel[:n_per_band - len(out[k])].tolist())
    return {k: np.array(v[:n_per_band]) for k, v in out.items()}, drawn


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
    print(f"[E1] eval dataset: {EVAL_DATASET.name}")
    reward_cfg = load_reward_config(str(CFG_PATH))

    obs_mean, obs_std = training_standardization(EVAL_DATASET)
    print(f"[E1] scaler mean[0]={obs_mean[0]:.6f} std[0]={obs_std[0]:.6f}")

    obs_list, _ = load_episodes(EVAL_DATASET)
    from _common import training_split_indices
    tr = training_split_indices(len(obs_list))
    train_obs = np.concatenate([obs_list[i] for i in tr], axis=0).astype(np.float32)
    mu, inv_cov = training_support(train_obs)
    th = band_thresholds(df=train_obs.shape[1])
    cov = np.linalg.inv(inv_cov)

    rng = np.random.RandomState(RNG_SEED)
    bands, _ = sample_bands(mu, cov, inv_cov, th, rng)
    for k, v in bands.items():
        print(f"[E1] band {k:9s}: {len(v)}")

    action_fns = {}
    for seed in SEEDS:
        p = BC_DIR / f"bc_seed{seed}.d3"
        if p.exists():
            pol = d3rlpy.load_learnable(str(p))
            action_fns[f"bc_seed{seed}"] = make_action_fn(pol, obs_mean, obs_std)
    for label, d in CQL_DIRS.items():
        for seed in SEEDS:
            p = d / f"cql_seed{seed}.d3"
            if p.exists():
                pol = d3rlpy.load_learnable(str(p))
                action_fns[f"{label}_seed{seed}"] = make_action_fn(pol, obs_mean, obs_std)

    results = {"per_policy_per_band": {}}
    for level, states in bands.items():
        for name, fn in action_fns.items():
            rng_l = np.random.RandomState(RNG_SEED + int(name.split("seed")[-1]))
            rets = np.array([rollout_return(fn, s, reward_cfg, rng_l, HORIZON)
                             for s in states])
            results["per_policy_per_band"].setdefault(name, {})[level] = \
                float(rets.mean())

    bc = {lvl: [] for lvl in bands}
    for k, v in results["per_policy_per_band"].items():
        if k.startswith("bc_"):
            for lvl, r in v.items():
                bc[lvl].append(r)
    bc_mean = {lvl: float(np.mean(v)) for lvl, v in bc.items()}

    print("\nband        BC        cql_main  cql_qr32  cql_meanq    CQL-BC")
    print("-" * 70)
    summary = {}
    for lvl in ["id", "mild", "moderate", "severe"]:
        cm = {k: float(np.mean([d[lvl] for n, d in results["per_policy_per_band"].items()
                                if n.startswith(k + "_")]))
              for k in CQL_DIRS}
        print(f"{lvl:<10} {bc_mean[lvl]:>10.4f} {cm['cql_main']:>10.4f} "
              f"{cm['cql_qr32_verify']:>10.4f} {cm['cql_meanq_verify']:>10.4f} "
              f"{cm['cql_main'] - bc_mean[lvl]:>10.4f}")
        summary[lvl] = {"bc": bc_mean[lvl], "cql_main": cm["cql_main"],
                        "cql_minus_bc_main": cm["cql_main"] - bc_mean[lvl]}

    (HERE / "banded_epsilon_corrected.json").write_text(
        json.dumps({"summary": summary,
                    "per_policy_per_band": results["per_policy_per_band"]}, indent=2))
    print(f"\n[E1] in-distribution CQL-BC (main) = "
          f"{summary['id']['cql_minus_bc_main']:+.4f}")
    print(f"[E1] paper's epsilon rollout CQL-BC = +0.3402")


if __name__ == "__main__":
    main()