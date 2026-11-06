"""
Corrected Mean-Q OOD banded diagnostic.

BC checkpoints from frozen_models/; CQL checkpoints from cql_mean/.
Same scaler (both trained on dataset_normal_train.h5).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import h5py, numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import d3rlpy
from src.rl.synthetic_reward import SimState, step, load_reward_config

DATASET      = REPO / "data" / "processed" / "offline_rl_dataset.h5"
BC_ROOT      = REPO / "reports" / "paper_exp" / "frozen_models" / "bc"
CQL_ROOT     = REPO / "reports" / "paper_exp" / "cql_mean"
MANIFEST     = REPO / "reports" / "paper_exp" / "frozen_models" / "normalization_manifest.json"
CFG_PATH     = REPO / "config" / "rl_config.yaml"

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
        return [f[f"observations_{e}"][:] for e in ids]


def training_support(train_obs):
    mu = train_obs.mean(axis=0)
    cov = np.cov(train_obs, rowvar=False) + np.eye(train_obs.shape[1]) * 1e-8
    return mu, np.linalg.inv(cov)


def band_thresholds(df):
    return {"id_max": float(stats.chi2.ppf(0.68, df)),
            "mild_max": float(stats.chi2.ppf(0.95, df)),
            "moderate_max": float(stats.chi2.ppf(0.997, df))}


def physical_mask(s):
    ok = np.ones(len(s), dtype=bool)
    for j, name in enumerate(DIM_NAMES):
        lo, hi = PHYSICAL_BOUNDS[name]
        ok &= (s[:, j] >= lo) & (s[:, j] <= hi)
    return ok


def unit_circle(s):
    a, b = s[:, 3], s[:, 4]
    r = np.sqrt(a ** 2 + b ** 2); r[r < 1e-12] = 1.0
    s[:, 3] = a / r; s[:, 4] = b / r
    return s


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


def make_action_fn(policy, obs_mean, obs_std):
    def fn(sv):
        raw = np.asarray(sv, dtype=np.float32)
        return int(np.asarray(policy.predict(((raw - obs_mean) / obs_std).reshape(1, -1))).reshape(-1)[0])
    return fn


def rollout_return(action_fn, start_arr, reward_cfg, rng, horizon):
    s = array_to_state(start_arr)
    total = 0.0
    for _ in range(horizon):
        a = action_fn(state_to_array(s))
        s, r, done = step(s, a, config=reward_cfg, n_actions=N_ACTIONS, rng=rng)
        total += float(r)
        if done:
            break
    return total


def main():
    manifest = json.load(open(MANIFEST))
    obs_mean = np.array(manifest["mean"], dtype=np.float32)
    obs_std = np.array(manifest["std"], dtype=np.float32)
    print(f"[OOD-MeanQ] scaler mean[0]={obs_mean[0]:.6f} std[0]={obs_std[0]:.6f}")

    obs_list = load_episodes(DATASET)
    rng_split = np.random.RandomState(SPLIT_SEED)
    idx = np.arange(len(obs_list)); rng_split.shuffle(idx)
    train_obs = np.concatenate([obs_list[i] for i in idx[:N_TRAIN]], axis=0).astype(np.float32)

    reward_cfg = load_reward_config(str(CFG_PATH))
    mu, inv_cov = training_support(train_obs)
    cov = np.linalg.inv(inv_cov)
    th = band_thresholds(df=train_obs.shape[1])
    rng = np.random.RandomState(RNG_SEED)
    bands = sample_bands(mu, cov, inv_cov, th, rng)

    policies = {}
    for seed in SEEDS:
        p = BC_ROOT / f"bc_seed{seed}.d3"
        if p.exists():
            policies[("bc", seed)] = make_action_fn(d3rlpy.load_learnable(str(p)), obs_mean, obs_std)
    for seed in SEEDS:
        p = CQL_ROOT / f"cql_seed{seed}.d3"
        if p.exists():
            policies[("cql", seed)] = make_action_fn(d3rlpy.load_learnable(str(p)), obs_mean, obs_std)

    results = {}
    for level, states in bands.items():
        for (algo, seed), fn in policies.items():
            rng_l = np.random.RandomState(RNG_SEED + seed)
            rets = [rollout_return(fn, s, reward_cfg, rng_l, HORIZON) for s in states]
            results.setdefault((algo, seed), {})[level] = float(np.mean(rets))

    print("\nband        BC          CQL(MeanQ)  CQL-BC")
    print("-" * 50)
    summary = {}
    for level in ["id", "mild", "moderate", "severe"]:
        bc = np.mean([d[level] for (a, s), d in results.items() if a == "bc"])
        cq = np.mean([d[level] for (a, s), d in results.items() if a == "cql"])
        print(f"{level:<10} {bc:>10.4f} {cq:>10.4f} {cq - bc:>10.4f}")
        summary[level] = {"bc": float(bc), "cql": float(cq), "cql_minus_bc": float(cq - bc)}

    (HERE / "ood_scaled_meanq_result.json").write_text(json.dumps(
        {"scaler": {"mean": obs_mean.tolist(), "std": obs_std.tolist()},
         "summary": summary}, indent=2))
    print(f"\n[OOD-MeanQ] wrote {HERE/'ood_scaled_meanq_result.json'}")
    print(f"[OOD-MeanQ] paper's Table XI: id -8.64  severe -18.17")


if __name__ == "__main__":
    main()