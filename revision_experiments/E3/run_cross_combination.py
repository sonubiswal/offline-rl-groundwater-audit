"""
revision_experiments/E3/run_cross_combination.py

E3: cross-combination with correct per-family scaler applied.

Each policy family is evaluated with the scaler computed from the dataset
it was trained on, matching compare_policies.load_training_standardization.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

import d3rlpy
from src.rl.synthetic_reward import SimState, step, load_reward_config
from _common import training_standardization, make_action_fn

CFG_PATH = REPO / "config" / "rl_config.yaml"
BASE_DS  = REPO / "data" / "processed" / "offline_rl_dataset.h5"
NOO_DS   = REPO / "data" / "processed" / "offline_rl_dataset_no_oracle.h5"
EPS_DS   = REPO / "data" / "processed" / "offline_rl_dataset_epsilon.h5"

CHECKPOINTS = {
    "base_BC":       (REPO / "models" / "bc",            "bc",  "base"),
    "base_CQL":      (REPO / "models" / "cql",           "cql", "base"),
    "no_oracle_BC":  (REPO / "models_no_oracle" / "bc",  "bc",  "no_oracle"),
    "no_oracle_CQL": (REPO / "models_no_oracle" / "cql", "cql", "no_oracle"),
}

SEEDS = [42, 123, 2024]
N_EVAL = 500
HORIZON = 20
RNG_SEED = 2026
N_ACTIONS = 5


def load_starts(path: Path):
    with h5py.File(path, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        starts = np.array([f[f"observations_{e}"][0] for e in ep_ids],
                          dtype=np.float32)
    return starts


def array_to_state(arr):
    return SimState(*[float(x) for x in arr])


def state_to_array(s):
    return np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                     s.month_sin, s.month_cos, s.extraction_rate],
                    dtype=np.float32)


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
    reward_cfg = load_reward_config(str(CFG_PATH))

    print("[E3] computing scalers ...")
    scalers = {
        "base":      training_standardization(BASE_DS),
        "no_oracle": training_standardization(NOO_DS),
    }
    for k, (m, s) in scalers.items():
        print(f"[E3] {k} scaler mean[0]={m[0]:.6f} std[0]={s[0]:.6f}")

    action_fns = {}
    for family, (dir_, stem, key) in CHECKPOINTS.items():
        m, s = scalers[key]
        for seed in SEEDS:
            p = dir_ / f"{stem}_seed{seed}.d3"
            if not p.exists():
                print(f"[E3] missing {p}")
                continue
            pol = d3rlpy.load_learnable(str(p))
            action_fns[(family, seed)] = make_action_fn(pol, m, s)

    for start_label, start_path in [("eps", EPS_DS), ("base", BASE_DS),
                                     ("no_oracle", NOO_DS)]:
        print(f"\n=== Evaluation starts: {start_label} ({start_path.name}) ===")
        starts_all = load_starts(start_path)
        rng = np.random.RandomState(RNG_SEED)
        idx = np.arange(len(starts_all)); rng.shuffle(idx)
        starts = starts_all[idx[:N_EVAL]]

        per_seed = {}
        for (family, seed), fn in action_fns.items():
            rets = []
            for i, s in enumerate(starts):
                local_rng = np.random.RandomState(RNG_SEED + i)
                rets.append(rollout_return(fn, s, reward_cfg, local_rng, HORIZON))
            per_seed[(family, seed)] = float(np.mean(rets))

        fam_mean, fam_std = {}, {}
        for family in CHECKPOINTS:
            vals = [per_seed[(family, s)] for s in SEEDS
                    if (family, s) in per_seed]
            fam_mean[family] = float(np.mean(vals)) if vals else float("nan")
            fam_std[family] = float(np.std(vals)) if vals else float("nan")
            print(f"  {family:<14} mean={fam_mean[family]:+.4f}  "
                  f"std={fam_std[family]:.4f}")

        print("  2x2 cross-combination (CQL - BC)")
        print(f"    {'':<14} base_CQL     no_oracle_CQL")
        for bc in ("base_BC", "no_oracle_BC"):
            d1 = fam_mean["base_CQL"] - fam_mean[bc]
            d2 = fam_mean["no_oracle_CQL"] - fam_mean[bc]
            print(f"    {bc:<14} {d1:+.4f}      {d2:+.4f}")

        (HERE / f"cross_combination_{start_label}.json").write_text(
            json.dumps({"per_seed": {f"{f}|{s}": v
                                      for (f, s), v in per_seed.items()},
                        "family_mean": fam_mean,
                        "family_std": fam_std}, indent=2))


if __name__ == "__main__":
    main()