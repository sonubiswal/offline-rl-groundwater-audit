"""
E2 — FQE convergence diagnostic and per-state paired bootstrap.

Trains FQE (d3rlpy.ope.DiscreteFQE) on all 10 ε BC/CQL checkpoints,
computes per-state Q-values via fqe.predict_value(x, action), and pools
across seeds for a per-state paired bootstrap.

Uses only the verified d3rlpy 2.8.1 API:
    from d3rlpy.ope import DiscreteFQE, FQEConfig
    fqe = DiscreteFQE(algo=target_algo, config=FQEConfig(...), device="cpu")
    fqe.predict_value(x, action)  # action shape (batch, 1) int64
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "rl"))

import d3rlpy
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
from d3rlpy.optimizers import AdamFactory

from src.rl.policy_io import load_policy_for_eval, compute_scaler_from_dataset
from src.rl.bc_baseline import (
    split_episode_indices, episodes_to_arrays, build_dataset, SPLIT_SEED,
)


DATASET = REPO / "data" / "processed" / "offline_rl_dataset_epsilon.h5"
BC_DIR = REPO / "models_10seed" / "epsilon" / "bc"
CQL_DIR = REPO / "models_10seed" / "epsilon" / "cql"
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]

N_ACTIONS = 5
FQE_N_STEPS = 20_000
FQE_N_STEPS_PER_EPOCH = 2_000
FQE_LR = 3e-4
FQE_BATCH = 100
FQE_TARGET_UPDATE_INTERVAL = 8000
N_EVAL_STATES = 500
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 5000


def load_dataset_arrays():
    with open(DATASET, "rb") as f:
        ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(ds.episodes)
    train_idx, _, test_idx = split_episode_indices(len(episodes), seed=SPLIT_SEED)

    obs_mean, obs_std = compute_scaler_from_dataset(DATASET)

    train_eps = [episodes[i] for i in train_idx]
    train_ds = build_dataset(train_eps, obs_mean, obs_std)

    test_eps = [episodes[i] for i in test_idx]
    test_obs, _, _, _, _ = episodes_to_arrays(test_eps, obs_mean, obs_std)

    return train_ds, test_obs


def train_fqe_on_algo(target_algo, train_ds, seed):
    """Fit a DiscreteFQE critic on the given (already-trained) algo."""
    d3rlpy.seed(seed)
    fqe_config = FQEConfig(
        batch_size=FQE_BATCH,
        learning_rate=FQE_LR,
        target_update_interval=FQE_TARGET_UPDATE_INTERVAL,
        optim_factory=AdamFactory(clip_grad_norm=1.0),
    )
    fqe = DiscreteFQE(algo=target_algo, config=fqe_config, device="cpu")
    fqe.fit(
        train_ds,
        n_steps=FQE_N_STEPS,
        n_steps_per_epoch=FQE_N_STEPS_PER_EPOCH,
        show_progress=False,
    )
    return fqe


def q_at_state(fqe, standardized_state, target_algo):
    """Q-value that the FQE critic assigns to the target policy's action
    at this standardized state.

    d3rlpy 2.8.1 signature: predict_value(x, action) where action is
    shape (batch, 1) int64.
    """
    obs = np.asarray(standardized_state, dtype=np.float32).reshape(1, -1)

    a = int(np.asarray(target_algo.predict(obs)).reshape(-1)[0])
    action_arr = np.array([[a]], dtype=np.int64)

    q = fqe.predict_value(obs, action_arr)
    return float(np.asarray(q).reshape(-1)[0])


def main():
    print("[E2] loading dataset")
    train_ds, test_obs = load_dataset_arrays()

    n_trans = sum(len(ep.observations) for ep in train_ds.episodes)
    print(f"[E2] train transitions: {n_trans}")
    print(f"[E2] test obs: {test_obs.shape}")

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    idx = rng.choice(len(test_obs), size=N_EVAL_STATES, replace=False)
    eval_states = test_obs[idx]
    print(f"[E2] eval states: {eval_states.shape}")

    results = {
        "protocol": {
            "dataset": str(DATASET),
            "seeds": SEEDS,
            "n_eval_states": int(len(eval_states)),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": N_BOOTSTRAP,
            "fqe_n_steps": FQE_N_STEPS,
            "fqe_learning_rate": FQE_LR,
        },
        "per_seed": {},
        "paired_bootstrap": {},
    }

    bc_q_per_seed = []
    cql_q_per_seed = []

    for seed in SEEDS:
        print(f"\n[E2] seed {seed}")

        bc_scaled = load_policy_for_eval(BC_DIR / f"bc_seed{seed}.d3", DATASET)
        cql_scaled = load_policy_for_eval(CQL_DIR / f"cql_seed{seed}.d3", DATASET)
        bc_algo = bc_scaled.policy
        cql_algo = cql_scaled.policy

        print(f"[E2]   fitting FQE on BC seed {seed} ...")
        fqe_bc = train_fqe_on_algo(bc_algo, train_ds, seed)

        print(f"[E2]   fitting FQE on CQL seed {seed} ...")
        fqe_cql = train_fqe_on_algo(cql_algo, train_ds, seed)

        bc_q = np.array([q_at_state(fqe_bc, s, bc_algo) for s in eval_states])
        cql_q = np.array([q_at_state(fqe_cql, s, cql_algo) for s in eval_states])

        diff = cql_q - bc_q
        diff_mean = float(diff.mean())
        diff_std = float(diff.std())
        print(f"[E2]   BC mean Q={bc_q.mean():.4f}  CQL mean Q={cql_q.mean():.4f}  "
              f"CQL-BC={diff_mean:+.4f}  std={diff_std:.4f}")

        results["per_seed"][str(seed)] = {
            "bc_mean_q": float(bc_q.mean()),
            "cql_mean_q": float(cql_q.mean()),
            "diff_mean": diff_mean,
            "diff_std": diff_std,
        }
        bc_q_per_seed.append(bc_q)
        cql_q_per_seed.append(cql_q)

    bc_pool = np.concatenate(bc_q_per_seed)
    cql_pool = np.concatenate(cql_q_per_seed)
    diff_pool = cql_pool - bc_pool

    rng_boot = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(diff_pool)
    boot_means = np.array([
        diff_pool[rng_boot.randint(0, n, n)].mean()
        for _ in range(N_BOOTSTRAP)
    ])
    obs = float(diff_pool.mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    results["paired_bootstrap"] = {
        "observed_difference": obs,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "ci_contains_zero": bool(lo <= 0 <= hi),
        "n_observations": int(n),
        "resampling_unit": "state (per-state FQE value, pooled across seeds)",
    }

    print("\n[E2] pooled FQE result")
    print(f"[E2]   CQL-BC = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'SIG' if not (lo <= 0 <= hi) else 'ns'}")
    print(f"[E2]   n = {n}, {N_BOOTSTRAP} bootstrap replicates")

    out = REPO / "revision_experiments" / "e2_fqe_convergence.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[E2] wrote {out}")


if __name__ == "__main__":
    main()