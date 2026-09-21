"""
src/rl/rollout_eval.py

Phase 5: exact-return policy evaluation via direct simulator rollout.

WHY THIS EXISTS ALONGSIDE FQE (not instead of it): genuine offline RL
cannot interact with the real environment during evaluation -- FQE
exists precisely because of that constraint. This project is not in that
situation: synthetic_reward.step() IS the actual environment the offline
dataset was generated from, fully known and controlled. Using it directly
gives EXACT return estimates with no function-approximation error, which
FQE cannot match. Both are used in this project deliberately:
  - Rollout (this file): the ground-truth number, primary result.
  - FQE (fqe_eval.py, separate script): reported alongside as "what a
    real offline-RL practitioner, without simulator access, would have
    had to rely on" -- and as a validity check on FQE itself (does its
    estimate track the true rollout return?).

This evaluates ANY policy through one uniform interface -- a callable
`action_fn(state: SimState, rng) -> int` -- so BC, CQL, and the three
hand-coded behavior policies (random/greedy_extraction/conservative) are
all scored identically, on identical initial states, with identical
simulator dynamics. Comparing across policy TYPES (learned vs.
hand-coded) this way is only fair because they share the exact same
evaluation procedure.

ANTI-LEAKAGE NOTE: initial states are sampled from the SAME real-data
seed table (reports/rf_grid_predictions.csv) used during training-data
generation (validation_methodology.md Section 26), but with a DISTINCT
random seed (EVAL_SEED, not simulation.random_seed=42) so evaluation
episodes are not simply replays of training episodes. This does not
touch the CGWB held-out set in any way -- it reuses Phase 2's RF
predictions, not held-out well readings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_reward import SimState, load_reward_config, step  # noqa: E402
from generate_offline_dataset import sample_initial_state  # noqa: E402

EVAL_SEED = 9999  # distinct from simulation.random_seed (42) used for training-data generation


def make_model_action_fn(algo, obs_mean: np.ndarray, obs_std: np.ndarray) -> Callable:
    """Wraps a trained d3rlpy model (BC or CQL) into the uniform
    action_fn(state, rng) -> int interface. Applies the SAME
    standardization the model was trained under -- obs_mean/obs_std MUST
    be the exact training-split statistics saved alongside that model,
    not recomputed here, or this silently evaluates the model under a
    different input distribution than it was trained on."""
    def action_fn(state: SimState, rng: np.random.RandomState) -> int:
        obs = state.to_array().reshape(1, -1)
        obs = (obs - obs_mean) / obs_std
        action = algo.predict(obs)
        return int(np.asarray(action).reshape(-1)[0])
    return action_fn


def make_handcoded_action_fn(policy_fn: Callable, n_actions: int) -> Callable:
    """Wraps one of generate_offline_dataset.py's policy_random /
    policy_greedy_extraction / policy_conservative functions into the
    same uniform interface, so hand-coded and learned policies are
    scored through identical rollout code."""
    def action_fn(state: SimState, rng: np.random.RandomState) -> int:
        return policy_fn(state, n_actions, rng)
    return action_fn


def rollout_policy(
    action_fn: Callable,
    rf_df: pd.DataFrame,
    n_actions: int,
    trajectory_length: int,
    n_episodes: int,
    reward_config: dict,
    gamma: float = 0.99,
    seed: int = EVAL_SEED,
) -> dict:
    """Runs n_episodes rollouts of action_fn through the real simulator,
    from real-data-anchored initial states, and returns return statistics.

    Mirrors generate_trajectory()'s exact calendar-advancement logic
    (synthetic_reward.step() does not advance month_sin/month_cos itself
    -- see its own docstring) so evaluation dynamics are identical to
    training-data generation dynamics, not a subtly different variant.
    """
    rng = np.random.RandomState(seed)              # drives sample_initial_state + step() only
    action_rng = np.random.RandomState(seed + 1)   # independent stream for action_fn's own randomness
    undiscounted_returns = []
    discounted_returns = []

    for ep in range(n_episodes):
        state = sample_initial_state(rf_df, {}, rng)
        undiscounted_total = 0.0
        discounted_total = 0.0
        discount = 1.0

        for t in range(trajectory_length):
            action = action_fn(state, action_rng)
            next_state, reward, done = step(
                state, action, config=reward_config, n_actions=n_actions, rng=rng
            )

            cur_month_angle = np.arctan2(state.month_sin, state.month_cos)
            next_month_angle = cur_month_angle + (2 * np.pi / 4)
            next_state.month_sin = float(np.sin(next_month_angle))
            next_state.month_cos = float(np.cos(next_month_angle))

            undiscounted_total += reward
            discounted_total += discount * reward
            discount *= gamma
            state = next_state

        undiscounted_returns.append(undiscounted_total)
        discounted_returns.append(discounted_total)

    undiscounted_returns = np.array(undiscounted_returns)
    discounted_returns = np.array(discounted_returns)

    return {
        "n_episodes": n_episodes,
        "undiscounted_return_mean": float(undiscounted_returns.mean()),
        "undiscounted_return_std": float(undiscounted_returns.std()),
        "discounted_return_mean": float(discounted_returns.mean()),
        "discounted_return_std": float(discounted_returns.std()),
        "per_episode_undiscounted": undiscounted_returns,  # kept for paired bootstrap later
        "per_episode_discounted": discounted_returns,
    }


def load_rf_seed_table(path: str = "reports/rf_grid_predictions.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_config(config_path: str = "config/rl_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Smoke-test entry point: evaluates the three hand-coded policies
    # only (no trained model files needed), so this can run standalone
    # to sanity-check rollout mechanics before any BC/CQL model exists.
    from generate_offline_dataset import policy_random, policy_greedy_extraction, policy_conservative

    cfg = load_config()
    reward_cfg = cfg["reward"]
    n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    rf_df = load_rf_seed_table()

    for name, fn in [
        ("random", policy_random),
        ("greedy_extraction", policy_greedy_extraction),
        ("conservative", policy_conservative),
    ]:
        action_fn = make_handcoded_action_fn(fn, n_actions)
        result = rollout_policy(action_fn, rf_df, n_actions, traj_len,
                                 n_episodes=200, reward_config=reward_cfg)
        print(f"[rollout_eval] {name}: "
              f"undiscounted={result['undiscounted_return_mean']:.3f} "
              f"± {result['undiscounted_return_std']:.3f}, "
              f"discounted={result['discounted_return_mean']:.3f} "
              f"± {result['discounted_return_std']:.3f}")