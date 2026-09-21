"""
dump_q_vs_return.py -- dump raw per-episode (Q_predicted, realized_return)
pairs for CQL, per seed, to CSV. The aggregated correlation number alone
can't distinguish between:
  (a) a GENUINE INVERSION -- Q ranks episodes backwards, or
  (b) a COMPRESSED/NOISY CRITIC -- Q barely varies (e.g. due to a large
      CQL conservatism penalty squashing the range), so whatever small
      variation remains is dominated by noise and can look anti-
      correlated with the real, much-larger-variance return signal
      even without any real inversion in the underlying policy quality.
Looking at the actual per-episode pairs (not just the summary stat)
tells these apart -- e.g. sorted by predicted Q, do the LOWEST-Q
episodes cluster at the HIGH end of realized return (genuine inversion),
or is there no clear pattern at all (just noise/compression)?

This reuses the exact same rollout + standardization + predict_value
call path as diagnostics_distribution_shift_6seed.py -- nothing about
the rollout mechanics changes here, this only ADDS raw-array export.

USAGE
-----
python dump_q_vs_return.py `
    --dataset data/processed/offline_rl_dataset_epsilon.h5 `
    --cql-dir models_epsilon/cql `
    --seeds 42 123 2024 `
    --n-eval-episodes 500 `
    --obs-mean 15.075033 36.681023 0.391654 -0.000000000002838 0.000000000002838 0.423686 `
    --obs-std 7.286923 41.271557 0.253303 0.707107 0.707107 0.194976 `
    --out-dir reports/epsilon/q_vs_return
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import d3rlpy
except ImportError:
    d3rlpy = None

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rl"))

try:
    from synthetic_reward import SimState, step
    from generate_offline_dataset import sample_initial_state
    from rollout_eval import make_model_action_fn, EVAL_SEED
except ImportError as e:
    raise ImportError(
        "Could not import synthetic_reward / generate_offline_dataset / "
        "rollout_eval from src/rl."
    ) from e


def standardize(obs, mean, std):
    return (obs - mean) / np.where(std == 0, 1.0, std)


def rollout_for_q_dump(action_fn, rf_df, n_actions, trajectory_length, n_episodes,
                        reward_config, gamma, seed):
    rng = np.random.RandomState(seed)
    action_rng = np.random.RandomState(seed + 1)

    initial_states, first_actions, discounted_returns = [], [], []

    for ep in range(n_episodes):
        state = sample_initial_state(rf_df, {}, rng)
        initial_states.append(state.to_array())

        discounted_total = 0.0
        discount = 1.0
        first_action_recorded = False

        for t in range(trajectory_length):
            action = action_fn(state, action_rng)
            if not first_action_recorded:
                first_actions.append(action)
                first_action_recorded = True

            next_state, reward, done = step(
                state, action, config=reward_config, n_actions=n_actions, rng=rng
            )

            cur_month_angle = np.arctan2(state.month_sin, state.month_cos)
            next_month_angle = cur_month_angle + (2 * np.pi / 4)
            next_state.month_sin = float(np.sin(next_month_angle))
            next_state.month_cos = float(np.cos(next_month_angle))

            discounted_total += discount * reward
            discount *= gamma
            state = next_state

        discounted_returns.append(discounted_total)

    return (
        np.array(initial_states),
        np.array(first_actions),
        np.array(discounted_returns),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Unused for this check but kept for symmetry with the main script.")
    ap.add_argument("--cql-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--rf-seed-table", default="reports/rf_grid_predictions.csv")
    ap.add_argument("--config", default="config/rl_config.yaml")
    ap.add_argument("--n-eval-episodes", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--obs-mean", nargs=6, type=float, required=True)
    ap.add_argument("--obs-std", nargs=6, type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    obs_mean = np.array(args.obs_mean, dtype=float)
    obs_std = np.array(args.obs_std, dtype=float)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    reward_config = cfg["reward"]
    n_actions = cfg["action_space"]["n_actions"]
    trajectory_length = cfg["simulation"]["trajectory_length"]

    rf_df = pd.read_csv(args.rf_seed_table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        ckpt = Path(args.cql_dir) / f"cql_seed{seed}.d3"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        cql = d3rlpy.load_learnable(str(ckpt))
        action_fn = make_model_action_fn(cql, obs_mean, obs_std)

        initial_states, first_actions, returns = rollout_for_q_dump(
            action_fn, rf_df, n_actions, trajectory_length,
            args.n_eval_episodes, reward_config, args.gamma, EVAL_SEED,
        )

        initial_states_std = standardize(initial_states, obs_mean, obs_std)
        q_values = np.asarray(cql.predict_value(initial_states_std, first_actions)).ravel()

        df = pd.DataFrame({
            "episode": np.arange(len(returns)),
            "q_predicted": q_values,
            "realized_return": returns,
            "first_action": first_actions.ravel() if first_actions.ndim > 1 else first_actions,
        })
        df["rank_by_q"] = df["q_predicted"].rank()
        df["rank_by_return"] = df["realized_return"].rank()
        df = df.sort_values("q_predicted").reset_index(drop=True)

        out_path = out_dir / f"q_vs_return_seed{seed}.csv"
        df.to_csv(out_path, index=False)

        # Bucket episodes into quintiles by predicted Q, report mean
        # realized return per bucket -- this shows the shape of the
        # relationship directly instead of collapsing it to one number.
        df["q_quintile"] = pd.qcut(df["q_predicted"], 5, labels=False, duplicates="drop")
        bucket_means = df.groupby("q_quintile")["realized_return"].agg(["mean", "std", "count"])

        print(f"\n=== seed {seed} ===")
        print(f"Wrote {out_path}  (n={len(df)} episodes)")
        print(f"q_predicted:     min={q_values.min():.3f} max={q_values.max():.3f} std={q_values.std():.3f}")
        print(f"realized_return: min={returns.min():.3f} max={returns.max():.3f} std={returns.std():.3f}")
        print(f"variance ratio (return_std / q_std): {returns.std() / max(q_values.std(), 1e-9):.2f}")
        print("Mean realized return by predicted-Q quintile (quintile 0 = lowest predicted Q):")
        print(bucket_means.to_string())


if __name__ == "__main__":
    main()