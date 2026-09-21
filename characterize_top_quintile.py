"""
characterize_top_quintile.py -- find out WHAT distinguishes the
catastrophic top-Q-quintile episodes (high predicted Q, terrible
realized return) from the rest. Reruns the same CQL rollout as
dump_q_vs_return.py, but additionally keeps each episode's raw initial
state features (gw_level, recent_rainfall, crop_water_demand,
month_sin, month_cos, extraction_rate) and first action, then reports
per-quintile means for every feature so you can see what's different
about the dangerous top bucket.

USAGE
-----
python characterize_top_quintile.py `
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

STATE_FIELDS = [
    "gw_level", "recent_rainfall", "crop_water_demand",
    "month_sin", "month_cos", "extraction_rate",
]


def standardize(obs, mean, std):
    return (obs - mean) / np.where(std == 0, 1.0, std)


def rollout_with_state_features(action_fn, rf_df, n_actions, trajectory_length,
                                 n_episodes, reward_config, gamma, seed):
    rng = np.random.RandomState(seed)
    action_rng = np.random.RandomState(seed + 1)

    initial_states, first_actions, discounted_returns = [], [], []
    min_gw_level_seen = []  # track worst groundwater level reached in-episode

    for ep in range(n_episodes):
        state = sample_initial_state(rf_df, {}, rng)
        initial_states.append(state.to_array())

        discounted_total = 0.0
        discount = 1.0
        first_action_recorded = False
        ep_min_gw = state.gw_level

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
            ep_min_gw = min(ep_min_gw, state.gw_level)

        discounted_returns.append(discounted_total)
        min_gw_level_seen.append(ep_min_gw)

    return (
        np.array(initial_states),
        np.array(first_actions),
        np.array(discounted_returns),
        np.array(min_gw_level_seen),
    )


def main():
    ap = argparse.ArgumentParser()
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

    all_seeds_df = []

    for seed in args.seeds:
        ckpt = Path(args.cql_dir) / f"cql_seed{seed}.d3"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        cql = d3rlpy.load_learnable(str(ckpt))
        action_fn = make_model_action_fn(cql, obs_mean, obs_std)

        initial_states, first_actions, returns, min_gw = rollout_with_state_features(
            action_fn, rf_df, n_actions, trajectory_length,
            args.n_eval_episodes, reward_config, args.gamma, EVAL_SEED,
        )

        initial_states_std = standardize(initial_states, obs_mean, obs_std)
        q_values = np.asarray(cql.predict_value(initial_states_std, first_actions)).ravel()

        df = pd.DataFrame(initial_states, columns=STATE_FIELDS)
        df["seed"] = seed
        df["first_action"] = first_actions.ravel() if first_actions.ndim > 1 else first_actions
        df["q_predicted"] = q_values
        df["realized_return"] = returns
        df["min_gw_level_in_episode"] = min_gw
        df["q_quintile"] = pd.qcut(df["q_predicted"], 5, labels=False, duplicates="drop")

        all_seeds_df.append(df)

    combined = pd.concat(all_seeds_df, ignore_index=True)
    out_path = out_dir / "quintile_characterization_all_seeds.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows across {len(args.seeds)} seeds)\n")

    print("=== Per-quintile means (pooled across all seeds) ===")
    cols_to_show = STATE_FIELDS + ["first_action", "q_predicted", "realized_return", "min_gw_level_in_episode"]
    summary = combined.groupby("q_quintile")[cols_to_show].mean()
    print(summary.to_string())

    print("\n=== First-action distribution within each quintile ===")
    action_dist = combined.groupby("q_quintile")["first_action"].value_counts(normalize=True).unstack(fill_value=0)
    print(action_dist.to_string())

    print("\n=== Fraction of episodes ending with min_gw_level below a low threshold, by quintile ===")
    # Uses the 10th percentile of ALL episodes' min_gw_level as a generic
    # "danger zone" cutoff -- not a domain-specific threshold, just a
    # relative marker to see if catastrophic episodes cluster there.
    low_gw_cutoff = combined["min_gw_level_in_episode"].quantile(0.10)
    combined["hit_low_gw"] = combined["min_gw_level_in_episode"] <= low_gw_cutoff
    print(f"(cutoff = 10th percentile of min_gw_level_in_episode = {low_gw_cutoff:.3f})")
    print(combined.groupby("q_quintile")["hit_low_gw"].mean().to_string())


if __name__ == "__main__":
    main()