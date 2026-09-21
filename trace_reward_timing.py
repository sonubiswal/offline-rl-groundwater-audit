"""
trace_reward_timing.py -- diagnose WHERE in the episode the negative
reward for quintile-4-like states (high crop_water_demand, action 3)
actually lands, to decide between the two remaining fixes:

  - If the penalty is concentrated near the terminal step: TD
    bootstrapping at gamma=0.99 has to propagate it backward through
    ~20 steps, and with only 10 epochs / 20000 gradient steps and a
    rare state (103/1050 train episodes hit this corner), the value
    function may simply not have converged there yet. Fix: add
    potential-based reward shaping on min_gw_level so the signal
    arrives earlier in the trajectory instead of only at the end.

  - If the penalty is already spread roughly evenly across steps:
    bootstrapping isn't the bottleneck, and the more likely fix is
    data density -- try --oversample-factor 10 or 20 so the critic
    simply sees this corner of state space enough times to fit it.

This does NOT touch the trained CQL checkpoint at all. It rolls out
trajectories starting from quintile-4-like initial states (high
crop_water_demand) under a fixed policy (always action 3, matching
what quintile 4's episodes actually did ~100% of the time) using the
same environment dynamics as generate_offline_dataset.py, and reports
the per-step reward profile directly from the simulator -- this is
ground truth, no model involved.

USAGE
-----
python trace_reward_timing.py `
    --rf-seed-table reports/rf_grid_predictions.csv `
    --config config/rl_config.yaml `
    --n-episodes 500 `
    --demand-cutoff 0.5871 `
    --out-dir reports/epsilon_verify_qr32/reward_timing
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rl"))

try:
    from synthetic_reward import SimState, step
    from generate_offline_dataset import sample_initial_state
except ImportError as e:
    raise ImportError(
        "Could not import synthetic_reward / generate_offline_dataset from src/rl."
    ) from e


def rollout_fixed_action(rf_df, n_actions, trajectory_length, n_episodes,
                          reward_config, action_to_take, demand_cutoff, seed):
    """
    Roll out episodes starting from high-demand initial states (the
    quintile-4 profile), always taking `action_to_take`, and record
    the per-timestep reward for each episode. Filters to episodes
    whose INITIAL crop_water_demand exceeds demand_cutoff, matching
    the oversampling script's own definition of the dangerous corner
    (see train_cql.py's oversampling log: demand_cutoff=0.5871).
    """
    rng = np.random.RandomState(seed)

    per_episode_rewards = []  # list of (trajectory_length,) arrays
    n_matched = 0
    n_tried = 0

    # Oversample initial-state draws until we have n_episodes that
    # actually match the high-demand profile, since most draws won't.
    while n_matched < n_episodes and n_tried < n_episodes * 50:
        n_tried += 1
        state = sample_initial_state(rf_df, {}, rng)
        if state.crop_water_demand <= demand_cutoff:
            continue

        n_matched += 1
        rewards = np.zeros(trajectory_length)
        for t in range(trajectory_length):
            next_state, reward, done = step(
                state, action_to_take, config=reward_config, n_actions=n_actions, rng=rng
            )
            cur_month_angle = np.arctan2(state.month_sin, state.month_cos)
            next_month_angle = cur_month_angle + (2 * np.pi / 4)
            next_state.month_sin = float(np.sin(next_month_angle))
            next_state.month_cos = float(np.cos(next_month_angle))
            rewards[t] = reward
            state = next_state

        per_episode_rewards.append(rewards)

    if n_matched < n_episodes:
        print(f"WARNING: only found {n_matched}/{n_episodes} episodes matching "
              f"demand_cutoff={demand_cutoff} after {n_tried} draws -- this "
              "cutoff may be rarer than expected, or the RF seed table's "
              "demand distribution has shifted.")

    return np.array(per_episode_rewards)  # (n_matched, trajectory_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-seed-table", default="reports/rf_grid_predictions.csv")
    ap.add_argument("--config", default="config/rl_config.yaml")
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--demand-cutoff", type=float, default=0.5871,
                     help="Matches the demand_cutoff logged by train_cql.py's "
                          "oversampling step -- the threshold defining the "
                          "dangerous high-demand corner.")
    ap.add_argument("--action", type=int, default=3,
                     help="Action to take every step -- 3 matches quintile 4's "
                          "~100%% first-action rate in the original diagnostic.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    reward_config = cfg["reward"]
    n_actions = cfg["action_space"]["n_actions"]
    trajectory_length = cfg["simulation"]["trajectory_length"]

    rf_df = pd.read_csv(args.rf_seed_table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards = rollout_fixed_action(
        rf_df, n_actions, trajectory_length, args.n_episodes,
        reward_config, args.action, args.demand_cutoff, args.seed,
    )

    n_episodes_found = rewards.shape[0]
    per_step_mean = rewards.mean(axis=0)
    per_step_total_negative = np.where(rewards < 0, rewards, 0).sum(axis=0)
    total_negative = per_step_total_negative.sum()
    cumulative_negative_frac = (
        np.cumsum(per_step_total_negative) / total_negative if total_negative != 0 else
        np.zeros(trajectory_length)
    )

    df = pd.DataFrame({
        "timestep": np.arange(trajectory_length),
        "mean_reward": per_step_mean,
        "total_negative_reward": per_step_total_negative,
        "cumulative_negative_reward_frac": cumulative_negative_frac,
    })
    out_path = out_dir / "reward_timing_profile.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({n_episodes_found} episodes matching demand_cutoff={args.demand_cutoff}, "
          f"action={args.action})\n")

    print("=== Per-timestep mean reward ===")
    print(df[["timestep", "mean_reward"]].to_string(index=False))

    # Find the timestep by which 50% and 90% of total negative reward
    # has accrued -- this is the actual diagnostic signal.
    def timestep_at_frac(frac):
        idx = np.searchsorted(cumulative_negative_frac, frac)
        return int(idx) if idx < trajectory_length else trajectory_length - 1

    t50 = timestep_at_frac(0.5)
    t90 = timestep_at_frac(0.9)

    print(f"\n50% of total negative reward accrued by timestep {t50} / {trajectory_length - 1}")
    print(f"90% of total negative reward accrued by timestep {t90} / {trajectory_length - 1}")

    midpoint = trajectory_length / 2
    print("\n=== Interpretation ===")
    if t90 >= trajectory_length - 3:
        print(f"-> Negative reward is BACKLOADED: 90% of it lands in the final few "
              f"steps (t={t90} of {trajectory_length - 1}). With gamma=0.99 this still "
              "discounts only mildly, but TD bootstrapping still needs enough training "
              "steps to propagate a late, rare signal backward to the initial-state "
              "value. Given this corner is only ~10% of train episodes (103/1050), "
              "convergence there over 10 epochs / 20000 steps is plausible to be "
              "incomplete. This favors REWARD RESHAPING (potential-based shaping on "
              "min_gw_level) so the signal shows up earlier in the trajectory, on top "
              "of or instead of raising --oversample-factor.")
    elif t50 <= midpoint:
        print(f"-> Negative reward is roughly SPREAD ACROSS the episode: 50% of it "
              f"accrues by timestep {t50} of {trajectory_length - 1}, close to or "
              "before the midpoint. Bootstrapping delay is less likely to be the "
              "core issue here -- the more likely bottleneck is simply that the "
              "critic hasn't seen enough of these states. This favors trying a "
              "higher --oversample-factor (10 or 20) before reaching for reward "
              "shaping.")
    else:
        print(f"-> Mixed profile: 50% of negative reward by t={t50}, 90% by t={t90}. "
              "Not cleanly backloaded or spread -- worth inspecting the full "
              "per-timestep table above directly rather than relying on this "
              "summary, and consider trying oversampling first since it's the "
              "cheaper change to test.")


if __name__ == "__main__":
    main()