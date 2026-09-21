"""
src/rl/generate_offline_dataset.py

Phase 4: runs synthetic_reward.py's simulator across multiple behavior
policies (random, greedy, conservative, epsilon_greedy_extraction) to
build a diverse offline RL dataset -- offline RL fails badly on
narrow/biased datasets (per the roadmap's own Risk Register), so
state-action coverage diversity is a requirement, not a nice-to-have.

Initial states are seeded from REAL data where possible, not pure
synthetic invention: gw_level is drawn from actual RF-predicted grid
values (Phase 2) at randomly sampled grid cells/dates, and
recent_rainfall from actual CHIRPS rolling-average values at the same
cell/date. This anchors the synthetic trajectories' STARTING points in
real observed conditions, even though everything downstream of the first
step is governed by the (explicitly documented, assumption-numbered)
water-balance simulator in synthetic_reward.py.

ADDED (Phase 5 coverage ablation): epsilon_greedy_extraction. The
original 3-policy set (random, greedy_extraction, conservative) was
found to have an asymmetry: conservative reads the reward config's own
sustainability_threshold_m directly (see its docstring), making it a
near-oracle policy rather than an independent behavioral baseline. A
"_no_oracle" dataset excluding conservative was generated to test
whether BC/CQL could discover good policy without oracle access -- but
that left only random + greedy_extraction as behavior policies, neither
of which visits near-threshold states while acting well there, so CQL
had no (near-threshold, low-pumping, high-reward) transitions to learn
from. epsilon_greedy_extraction closes that specific gap: it is
greedy_extraction with injected uniform-random exploration noise
(EPSILON_GREEDY_EXTRACTION_EPSILON fraction of the time), giving the
dataset occasional low-pumping actions in states greedy_extraction
would otherwise always pump heavily in -- WITHOUT reading the reward
config or any sustainability threshold. It is exploration noise, not
oracle knowledge.

Usage:
    python src/rl/generate_offline_dataset.py --config config/rl_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "downscaling"))
from synthetic_reward import SimState, load_reward_config, step  # noqa: E402

try:
    from d3rlpy.dataset import MDPDataset
    D3RLPY_AVAILABLE = True
except ImportError:
    D3RLPY_AVAILABLE = False


def policy_random(state: SimState, n_actions: int, rng: np.random.RandomState) -> int:
    return int(rng.randint(0, n_actions))


def policy_greedy_extraction(state: SimState, n_actions: int, rng: np.random.RandomState) -> int:
    target_fraction = state.crop_water_demand
    action = int(round(target_fraction * (n_actions - 1)))
    return int(np.clip(action, 0, n_actions - 1))


def policy_conservative(state: SimState, n_actions: int, rng: np.random.RandomState) -> int:
    cfg = load_reward_config()
    threshold = cfg["sustainability_threshold_m"]
    if state.gw_level >= threshold * 0.8:
        return 0
    target_fraction = min(state.crop_water_demand, 0.5)
    action = int(round(target_fraction * (n_actions - 1)))
    return int(np.clip(action, 0, n_actions - 1))


EPSILON_GREEDY_EXTRACTION_EPSILON = 0.3


def policy_epsilon_greedy_extraction(state: SimState, n_actions: int, rng: np.random.RandomState) -> int:
    """Epsilon-noisy version of greedy_extraction: with probability
    EPSILON_GREEDY_EXTRACTION_EPSILON, takes a uniformly random action
    instead of the greedy one. Does NOT read the reward config or any
    sustainability threshold -- unlike policy_conservative, this has zero
    oracle access."""
    if rng.random_sample() < EPSILON_GREEDY_EXTRACTION_EPSILON:
        return int(rng.randint(0, n_actions))
    return policy_greedy_extraction(state, n_actions, rng)


POLICY_MAP = {
    "random": policy_random,
    "greedy_extraction": policy_greedy_extraction,
    "conservative": policy_conservative,
    "epsilon_greedy_extraction": policy_epsilon_greedy_extraction,
}


def sample_initial_state(
    rf_predictions_df, chirps_stack: dict, rng: np.random.RandomState,
    block_col: str = "spatial_block",
) -> "SimState":
    if block_col not in rf_predictions_df.columns:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
        from spatial_block_cv import assign_spatial_block

        import yaml as _yaml
        with open("config/data_config.yaml") as f:
            bbox = tuple(_yaml.safe_load(f)["region"]["bbox"])
        rf_predictions_df[block_col] = rf_predictions_df.apply(
            lambda r: assign_spatial_block(r["lat"], r["lon"], bbox, n_grid=8), axis=1
        )

    available_blocks = rf_predictions_df[block_col].unique()
    chosen_block = available_blocks[rng.randint(0, len(available_blocks))]
    block_rows = rf_predictions_df[rf_predictions_df[block_col] == chosen_block]
    row = block_rows.sample(1, random_state=rng.randint(0, 2**31 - 1)).iloc[0]

    month = int(row.get("month", rng.randint(1, 13)))
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return SimState(
        gw_level=float(row["gw_level_pred"]),
        recent_rainfall=float(row.get("chirps_roll3", rng.uniform(20, 300))),
        crop_water_demand=float(np.clip(0.5 - 0.3 * month_cos, 0.0, 1.0)),
        month_sin=month_sin,
        month_cos=month_cos,
        extraction_rate=float(rng.uniform(0.1, 0.5)),
    )


def generate_trajectory(
    initial_state: SimState, policy_fn, n_actions: int, traj_len: int,
    reward_config: dict, rng: np.random.RandomState,
) -> list[tuple[np.ndarray, int, float, np.ndarray, bool]]:
    transitions = []
    state = initial_state
    for t in range(traj_len):
        action = policy_fn(state, n_actions, rng)
        next_state, reward, done = step(state, action, config=reward_config, n_actions=n_actions, rng=rng)

        cur_month_angle = np.arctan2(state.month_sin, state.month_cos)
        next_month_angle = cur_month_angle + (2 * np.pi / 4)
        next_state.month_sin = float(np.sin(next_month_angle))
        next_state.month_cos = float(np.cos(next_month_angle))

        is_last = (t == traj_len - 1)
        transitions.append((state.to_array(), action, reward, next_state.to_array(), is_last))
        state = next_state
    return transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/rl_config.yaml")
    ap.add_argument("--rf-predictions-csv", default="reports/rf_downscale_validation.csv")
    ap.add_argument("--out", default="data/processed/offline_rl_dataset.h5")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_actions = cfg["action_space"]["n_actions"]
    sim_cfg = cfg["simulation"]
    reward_cfg = cfg["reward"]
    rng = np.random.RandomState(sim_cfg["random_seed"])

    rf_df = None
    try:
        import pandas as pd
        rf_df = pd.read_csv(args.rf_predictions_csv)
        print(f"Loaded {len(rf_df)} real RF-predicted rows to seed initial states from "
              f"{args.rf_predictions_csv}.")
    except FileNotFoundError:
        print(f"WARNING: {args.rf_predictions_csv} not found -- falling back to fully "
              f"synthetic initial states.")

    all_observations, all_actions, all_rewards, all_terminals = [], [], [], []

    for policy_name in sim_cfg["behavior_policies"]:
        policy_fn = POLICY_MAP[policy_name]
        print(f"Generating {sim_cfg['n_trajectories_per_policy']} trajectories "
              f"under '{policy_name}' policy...")

        for _ in range(sim_cfg["n_trajectories_per_policy"]):
            if rf_df is not None:
                init_state = sample_initial_state(rf_df, {}, rng)
            else:
                month = rng.randint(1, 13)
                init_state = SimState(
                    gw_level=float(rng.uniform(2, 30)),
                    recent_rainfall=float(rng.uniform(20, 300)),
                    crop_water_demand=float(rng.uniform(0.2, 0.8)),
                    month_sin=float(np.sin(2 * np.pi * month / 12)),
                    month_cos=float(np.cos(2 * np.pi * month / 12)),
                    extraction_rate=float(rng.uniform(0.1, 0.5)),
                )

            transitions = generate_trajectory(
                init_state, policy_fn, n_actions, sim_cfg["trajectory_length"], reward_cfg, rng
            )
            for obs, action, reward, next_obs, done in transitions:
                all_observations.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_terminals.append(done)

    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)
    rewards = np.array(all_rewards, dtype=np.float32)

    terminals = np.zeros(len(observations), dtype=np.float32)
    timeouts = np.array(all_terminals, dtype=np.float32)

    n_trajectories = len(sim_cfg["behavior_policies"]) * sim_cfg["n_trajectories_per_policy"]
    print(f"\nGenerated {len(observations)} transitions across {n_trajectories} trajectories "
          f"({len(sim_cfg['behavior_policies'])} policies x "
          f"{sim_cfg['n_trajectories_per_policy']} each).")
    print(f"Action distribution: {np.bincount(actions, minlength=n_actions)}")
    print(f"Reward stats: mean={rewards.mean():.3f}, std={rewards.std():.3f}, "
          f"min={rewards.min():.3f}, max={rewards.max():.3f}")
    print(f"Terminals: all-zero (simulator never truly terminates) -- {int(terminals.sum())} true terminals.")
    print(f"Timeouts: {int(timeouts.sum())} episode-boundary cutoffs "
          f"(should equal n_trajectories = {n_trajectories}).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if D3RLPY_AVAILABLE:
        dataset = MDPDataset(
            observations=observations, actions=actions,
            rewards=rewards, terminals=terminals, timeouts=timeouts,
        )
        dataset.dump(args.out)
        print(f"Saved d3rlpy MDPDataset -> {args.out}")
    else:
        npz_path = str(Path(args.out).with_suffix(".npz"))
        np.savez_compressed(npz_path, observations=observations, actions=actions,
                             rewards=rewards, terminals=terminals, timeouts=timeouts)
        print(f"d3rlpy not installed -- saved raw arrays -> {npz_path} instead.")


if __name__ == "__main__":
    main()
