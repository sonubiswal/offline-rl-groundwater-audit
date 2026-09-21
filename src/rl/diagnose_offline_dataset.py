"""
src/rl/diagnose_offline_dataset.py

Phase 4 diagnostic: inspects the generated offline RL dataset BEFORE any
CQL/BC training begins. Produces a transition-level breakdown by action
and by behavior policy, and flags obvious red flags automatically.

This answers a narrower question than "is this scientifically valid" --
it only checks internal consistency (does higher pumping mechanically
produce more drawdown, is reward actually action-sensitive, are
depletion events occurring at a plausible rate). It does NOT validate
synthetic_reward.py's six numbered assumptions (A1-A6, see
limitations.md Section 11) -- those are open by design and no amount of
this script passing clean resolves them.

SEPARATE PRECONDITION, NOT CHECKED BY THIS SCRIPT: whatever produced
reports/rf_grid_predictions.csv (used to seed the 61,913 initial states)
must never have read held_out_ids.csv or held-out wells' true readings
directly. Trace that script's provenance before trusting anything below
-- a leak there would invalidate the whole dataset regardless of what
this diagnostic shows.

CAVEAT ON THE PER-POLICY BREAKDOWN: policy labels are inferred from
EPISODE ORDER ONLY (see POLICY_ORDER / N_TRAJ_PER_POLICY below),
assuming generate_offline_dataset.py saved episodes in the same
sequential order the terminal log printed them (random, then
greedy_extraction, then conservative, 500 each). This has NOT been
verified against that script's actual save logic. If episodes are
interleaved or shuffled before saving, this breakdown is silently wrong
even though the total counts would still look correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Must match src/rl/synthetic_reward.py's SimState dataclass field order exactly.
STATE_FIELDS = [
    "gw_level", "recent_rainfall", "crop_water_demand",
    "month_sin", "month_cos", "extraction_rate",
]
GW_LEVEL_IDX = STATE_FIELDS.index("gw_level")

N_ACTIONS = 5
ACTION_MEANINGS = {
    0: "0% pumping (no extraction)",
    1: "25% pumping (light)",
    2: "50% pumping (moderate)",
    3: "75% pumping (heavy)",
    4: "100% pumping (max)",
}

N_TRAJ_PER_POLICY = 500
POLICY_ORDER = ["random", "greedy_extraction", "conservative"]  # per printed generation log order -- UNVERIFIED, see module docstring


def load_dataset(path: str) -> dict:
    """
    Loads the saved offline RL dataset into flat numpy arrays. Tries
    d3rlpy's API first (handling both the v2.x ReplayBuffer format and
    the older v1.x MDPDataset format), falling back to a raw .npz with
    the same file stem if d3rlpy isn't importable or the load fails.

    Returns a dict with:
      observations: (N, 6) float array, columns = STATE_FIELDS order
      actions:      (N,)   int array, 0..4
      rewards:      (N,)   float array
      terminals:    (N,)   bool array, True at the last transition of each episode
      episode_id:   (N,)   int array, which episode (0..1499) each row belongs to
    """
    path = Path(path)

    if path.suffix in (".h5", ".hdf5"):
        try:
            import d3rlpy
            print(f"Loading via d3rlpy ({d3rlpy.__version__}) from {path} ...")
            episodes = None
            try:
                # d3rlpy >= 2.x
                from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
                with open(path, "rb") as f:
                    buffer = ReplayBuffer.load(f, InfiniteBuffer())
                episodes = buffer.episodes
            except Exception as e_v2:
                print(f"  v2-style load failed ({e_v2}), trying v1 MDPDataset ...")
                from d3rlpy.dataset import MDPDataset
                dataset = MDPDataset.load(str(path))
                episodes = dataset.episodes

            obs_list, act_list, rew_list, term_list, ep_id_list = [], [], [], [], []
            for ep_idx, ep in enumerate(episodes):
                observations = np.asarray(ep.observations, dtype=np.float32)
                actions = np.asarray(ep.actions).reshape(-1)
                rewards = np.asarray(ep.rewards, dtype=np.float64).reshape(-1)
                n = len(rewards)
                terminals = np.zeros(n, dtype=bool)
                terminals[-1] = bool(getattr(ep, "terminated", True))
                obs_list.append(observations[:n])
                act_list.append(actions[:n])
                rew_list.append(rewards)
                term_list.append(terminals)
                ep_id_list.append(np.full(n, ep_idx, dtype=int))

            return {
                "observations": np.concatenate(obs_list, axis=0),
                "actions": np.concatenate(act_list, axis=0),
                "rewards": np.concatenate(rew_list, axis=0),
                "terminals": np.concatenate(term_list, axis=0),
                "episode_id": np.concatenate(ep_id_list, axis=0),
            }
        except Exception as e:
            print(f"d3rlpy load failed ({e}); trying npz fallback at the same stem ...", file=sys.stderr)
            path = path.with_suffix(".npz")

    if path.suffix == ".npz":
        print(f"Loading raw npz from {path} ...")
        data = np.load(path)
        required = {"observations", "actions", "rewards", "terminals"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"npz is missing required arrays: {missing}. Found: {data.files}")

        observations = data["observations"]
        actions = data["actions"].reshape(-1).astype(int)
        rewards = data["rewards"].reshape(-1).astype(float)
        terminals = data["terminals"].reshape(-1).astype(bool)

        if "episode_id" in data.files:
            episode_id = data["episode_id"].astype(int)
        else:
            # Reconstruct from terminal flags: a new episode starts right
            # after each terminal=True row.
            episode_id = np.zeros(len(terminals), dtype=int)
            episode_id[1:] = np.cumsum(terminals[:-1])

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "episode_id": episode_id,
        }

    raise ValueError(f"Unrecognized dataset format: {path}")


def infer_policy_labels(episode_id: np.ndarray) -> np.ndarray:
    """Maps each transition's episode_id to a policy name by sequential
    order. See module docstring CAVEAT -- this is unverified against the
    generator's actual save order."""
    n_episodes_seen = int(episode_id.max()) + 1
    expected = N_TRAJ_PER_POLICY * len(POLICY_ORDER)
    if n_episodes_seen != expected:
        print(
            f"WARNING: found {n_episodes_seen} episodes, expected {expected} "
            f"({N_TRAJ_PER_POLICY} x {len(POLICY_ORDER)}). Policy-order "
            f"inference below is likely WRONG -- do not trust the per-policy "
            f"breakdown until this is reconciled.",
            file=sys.stderr,
        )

    policy_per_episode = np.empty(n_episodes_seen, dtype=object)
    for i, policy in enumerate(POLICY_ORDER):
        lo = i * N_TRAJ_PER_POLICY
        hi = min(lo + N_TRAJ_PER_POLICY, n_episodes_seen)
        policy_per_episode[lo:hi] = policy

    return policy_per_episode[episode_id]


def compute_depletion_events(next_gw_level: np.ndarray, threshold_m: float) -> np.ndarray:
    """Matches synthetic_reward.py's depletion_penalty() > 0 condition exactly:
    strictly deeper than the sustainability threshold."""
    return next_gw_level > threshold_m


def build_transition_table(data: dict, threshold_m: float) -> pd.DataFrame:
    obs = data["observations"]
    actions = data["actions"].astype(int)
    rewards = data["rewards"].astype(float)
    terminals = data["terminals"].astype(bool)
    episode_id = data["episode_id"].astype(int)

    gw_before = obs[:, GW_LEVEL_IDX].astype(float)

    # "After" state = the next row's gw_level, but ONLY when that next row
    # is (a) not across an episode boundary and (b) genuinely the next
    # step of the same episode. Terminal transitions get NaN for gw_after
    # since there is no valid next state within this array.
    gw_after = np.full_like(gw_before, np.nan)
    shifted_gw = np.roll(gw_before, -1)
    shifted_ep = np.roll(episode_id, -1)
    valid = (~terminals) & (shifted_ep == episode_id)
    gw_after[valid] = shifted_gw[valid]

    gw_change = gw_after - gw_before
    depletion_event = np.where(
        np.isnan(gw_after), False, compute_depletion_events(gw_after, threshold_m)
    )

    policy = infer_policy_labels(episode_id)

    return pd.DataFrame({
        "episode_id": episode_id,
        "policy": policy,
        "action": actions,
        "reward": rewards,
        "gw_before": gw_before,
        "gw_after": gw_after,
        "gw_change": gw_change,
        "depletion_event": depletion_event,
        "terminal": terminals,
    })


def _agg(group: pd.DataFrame) -> pd.Series:
    valid = group.dropna(subset=["gw_after"])
    return pd.Series({
        "n_transitions": len(group),
        "mean_reward": group["reward"].mean(),
        "median_reward": group["reward"].median(),
        "mean_gw_before": group["gw_before"].mean(),
        "mean_gw_after": valid["gw_after"].mean() if len(valid) else np.nan,
        "mean_gw_change": valid["gw_change"].mean() if len(valid) else np.nan,
        "depletion_event_count": int(group["depletion_event"].sum()),
        "depletion_event_rate": group["depletion_event"].mean(),
        "terminal_event_count": int(group["terminal"].sum()),
    })


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = df.groupby("action").apply(_agg).reset_index()
    overall["action_meaning"] = overall["action"].map(ACTION_MEANINGS)
    overall = overall[[
        "action", "action_meaning", "n_transitions", "mean_reward", "median_reward",
        "mean_gw_before", "mean_gw_after", "mean_gw_change",
        "depletion_event_count", "depletion_event_rate", "terminal_event_count",
    ]]

    by_policy = df.groupby(["policy", "action"]).apply(_agg).reset_index()
    by_policy["action_meaning"] = by_policy["action"].map(ACTION_MEANINGS)
    by_policy = by_policy[[
        "policy", "action", "action_meaning", "n_transitions", "mean_reward", "median_reward",
        "mean_gw_before", "mean_gw_after", "mean_gw_change",
        "depletion_event_count", "depletion_event_rate", "terminal_event_count",
    ]]

    return overall, by_policy


def print_flags(overall: pd.DataFrame) -> None:
    print("\n=== Sanity flags ===")
    flags = []

    if overall["mean_reward"].nunique() == 1:
        flags.append(
            "All actions have IDENTICAL mean reward -- reward may not be "
            "action-sensitive. Do not train CQL until this is explained."
        )

    if (overall["depletion_event_rate"] == 0).all():
        flags.append(
            "ZERO depletion events across ALL actions -- the depletion "
            "penalty may never trigger. Check sustainability_threshold_m "
            "against the observed gw_level range (see mean_gw_after above)."
        )
    elif (overall["depletion_event_rate"] == 1).all():
        flags.append(
            "100% depletion events across ALL actions -- threshold_m may "
            "be set too low relative to typical gw_level; the penalty term "
            "may be dominating every transition regardless of action."
        )

    sorted_by_action = overall.sort_values("action")
    if not sorted_by_action["mean_gw_change"].is_monotonic_increasing:
        flags.append(
            "mean_gw_change is NOT monotonically increasing with action "
            "index. Higher pumping should mechanically produce equal-or-"
            "larger drawdown (see synthetic_reward.py's linear A5 model) -- "
            "a non-monotonic result here means either noise is swamping the "
            "signal at this sample size, or something is wrong upstream. "
            "Investigate before trusting the environment."
        )

    if not flags:
        flags.append(
            "No obvious internal-consistency red flags found. This does "
            "NOT establish physical realism or scientific validity -- see "
            "limitations.md Section 11 (A1-A6). It only means action, "
            "reward, and state transitions are behaving self-consistently."
        )

    for f in flags:
        print(f"- {f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/processed/offline_rl_dataset.h5",
                         help="Path to the saved offline RL dataset (.h5 or .npz)")
    parser.add_argument("--config", default="config/rl_config.yaml",
                         help="Path to rl_config.yaml, for sustainability_threshold_m")
    parser.add_argument("--out-prefix", default="reports/phase4_diagnostics",
                         help="Prefix for output CSVs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    threshold_m = cfg["reward"]["sustainability_threshold_m"]
    print(f"Using sustainability_threshold_m = {threshold_m} from {args.config}")

    data = load_dataset(args.dataset)
    n = len(data["rewards"])
    print(f"Loaded {n} transitions across {int(data['episode_id'].max()) + 1} episodes.")

    df = build_transition_table(data, threshold_m)
    overall, by_policy = summarize(df)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("\n=== Overall, by action (all policies pooled) ===")
    print(overall.to_string(index=False))

    print("\n=== By policy and action ===")
    print(by_policy.to_string(index=False))

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    overall_path = f"{args.out_prefix}_by_action.csv"
    by_policy_path = f"{args.out_prefix}_by_policy_action.csv"
    overall.to_csv(overall_path, index=False)
    by_policy.to_csv(by_policy_path, index=False)
    print(f"\nSaved: {overall_path}")
    print(f"Saved: {by_policy_path}")

    print_flags(overall)


if __name__ == "__main__":
    main()