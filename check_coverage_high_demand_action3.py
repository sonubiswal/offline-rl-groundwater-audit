"""
check_coverage_high_demand_action3.py -- checks whether the offline
dataset actually has enough (state, action) coverage in the specific
corner implicated by characterize_top_quintile.py: HIGH
crop_water_demand states paired with ACTION 3.

Two things this reports, separately (do not conflate them):
  1. MARGINAL coverage: how often does high-demand occur at all in the
     data, and how often is action 3 taken at all. (You already know
     rare_action_rate_cql == 0.0 overall -- this is NOT that. This is
     about the JOINT combination.)
  2. CONDITIONAL/JOINT coverage: among only the HIGH-demand transitions,
     what fraction took action 3? If this is a small number of
     transitions, or if action 3 is rarely paired with high demand in
     the data even though action 3 itself is common overall, that is
     exactly the kind of thin-coverage corner that lets a Q-function
     extrapolate confidently and wrongly, which CQL's penalty is
     supposed to guard against but scales with OVERALL action rarity,
     not per-region rarity, unless conditioned explicitly.

This script does NOT decide anything about the reward function itself
(that requires reading config/rl_config.yaml / synthetic_reward.py by
hand) -- it only answers the data-coverage question directly from the
offline dataset.

USAGE
-----
python check_coverage_high_demand_action3.py `
    --dataset data/processed/offline_rl_dataset_epsilon.h5 `
    --demand-quantile 0.75
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")

STATE_FIELDS = [
    "gw_level", "recent_rainfall", "crop_water_demand",
    "month_sin", "month_cos", "extraction_rate",
]
CROP_DEMAND_IDX = STATE_FIELDS.index("crop_water_demand")


def load_full_dataset(path: str):
    with h5py.File(path, "r") as f:
        ep_indices = sorted(
            int(m.group(1))
            for k in f.keys()
            if (m := re.match(r"^observations_(\d+)$", k))
        )
        obs_chunks, act_chunks = [], []
        for i in ep_indices:
            obs_chunks.append(np.array(f[f"observations_{i}"]))
            act_chunks.append(np.array(f[f"actions_{i}"]))

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(act_chunks, axis=0).astype(int).ravel()

    return observations, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--demand-quantile", type=float, default=0.75,
        help=(
            "crop_water_demand values at or above this quantile of the "
            "OFFLINE DATA's own distribution are treated as 'high demand'. "
            "Default 0.75 (top quartile). The catastrophic quintile in "
            "the rollout had mean crop_water_demand ~0.74, so 0.75 is a "
            "reasonable starting cutoff -- adjust and rerun if you want "
            "to probe a narrower/wider band."
        ),
    )
    ap.add_argument(
        "--target-action", type=int, default=3,
        help="The action implicated by characterize_top_quintile.py.",
    )
    args = ap.parse_args()

    observations, actions = load_full_dataset(args.dataset)
    demand = observations[:, CROP_DEMAND_IDX]

    cutoff = np.quantile(demand, args.demand_quantile)
    high_demand_mask = demand >= cutoff

    n_total = len(demand)
    n_high_demand = int(high_demand_mask.sum())

    print("=" * 70)
    print(f"Total offline transitions: {n_total}")
    print(f"crop_water_demand cutoff for 'high demand' (>= {args.demand_quantile:.2f} quantile): {cutoff:.4f}")
    print(f"High-demand transitions: {n_high_demand} ({100 * n_high_demand / n_total:.2f}% of all data)")
    print("=" * 70)

    print("\n--- 1. MARGINAL action distribution (all data, for reference) ---")
    unique_actions = sorted(np.unique(actions))
    for a in unique_actions:
        frac = np.mean(actions == a)
        print(f"  action {a}: {frac:.4f} of ALL transitions ({int((actions == a).sum())} rows)")

    print("\n--- 2. CONDITIONAL action distribution, GIVEN high demand ---")
    if n_high_demand == 0:
        print("  No high-demand transitions found at this cutoff -- try a lower --demand-quantile.")
    else:
        actions_in_high_demand = actions[high_demand_mask]
        for a in unique_actions:
            count = int((actions_in_high_demand == a).sum())
            frac = count / n_high_demand
            print(f"  action {a}: {frac:.4f} of high-demand transitions ({count} rows)")

    print(f"\n--- 3. JOINT coverage: (high demand AND action {args.target_action}) ---")
    joint_mask = high_demand_mask & (actions == args.target_action)
    n_joint = int(joint_mask.sum())
    print(f"  Transitions with high demand AND action {args.target_action}: {n_joint} "
          f"({100 * n_joint / n_total:.3f}% of all {n_total} transitions)")

    print("\n" + "=" * 70)
    print("READ THIS")
    print("=" * 70)
    if n_joint < 30:
        print(
            f"  n={n_joint} is a THIN sample (rule of thumb: <30 is thin for "
            f"a function approximator to learn a reliable value estimate in "
            f"a specific region). This supports the hypothesis that CQL's "
            f"critic extrapolated into this corner with too much confidence "
            f"because it barely saw this exact combination during training. "
            f"A higher CQL conservatism penalty (alpha), or deliberately "
            f"collecting/oversampling more (high-demand, action {args.target_action}) "
            f"transitions, are both reasonable next steps -- alpha is faster "
            f"to test, more data is the more durable fix."
        )
    else:
        print(
            f"  n={n_joint} is not obviously thin. If CQL is still "
            f"overvaluing this region despite reasonable coverage, the "
            f"issue is more likely in HOW those transitions' rewards/next-"
            f"states are structured (e.g. check whether reward computation "
            f"in synthetic_reward.step() for this action in high-demand "
            f"states matches what generated the offline dataset) rather "
            f"than a pure coverage problem. Increasing alpha may help less "
            f"here than expected -- inspect the reward function directly "
            f"before spending a training run on it."
        )


if __name__ == "__main__":
    main()