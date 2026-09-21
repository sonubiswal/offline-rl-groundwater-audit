"""
inspect_bucket_rewards.py -- for the specific (high crop_water_demand,
action 3) bucket already confirmed to have reasonable coverage, look at
the ACTUAL per-step rewards stored in the offline dataset for those
transitions. This distinguishes two remaining hypotheses:

  HYPOTHESIS A (reward/config mismatch): stored per-step rewards for
    this bucket already look bad/negative in the offline data. If so,
    CQL's Q-function SHOULD have learned this is a poor action here --
    if it didn't, something is wrong in training (e.g. Q-function
    underfit, or the standardization/action encoding fed to CQL at
    train time doesn't match what's used at eval time).

  HYPOTHESIS B (tail risk / averaging problem): stored per-step rewards
    for this bucket look FINE on average (similar to other buckets).
    In that case CQL is not "wrong" about the one-step reward -- the
    problem is that full 20-step EPISODE returns occasionally crater
    due to a rare within-episode compounding effect that a single-step
    reward average doesn't capture, and standard (non-risk-sensitive)
    Q-learning is expected-value-based, so it would need very good
    exploration of full trajectories, not just single transitions, to
    learn this -- which is a much harder, more structural problem than
    an alpha tweak.

This script reports per-step reward stats for the bucket vs. the rest
of the dataset. It does NOT run any simulation or retraining -- it only
reads numbers already stored in the offline HDF5 file.

USAGE
-----
python inspect_bucket_rewards.py `
    --dataset data/processed/offline_rl_dataset_epsilon.h5 `
    --demand-quantile 0.75 `
    --target-action 3
"""

import argparse
import re
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


def load_full_dataset_with_rewards(path: str):
    with h5py.File(path, "r") as f:
        ep_indices = sorted(
            int(m.group(1))
            for k in f.keys()
            if (m := re.match(r"^observations_(\d+)$", k))
        )
        obs_chunks, act_chunks, rew_chunks, ep_id_chunks, t_chunks = [], [], [], [], []
        for i in ep_indices:
            obs = np.array(f[f"observations_{i}"])
            obs_chunks.append(obs)
            act_chunks.append(np.array(f[f"actions_{i}"]))
            rew_chunks.append(np.array(f[f"rewards_{i}"]))
            ep_id_chunks.append(np.full(obs.shape[0], i))
            t_chunks.append(np.arange(obs.shape[0]))

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(act_chunks, axis=0).astype(int).ravel()
        rewards = np.concatenate(rew_chunks, axis=0).astype(float).ravel()
        episode_id = np.concatenate(ep_id_chunks, axis=0)
        timestep = np.concatenate(t_chunks, axis=0)

    return observations, actions, rewards, episode_id, timestep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--demand-quantile", type=float, default=0.75)
    ap.add_argument("--target-action", type=int, default=3)
    args = ap.parse_args()

    observations, actions, rewards, episode_id, timestep = load_full_dataset_with_rewards(args.dataset)
    demand = observations[:, CROP_DEMAND_IDX]
    cutoff = np.quantile(demand, args.demand_quantile)

    bucket_mask = (demand >= cutoff) & (actions == args.target_action)
    rest_mask = ~bucket_mask

    bucket_rewards = rewards[bucket_mask]
    rest_rewards = rewards[rest_mask]

    print("=" * 70)
    print(f"Bucket: crop_water_demand >= {cutoff:.4f} (top {100*(1-args.demand_quantile):.0f}%) AND action == {args.target_action}")
    print(f"Bucket size: {len(bucket_rewards)} transitions")
    print("=" * 70)

    print("\n--- Per-step reward stats: BUCKET vs REST OF DATASET ---")
    print(f"{'':20s}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'median':>10}")
    print(f"{'bucket':20s}{bucket_rewards.mean():>10.4f}{bucket_rewards.std():>10.4f}"
          f"{bucket_rewards.min():>10.4f}{bucket_rewards.max():>10.4f}{np.median(bucket_rewards):>10.4f}")
    print(f"{'rest of data':20s}{rest_rewards.mean():>10.4f}{rest_rewards.std():>10.4f}"
          f"{rest_rewards.min():>10.4f}{rest_rewards.max():>10.4f}{np.median(rest_rewards):>10.4f}")

    print("\n--- Reward distribution within the bucket (percentiles) ---")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{pct:>3}: {np.percentile(bucket_rewards, pct):.4f}")

    print("\n--- Where in each episode does this bucket occur? (timestep distribution) ---")
    bucket_timesteps = timestep[bucket_mask]
    print(f"  mean timestep: {bucket_timesteps.mean():.2f}  "
          f"(episode length is 20, so this shows if the bucket clusters "
          f"early/mid/late in episodes)")
    print(f"  timestep histogram (0-19, grouped by 5s):")
    for lo in range(0, 20, 5):
        hi = lo + 5
        frac = np.mean((bucket_timesteps >= lo) & (bucket_timesteps < hi))
        print(f"    steps {lo:>2}-{hi-1:>2}: {frac:.3f}")

    print("\n" + "=" * 70)
    print("READ THIS")
    print("=" * 70)
    mean_gap = bucket_rewards.mean() - rest_rewards.mean()
    print(f"Mean reward gap (bucket - rest): {mean_gap:+.4f}")
    if bucket_rewards.mean() < rest_rewards.mean() - 0.05:
        print(
            "  Bucket's stored per-step rewards are ALREADY noticeably worse "
            "than the rest of the dataset on average. This supports "
            "HYPOTHESIS A: the training signal itself already says this "
            "action-in-this-state is bad, and yet CQL's Q-function rates it "
            "as the single most valuable region it sees. That is NOT a "
            "conservatism/coverage issue -- that's a critic fitting/training "
            "problem (check training loss curves, learning rate, critic "
            "network capacity, or whether the exact same standardization "
            "used here was used at CQL train time). More alpha will not fix "
            "an underfit or miscalibrated critic."
        )
    else:
        print(
            "  Bucket's stored per-step rewards look SIMILAR to the rest of "
            "the dataset on average -- no obvious single-step penalty. This "
            "supports HYPOTHESIS B: the catastrophic outcome only shows up "
            "when this action is taken repeatedly / compounds across a full "
            "20-step episode (e.g. cumulative effect), which single-step "
            "reward-matching during Q-learning would not obviously catch. "
            "This is a harder problem: consider whether the reward should "
            "be reshaped to penalize the compounding effect earlier, or "
            "whether a longer effective planning horizon / better critic "
            "architecture is needed -- raising CQL's alpha is unlikely to "
            "resolve a genuinely compounding, delayed-penalty dynamic."
        )


if __name__ == "__main__":
    main()