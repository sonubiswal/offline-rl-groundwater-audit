"""
run_fqe_check.py -- train Fitted Q-Evaluation (FQE) to independently
estimate the value of CQL's policy on the SAME offline dataset, then
compare FQE's estimate for the (high-demand, action-3) bucket against
CQL's own critic. This is the standard "second opinion" for offline
value estimates:

  - If FQE ALSO rates this bucket highly (agrees with CQL's critic),
    the miscalibration is likely inherent to fitting THIS reward
    distribution with a standard expected-value TD objective on this
    amount of data -- not specific to CQL's training run. Fix path:
    distributional/risk-aware critic, more data in this region, or
    reward reshaping -- not just retraining CQL differently.

  - If FQE correctly rates this bucket LOW (disagrees with CQL),
    the problem is specific to CQL's own critic training (e.g.
    undertrained, bad alpha/hyperparameters, insufficient capacity).
    Fix path: retrain CQL with adjusted hyperparameters/more epochs,
    the data and objective are fine.

FIXES applied in this version (see change notes inline, tagged FIX):
  FIX 1: FQEConfig now uses reward scaling + lower LR + 2 critics.
         The previous all-defaults FQEConfig() diverged (loss went
         0.30 -> 39,206 over 20 epochs) because unscaled bootstrapped
         TD targets over a 20-step, gamma=0.99 horizon with rewards
         ranging to -7.48 blow up without reward scaling.
  FIX 2: verdict logic now uses a scale-aware (standardized) gap
         instead of a fixed +/-0.1 absolute threshold. +/-0.1 was
         calibrated for a per-step reward scale; real cumulative
         value gaps here are naturally tens-to-hundreds in magnitude,
         so the old thresholds silently fell into "ambiguous" for
         almost any real result and could not surface a sign
         mismatch between the FQE gap and the CQL gap.
  FIX 3: reward scaling + lower LR alone (FIX 1) reduced but did NOT
         stop the divergence (loss still went 0.7 -> 19,855 over 20
         epochs on a rerun). The remaining driver is FQE's hard
         target-network updates compounding bootstrap error with no
         gradient clipping and no conservative penalty (unlike CQL,
         plain FQE has nothing fighting extrapolation error on
         under-covered next-state actions). Added gradient clipping
         via AdamFactory(clip_grad_norm=...), a lower learning rate,
         and a much less frequent target_update_interval so the
         target network re-anchors less often to an already-drifting
         online network. Also added an explicit divergence sanity
         check against the theoretical max value implied by the
         dataset's own reward bounds and horizon -- if FQE's output
         is wildly outside that bound, the run is flagged as diverged
         and the interpretive verdict is skipped rather than printed
         as if it were meaningful.

NOTE: this trains a small extra model (FQE), so it takes real compute
time, unlike every other script in this investigation so far. Reduce
--n-epochs for a faster/rougher check.

USAGE
-----
python run_fqe_check.py `
    --dataset data/processed/offline_rl_dataset_epsilon.h5 `
    --cql-dir models_epsilon/cql `
    --seed 42 `
    --n-epochs 20 `
    --demand-quantile 0.75 `
    --target-action 3
"""

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.ope import DiscreteFQE, FQEConfig
    from d3rlpy.preprocessing import StandardRewardScaler  # FIX 1
    from d3rlpy.optimizers import AdamFactory  # FIX 3
except ImportError as e:
    raise ImportError(
        "d3rlpy (with discrete FQE support) is required. This project "
        "uses a DISCRETE action space (5 actions). In this d3rlpy version "
        "there is only one shared FQEConfig, but a separate DiscreteFQE "
        "algorithm class selects discrete-action-space behavior."
    ) from e

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
    """Returns observations, actions, rewards, terminals, timeouts.

    IMPORTANT: this dataset's episodes all run a FIXED 20-step horizon
    (see diagnostics_distribution_shift.py's rollout mechanics) -- most
    or all episodes never hit a true terminated_i == True. d3rlpy's
    MDPDataset requires episode boundaries to be marked via EITHER
    terminals (true MDP termination) OR timeouts (horizon cutoff, NOT
    a real terminal state -- the value function should still bootstrap
    past a timeout, unlike a true terminal). We mark the last timestep
    of every episode as a timeout whenever it wasn't a true termination,
    so d3rlpy can find episode boundaries at all.
    """
    with h5py.File(path, "r") as f:
        ep_indices = sorted(
            int(m.group(1))
            for k in f.keys()
            if (m := re.match(r"^observations_(\d+)$", k))
        )
        observations, actions, rewards, terminals, timeouts = [], [], [], [], []
        for i in ep_indices:
            obs = np.array(f[f"observations_{i}"])
            act = np.array(f[f"actions_{i}"]).ravel()
            rew = np.array(f[f"rewards_{i}"]).ravel()
            terminated_flag = bool(np.array(f[f"terminated_{i}"]))

            term_arr = np.zeros(obs.shape[0], dtype=bool)
            timeout_arr = np.zeros(obs.shape[0], dtype=bool)

            if terminated_flag:
                term_arr[-1] = True
            else:
                timeout_arr[-1] = True  # fixed-horizon cutoff, not a real terminal

            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            terminals.append(term_arr)
            timeouts.append(timeout_arr)

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        timeouts = np.concatenate(timeouts, axis=0)

    n_true_terminations = int(terminals.sum())
    n_timeouts = int(timeouts.sum())
    print(
        f"Loaded {len(ep_indices)} episodes: {n_true_terminations} ended via "
        f"true termination, {n_timeouts} ended via horizon timeout."
    )

    return observations, actions, rewards, terminals, timeouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cql-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--demand-quantile", type=float, default=0.75)
    ap.add_argument("--target-action", type=int, default=3)
    ap.add_argument("--fqe-lr", type=float, default=1e-5,
                     help="FQE learning rate. Lower this further if the "
                          "loss column is still climbing epoch over epoch.")
    ap.add_argument("--fqe-target-update-interval", type=int, default=1000,
                     help="Steps between hard target-network copies. Higher "
                          "= more stable but slower-adapting.")
    args = ap.parse_args()

    observations, actions, rewards, terminals, timeouts = load_full_dataset(args.dataset)

    dataset = MDPDataset(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.int32),
        rewards=rewards.astype(np.float32),
        terminals=terminals,
        timeouts=timeouts,
    )

    cql_ckpt = Path(args.cql_dir) / f"cql_seed{args.seed}.d3"
    if not cql_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cql_ckpt}")
    cql = d3rlpy.load_learnable(str(cql_ckpt))

    print(f"Training FQE to evaluate CQL seed {args.seed}'s policy "
          f"({args.n_epochs} epochs)...")

    # FIX 1: reward scaling + lower LR + 2 critics instead of all-defaults.
    # Defaults (reward_scaler=none, lr=1e-4, n_critics=1) diverged: loss
    # went 0.30 -> 39,206 over 20 epochs on this reward distribution
    # (rewards range to -7.48, 20-step horizon, gamma=0.99 -- unscaled
    # bootstrapped TD targets blow up under hard target updates with a
    # single critic to average against).
    # FIX 3: gradient clipping + lower LR + far less frequent hard target
    # updates. Reward scaling + a moderately lower LR alone (previous
    # version of this script) reduced the divergence rate but did not
    # stop it -- loss still went from 0.7 to 19,855 over 20 epochs. The
    # remaining driver is hard target-network updates (every 100 steps)
    # compounding bootstrap error, unregularized, when FQE evaluates
    # CQL's policy at next-states/actions that are under-covered in the
    # offline data. Gradient clipping caps how far the network can move
    # per update; a longer target_update_interval means the target
    # network re-anchors less often to an already-diverging online net.
    fqe_config = FQEConfig(
        reward_scaler=StandardRewardScaler(),
        learning_rate=args.fqe_lr,
        n_critics=2,
        target_update_interval=args.fqe_target_update_interval,
        optim_factory=AdamFactory(clip_grad_norm=1.0),
    )
    fqe = DiscreteFQE(algo=cql, config=fqe_config, device=None)
    fqe.fit(dataset, n_steps=args.n_epochs * 1000, n_steps_per_epoch=1000)

    # Evaluate FQE's value estimate on the SAME bucket that CQL's own
    # critic overvalued: high crop_water_demand, action == target_action.
    demand = observations[:, CROP_DEMAND_IDX]
    cutoff = np.quantile(demand, args.demand_quantile)

    # FIX 4: metric-definition check, part 1. The original finding was
    # about CQL's ARGMAX action in its top-Q-quintile states (~98.7% of
    # them pick action 3 in the high-demand bucket). Check whether a
    # simple raw-feature-quantile filter matches that population at all.
    high_demand_mask = demand >= cutoff
    high_demand_obs = observations[high_demand_mask].astype(np.float32)
    logged_action_rate = float((actions[high_demand_mask] == args.target_action).mean())
    cql_greedy_actions = np.asarray(cql.predict(high_demand_obs)).ravel()
    cql_greedy_rate = float((cql_greedy_actions == args.target_action).mean())
    print(
        f"\nIn the high-demand region (crop_water_demand >= {cutoff:.4f}, n={high_demand_mask.sum()}):\n"
        f"  fraction of LOGGED (behavior) actions == {args.target_action}: {logged_action_rate:.4f}\n"
        f"  fraction of CQL's GREEDY (argmax) actions == {args.target_action}: {cql_greedy_rate:.4f}\n"
        "  Both near the base rate (~1/5 actions, or the behavior rate) means "
        "raw crop_water_demand quantile alone does NOT reproduce the original "
        "~98.7% figure -- that finding was about CQL's own top-Q-quintile "
        "states, a narrower and differently-defined population. Redefining "
        "the bucket below accordingly."
    )

    # FIX 7: the quintile itself was still built on the wrong value.
    # top_q_action3_rate came back ~0.497 last run -- basically the base
    # rate, not the ~0.987 target -- because states were ranked by
    # predict_value(obs, LOGGED action), i.e. the value of whatever
    # action the behavior policy happened to take. "CQL's top-Q-quintile"
    # should rank states by CQL's OWN greedy action's value (max_a
    # Q(s,a)), since that's what "CQL rates this state highly" means.
    # Compute the greedy action first, then value THAT action.
    cql_greedy_all = np.asarray(cql.predict(observations.astype(np.float32))).ravel()
    cql_greedy_values = np.asarray(cql.predict_value(
        observations.astype(np.float32), cql_greedy_all.astype(np.int32)
    )).ravel()
    q80 = np.quantile(cql_greedy_values, 0.8)
    top_q_quintile_mask = cql_greedy_values >= q80
    top_q_action3_rate = float(
        (cql_greedy_all[top_q_quintile_mask] == args.target_action).mean()
    ) if top_q_quintile_mask.any() else float("nan")
    top_q_high_demand_rate = float(
        (demand[top_q_quintile_mask] >= cutoff).mean()
    ) if top_q_quintile_mask.any() else float("nan")
    # FIX 8: the quintile was still the wrong population. n=6000/30000=20%
    # confirms it's an UNCONDITIONAL top-Q-quintile across the WHOLE
    # dataset -- and only 4.78% of that quintile is even high-demand,
    # meaning it's finding states that are high-value for reasons
    # unrelated to demand (e.g. low-demand states are just easier to get
    # a good outcome in). Re-reading the original finding -- "high
    # crop_water_demand states where CQL picks action 3 (~98.7% of CQL's
    # top-Q-quintile episodes)" -- the quintile was almost certainly
    # computed WITHIN the high-demand subset, not across all states.
    # Compute both definitions and report both, so this doesn't require
    # another manual round trip: use whichever actually reproduces ~0.987.
    high_demand_greedy_actions = cql_greedy_all[high_demand_mask]
    high_demand_greedy_values = cql_greedy_values[high_demand_mask]
    hd_q80 = np.quantile(high_demand_greedy_values, 0.8)
    conditional_top_q_mask_local = high_demand_greedy_values >= hd_q80
    conditional_action3_rate = float(
        (high_demand_greedy_actions[conditional_top_q_mask_local] == args.target_action).mean()
    ) if conditional_top_q_mask_local.any() else float("nan")
    # map the local (within-high-demand) mask back to a full-length mask
    conditional_top_q_mask = np.zeros_like(high_demand_mask)
    conditional_top_q_mask[np.where(high_demand_mask)[0][conditional_top_q_mask_local]] = True

    print(
        f"\nConditional check -- top-Q-quintile WITHIN the high-demand subset "
        f"(n={int(conditional_top_q_mask.sum())} of {int(high_demand_mask.sum())} high-demand states):\n"
        f"  fraction where CQL's GREEDY action == {args.target_action}: {conditional_action3_rate:.4f}\n"
        "  This is the population that best matches the original wording "
        "('high crop_water_demand states where CQL picks action 3 in its "
        "top-Q-quintile'). Compare this number to ~0.987 rather than the "
        "unconditional rate above."
    )

    # Pick whichever definition is actually closest to the reported 0.987,
    # rather than assuming -- and say so explicitly in the bucket label.
    candidates = [
        ("unconditional top-Q-quintile (all states)", top_q_quintile_mask, top_q_action3_rate),
        ("top-Q-quintile within high-demand states", conditional_top_q_mask, conditional_action3_rate),
    ]
    valid_candidates = [c for c in candidates if not np.isnan(c[2])]
    best_label, best_mask, best_rate = min(
        valid_candidates, key=lambda c: abs(c[2] - 0.987)
    )
    print(
        f"\nUsing bucket definition: {best_label} "
        f"(action=={args.target_action} rate {best_rate:.4f}, "
        f"distance from 0.987: {abs(best_rate - 0.987):.4f})"
    )
    if abs(best_rate - 0.987) > 0.1:
        print(
            "  WARNING: even the closer of the two definitions is still far "
            "from 0.987. Neither reproduces the original finding well -- this "
            "likely means the original ~98.7% figure used a different "
            "quintile cutoff, a per-seed average across all 3 seeds (this "
            "script only uses seed 42), or a different value source "
            "entirely. Proceeding with the closer definition below, but "
            "treat the resulting comparison as exploratory, not confirmatory, "
            "until the original analysis script is checked directly."
        )

    top_q_quintile_mask = best_mask
    top_q_action3_rate = best_rate

    # Bucket = CQL's own top-Q-quintile as defined above (per the original
    # finding, this population is already ~98.7% action 3 on its own, so no
    # further action filter is applied here -- filtering to == target_action
    # would double-select and shrink an already-small population).
    bucket_mask = top_q_quintile_mask
    rest_mask = ~bucket_mask
    bucket_description = (
        f"{best_label} (action=={args.target_action} rate: {top_q_action3_rate:.4f})"
    )
    if bucket_mask.sum() == 0:
        print(
            "\n  WARNING: the top-Q-quintile bucket is EMPTY. Falling back to "
            "the raw crop_water_demand-quantile bucket so the script still "
            "produces output, but treat this as a separate finding worth "
            "investigating on its own."
        )
        bucket_mask = (demand >= cutoff) & (actions == args.target_action)
        rest_mask = ~bucket_mask
        bucket_description = (
            f"FALLBACK: crop_water_demand >= {cutoff:.4f}, action == {args.target_action}"
        )

    bucket_obs = observations[bucket_mask].astype(np.float32)
    bucket_actions = actions[bucket_mask].astype(np.int32)
    rest_obs = observations[rest_mask].astype(np.float32)
    rest_actions = actions[rest_mask].astype(np.int32)

    fqe_bucket_values = np.asarray(fqe.predict_value(bucket_obs, bucket_actions)).ravel()
    fqe_rest_values = np.asarray(fqe.predict_value(rest_obs, rest_actions)).ravel()

    cql_bucket_values = np.asarray(cql.predict_value(bucket_obs, bucket_actions)).ravel()
    cql_rest_values = np.asarray(cql.predict_value(rest_obs, rest_actions)).ravel()

    # FIX 3: divergence sanity check. With a fixed 20-step horizon and
    # gamma=0.99 (must match FQEConfig's gamma above), no true value can
    # exceed max|reward| * sum_{t=0..19} gamma^t, regardless of policy.
    # If FQE's output blows past this by a wide margin, the run diverged
    # numerically and the values are meaningless -- report that plainly
    # instead of feeding them into the interpretive verdict below.
    GAMMA = 0.99  # keep in sync with fqe_config gamma (uses d3rlpy default)
    HORIZON = 20
    max_abs_reward = float(np.max(np.abs(rewards)))
    theoretical_bound = max_abs_reward * sum(GAMMA ** t for t in range(HORIZON))
    SAFETY_MULTIPLIER = 3.0  # generous slack for FQE/CQL approximation error
    diverged = (
        np.abs(fqe_bucket_values).max() > theoretical_bound * SAFETY_MULTIPLIER
        or np.abs(fqe_rest_values).max() > theoretical_bound * SAFETY_MULTIPLIER
    )

    fqe_gap = fqe_bucket_values.mean() - fqe_rest_values.mean()
    cql_gap = cql_bucket_values.mean() - cql_rest_values.mean()

    # FIX 2: scale-aware (standardized) gaps instead of fixed +/-0.1.
    # +/-0.1 was calibrated for a per-step reward scale; real cumulative
    # value gaps here are naturally tens-to-hundreds in magnitude, so the
    # old thresholds fell into "ambiguous" almost regardless of outcome
    # and could not flag a sign mismatch between fqe_gap and cql_gap.
    fqe_pooled_std = np.std(np.concatenate([fqe_bucket_values, fqe_rest_values])) + 1e-8
    cql_pooled_std = np.std(np.concatenate([cql_bucket_values, cql_rest_values])) + 1e-8
    fqe_gap_z = fqe_gap / fqe_pooled_std
    cql_gap_z = cql_gap / cql_pooled_std
    Z_THRESH = 0.1  # effect-size threshold in pooled-std units, not raw value units

    print("\n" + "=" * 70)
    print(f"Bucket: {bucket_description}, n={bucket_mask.sum()}")
    print("=" * 70)
    print(f"{'':30s}{'bucket mean':>15}{'rest mean':>15}{'gap':>12}{'gap (z)':>12}")
    print(f"{'FQE predicted value':30s}{fqe_bucket_values.mean():>15.4f}{fqe_rest_values.mean():>15.4f}"
          f"{fqe_gap:>12.4f}{fqe_gap_z:>12.4f}")
    print(f"{'CQL predicted value':30s}{cql_bucket_values.mean():>15.4f}{cql_rest_values.mean():>15.4f}"
          f"{cql_gap:>12.4f}{cql_gap_z:>12.4f}")

    print("\n" + "=" * 70)
    print("READ THIS")
    print("=" * 70)

    if diverged:
        print(
            f"  DIVERGED -- FQE's predicted values exceed the theoretical max "
            f"possible value (~{theoretical_bound:.1f}, from max|reward|="
            f"{max_abs_reward:.2f} over a {HORIZON}-step horizon at gamma="
            f"{GAMMA}, x{SAFETY_MULTIPLIER} safety margin) by a wide margin. "
            "These numbers are numerical blowup, not a real value estimate -- "
            "do NOT use them to draw any conclusion about CQL. Check that the "
            "loss column above is flat/decreasing, not climbing, before "
            "rerunning. If it's still climbing, lower --learning-rate further "
            "or increase target_update_interval again."
        )
        return

    # Flag a sign mismatch explicitly -- this is informative on its own
    # even before applying the effect-size thresholds below.
    if np.sign(fqe_gap_z) != np.sign(cql_gap_z) and abs(cql_gap_z) > Z_THRESH and abs(fqe_gap_z) > Z_THRESH:
        print(
            f"  NOTE: fqe_gap_z ({fqe_gap_z:.4f}) and cql_gap_z ({cql_gap_z:.4f}) "
            "have OPPOSITE signs. Double-check that this bucket-mean comparison "
            "matches how the original top-Q-quintile finding was computed (e.g. "
            "argmax-over-actions per state vs. mean value of the behavior-taken "
            "action across states are different quantities and can disagree in "
            "sign even when both are individually correct)."
        )

    if fqe_gap_z < -Z_THRESH and cql_gap_z > Z_THRESH:
        print(
            "  FQE rates the bucket LOWER than the rest (agreeing with the "
            "actual reward data), while CQL's OWN critic rates it HIGHER. "
            "This means the miscalibration is SPECIFIC TO CQL'S TRAINING "
            "RUN, not inherent to the data or objective. Fix path: retrain "
            "CQL -- check convergence (more epochs), try different alpha, "
            "verify critic network capacity/architecture. The data and "
            "reward function are fine; this checkpoint's critic is not."
        )
    elif fqe_gap_z > Z_THRESH:
        print(
            "  FQE ALSO rates the bucket higher than the rest, agreeing "
            "with CQL's (wrong) assessment. This means even an independent "
            "estimator trained on the same data reaches the same wrong "
            "conclusion -- the miscalibration is likely INHERENT to fitting "
            "this reward distribution with a standard expected-value TD "
            "objective on this data, not a CQL-specific training bug. Fix "
            "path: a distributional/risk-aware critic (e.g. quantile "
            "regression), more data specifically covering the negative "
            "tail of this bucket, or reward reshaping -- simply retraining "
            "CQL with different hyperparameters is unlikely to resolve this "
            "on its own."
        )
    elif fqe_gap_z < -Z_THRESH and cql_gap_z < -Z_THRESH:
        # FIX 6: this branch was missing entirely. Both estimators agreeing
        # the bucket is LOWER value than the rest is a real, informative
        # outcome -- not ambiguous -- but the original either/or framing
        # (does CQL overvalue this bucket, and if so is that CQL-specific
        # or inherent to the data) doesn't actually cover this outcome.
        # It means: under THIS bucket definition, CQL does not overvalue
        # the bucket at all -- both critics agree it's worse than average.
        print(
            "  Both FQE and CQL's own critic rate this bucket LOWER than "
            "the rest (they AGREE, in the negative direction). This does "
            "NOT match the original hypothesis being tested (CQL overvaluing "
            "this bucket) -- under this bucket's definition, neither critic "
            "shows the overvaluation pattern. Two likely explanations: (1) "
            "the bucket_mask above still isn't matching the original "
            "top-Q-quintile population that showed the ~98.7% action-3 "
            "concentration -- check the printed top-Q-quintile action-3 rate "
            "against ~0.987 before trusting this result, or (2) this "
            "particular checkpoint/seed genuinely doesn't reproduce the "
            "original overvaluation finding. Either way, this result does "
            "not confirm the CQL-critic-training-bug hypothesis as written."
        )
    else:
        print(
            "  Results are close/ambiguous under the standardized-gap "
            "threshold -- rerun with more --n-epochs (after confirming the "
            "FQE loss curve has actually flattened, not just changed) for a "
            "more reliable fit before concluding either way."
        )


if __name__ == "__main__":
    main()
