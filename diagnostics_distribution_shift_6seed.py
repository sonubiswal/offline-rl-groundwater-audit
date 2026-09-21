"""
diagnostics_distribution_shift_6seed.py -- Trishna-OPAL Phase 5c, N-seed runner
(built on the validated 3-seed diagnostics_distribution_shift.py; same
rollout mechanics, same episodic-HDF5 loader, same action/state/Q-value
reports. Two additions only:

  1. SEED AUTO-DISCOVERY: if --seeds is omitted, this script scans
     --bc-dir and --cql-dir for files matching bc_seed{N}.d3 /
     cql_seed{N}.d3, takes the INTERSECTION of seed numbers present in
     both directories (so it never tries to load a BC or CQL checkpoint
     that doesn't exist), sorts them, and uses that as the seed list.
     This avoids guessing seed values and failing at load_policy().
     You can still pass --seeds explicitly to override.

  2. CROSS-SEED AGGREGATION: after the per-seed loop, walks every
     numeric leaf in the per-seed results (e.g. tv_distance_cql_vs_behavior,
     mean_bias_q_minus_realized_return, correlation_q_vs_realized_return,
     etc.) and reports mean/std/min/max across seeds for each metric,
     written under results["aggregate"]. This does NOT invent or impute
     values for seeds where a metric was unavailable (e.g. BC's
     q_value_bc.available == false) -- those seeds are simply excluded
     from that metric's aggregate, and the aggregate entry records how
     many seeds contributed (n).

USAGE
-----
# Auto-discover seeds from whatever checkpoints exist:
python diagnostics_distribution_shift_6seed.py `
    --dataset data/processed/offline_rl_dataset_epsilon.h5 `
    --bc-dir models_epsilon/bc `
    --cql-dir models_epsilon/cql `
    --n-eval-episodes 500 `
    --obs-mean 15.075033 36.681023 0.391654 -0.000000000002838 0.000000000002838 0.423686 `
    --obs-std 7.286923 41.271557 0.253303 0.707107 0.707107 0.194976 `
    --out reports/epsilon/distribution_shift_6seed.json

# Or pin the exact 6 seeds explicitly:
python diagnostics_distribution_shift_6seed.py `
    ... (same as above) `
    --seeds 42 123 2024 7 99 555 `
    --out reports/epsilon/distribution_shift_6seed.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

try:
    import h5py
except ImportError:
    h5py = None

try:
    import d3rlpy
except ImportError:
    d3rlpy = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rl"))

try:
    from synthetic_reward import SimState, step
    from generate_offline_dataset import sample_initial_state
    from rollout_eval import make_model_action_fn, EVAL_SEED
except ImportError as e:
    raise ImportError(
        "Could not import synthetic_reward / generate_offline_dataset / "
        "rollout_eval from src/rl. This script must run from a location "
        "where those modules are importable -- adjust the sys.path.insert "
        "above to point at your actual src/rl directory."
    ) from e


RARE_ACTION_THRESHOLD = 0.01
KNN_K = 10


# ---------------------------------------------------------------------------
# Seed discovery
# ---------------------------------------------------------------------------

def discover_seeds(bc_dir: str, cql_dir: str):
    """Find seeds with BOTH a bc_seed{N}.d3 and cql_seed{N}.d3 present."""
    bc_seeds = {
        int(m.group(1))
        for p in Path(bc_dir).glob("bc_seed*.d3")
        if (m := re.match(r"bc_seed(\d+)\.d3$", p.name))
    }
    cql_seeds = {
        int(m.group(1))
        for p in Path(cql_dir).glob("cql_seed*.d3")
        if (m := re.match(r"cql_seed(\d+)\.d3$", p.name))
    }

    both = sorted(bc_seeds & cql_seeds)
    bc_only = sorted(bc_seeds - cql_seeds)
    cql_only = sorted(cql_seeds - bc_seeds)

    if bc_only:
        print(f"WARNING: seeds with a BC checkpoint but no matching CQL checkpoint, skipped: {bc_only}", file=sys.stderr)
    if cql_only:
        print(f"WARNING: seeds with a CQL checkpoint but no matching BC checkpoint, skipped: {cql_only}", file=sys.stderr)

    if not both:
        raise FileNotFoundError(
            f"No seed had BOTH bc_seed{{N}}.d3 in {bc_dir} AND cql_seed{{N}}.d3 "
            f"in {cql_dir}. Pass --seeds explicitly, or check your checkpoint "
            f"directories/filenames."
        )

    print(f"Auto-discovered {len(both)} usable seed(s): {both}")
    return both


# ---------------------------------------------------------------------------
# Loading (unchanged from the validated episodic-HDF5 loader)
# ---------------------------------------------------------------------------

def load_h5_dataset(path: str):
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(p, "r") as f:
        keys = list(f.keys())

        ep_indices = sorted(
            int(m.group(1))
            for k in keys
            if (m := re.match(r"^observations_(\d+)$", k))
        )

        if not ep_indices:
            raise KeyError(
                f"No 'observations_N' keys found in {path}. This loader "
                "expects d3rlpy's per-episode HDF5 export format."
            )

        if "num_episodes" in f:
            declared = int(np.array(f["num_episodes"]))
            if declared != len(ep_indices):
                print(
                    f"WARNING: num_episodes attr says {declared} but found "
                    f"{len(ep_indices)} observations_N keys. Using the "
                    f"{len(ep_indices)} keys actually present.",
                    file=sys.stderr,
                )

        obs_chunks, act_chunks, rew_chunks, term_chunks = [], [], [], []

        for i in ep_indices:
            obs_key = f"observations_{i}"
            act_key = f"actions_{i}"
            rew_key = f"rewards_{i}"
            term_key = f"terminated_{i}"

            for required in (obs_key, act_key, rew_key, term_key):
                if required not in f:
                    raise KeyError(
                        f"Episode {i} is missing expected key {required!r}."
                    )

            obs = np.array(f[obs_key])
            act = np.array(f[act_key])
            rew = np.array(f[rew_key])
            terminated_flag = bool(np.array(f[term_key]))

            ep_len = obs.shape[0]

            if act.shape[0] != ep_len or rew.shape[0] != ep_len:
                raise ValueError(
                    f"Episode {i}: observations has {ep_len} timesteps "
                    f"but actions has {act.shape[0]} and rewards has "
                    f"{rew.shape[0]}."
                )

            term_arr = np.zeros(ep_len, dtype=bool)
            term_arr[-1] = terminated_flag

            obs_chunks.append(obs)
            act_chunks.append(act)
            rew_chunks.append(rew)
            term_chunks.append(term_arr)

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(act_chunks, axis=0)
        rewards = np.concatenate(rew_chunks, axis=0)
        terminals = np.concatenate(term_chunks, axis=0)

        version = f["version"][()] if "version" in f else None

    print(
        f"Loaded {len(ep_indices)} episodes, {observations.shape[0]} "
        f"total timesteps (obs dim={observations.shape[1]}) from {p}"
        + (f" [dataset version: {version}]" if version is not None else "")
    )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
    }


def load_policy(algo_dir: str, algo_name: str, seed: int):
    if d3rlpy is None:
        raise ImportError("d3rlpy is required: pip install d3rlpy")

    ckpt = Path(algo_dir) / f"{algo_name}_seed{seed}.d3"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    return d3rlpy.load_learnable(str(ckpt))


# ---------------------------------------------------------------------------
# 1. Action-support diagnostics
# ---------------------------------------------------------------------------

def action_distribution(actions: np.ndarray, n_actions: int) -> np.ndarray:
    counts = np.bincount(actions.astype(int).ravel(), minlength=n_actions)
    return counts / max(counts.sum(), 1)


def total_variation_distance(p, q) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def kl_divergence(p, q, eps: float = 1e-8) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def rare_action_rate(policy_actions, behavior_dist) -> float:
    rare_mask = behavior_dist < RARE_ACTION_THRESHOLD
    chosen = policy_actions.astype(int).ravel()
    in_range = (chosen >= 0) & (chosen < len(behavior_dist))
    return (
        float(np.mean(rare_mask[chosen[in_range]]))
        if in_range.any() else float("nan")
    )


def action_support_report(behavior_actions, bc_actions, cql_actions, n_actions):
    p_behavior = action_distribution(behavior_actions, n_actions)
    p_bc = action_distribution(bc_actions, n_actions)
    p_cql = action_distribution(cql_actions, n_actions)

    return {
        "behavior_action_distribution": p_behavior.tolist(),
        "bc_action_distribution": p_bc.tolist(),
        "cql_action_distribution": p_cql.tolist(),
        "tv_distance_bc_vs_behavior": total_variation_distance(p_bc, p_behavior),
        "tv_distance_cql_vs_behavior": total_variation_distance(p_cql, p_behavior),
        "kl_bc_from_behavior": kl_divergence(p_bc, p_behavior),
        "kl_cql_from_behavior": kl_divergence(p_cql, p_behavior),
        "rare_action_rate_bc": rare_action_rate(bc_actions, p_behavior),
        "rare_action_rate_cql": rare_action_rate(cql_actions, p_behavior),
    }


# ---------------------------------------------------------------------------
# 2. State-distribution shift diagnostics
# ---------------------------------------------------------------------------

def standardize(obs, mean, std):
    return (obs - mean) / np.where(std == 0, 1.0, std)


def mean_knn_distance(query_states, reference_states, k: int = KNN_K) -> float:
    ref = reference_states

    if ref.shape[0] > 20000:
        idx = np.random.default_rng(0).choice(ref.shape[0], 20000, replace=False)
        ref = ref[idx]

    k_eff = min(k, ref.shape[0])

    if cKDTree is not None:
        tree = cKDTree(ref)
        dists, _ = tree.query(query_states, k=k_eff)
        return float(np.mean(dists))

    dists = []
    for s in query_states:
        d = np.linalg.norm(ref - s, axis=1)
        d.sort()
        dists.append(d[:k_eff].mean())

    return float(np.mean(dists))


def state_shift_report(behavior_states, bc_rollout_states, cql_rollout_states, obs_mean, obs_std):
    beh_z = standardize(behavior_states, obs_mean, obs_std)
    bc_z = standardize(bc_rollout_states, obs_mean, obs_std)
    cql_z = standardize(cql_rollout_states, obs_mean, obs_std)

    return {
        "mean_abs_zscore_behavior": float(np.mean(np.abs(beh_z))),
        "mean_abs_zscore_bc_rollout": float(np.mean(np.abs(bc_z))),
        "mean_abs_zscore_cql_rollout": float(np.mean(np.abs(cql_z))),
        "knn_distance_bc_to_behavior_support": mean_knn_distance(bc_z, beh_z),
        "knn_distance_cql_to_behavior_support": mean_knn_distance(cql_z, beh_z),
        "note": (
            "Higher mean |z-score| or higher k-NN distance-to-behavior-support "
            "for a policy indicates it visits states farther from what the "
            "offline dataset covered -- a direct proxy for distribution shift."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Q-value diagnostics
# ---------------------------------------------------------------------------

def q_value_report(policy, initial_states_raw, chosen_actions, realized_discounted_returns, obs_mean, obs_std):
    if not hasattr(policy, "predict_value"):
        return {"available": False, "reason": "policy has no learned Q-function (e.g. BC)"}

    initial_states_std = standardize(initial_states_raw, obs_mean, obs_std)

    # BC defines predict_value (inherited from the base class) but raises
    # NotImplementedError when called, since it has no learned Q-function.
    # hasattr() alone can't detect this -- only calling it can.
    try:
        q_values = policy.predict_value(initial_states_std, chosen_actions)
    except NotImplementedError as e:
        return {"available": False, "reason": f"policy has no learned Q-function ({e})"}

    q_values = np.asarray(q_values).ravel()
    residual = q_values - realized_discounted_returns

    return {
        "available": True,
        "q_mean": float(np.mean(q_values)),
        "q_std": float(np.std(q_values)),
        "q_min": float(np.min(q_values)),
        "q_max": float(np.max(q_values)),
        "mean_bias_q_minus_realized_return": float(np.mean(residual)),
        "rmse_q_vs_realized_return": float(np.sqrt(np.mean(residual ** 2))),
        "correlation_q_vs_realized_return": (
            float(np.corrcoef(q_values, realized_discounted_returns)[0, 1])
            if len(q_values) > 1 else float("nan")
        ),
        "interpretation": (
            "A large positive mean_bias indicates Q-value overestimation "
            "relative to what the simulator actually delivers. Low or "
            "negative correlation indicates the learned Q-function is not "
            "tracking real policy performance, which would undermine any "
            "conclusion drawn from FQE / offline value estimates alone."
        ),
    }


# ---------------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------------

def rollout_policy_with_diagnostics(
    action_fn, rf_df, n_actions, trajectory_length, n_episodes,
    reward_config, gamma=0.99, seed=EVAL_SEED,
):
    rng = np.random.RandomState(seed)
    action_rng = np.random.RandomState(seed + 1)

    initial_states = []
    first_actions = []
    visited_states = []
    visited_actions = []
    discounted_returns = []

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

            visited_states.append(state.to_array())
            visited_actions.append(action)

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

    return {
        "initial_states": np.array(initial_states),
        "first_actions": np.array(first_actions),
        "visited_states": np.array(visited_states),
        "visited_actions": np.array(visited_actions),
        "discounted_returns": np.array(discounted_returns),
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------

def collect_numeric_leaves(per_seed_results: dict):
    """
    Walk per_seed_results[seed][section][metric] and collect numeric
    leaves keyed by 'section.metric' across seeds. Skips non-numeric
    leaves (lists, strings, bools, the 'available'/'note'/'interpretation'/
    'reason' fields) and skips a seed's value for a metric entirely if
    that seed's section reports available == False, rather than
    inventing a number for it.
    """
    by_metric = defaultdict(list)

    for seed, sections in per_seed_results.items():
        for section_name, section in sections.items():
            if isinstance(section, dict) and section.get("available") is False:
                continue  # e.g. q_value_bc unavailable for this seed

            if not isinstance(section, dict):
                continue

            for metric_name, value in section.items():
                if metric_name in ("note", "interpretation", "reason", "available"):
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    key = f"{section_name}.{metric_name}"
                    by_metric[key].append(float(value))
                # lists (e.g. action distributions) intentionally skipped --
                # aggregating per-bin across seeds is a separate question
                # from these scalar diagnostics and shouldn't be silently
                # flattened here.

    return by_metric


def aggregate_across_seeds(per_seed_results: dict):
    by_metric = collect_numeric_leaves(per_seed_results)

    aggregate = {}
    for key, values in sorted(by_metric.items()):
        arr = np.array(values, dtype=float)
        aggregate[key] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return aggregate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bc-dir", required=True)
    ap.add_argument("--cql-dir", required=True)
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=(
            "Which BC/CQL training-seed checkpoints to diagnose. If "
            "omitted, seeds are AUTO-DISCOVERED from every bc_seed{N}.d3 "
            "/ cql_seed{N}.d3 pair present in --bc-dir/--cql-dir. All "
            "seeds are rolled out under rollout_eval.EVAL_SEED for a "
            "fair, identical-dynamics comparison."
        ),
    )
    ap.add_argument("--rf-seed-table", default="reports/rf_grid_predictions.csv")
    ap.add_argument("--config", default="config/rl_config.yaml")
    ap.add_argument("--n-eval-episodes", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)

    ap.add_argument("--obs-mean", nargs=6, type=float, required=True,
        metavar=("M1", "M2", "M3", "M4", "M5", "M6"))
    ap.add_argument("--obs-std", nargs=6, type=float, required=True,
        metavar=("S1", "S2", "S3", "S4", "S5", "S6"))

    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    obs_mean = np.array(args.obs_mean, dtype=float)
    obs_std = np.array(args.obs_std, dtype=float)

    if len(obs_mean) != len(obs_std):
        print(
            f"ERROR: --obs-mean has {len(obs_mean)} values but --obs-std has "
            f"{len(obs_std)} values; they must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    reward_config = cfg["reward"]
    n_actions = cfg["action_space"]["n_actions"]
    trajectory_length = cfg["simulation"]["trajectory_length"]

    rf_table_path = Path(args.rf_seed_table)
    if not rf_table_path.exists():
        raise FileNotFoundError(f"RF seed table not found: {rf_table_path}")

    rf_df = pd.read_csv(rf_table_path)

    behavior = load_h5_dataset(args.dataset)
    behavior_actions = behavior["actions"]
    behavior_states = behavior["observations"]

    if behavior_states.shape[1] != len(obs_mean):
        print(
            f"ERROR: dataset observations have dimension "
            f"{behavior_states.shape[1]} but --obs-mean/--obs-std have "
            f"dimension {len(obs_mean)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    seeds = args.seeds if args.seeds is not None else discover_seeds(args.bc_dir, args.cql_dir)

    results = {
        "per_seed": {},
        "config": {
            **vars(args),
            "seeds_used": seeds,
            "n_actions": n_actions,
            "trajectory_length": trajectory_length,
            "eval_seed": EVAL_SEED,
        },
    }

    for seed in seeds:
        print(f"--- seed {seed} ---")

        bc = load_policy(args.bc_dir, "bc", seed)
        cql = load_policy(args.cql_dir, "cql", seed)

        bc_action_fn = make_model_action_fn(bc, obs_mean, obs_std)
        cql_action_fn = make_model_action_fn(cql, obs_mean, obs_std)

        bc_roll = rollout_policy_with_diagnostics(
            bc_action_fn, rf_df, n_actions, trajectory_length,
            args.n_eval_episodes, reward_config, gamma=args.gamma, seed=EVAL_SEED,
        )
        cql_roll = rollout_policy_with_diagnostics(
            cql_action_fn, rf_df, n_actions, trajectory_length,
            args.n_eval_episodes, reward_config, gamma=args.gamma, seed=EVAL_SEED,
        )

        action_report = action_support_report(
            behavior_actions, bc_roll["visited_actions"], cql_roll["visited_actions"], n_actions
        )
        state_report = state_shift_report(
            behavior_states, bc_roll["visited_states"], cql_roll["visited_states"], obs_mean, obs_std
        )
        q_report_cql = q_value_report(
            cql, cql_roll["initial_states"], cql_roll["first_actions"],
            cql_roll["discounted_returns"], obs_mean, obs_std,
        )
        q_report_bc = q_value_report(
            bc, bc_roll["initial_states"], bc_roll["first_actions"],
            bc_roll["discounted_returns"], obs_mean, obs_std,
        )

        results["per_seed"][str(seed)] = {
            "action_support": action_report,
            "state_shift": state_report,
            "q_value_cql": q_report_cql,
            "q_value_bc": q_report_bc,
        }

    results["aggregate"] = aggregate_across_seeds(results["per_seed"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_path}")

    print("\n=== Cross-seed summary (mean +/- std, n seeds) ===")
    for key, stats in results["aggregate"].items():
        print(f"{key:55s}  {stats['mean']:+.4f} +/- {stats['std']:.4f}  (n={stats['n']})")


if __name__ == "__main__":
    main()