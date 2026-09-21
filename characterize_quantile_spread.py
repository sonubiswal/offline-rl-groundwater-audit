"""
characterize_quantile_spread.py -- extends characterize_top_quintile.py
with a per-quantile spread diagnostic for QR-based CQL checkpoints.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import d3rlpy
    import torch
except ImportError:
    d3rlpy = None
    torch = None

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
    min_gw_level_seen = []

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


def try_extract_quantiles(cql, states_std, actions):
    if torch is None:
        return None, "torch not available"

    impl = getattr(cql, "impl", None)
    if impl is None:
        return None, "cql.impl not found on loaded learnable"

    x = torch.as_tensor(states_std, dtype=torch.float32)
    a = torch.as_tensor(actions, dtype=torch.long)

    last_err = None

    def gather_taken_action(out_3d):
        idx = a.view(-1, 1, 1).expand(-1, 1, out_3d.shape[-1])
        return out_3d.gather(1, idx).squeeze(1)

    def unpack_output(out):
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out
            raise ValueError(f"Tensor output has shape {tuple(out.shape)}, not 3D.")

        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    return item
            shapes = [tuple(i.shape) for i in out if isinstance(i, torch.Tensor)]
            raise ValueError(f"No 3D tensor found among tuple/list items. Shapes seen: {shapes}")

        found = {}
        for attr_name in ("quantiles", "q_value", "values", "logits", "taus"):
            val = getattr(out, attr_name, None)
            if isinstance(val, torch.Tensor):
                found[attr_name] = tuple(val.shape)

        for attr_name, shape in found.items():
            if len(shape) == 3:
                return getattr(out, attr_name)

        raise ValueError(
            f"No 3D (batch, n_actions, n_quantiles) tensor found among "
            f"candidate attributes. Shapes seen: {found}"
        )

    for qfunc_attr in ("q_function", "_q_func", "q_func"):
        qfunc_container = getattr(impl, qfunc_attr, None)
        if qfunc_container is None:
            continue

        if isinstance(qfunc_container, (torch.nn.ModuleList, list, tuple)):
            submodules = list(qfunc_container)
        else:
            submodules = [qfunc_container]

        per_critic_quantiles = []
        try:
            with torch.no_grad():
                for sub in submodules:
                    out = sub(x)
                    out_3d = unpack_output(out)
                    per_critic_quantiles.append(gather_taken_action(out_3d))
            if per_critic_quantiles:
                stacked = torch.stack(per_critic_quantiles, dim=0).mean(dim=0)
                return stacked.cpu().numpy(), None
        except Exception:
            last_err = traceback.format_exc()
            continue

    return None, last_err or "no known q-function accessor found on cql.impl"


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
    quantile_extraction_ok = None

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

        assert len(initial_states) == args.n_eval_episodes, (
            f"Expected {args.n_eval_episodes} rows, got {len(initial_states)} "
            "-- rollout produced an unexpected shape before we even get to "
            "predict_value(), investigate rollout_with_state_features first."
        )

        initial_states_std = standardize(initial_states, obs_mean, obs_std)

        q_values = np.asarray(cql.predict_value(initial_states_std, first_actions)).ravel()
        if len(q_values) != args.n_eval_episodes:
            raise ValueError(
                f"predict_value() returned {len(q_values)} values for "
                f"{args.n_eval_episodes} states -- this is exactly the "
                "shape-mismatch risk flagged for QR checkpoints. Do not "
                "trust downstream quintiles from this run; inspect "
                "cql.predict_value's raw (pre-ravel) shape before proceeding."
            )

        df = pd.DataFrame(initial_states, columns=STATE_FIELDS)
        df["seed"] = seed
        df["first_action"] = first_actions.ravel() if first_actions.ndim > 1 else first_actions
        df["q_predicted"] = q_values
        df["realized_return"] = returns
        df["min_gw_level_in_episode"] = min_gw
        df["q_quintile"] = pd.qcut(df["q_predicted"], 5, labels=False, duplicates="drop")

        quantiles, err = try_extract_quantiles(cql, initial_states_std, first_actions)
        if quantiles is not None:
            quantile_extraction_ok = True
            df["q_quantile_std"] = quantiles.std(axis=1)
            df["q_quantile_min"] = quantiles.min(axis=1)
            df["q_quantile_max"] = quantiles.max(axis=1)
        else:
            if quantile_extraction_ok is None:
                print(f"[seed {seed}] Per-quantile extraction failed, continuing "
                      f"with point-estimate q_predicted only. Reason:\n{err}\n")
            quantile_extraction_ok = False
            df["q_quantile_std"] = np.nan
            df["q_quantile_min"] = np.nan
            df["q_quantile_max"] = np.nan

        all_seeds_df.append(df)

    combined = pd.concat(all_seeds_df, ignore_index=True)
    out_path = out_dir / "quintile_characterization_with_spread.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows across {len(args.seeds)} seeds)\n")

    print("=== Per-quintile means (pooled across all seeds) ===")
    cols_to_show = STATE_FIELDS + [
        "first_action", "q_predicted", "realized_return",
        "min_gw_level_in_episode", "q_quantile_std",
    ]
    summary = combined.groupby("q_quintile")[cols_to_show].mean()
    print(summary.to_string())

    print("\n=== First-action distribution within each quintile ===")
    action_dist = combined.groupby("q_quintile")["first_action"].value_counts(normalize=True).unstack(fill_value=0)
    print(action_dist.to_string())

    print("\n=== Fraction of episodes ending with min_gw_level below a low threshold, by quintile ===")
    low_gw_cutoff = combined["min_gw_level_in_episode"].quantile(0.10)
    combined["hit_low_gw"] = combined["min_gw_level_in_episode"] <= low_gw_cutoff
    print(f"(cutoff = 10th percentile of min_gw_level_in_episode = {low_gw_cutoff:.3f})")
    print(combined.groupby("q_quintile")["hit_low_gw"].mean().to_string())

    if quantile_extraction_ok:
        print("\n=== Interpreting q_quantile_std for the top quintile ===")
        top_std = combined.loc[combined["q_quintile"] == combined["q_quintile"].max(), "q_quantile_std"]
        other_std = combined.loc[combined["q_quintile"] != combined["q_quintile"].max(), "q_quantile_std"]
        print(f"Top quintile mean quantile-std: {top_std.mean():.4f}")
        print(f"Other quintiles mean quantile-std: {other_std.mean():.4f}")
        if top_std.mean() > other_std.mean() * 1.5:
            print("-> Top quintile shows distinctly WIDER quantile spread than the "
                  "rest: consistent with QR correctly flagging this as a "
                  "high-uncertainty region. Whether the fix works depends on "
                  "whether action selection actually uses the lower quantiles "
                  "to penalize it, not just whether the spread exists.")
        else:
            print("-> Top quintile spread is NOT distinctly wider than other "
                  "quintiles: if q_predicted improved here anyway, that "
                  "improvement may not be coming from QR's uncertainty "
                  "mechanism specifically -- worth re-checking under a "
                  "different alpha/oversample-factor/seed before trusting it.")
    else:
        print("\n[Per-quantile spread diagnostic skipped -- extraction did not "
              "work for this d3rlpy version. q_predicted/quintile results above "
              "are still valid; only the spread interpretation is missing.]")


if __name__ == "__main__":
    main()