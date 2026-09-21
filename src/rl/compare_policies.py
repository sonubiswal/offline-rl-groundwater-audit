"""
src/rl/compare_policies.py

Phase 5 -- Policy Comparison
============================

Comparison of:

    1. Behavior Cloning (BC)      -- seeds 42, 123, 2024
    2. Conservative Q-Learning     -- seeds 42, 123, 2024
    3. Hand-coded policies         -- random, greedy_extraction, conservative

Evaluation
----------
Primary: exact controlled simulator rollout (synthetic_reward.step()).
Statistical comparison: paired bootstrap of CQL - BC episode returns.

Default protocol: 500 evaluation episodes, 5000 bootstrap replicates,
evaluation seed = 9999.

--- CORRECTIONS FROM AN EARLIER DRAFT OF THIS SCRIPT ---
An earlier version of this file had several defects that would have
either crashed immediately or silently produced meaningless numbers.
Documented here rather than silently fixed, per this project's own
practice of recording bugs found and fixed (validation_methodology.md
Sections 9/10/17/27):

1. load_reward_config() takes a FILE PATH (it opens and parses the YAML
   itself) -- the earlier draft passed an already-loaded dict, which
   would crash with TypeError. Fixed: reward_cfg = config["reward"]
   directly (config is already loaded once in main()).
2. SimState's real fields (confirmed directly from synthetic_reward.py)
   are gw_level, recent_rainfall, crop_water_demand, month_sin,
   month_cos, extraction_rate. The earlier draft accessed .groundwater,
   .rainfall, .soil_moisture, .demand -- none of which exist
   (soil_moisture in particular is a GLDAS covariate name from Phase
   1/2, unrelated to the RL state space) -- and dropped extraction_rate
   entirely. Fixed: SimState.from_array()/.to_array() are used
   throughout, so field order and names are never hand-typed twice.
3. step() always returns done=False (confirmed -- this simulator has no
   terminal condition, only fixed-length trajectories). The earlier
   draft's rollout loop was `while True: ... if terminated: break`,
   which can never exit via that condition -- every episode would run
   to the 10,000-step safety guard and crash. Fixed: a bounded
   `for t in range(trajectory_length)` loop, matching
   generate_offline_dataset.py's own convention exactly.
4. step()'s rng defaulted to an unseeded fresh RandomState() every call
   in the earlier draft, breaking the claimed evaluation reproducibility.
   Fixed: a seeded RNG is threaded through every step() call.
5. The earlier draft's hand-coded greedy_extraction/conservative
   policies used normalized 0-1 thresholds against gw_level, which is
   actually a depth in meters (range ~0-46) -- this silently evaluated a
   DIFFERENT policy than the one that generated the offline dataset.
   Fixed: the real policy_random/policy_greedy_extraction/
   policy_conservative functions are imported directly from
   generate_offline_dataset.py and wrapped via SimState.from_array(),
   not reimplemented.
6. Model paths corrected to models/{bc,cql}/{bc,cql}_seed{N}.d3,
   matching where bc_baseline.py and train_cql.py actually save them
   (their --model-dir / model_dir defaults are models/bc and
   models/cql respectively, and both scripts' own docstrings/CLI
   defaults confirm this -- an earlier version of this comment
   claimed a flat data/processed/{bc,cql}_seed{N}.d3 layout, which
   was never true of either training script and was corrected after
   compare_policies.py failed with FileNotFoundError against the
   actual on-disk layout).
7. BUGFIX (this revision) -- RIGID bc/cql SUBFOLDER ASSUMPTION:
   bc_paths/cql_paths were previously ALWAYS derived from a single
   --model-dir as model_dir/"bc"/... and model_dir/"cql"/..., which
   silently assumes both families live under one shared parent with
   exactly those two subfolder names. That assumption breaks the
   moment CQL is retrained into a separate directory for a different
   hyperparameter -- e.g. running train_cql.py with
   --model-dir models_no_oracle_alpha1 (per that script's own
   --model-dir flag, see its BUGFIX 2/model-dir addition) saves
   directly to models_no_oracle_alpha1/cql_seed{N}.d3, NOT
   models_no_oracle_alpha1/cql/cql_seed{N}.d3 -- there is no
   guarantee the two families ever share a parent, especially when
   comparing multiple CQL alpha candidates against the same fixed BC
   baseline. This revision adds independent --bc-dir and --cql-dir
   arguments (each defaulting to model_dir/"bc" and model_dir/"cql"
   respectively, so existing single-model-dir usage is unchanged) so
   the two families can point at arbitrary, independent directories
   without any file copying or restructuring. bc_paths/cql_paths are
   now built from bc_dir/cql_dir directly, and both resolved
   directories are recorded in the JSON report's "models" section so
   it's unambiguous after the fact which BC dir was paired with which
   CQL dir for a given comparison run.
---------------------------------------------------------

Usage:
    python -m src.rl.compare_policies --n-episodes 500 --n-bootstrap 5000

    # BC and CQL saved under a shared parent (original behavior):
    python -m src.rl.compare_policies --model-dir models_no_oracle

    # BC and CQL in independent directories (e.g. comparing a CQL
    # alpha candidate trained into its own folder against a fixed BC
    # baseline elsewhere):
    python -m src.rl.compare_policies \
        --bc-dir models_no_oracle/bc \
        --cql-dir models_no_oracle_alpha10 \
        --output-dir reports/no_oracle_alpha10
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import d3rlpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_reward import SimState, step  # noqa: E402
from generate_offline_dataset import (  # noqa: E402
    sample_initial_state, policy_random, policy_greedy_extraction, policy_conservative,
)
from bc_baseline import split_episode_indices, episodes_to_arrays, SPLIT_SEED  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "rl_config.yaml"
DEFAULT_RF_TABLE = PROJECT_ROOT / "reports" / "rf_grid_predictions.csv"
# Corrected: bc_baseline.py and train_cql.py save into models/bc/ and
# models/cql/ respectively (see change log item 6 above), not a flat
# data/processed/ directory. This is still the DEFAULT parent used to
# derive --bc-dir/--cql-dir when those are not given explicitly (see
# change log item 7) -- not the only supported layout anymore.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "offline_rl_dataset.h5"

EVAL_SEED = 9999
TRAINING_SEEDS = [42, 123, 2024]
POLICY_ORDER = ["random", "greedy_extraction", "conservative"]
HANDCODED_POLICY_FN = {
    "random": policy_random,
    "greedy_extraction": policy_greedy_extraction,
    "conservative": policy_conservative,
}

OBSERVATION_DIM = 6
DEFAULT_N_EPISODES = 500
DEFAULT_N_BOOTSTRAP = 5000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_GAMMA = 0.99


# =============================================================================
# General utilities
# =============================================================================

def resolve_path(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def load_config(config_path) -> Dict[str, Any]:
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found:\n{path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a top-level mapping.")
    return config


def load_rf_seed_table(path=DEFAULT_RF_TABLE) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"RF prediction table not found:\n{path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"RF prediction table is empty:\n{path}")
    return df


# =============================================================================
# Action functions -- ONE uniform interface for learned AND hand-coded
# policies: action_fn(state_vector: np.ndarray) -> int. Hand-coded
# policies are bridged to this interface via SimState.from_array(),
# reusing the REAL policy functions rather than reimplementing them
# against raw array indices with guessed units (see change log item 5).
# =============================================================================

def make_model_action_fn(algo: Any, obs_mean: np.ndarray, obs_std: np.ndarray) -> Callable:
    obs_mean = np.asarray(obs_mean, dtype=np.float32)
    obs_std = np.asarray(obs_std, dtype=np.float32)
    if obs_mean.shape != (OBSERVATION_DIM,) or obs_std.shape != (OBSERVATION_DIM,):
        raise ValueError("obs_mean/obs_std must have shape (OBSERVATION_DIM,).")
    if np.any(obs_std <= 0):
        raise ValueError("Observation standard deviations must be positive.")

    def action_fn(state_vector: np.ndarray) -> int:
        raw = np.asarray(state_vector, dtype=np.float32)
        standardized = ((raw - obs_mean) / obs_std).reshape(1, OBSERVATION_DIM)
        pred = np.asarray(algo.predict(standardized)).reshape(-1)
        action = int(pred[0])
        if not 0 <= action < 5:
            raise RuntimeError(f"Policy produced invalid action {action}.")
        return action

    return action_fn


def make_handcoded_action_fn(policy_name: str, n_actions: int, rng: np.random.RandomState) -> Callable:
    if policy_name not in HANDCODED_POLICY_FN:
        raise ValueError(f"Unknown hand-coded policy '{policy_name}'. Expected one of {POLICY_ORDER}.")
    policy_fn = HANDCODED_POLICY_FN[policy_name]

    def action_fn(state_vector: np.ndarray) -> int:
        state = SimState.from_array(np.asarray(state_vector, dtype=np.float32))
        return policy_fn(state, n_actions, rng)

    return action_fn


# =============================================================================
# Direct simulator rollout
# =============================================================================

def rollout_policy(
    policy_name: str,
    action_fn: Callable[[np.ndarray], int],
    rf_df: pd.DataFrame,
    reward_cfg: Dict[str, Any],
    n_actions: int,
    trajectory_length: int,
    n_episodes: int,
    seed: int,
    gamma: float = DEFAULT_GAMMA,
) -> Dict[str, Any]:
    """Evaluate a policy via exact rollout through the real simulator.

    FIXED-LENGTH episodes (trajectory_length steps), matching
    generate_offline_dataset.py's own convention -- NOT an
    environment-signaled termination loop, since step() always returns
    done=False for this simulator (see change log item 3).

    A single seeded RNG stream feeds BOTH sample_initial_state() and
    step()'s internal noise, and is reused identically (same seed) for
    every policy this function is called for -- this is what makes the
    paired bootstrap comparison valid: all policies see the exact same
    sequence of initial states and the exact same simulator noise
    realizations, differing only in which action each one chooses.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}.")

    rng = np.random.RandomState(seed)

    episode_returns = np.zeros(n_episodes, dtype=np.float64)
    discounted_returns = np.zeros(n_episodes, dtype=np.float64)

    for ep in range(n_episodes):
        state = sample_initial_state(rf_df, {}, rng)
        undiscounted = 0.0
        discounted = 0.0
        discount = 1.0

        for _ in range(trajectory_length):
            action = int(action_fn(state.to_array()))
            if not 0 <= action < n_actions:
                raise RuntimeError(f"Policy '{policy_name}' produced invalid action {action}.")

            next_state, reward, done = step(state, action, config=reward_cfg, n_actions=n_actions, rng=rng)
            reward = float(reward)
            if not math.isfinite(reward):
                raise RuntimeError(f"Simulator returned non-finite reward: {reward}")

            # Calendar advancement -- step() itself leaves month_sin/cos
            # unchanged by design (see synthetic_reward.py docstring);
            # the trajectory generator is responsible for advancing it,
            # exactly as generate_offline_dataset.py's generate_trajectory() does.
            cur_angle = math.atan2(state.month_sin, state.month_cos)
            next_angle = cur_angle + (2 * math.pi / 4)
            next_state.month_sin = math.sin(next_angle)
            next_state.month_cos = math.cos(next_angle)

            undiscounted += reward
            discounted += discount * reward
            discount *= gamma
            state = next_state

        episode_returns[ep] = undiscounted
        discounted_returns[ep] = discounted

    return {
        "policy": policy_name,
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "gamma": float(gamma),
        "episode_returns": episode_returns,
        "discounted_returns": discounted_returns,
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns, ddof=1)) if n_episodes > 1 else 0.0,
        "discounted_return_mean": float(np.mean(discounted_returns)),
        "discounted_return_std": float(np.std(discounted_returns, ddof=1)) if n_episodes > 1 else 0.0,
    }


# =============================================================================
# d3rlpy model loading
# =============================================================================

def load_d3rlpy_model(model_path: Path, label: str) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"{label} model does not exist:\n{model_path}")
    if model_path.stat().st_size == 0:
        raise RuntimeError(f"{label} model file is empty:\n{model_path}")
    print(f"[compare] Loading {label}: {model_path}")
    try:
        return d3rlpy.load_learnable(str(model_path), device="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {label} model:\n  {model_path}\n\n"
            f"d3rlpy version: {getattr(d3rlpy, '__version__', 'unknown')}\n\n"
            f"Original error:\n{exc}"
        ) from exc


# =============================================================================
# Standardization -- reuses bc_baseline.py's exact split/statistics
# functions directly (fixed from the earlier draft's hand-copied
# reimplementation -- see change log item 7 discussion).
# =============================================================================

def load_training_standardization(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer

    if not dataset_path.exists():
        raise FileNotFoundError(f"Offline dataset not found:\n{dataset_path}")

    with dataset_path.open("rb") as f:
        dataset = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(dataset.episodes)
    if not episodes:
        raise ValueError("Offline dataset contains zero episodes.")

    train_idx, _, _ = split_episode_indices(len(episodes), seed=SPLIT_SEED)
    train_episodes = [episodes[i] for i in train_idx]

    train_obs, _, _, _, _ = episodes_to_arrays(train_episodes)
    obs_mean = train_obs.mean(axis=0).astype(np.float32)
    obs_std = train_obs.std(axis=0).astype(np.float32)
    obs_std[obs_std < 1e-6] = 1.0
    return obs_mean, obs_std


# =============================================================================
# Paired bootstrap
# =============================================================================

def paired_bootstrap_difference(
    returns_a: np.ndarray, returns_b: np.ndarray, n_bootstrap: int, seed: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    returns_a = np.asarray(returns_a, dtype=np.float64)
    returns_b = np.asarray(returns_b, dtype=np.float64)
    if returns_a.shape != returns_b.shape:
        raise ValueError("Paired bootstrap requires equal-length return arrays.")
    if len(returns_a) < 2:
        raise ValueError("At least two episodes are required.")
    if not np.all(np.isfinite(returns_a)) or not np.all(np.isfinite(returns_b)):
        raise ValueError("Inputs contain non-finite values.")

    paired_difference = returns_a - returns_b
    observed_difference = float(np.mean(paired_difference))

    rng = np.random.RandomState(seed)
    sampled_indices = rng.randint(0, len(paired_difference), size=(n_bootstrap, len(paired_difference)))
    bootstrap_differences = np.mean(paired_difference[sampled_indices], axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_differences, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(bootstrap_differences, 100.0 * (1.0 - alpha / 2.0)))
    probability_positive = float(np.mean(bootstrap_differences > 0.0))

    return {
        "observed_difference": observed_difference,
        "confidence_level": confidence_level,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "probability_bootstrap_difference_positive": probability_positive,
        "ci_contains_zero": bool(lower <= 0.0 <= upper),
        "bootstrap_differences": bootstrap_differences,
    }


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_stem.with_suffix(".png")), dpi=400, bbox_inches="tight")
    fig.savefig(str(output_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 comparison of BC, CQL, and hand-coded policies.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--rf-table", default=str(DEFAULT_RF_TABLE))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                         help="Parent directory containing bc/ and cql/ subfolders "
                              "with each seed's saved .d3 model, matching "
                              "bc_baseline.py's and train_cql.py's own "
                              "--model-dir / model_dir defaults. Only used to "
                              "derive --bc-dir/--cql-dir when those are not "
                              "given explicitly (see --bc-dir/--cql-dir).")
    parser.add_argument("--bc-dir", default=None,
                         help="Directory containing bc_seed{N}.d3 files directly "
                              "(NOT a parent of a bc/ subfolder -- this IS that "
                              "folder). Overrides --model-dir/'bc'. Use this to "
                              "point at a BC baseline that lives independently "
                              "of wherever CQL models are stored.")
    parser.add_argument("--cql-dir", default=None,
                         help="Directory containing cql_seed{N}.d3 files directly. "
                              "Overrides --model-dir/'cql'. Use this to compare "
                              "different CQL runs (e.g. different alpha values, "
                              "each trained via train_cql.py --model-dir "
                              "<its own folder>) against the same fixed BC "
                              "baseline without moving any files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--exclude-policy", action="append", default=[],
        choices=POLICY_ORDER,
        help="Hand-coded policy to drop from the direct-rollout comparison "
             "and plots (repeatable). Does NOT change what BC/CQL were "
             "trained on -- only what's evaluated/plotted here. Use e.g. "
             "--exclude-policy conservative to compare BC/CQL against only "
             "the non-oracle baselines (random, greedy_extraction), since "
             "policy_conservative reads the reward config's own "
             "sustainability_threshold_m and is not an independent "
             "behavioral baseline -- see module docstring change log."
    )
    args = parser.parse_args()

    if args.n_episodes <= 0 or args.n_bootstrap <= 0 or not 0.0 < args.gamma <= 1.0:
        raise ValueError("Invalid --n-episodes/--n-bootstrap/--gamma.")

    config_path = resolve_path(args.config)
    dataset_path = resolve_path(args.dataset)
    rf_table_path = resolve_path(args.rf_table)
    model_dir = resolve_path(args.model_dir)
    # BUGFIX (change log item 7): bc_dir/cql_dir are now independent of
    # each other. Each defaults to model_dir/"bc" or model_dir/"cql" for
    # backward compatibility with the original single --model-dir usage,
    # but either can be overridden separately so BC and CQL don't have
    # to share a parent directory or subfolder naming convention.
    bc_dir = resolve_path(args.bc_dir) if args.bc_dir is not None else model_dir / "bc"
    cql_dir = resolve_path(args.cql_dir) if args.cql_dir is not None else model_dir / "cql"
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80 + "\nPHASE 5 -- POLICY COMPARISON\n" + "=" * 80)
    print(f"d3rlpy version : {getattr(d3rlpy, '__version__', 'unknown')}")
    print(f"Evaluation seed: {EVAL_SEED}")
    print(f"Episodes       : {args.n_episodes}")
    print(f"Bootstrap reps : {args.n_bootstrap}")
    print(f"Gamma          : {args.gamma}")
    print(f"BC dir         : {bc_dir}")
    print(f"CQL dir        : {cql_dir}")

    config = load_config(config_path)
    reward_cfg = config["reward"]  # FIXED: was load_reward_config(config), see change log item 1
    n_actions = config["action_space"]["n_actions"]
    trajectory_length = config["simulation"]["trajectory_length"]

    print("\n[compare] Loading RF grid predictions...")
    rf_df = load_rf_seed_table(rf_table_path)
    print(f"[compare] RF table rows: {len(rf_df)}")

    print("[compare] Recovering train-only observation statistics...")
    obs_mean, obs_std = load_training_standardization(dataset_path)
    print(f"[compare] Observation mean: {obs_mean}")
    print(f"[compare] Observation std : {obs_std}")

    # Corrected (change log item 6, refined in item 7): bc_seed{N}.d3 and
    # cql_seed{N}.d3 files are read from bc_dir/cql_dir directly -- these
    # are independent directories, not necessarily subfolders of a shared
    # --model-dir parent (see item 7).
    bc_paths = {seed: bc_dir / f"bc_seed{seed}.d3" for seed in TRAINING_SEEDS}
    cql_paths = {seed: cql_dir / f"cql_seed{seed}.d3" for seed in TRAINING_SEEDS}

    bc_models = {seed: load_d3rlpy_model(bc_paths[seed], f"BC seed {seed}") for seed in TRAINING_SEEDS}
    cql_models = {seed: load_d3rlpy_model(cql_paths[seed], f"CQL seed {seed}") for seed in TRAINING_SEEDS}

    action_functions: Dict[str, Callable] = {}
    for seed in TRAINING_SEEDS:
        action_functions[f"BC_seed{seed}"] = make_model_action_fn(bc_models[seed], obs_mean, obs_std)
        action_functions[f"CQL_seed{seed}"] = make_model_action_fn(cql_models[seed], obs_mean, obs_std)

    # Each hand-coded policy gets its own RNG stream (only "random" actually
    # consumes it -- greedy/conservative are deterministic, per the real
    # implementations in generate_offline_dataset.py).
    for policy_name in POLICY_ORDER:
        policy_rng = np.random.RandomState(EVAL_SEED + 1)
        action_functions[policy_name] = make_handcoded_action_fn(policy_name, n_actions, policy_rng)

    evaluation_order = (
        [f"BC_seed{s}" for s in TRAINING_SEEDS]
        + [f"CQL_seed{s}" for s in TRAINING_SEEDS]
        + [p for p in POLICY_ORDER if p not in args.exclude_policy]
    )
    if args.exclude_policy:
        print(f"\n[compare] Excluding hand-coded polic{'y' if len(args.exclude_policy) == 1 else 'ies'} "
              f"from this comparison: {', '.join(args.exclude_policy)}")

    print("\n" + "=" * 80 + "\nDIRECT SIMULATOR EVALUATION\n" + "=" * 80)
    rollout_results: Dict[str, Dict[str, Any]] = {}
    for policy_name in evaluation_order:
        print(f"\n[compare] Evaluating {policy_name}...")
        result = rollout_policy(
            policy_name=policy_name,
            action_fn=action_functions[policy_name],
            rf_df=rf_df,
            reward_cfg=reward_cfg,
            n_actions=n_actions,
            trajectory_length=trajectory_length,
            n_episodes=args.n_episodes,
            seed=EVAL_SEED,
            gamma=args.gamma,
        )
        rollout_results[policy_name] = result
        print(f"    return     = {result['return_mean']:.4f} ± {result['return_std']:.4f}")
        print(f"    discounted = {result['discounted_return_mean']:.4f} ± {result['discounted_return_std']:.4f}")




        # --- Save per-policy per-episode return arrays (VM §51.4 / LM §35.4) ---
    # Enables independent bootstrap recomputation from archived artifacts,
    # which was previously impossible because only summary stats were kept.
    # Filenames are lowercase to match the convention documented in
    # validation_methodology.md §51.4.
    arrays_dir = output_dir / "returns"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for policy_name, result in rollout_results.items():
        tag = policy_name.lower()
        np.save(arrays_dir / f"returns_{tag}.npy",
                np.asarray(result["episode_returns"], dtype=np.float64))
        np.save(arrays_dir / f"discounted_returns_{tag}.npy",
                np.asarray(result["discounted_returns"], dtype=np.float64))
    print(f"\n[compare] Saved {len(rollout_results)} per-policy return arrays -> {arrays_dir}")


    
    # --- Aggregate BC vs CQL across seeds ---
    bc_matrix = np.vstack([rollout_results[f"BC_seed{s}"]["episode_returns"] for s in TRAINING_SEEDS])
    cql_matrix = np.vstack([rollout_results[f"CQL_seed{s}"]["episode_returns"] for s in TRAINING_SEEDS])
    bc_avg = np.mean(bc_matrix, axis=0)
    cql_avg = np.mean(cql_matrix, axis=0)

    bootstrap_result = paired_bootstrap_difference(
        cql_avg, bc_avg, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed,
    )

    print("\n" + "=" * 80 + "\nCQL - BC PAIRED BOOTSTRAP\n" + "=" * 80)
    print(f"Observed difference : {bootstrap_result['observed_difference']:.4f}")
    print(f"95% CI              : [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print(f"CI contains zero    : {bootstrap_result['ci_contains_zero']}")
    print(f"Bootstrap P(diff>0) : {bootstrap_result['probability_bootstrap_difference_positive']:.4f}")

    bc_disc_matrix = np.vstack([rollout_results[f"BC_seed{s}"]["discounted_returns"] for s in TRAINING_SEEDS])
    cql_disc_matrix = np.vstack([rollout_results[f"CQL_seed{s}"]["discounted_returns"] for s in TRAINING_SEEDS])
    discounted_bootstrap = paired_bootstrap_difference(
        np.mean(cql_disc_matrix, axis=0), np.mean(bc_disc_matrix, axis=0),
        n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed + 1,
    )

    per_seed_differences = {}
    for seed in TRAINING_SEEDS:
        bc_r = rollout_results[f"BC_seed{seed}"]["episode_returns"]
        cql_r = rollout_results[f"CQL_seed{seed}"]["episode_returns"]
        per_seed_differences[str(seed)] = {
            "bc_mean": float(np.mean(bc_r)),
            "cql_mean": float(np.mean(cql_r)),
            "cql_minus_bc": float(np.mean(cql_r - bc_r)),
        }

    # --- Plots ---
    labels = [n.replace("_seed", "-") for n in evaluation_order]
    values = [rollout_results[n]["return_mean"] for n in evaluation_order]
    errors = [rollout_results[n]["return_std"] for n in evaluation_order]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean undiscounted episode return")
    ax.set_title("Phase 5 Policy Performance in Controlled Simulator")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "policy_comparison_mean_return")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(bootstrap_result["bootstrap_differences"], bins=50)
    ax.axvline(bootstrap_result["observed_difference"], linestyle="--", linewidth=2, label="Observed CQL - BC")
    ax.axvline(0.0, linestyle=":", linewidth=2, label="No difference")
    ax.axvline(bootstrap_result["ci_lower"], linestyle="--", linewidth=1, label="95% CI")
    ax.axvline(bootstrap_result["ci_upper"], linestyle="--", linewidth=1)
    ax.set_xlabel("Bootstrap mean return difference (CQL - BC)")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title("Paired Bootstrap Distribution: CQL - BC")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "cql_minus_bc_bootstrap")

    # --- CSV summary ---
    rows = []
    for policy_name in evaluation_order:
        r = rollout_results[policy_name]
        if policy_name.startswith("BC_seed"):
            family, seed = "BC", int(policy_name.replace("BC_seed", ""))
        elif policy_name.startswith("CQL_seed"):
            family, seed = "CQL", int(policy_name.replace("CQL_seed", ""))
        else:
            family, seed = "handcoded", None
        rows.append({
            "policy": policy_name, "family": family, "training_seed": seed,
            "evaluation_seed": EVAL_SEED, "n_episodes": args.n_episodes,
            "return_mean": r["return_mean"], "return_std": r["return_std"],
            "discounted_return_mean": r["discounted_return_mean"],
            "discounted_return_std": r["discounted_return_std"],
        })
    summary_csv = output_dir / "policy_comparison_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    # --- JSON report ---
    final_report = {
        "experiment": "Phase 5 learned-policy and hand-coded-policy comparison",
        "protocol": {
            "evaluation_method": "exact direct rollout in controlled simulator",
            "evaluation_seed": EVAL_SEED, "n_evaluation_episodes": args.n_episodes,
            "bootstrap_replicates": args.n_bootstrap, "bootstrap_seed": args.bootstrap_seed,
            "gamma": args.gamma, "training_seeds": TRAINING_SEEDS, "split_seed": SPLIT_SEED,
        },
        "software": {
            "python": platform.python_version(), "platform": platform.platform(),
            "d3rlpy": getattr(d3rlpy, "__version__", "unknown"),
        },
        "models": {
            "bc_dir": str(bc_dir),
            "cql_dir": str(cql_dir),
            "BC": {str(s): str(bc_paths[s]) for s in TRAINING_SEEDS},
            "CQL": {str(s): str(cql_paths[s]) for s in TRAINING_SEEDS},
        },
        "observation_standardization": {"fitted_on": "training episodes only",
                                          "mean": obs_mean.tolist(), "std": obs_std.tolist()},
        "direct_rollout": {n: {k: v for k, v in r.items() if k not in
                                 ("episode_returns", "discounted_returns")}
                            for n, r in rollout_results.items()},
        "per_seed_cql_minus_bc": per_seed_differences,
        "cql_minus_bc_paired_bootstrap": {k: v for k, v in bootstrap_result.items()
                                            if k != "bootstrap_differences"},
        "cql_minus_bc_discounted_bootstrap": {k: v for k, v in discounted_bootstrap.items()
                                                if k != "bootstrap_differences"},
        "scientific_caveat": (
            "Direct simulator rollout provides exact evaluation only for the "
            "controlled simulator used by this experiment. It is not evidence "
            "of real-world causal groundwater-management efficacy."
        ),
        "statistical_caveat": (
            "The bootstrap CI quantifies episode-level uncertainty for the "
            "evaluated policies. Three training seeds should not be treated "
            "as many independent training experiments."
        ),
    }
    json_path = output_dir / "policy_comparison.json"
    write_json(json_path, final_report)

    print("\n" + "=" * 80 + "\nFINAL POLICY COMPARISON\n" + "=" * 80)
    for policy_name in evaluation_order:
        r = rollout_results[policy_name]
        print(f"{policy_name:16s} {r['return_mean']:9.4f} ± {r['return_std']:8.4f}")
    print(f"\nCQL - BC observed difference: {bootstrap_result['observed_difference']:.4f}")
    print(f"CQL - BC 95% CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print(f"\nSummary CSV -> {summary_csv}")
    print(f"JSON report -> {json_path}")


if __name__ == "__main__":
    main()