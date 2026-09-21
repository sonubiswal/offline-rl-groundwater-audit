"""
src/rl/bc_baseline.py

Phase 5: Behavior Cloning baseline, trained on the Phase 4 offline
dataset, for comparison against CQL (train_cql.py).

VERIFIED API (not assumed) against d3rlpy 2.8.1, this session:
  - Actions are int64, shape (T, 1); d3rlpy auto-detected
    ActionSpace.DISCRETE, action_size=5 on load.
  - The correct class for discrete actions is DiscreteBCConfig, NOT
    plain BCConfig (which trains a continuous/regression policy and
    would silently be wrong for categorical pumping-level actions).
  - DiscreteBCConfig(batch_size, gamma, learning_rate, beta, ...).create()
    -> DiscreteBC. beta (default 0.5) is DiscreteBC's own regularization
    weight, distinct from CQL's conservative_weight -- do not confuse
    the two when reading rl_config.yaml.
  - .fit(dataset, n_steps, n_steps_per_epoch=10000, evaluators=None, ...)
    controls training duration via n_steps/n_steps_per_epoch, not a
    separate n_epochs argument.
  - Loading the saved dataset requires the ReplayBuffer.load() classmethod
    (NOT MDPDataset.load(), which shares the inherited method but has an
    incompatible __init__ signature and raises TypeError) -- confirmed by
    hitting that exact error and correcting it, not assumed.
  - ds.episodes is a plain list of Episode objects, each exposing
    .observations (T,6), .actions (T,1), .rewards (T,1), .terminated
    (bool, confirmed False for every episode -- see INVESTIGATION NOTE
    below for why that is CORRECT, not a bug).
  - d3rlpy.seed(seed) is the library's documented reproducibility entry
    point: it seeds Python's `random`, NumPy's global RNG, and PyTorch
    (CPU + CUDA) in one call. This is called at the top of
    run_one_seed(), before the algo is constructed.

NOT independently verified in this session (flagged, not assumed away):
  - algo.predict(observations) is assumed to follow d3rlpy's standard
    AlgoBase interface (returns predicted discrete action per row). This
    matches every other d3rlpy algo class's documented interface but was
    not spot-checked against DiscreteBC specifically before this script
    was written. If action-accuracy numbers come back nonsensical (e.g.
    constant predictions when the loss is clearly decreasing), check this
    first.

BUGFIX 1: the previous version of this file accepted a `seed` argument
into run_one_seed() and used it ONLY as a print/logging label -- it was
never passed to d3rlpy.seed(), np.random.seed(), torch.manual_seed(), or
anything else that actually controls RNG state. As a result, the three
"TRAINING_SEEDS = [42, 123, 2024]" runs were NOT actually independently
seeded. Fixed by calling d3rlpy.seed(seed) at the start of
run_one_seed(), before the algo is constructed.

BUGFIX 2: the previous version of run_one_seed() never called
algo.save() at all -- no BC model was ever written to disk.
compare_policies.py expects to load models/bc/bc_seed{seed}.d3 via
DiscreteBC.load(), so that path was silently broken. Fixed by saving
each seed's model to models/bc/bc_seed{seed}.d3, plus a
bc_seed{seed}.meta.json sidecar recording the seed and key
hyperparameters.

INVESTIGATION NOTE -- terminal vs. timeout (raised, then RULED OUT, this
session):
  While debugging an FQE divergence (loss growing, getting WORSE with
  shorter target_update_interval), it was hypothesized that marking
  every episode's last transition as a TIMEOUT rather than a TERMINAL
  was the cause -- reasoning that quarter 20 might be a true fixed
  planning horizon with zero continuation value past it, so bootstrapping
  V(obs[19]) as if the process continued would produce exactly this kind
  of runaway inflation.

  This was checked directly against synthetic_reward.py's step()
  function and docstring, which is the actual ground truth (a citation
  to validation_methodology.md Section 27 turned out to be WRONG --
  Section 27 is about the dataset validator's duplicate/monotonicity/
  sign-convention checks, unrelated to terminal-vs-timeout, and should
  not have been trusted without checking). step()'s own docstring is
  explicit: "done is always False here; episode length is controlled
  externally by the trajectory generator... not by any terminal
  condition in the simulator itself." The state dynamics (gw_level,
  rainfall, demand, extraction_rate) are all ongoing AR(1)-ish processes
  with no absorbing state -- calling step() an imagined 21st time would
  produce a well-formed next_state with real consequences. Quarter 20 is
  therefore a DATA-COLLECTION cutoff on a continuing process, not a true
  fixed horizon.

  CONCLUSION: the ORIGINAL code (this revision) -- terminals all-zero,
  timeouts[-1]=1 at the end of every episode -- is correct. Marking the
  last transition as a true terminal instead (briefly tried, then
  reverted) would have been a NEW, opposite-direction bug: it tells the
  critic there is zero continuation value past quarter 20, which is
  false, and would introduce systematic value UNDERESTIMATION near every
  episode boundary. Left this note in so nobody re-derives the terminal
  "fix" later without re-checking synthetic_reward.py first.

  This means the FQE divergence that prompted the investigation is NOT
  explained by terminal/timeout handling -- it needs a different root
  cause. Worth checking directly in fqe_eval.py and train_cql.py: this
  is standard territory for off-policy value-based bootstrapping
  instability (the "deadly triad" -- function approximation +
  bootstrapping + off-policy data), which FQE is known to be prone to
  even with correct terminal/timeout handling. Also worth checking
  concretely whether d3rlpy's timeout handling actually does what
  Pardo et al.'s time-limit bootstrapping fix intends (bootstrap through
  timeouts, not through true terminals) in this d3rlpy version, rather
  than assuming it from the API docs alone.

ANTI-LEAKAGE DESIGN:
  - Train/val/test split is by EPISODE, not by transition -- an episode's
    20 quarterly transitions are highly autocorrelated (each state is
    largely a deterministic function of the prior one), so splitting at
    the transition level would leak information across the boundary the
    same way un-grouped CV leaked spatial information in
    validation_methodology.md Section 11.
  - The split is computed ONCE (fixed seed, separate from the 3 training
    seeds below) and reused identically across all 3 seeds, so seed-to-
    seed variance reflects only training stochasticity, not which
    episodes each run happened to see.
  - Observation standardization (mean/std) is fit on the TRAIN split
    only and applied unchanged to val/test -- mirrors the point-scale
    LSTM's StandardScaler discipline (validation_methodology.md,
    Section 15.5), implemented manually here rather than via an
    unverified d3rlpy preprocessing class name.

Usage:
    python src/rl/bc_baseline.py --config config/rl_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import d3rlpy
from d3rlpy.algos import DiscreteBCConfig
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is the remainder (0.15)

SPLIT_SEED = 1  # fixed, independent of the 3 training seeds below
TRAINING_SEEDS = [42, 123, 2024]

N_STEPS = 20_000
N_STEPS_PER_EPOCH = 2_000  # -> 10 "epochs" worth of logging granularity

N_PER_POLICY = 500
POLICY_ORDER = ["random", "greedy_extraction", "conservative"]


def policy_for_episode_index(idx: int) -> str:
    return POLICY_ORDER[idx // N_PER_POLICY]


def split_episode_indices(n_episodes: int, seed: int = SPLIT_SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_episodes)
    n_train = int(n_episodes * TRAIN_FRAC)
    n_val = int(n_episodes * VAL_FRAC)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def episodes_to_arrays(episodes, obs_mean=None, obs_std=None):
    obs_list, act_list, rew_list, term_list, timeout_list = [], [], [], [], []
    for ep in episodes:
        obs = ep.observations.astype(np.float32)
        if obs_mean is not None:
            obs = (obs - obs_mean) / obs_std
        obs_list.append(obs)
        act_list.append(ep.actions.reshape(-1).astype(np.int64))
        rew_list.append(ep.rewards.reshape(-1).astype(np.float32))

        T = obs.shape[0]
        terminals = np.zeros(T, dtype=np.float32)
        timeouts = np.zeros(T, dtype=np.float32)
        if ep.terminated:
            terminals[-1] = 1.0
        else:
            timeouts[-1] = 1.0
        term_list.append(terminals)
        timeout_list.append(timeouts)

    return (
        np.concatenate(obs_list, axis=0),
        np.concatenate(act_list, axis=0),
        np.concatenate(rew_list, axis=0),
        np.concatenate(term_list, axis=0),
        np.concatenate(timeout_list, axis=0),
    )


def build_dataset(episodes, obs_mean=None, obs_std=None) -> MDPDataset:
    obs, act, rew, term, timeout = episodes_to_arrays(episodes, obs_mean, obs_std)
    return MDPDataset(observations=obs, actions=act, rewards=rew,
                       terminals=term, timeouts=timeout)


def action_accuracy(algo, episodes, obs_mean, obs_std, episode_indices=None) -> dict:
    correct, total = 0, 0
    per_policy_correct = {p: 0 for p in POLICY_ORDER}
    per_policy_total = {p: 0 for p in POLICY_ORDER}

    for i, ep in enumerate(episodes):
        obs = ep.observations.astype(np.float32)
        obs = (obs - obs_mean) / obs_std
        true_actions = ep.actions.reshape(-1).astype(np.int64)
        pred_actions = algo.predict(obs)
        n_correct = int(np.sum(pred_actions == true_actions))
        correct += n_correct
        total += len(true_actions)

        if episode_indices is not None:
            policy = policy_for_episode_index(episode_indices[i])
            per_policy_correct[policy] += n_correct
            per_policy_total[policy] += len(true_actions)

    result = {"overall": correct / total if total > 0 else float("nan")}
    if episode_indices is not None:
        for p in POLICY_ORDER:
            t = per_policy_total[p]
            result[p] = per_policy_correct[p] / t if t > 0 else float("nan")
    return result


def run_one_seed(seed: int, train_ds, val_episodes, val_indices,
                  test_episodes, test_indices, obs_mean, obs_std,
                  cfg: dict, model_dir: str = "models/bc") -> dict:
    print(f"\n[bc_baseline] === Seed {seed} ===")

    d3rlpy.seed(seed)

    algo = DiscreteBCConfig(
        batch_size=cfg.get("batch_size", 100),
        learning_rate=cfg.get("learning_rate", 0.001),
        beta=cfg.get("beta", 0.5),
    ).create(device=False)

    algo.fit(
        train_ds,
        n_steps=cfg.get("n_steps", N_STEPS),
        n_steps_per_epoch=cfg.get("n_steps_per_epoch", N_STEPS_PER_EPOCH),
        show_progress=True,
    )

    val_acc = action_accuracy(algo, val_episodes, obs_mean, obs_std, val_indices)
    test_acc = action_accuracy(algo, test_episodes, obs_mean, obs_std, test_indices)

    print(f"[bc_baseline] Seed {seed}: val_accuracy={val_acc['overall']:.4f}, "
          f"test_accuracy={test_acc['overall']:.4f}")
    print(f"[bc_baseline] Seed {seed} val by policy: "
          + ", ".join(f"{p}={val_acc[p]:.4f}" for p in POLICY_ORDER))
    print(f"[bc_baseline] Seed {seed} test by policy: "
          + ", ".join(f"{p}={test_acc[p]:.4f}" for p in POLICY_ORDER))

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"bc_seed{seed}.d3"
    algo.save(str(model_path))

    meta_path = out_dir / f"bc_seed{seed}.meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "seed": seed,
            "algo": "DiscreteBC",
            "n_steps": cfg.get("n_steps", N_STEPS),
            "n_steps_per_epoch": cfg.get("n_steps_per_epoch", N_STEPS_PER_EPOCH),
            "batch_size": cfg.get("batch_size", 100),
            "learning_rate": cfg.get("learning_rate", 0.001),
            "beta": cfg.get("beta", 0.5),
            "val_accuracy_overall": val_acc["overall"],
            "test_accuracy_overall": test_acc["overall"],
            "seeded_via": "d3rlpy.seed(seed), called before .create()",
        }, f, indent=2)

    print(f"[bc_baseline] Seed {seed}: saved -> {model_path}")

    return {
        "seed": seed,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "model_path": str(model_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/rl_config.yaml")
    ap.add_argument("--dataset", default="data/processed/offline_rl_dataset.h5")
    ap.add_argument("--out", default="reports/bc_baseline_results.json")
    ap.add_argument("--model-dir", default="models/bc",
                     help="Directory to save each seed's DiscreteBC model + "
                          "metadata sidecar.")
    args = ap.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    bc_cfg = full_cfg.get("bc", {})

    print(f"[bc_baseline] Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(ds.episodes)
    n_episodes = len(episodes)
    print(f"[bc_baseline] Loaded {n_episodes} episodes.")

    train_idx, val_idx, test_idx = split_episode_indices(n_episodes)
    print(f"[bc_baseline] Episode split (fixed seed={SPLIT_SEED}): "
          f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    train_episodes = [episodes[i] for i in train_idx]
    val_episodes = [episodes[i] for i in val_idx]
    test_episodes = [episodes[i] for i in test_idx]

    train_obs_raw, _, _, _, _ = episodes_to_arrays(train_episodes)
    obs_mean = train_obs_raw.mean(axis=0)
    obs_std = train_obs_raw.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0
    print(f"[bc_baseline] Train obs mean: {obs_mean}")
    print(f"[bc_baseline] Train obs std:  {obs_std}")

    train_ds = build_dataset(train_episodes, obs_mean, obs_std)

    results = []
    for seed in TRAINING_SEEDS:
        result = run_one_seed(seed, train_ds, val_episodes, val_idx,
                               test_episodes, test_idx, obs_mean, obs_std,
                               bc_cfg, model_dir=args.model_dir)
        results.append(result)

    val_overall = [r["val_accuracy"]["overall"] for r in results]
    test_overall = [r["test_accuracy"]["overall"] for r in results]
    summary = {
        "per_seed": results,
        "val_accuracy_mean": float(np.mean(val_overall)),
        "val_accuracy_std": float(np.std(val_overall)),
        "test_accuracy_mean": float(np.mean(test_overall)),
        "test_accuracy_std": float(np.std(test_overall)),
        "n_train_episodes": len(train_idx),
        "n_val_episodes": len(val_idx),
        "n_test_episodes": len(test_idx),
        "split_seed": SPLIT_SEED,
        "training_seeds": TRAINING_SEEDS,
        "policy_order_assumption": POLICY_ORDER,
        "n_per_policy_assumption": N_PER_POLICY,
        "model_dir": args.model_dir,
    }

    print("\n[bc_baseline] === Summary across 3 seeds ===")
    print(f"  Val accuracy:  {summary['val_accuracy_mean']:.4f} "
          f"± {summary['val_accuracy_std']:.4f}")
    print(f"  Test accuracy: {summary['test_accuracy_mean']:.4f} "
          f"± {summary['test_accuracy_std']:.4f}")
    for p in POLICY_ORDER:
        p_val = [r["val_accuracy"][p] for r in results]
        p_test = [r["test_accuracy"][p] for r in results]
        print(f"  [{p}] val: {np.mean(p_val):.4f} ± {np.std(p_val):.4f}   "
              f"test: {np.mean(p_test):.4f} ± {np.std(p_test):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bc_baseline] Saved -> {args.out}")


if __name__ == "__main__":
    main()
