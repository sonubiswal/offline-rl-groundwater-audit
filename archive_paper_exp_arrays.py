"""
archive_paper_exp_arrays.py

Best-effort archiver for the expanded Phase-5 report's per-episode
return arrays. Walks reports/paper_exp/*/, finds each config's BC and
CQL model directories, runs the standard compare_policies-style rollout
(500 episodes, eval seed 9999) for seeds 42/123/2024, and saves arrays to
reports/<config>/returns/.

This does NOT reproduce each paper_exp config's specific headline
protocol (nine-cell scenarios, adapted policies, weights sweeps, etc.) —
it archives a standard canonical-start-state rollout for each pair so
bootstrap CIs can be recomputed from raw data. Configs whose headline
protocol differs will produce arrays with different values than the
headline, but the raw data will exist.

Idempotent: skips configs whose returns/ dir already has all expected arrays.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "rl"))

import d3rlpy

from compare_policies import (
    load_training_standardization,
    make_model_action_fn,
    rollout_policy,
    EVAL_SEED, TRAINING_SEEDS, DEFAULT_GAMMA,
)

PAPER_EXP = Path("reports/paper_exp")
CONFIG_PATH = Path("config/rl_config.yaml")
RF_TABLE = Path("reports/rf_grid_predictions.csv")
DATASET = Path("data/processed/offline_rl_dataset.h5")

N_EPISODES = 500

cfg = yaml.safe_load(open(CONFIG_PATH))
reward_cfg = cfg["reward"]
n_actions = cfg["action_space"]["n_actions"]
traj_len = cfg["simulation"]["trajectory_length"]
rf_df = pd.read_csv(RF_TABLE)

print(f"[archive] RF rows: {len(rf_df)}")
obs_mean, obs_std = load_training_standardization(DATASET)
print(f"[archive] normalization recovered")

SKIP_DIRS = {"frozen_models", "ood_models"}


def find_model_dir(root: Path, algo: str):
    """Return a directory under `root` that contains <algo>_seed*.d3 files, or None."""
    for cand in root.rglob(f"{algo}_seed42.d3"):
        return cand.parent
    return None


configs_found = []
for cfg_dir in sorted(PAPER_EXP.iterdir()):
    if not cfg_dir.is_dir():
        continue
    if cfg_dir.name in SKIP_DIRS:
        continue

    # some configs have bc/ and cql/ or bc/ and cql_frozen/
    bc_dir = find_model_dir(cfg_dir, "bc")
    cql_dir = find_model_dir(cfg_dir, "cql")

    if bc_dir is None or cql_dir is None:
        # try single-subdir layout (frozen_models/bc + frozen_models/cql at top)
        continue

    configs_found.append((cfg_dir.name, bc_dir, cql_dir))

# also handle frozen_models/ at the top level (base paper_exp canonical)
frozen_root = PAPER_EXP / "frozen_models"
if frozen_root.exists():
    bc_dir = find_model_dir(frozen_root, "bc")
    cql_dir = find_model_dir(frozen_root, "cql")
    if bc_dir and cql_dir:
        configs_found.append(("frozen_models", bc_dir, cql_dir))

print(f"[archive] found {len(configs_found)} config(s) with paired BC+CQL:")
for name, b, c in configs_found:
    print(f"  {name:35s}  bc={b.relative_to(PAPER_EXP)}  cql={c.relative_to(PAPER_EXP)}")

total_saved = 0
for name, bc_dir, cql_dir in configs_found:
    out_dir = PAPER_EXP / name / "returns"
    expected = [
        out_dir / f"returns_{algo}_seed{s}.npy"
        for algo in ("bc", "cql") for s in TRAINING_SEEDS
    ]
    if all(p.exists() for p in expected):
        print(f"\n[archive] {name}: already archived, skipping")
        continue

    print(f"\n[archive] {name}: rolling out")
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in TRAINING_SEEDS:
        bc_path = bc_dir / f"bc_seed{seed}.d3"
        cql_path = cql_dir / f"cql_seed{seed}.d3"
        if not bc_path.exists() or not cql_path.exists():
            print(f"  seed {seed}: missing model, skipping")
            continue

        bc_model = d3rlpy.load_learnable(str(bc_path), device="cpu")
        cql_model = d3rlpy.load_learnable(str(cql_path), device="cpu")

        bc_fn = make_model_action_fn(bc_model, obs_mean, obs_std)
        cql_fn = make_model_action_fn(cql_model, obs_mean, obs_std)

        bc_res = rollout_policy(
            policy_name=f"BC_seed{seed}", action_fn=bc_fn, rf_df=rf_df,
            reward_cfg=reward_cfg, n_actions=n_actions,
            trajectory_length=traj_len, n_episodes=N_EPISODES,
            seed=EVAL_SEED, gamma=DEFAULT_GAMMA,
        )
        cql_res = rollout_policy(
            policy_name=f"CQL_seed{seed}", action_fn=cql_fn, rf_df=rf_df,
            reward_cfg=reward_cfg, n_actions=n_actions,
            trajectory_length=traj_len, n_episodes=N_EPISODES,
            seed=EVAL_SEED, gamma=DEFAULT_GAMMA,
        )

        np.save(out_dir / f"returns_bc_seed{seed}.npy",
                np.asarray(bc_res["episode_returns"], dtype=np.float64))
        np.save(out_dir / f"returns_cql_seed{seed}.npy",
                np.asarray(cql_res["episode_returns"], dtype=np.float64))
        np.save(out_dir / f"discounted_returns_bc_seed{seed}.npy",
                np.asarray(bc_res["discounted_returns"], dtype=np.float64))
        np.save(out_dir / f"discounted_returns_cql_seed{seed}.npy",
                np.asarray(cql_res["discounted_returns"], dtype=np.float64))

        total_saved += 4
        print(f"  seed {seed}: BC={bc_res['return_mean']:+.4f}  CQL={cql_res['return_mean']:+.4f}  "
              f"Δ={cql_res['return_mean']-bc_res['return_mean']:+.4f}")

print(f"\n[archive] saved {total_saved} array files across {len(configs_found)} configs")
print("[archive] to verify: python -c \"import numpy as np, glob; "
      "print(sum(1 for _ in glob.glob('reports/paper_exp/*/returns/*.npy')))\"")