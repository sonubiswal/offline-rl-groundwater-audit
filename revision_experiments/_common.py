"""
revision_experiments/_common.py

Shared helpers for E1 and E3.

Key design rule: this module must NOT reimplement the training split or the
standardisation statistics. It imports them directly from bc_baseline.py, the
same way compare_policies.py does, so that any evaluation that uses these
helpers sees exactly the scaler the checkpoints were trained with.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# --- ensure the repo root is importable --------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------- loading

def load_episodes(dataset_path: Path):
    """Read raw per-episode observations and actions from an .h5 dataset.

    Returns (obs_list, act_list) as lists of numpy arrays. Observations are
    raw (metres, mm, etc.) — no standardisation is applied here. This is
    used for band-definition (Mahalanobis support) which works on raw
    physical units, not on the standardised training inputs.
    """
    dataset_path = Path(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        obs = [f[f"observations_{e}"][:] for e in ep_ids]
        act = [f[f"actions_{e}"][:] for e in ep_ids]
    return obs, act


# --------------------------------------------------------- split + scaler

def training_standardization(dataset_path: Path):
    """Recompute obs_mean/obs_std exactly the way compare_policies.py does.

    compare_policies.load_training_standardization:
        1. loads the dataset as a ReplayBuffer (d3rlpy)
        2. calls bc_baseline.split_episode_indices(n, seed=SPLIT_SEED)
        3. calls bc_baseline.episodes_to_arrays(train_episodes)
        4. takes .mean(0) and .std(0) of the raw training observations
        5. clips std to >= 1e-6

    We reproduce that path verbatim, importing the canonical functions
    instead of reimplementing them. This is what guarantees the base and
    epsilon scalers match the paper's post-fix reports.
    """
    from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
    from src.rl.bc_baseline import (
        split_episode_indices,
        episodes_to_arrays,
        SPLIT_SEED,
    )

    dataset_path = Path(dataset_path)
    with open(dataset_path, "rb") as f:
        dataset = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(dataset.episodes)
    if not episodes:
        raise ValueError(f"no episodes in {dataset_path}")

    train_idx, _, _ = split_episode_indices(len(episodes), seed=SPLIT_SEED)
    train_eps = [episodes[i] for i in train_idx]

    # episodes_to_arrays with no mean/std returns raw arrays
    train_obs, _, _, _, _ = episodes_to_arrays(train_eps)
    train_obs = train_obs.astype(np.float32)

    mean = train_obs.mean(axis=0).astype(np.float32)
    std = train_obs.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def training_split_indices(n_episodes: int, seed: int = 1):
    """Convenience wrapper around the canonical split function.

    Only used by callers that already have a plain list of episodes and
    don't want to go through ReplayBuffer.load. Behaviour is identical to
    bc_baseline.split_episode_indices for the 1500-episode case (1050
    training episodes) and falls back to 70/15/15 otherwise.
    """
    try:
        from src.rl.bc_baseline import split_episode_indices, SPLIT_SEED as _S
        seed = _S if seed == 1 else seed
        train_idx, _, _ = split_episode_indices(n_episodes, seed=seed)
        return train_idx
    except Exception:
        # fallback that matches bc_baseline's convention when the import
        # is not possible (should not happen in practice)
        n_train = 1050 if n_episodes == 1500 else int(round(0.70 * n_episodes))
        rng = np.random.RandomState(seed)
        idx = np.arange(n_episodes)
        rng.shuffle(idx)
        return idx[:n_train]


# ---------------------------------------------------------- action wrapper

def make_action_fn(policy, obs_mean: np.ndarray, obs_std: np.ndarray):
    """Return a callable that standardises before predict.

    Mirrors compare_policies.make_model_action_fn exactly:
        standardized = ((raw - obs_mean) / obs_std).reshape(1, OBSERVATION_DIM)
        pred = algo.predict(standardized)
        action = int(pred[0])
    """
    obs_mean = np.asarray(obs_mean, dtype=np.float32)
    obs_std = np.asarray(obs_std, dtype=np.float32)

    def action_fn(state_vector: np.ndarray) -> int:
        raw = np.asarray(state_vector, dtype=np.float32)
        standardized = ((raw - obs_mean) / obs_std).reshape(1, -1)
        pred = np.asarray(policy.predict(standardized)).reshape(-1)
        return int(pred[0])

    return action_fn


# ------------------------------------------------------------ small helper

def describe_scaler(name: str, mean: np.ndarray, std: np.ndarray):
    """Print a short summary of a scaler for logs."""
    print(f"[scaler] {name:<14}  "
          f"gw_level mean={mean[0]:.6f} std={std[0]:.6f}  "
          f"rainfall mean={mean[1]:.4f} std={std[1]:.4f}  "
          f"extraction mean={mean[5]:.4f} std={std[5]:.4f}")