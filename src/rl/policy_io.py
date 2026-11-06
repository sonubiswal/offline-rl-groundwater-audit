"""
src/rl/policy_io.py

Canonical loader for trained offline-RL policies.

Every evaluation script must load policies through `load_policy_for_eval`,
which applies the observation standardisation used at training time before
calling the underlying d3rlpy model's predict(). This closes the class of
silent scaler-mismatch bug documented in the project's configuration audit.

Also provides the sidecar save/read helpers used by the training scripts
(bc_baseline.py, train_cql.py) to persist the training-time scaler next to
each checkpoint.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import d3rlpy
import numpy as np


# --------------------------------------------------------------------- scaler
def _load_training_split_stats(dataset_path: Path):
    """Compute obs_mean / obs_std from the training split of a dataset.

    Imports bc_baseline lazily to avoid a circular import: bc_baseline
    imports from this module.
    """
    from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
    from src.rl.bc_baseline import (
        split_episode_indices,
        episodes_to_arrays,
        SPLIT_SEED,
    )

    dataset_path = Path(dataset_path)
    with open(dataset_path, "rb") as f:
        ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(ds.episodes)
    train_idx, _, _ = split_episode_indices(len(episodes), seed=SPLIT_SEED)
    train_eps = [episodes[i] for i in train_idx]
    train_obs, _, _, _, _ = episodes_to_arrays(train_eps)
    train_obs = train_obs.astype(np.float32)
    mean = train_obs.mean(axis=0).astype(np.float32)
    std = train_obs.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def compute_scaler_from_dataset(dataset_path: Path):
    """Public alias used by scripts that want the mean/std directly."""
    return _load_training_split_stats(Path(dataset_path))


# ------------------------------------------------------------------ sidecars
def scaler_sidecar_path(checkpoint_path: Path) -> Path:
    """Return the .scaler.json path that sits next to a .d3 checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_suffix(".d3.scaler.json")


def save_scaler_sidecar(checkpoint_path: Path,
                        obs_mean: np.ndarray,
                        obs_std: np.ndarray,
                        dataset_path: Path,
                        extra: dict | None = None) -> Path:
    """Write the training-time scaler as JSON next to the checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    dataset_path = Path(dataset_path)
    payload = {
        "mean": np.asarray(obs_mean, dtype=np.float32).tolist(),
        "std": np.asarray(obs_std, dtype=np.float32).tolist(),
        "dataset": str(dataset_path),
        "created": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        payload.update(extra)
    out = scaler_sidecar_path(checkpoint_path)
    out.write_text(json.dumps(payload, indent=2))
    return out


def read_scaler_sidecar(checkpoint_path: Path) -> dict:
    p = scaler_sidecar_path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(
            f"No scaler sidecar next to {checkpoint_path.name}. "
            f"Expected {p}."
        )
    return json.loads(p.read_text())


# ---------------------------------------------------------------- the loader
class ScaledPolicy:
    """Wraps a d3rlpy policy so .predict(raw_obs) applies the scaler first."""

    def __init__(self, policy, obs_mean: np.ndarray, obs_std: np.ndarray):
        self.policy = policy
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)

    def predict(self, raw_obs):
        raw = np.asarray(raw_obs, dtype=np.float32)
        std = (raw - self.obs_mean) / self.obs_std
        return self.policy.predict(std)

    def action(self, state_vec) -> int:
        sv = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
        return int(np.asarray(self.predict(sv)).reshape(-1)[0])


def load_policy_for_eval(checkpoint_path: Path, dataset_path: Path,
                          strict: bool = True) -> ScaledPolicy:
    """
    Load a checkpoint and return a ScaledPolicy.

    strict=True: sidecar must exist and match a freshly recomputed scaler
                 from dataset_path's training split. Fails hard on mismatch.
    strict=False: warn on mismatch but proceed with the sidecar's value.
    """
    checkpoint_path = Path(checkpoint_path)
    dataset_path = Path(dataset_path)

    sidecar = read_scaler_sidecar(checkpoint_path)
    sidecar_mean = np.asarray(sidecar["mean"], dtype=np.float32)
    sidecar_std = np.asarray(sidecar["std"], dtype=np.float32)

    recomputed_mean, recomputed_std = _load_training_split_stats(dataset_path)

    if not np.allclose(sidecar_mean, recomputed_mean, atol=1e-4) or \
       not np.allclose(sidecar_std, recomputed_std, atol=1e-4):
        msg = (
            f"Scaler mismatch for {checkpoint_path.name}\n"
            f"  sidecar dataset:    {sidecar['dataset']}\n"
            f"  loaded dataset:     {dataset_path}\n"
            f"  sidecar mean[0]:    {sidecar_mean[0]:.6f}\n"
            f"  recomputed mean[0]: {recomputed_mean[0]:.6f}"
        )
        if strict:
            raise AssertionError(msg)
        else:
            print(f"[policy_io] WARNING: {msg}")

    policy = d3rlpy.load_learnable(str(checkpoint_path))
    return ScaledPolicy(policy, sidecar_mean, sidecar_std)