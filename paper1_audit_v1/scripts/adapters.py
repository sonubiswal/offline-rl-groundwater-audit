"""
adapters.py

Wires four functions to the trishna-opal project so that the analysis
scripts (run_interval_constructions, run_cross_combination_10seed,
run_bc_sweep, run_ood_w3_zero) run unchanged.

COLLISION FIX (revision): two RandomState seeds in this file previously
overlapped numerically across (seed, episode) pairs. Both are replaced
with disjoint ranges. See Section III-F and Section VII of the manuscript.

    _load_rf_starts : 9999 + seed              -> 100000 + seed
    rollout_returns : 9999 + seed + i          -> 200000 + 1000*seed + i

For the ten seeds used here (minimum 7, maximum 99999, all distinct), the
start-sampler range [100007, 199999] and the rollout range
[242000, 100199224] are disjoint, and two rollout generators with
different seeds are separated by at least 1000 integers, exceeding the
maximum episode index 224. No two streams therefore coincide. The argument
is stated in Section III-F.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "rl"))

from src.rl.policy_io import load_policy_for_eval, compute_scaler_from_dataset
from src.rl.synthetic_reward import SimState, step, load_reward_config

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]

DATASETS = {
    "base":       ROOT / "data" / "processed" / "offline_rl_dataset.h5",
    "epsilon":    ROOT / "data" / "processed" / "offline_rl_dataset_epsilon.h5",
    "no_oracle":  ROOT / "data" / "processed" / "offline_rl_dataset_no_oracle.h5",
}

MODELS_ROOT = ROOT / "models_10seed"

CKPT_FAMILIES = {
    "base":          (MODELS_ROOT / "base" / "bc",          MODELS_ROOT / "base" / "cql"),
    "epsilon":       (MODELS_ROOT / "epsilon" / "bc",       MODELS_ROOT / "epsilon" / "cql"),
    "no_oracle":     (MODELS_ROOT / "no_oracle" / "bc",     MODELS_ROOT / "no_oracle" / "cql"),
    "no_oracle_a1":  (MODELS_ROOT / "no_oracle_a1" / "bc",  MODELS_ROOT / "no_oracle_a1" / "cql"),
    "no_oracle_a10": (MODELS_ROOT / "no_oracle_a10" / "bc", MODELS_ROOT / "no_oracle_a10" / "cql"),
}

RF_TABLE = ROOT / "reports" / "rf_grid_predictions.csv"
CFG_PATH = ROOT / "config" / "rl_config.yaml"


# ------------------------------------------------------- sidecar helpers
def _sidecar_path(ckpt_path):
    """Return <ckpt>.scaler.json path (appended, not suffix-replaced)."""
    ckpt_path = Path(ckpt_path)
    return ckpt_path.parent / (ckpt_path.name + ".scaler.json")


def _ensure_sidecar(ckpt_path, dataset_path):
    """Make sure <ckpt>.scaler.json exists next to the checkpoint."""
    ckpt_path = Path(ckpt_path)
    sidecar = _sidecar_path(ckpt_path)
    if sidecar.exists():
        return sidecar

    ds_resolved = Path(dataset_path).resolve()
    for fam, d in DATASETS.items():
        if d.resolve() != ds_resolved:
            continue
        bc_dir, cql_dir = CKPT_FAMILIES[fam]
        for d2 in (bc_dir, cql_dir):
            d2 = Path(d2)
            if not d2.exists():
                continue
            for pattern in ("bc_seed*.d3.scaler.json",
                            "cql_seed*.d3.scaler.json",
                            "*.d3.scaler.json"):
                hits = sorted(d2.glob(pattern))
                if hits:
                    shutil.copy2(hits[0], sidecar)
                    return sidecar

    try:
        from src.rl.policy_io import write_scaler_sidecar  # type: ignore
        scaler = compute_scaler_from_dataset(Path(dataset_path))
        try:
            write_scaler_sidecar(scaler, sidecar)
        except TypeError:
            write_scaler_sidecar(sidecar, scaler)
        if sidecar.exists():
            return sidecar
    except ImportError:
        pass

    raise FileNotFoundError(
        f"No scaler sidecar for {ckpt_path.name} and no canonical "
        f"reference sidecar found for dataset {ds_resolved}. "
        f"Trained on a non-canonical dataset? "
        f"Place {sidecar.name} manually next to the checkpoint."
    )


# --------------------------------------------------------- adapter 1: policy
def load_policy(ckpt_path, dataset_path):
    """Return a callable obs(np.ndarray [N,6] or [6]) -> actions [N]."""
    ckpt_path = Path(ckpt_path)
    _ensure_sidecar(ckpt_path, dataset_path)
    sp = load_policy_for_eval(ckpt_path, Path(dataset_path))

    def _predict(obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]                 # (6,) -> (1, 6)
        return np.asarray(sp.predict(obs)).reshape(-1)

    return _predict


# ------------------------------------------------------- adapter 2: rollout
def _load_test_episode_starts(dataset_path, n=225):
    with h5py.File(dataset_path, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        rng = np.random.RandomState(1)
        idx = rng.permutation(len(ep_ids))
        test_idx = idx[1050 + 225 : 1050 + 225 + n]
        assert len(test_idx) == n, (
            f"expected {n} test starts, got {len(test_idx)} "
            f"(dataset has {len(ep_ids)} episodes)")
        starts = np.array([f[f"observations_{ep_ids[i]}"][0]
                           for i in test_idx], dtype=np.float32)
    return starts


def _load_rf_starts(n, seed):
    df = pd.read_csv(RF_TABLE)
    # COLLISION FIX: was RandomState(9999 + seed)
    rng = np.random.RandomState(100000 + seed)
    pick = rng.choice(len(df), size=n, replace=True)
    rows = df.iloc[pick].reset_index(drop=True)
    starts = []
    for _, row in rows.iterrows():
        month = float(row.get("month", 1))
        ang = 2 * np.pi * (month - 1) / 12.0
        starts.append(np.array([
            float(row.get("gw_level_pred", 15.0)),
            float(row.get("chirps_roll3", 30.0)),
            0.4,
            float(np.sin(ang)),
            float(np.cos(ang)),
            0.3,
        ], dtype=np.float32))
    return np.asarray(starts, dtype=np.float32)


def rollout_returns(policy, n_episodes, seed, start_set="rf_grid",
                    reward_weights=(1.0, 2.0, 0.5), horizon=20):
    """Return np.ndarray [n_episodes] of undiscounted 20-step returns."""
    cfg = load_reward_config(str(CFG_PATH))
    cfg["w1_crop_revenue"] = reward_weights[0]
    cfg["w2_depletion_penalty"] = reward_weights[1]
    cfg["w3_domestic_supply"] = reward_weights[2]

    if isinstance(start_set, np.ndarray):
        raw_starts = start_set
    elif start_set == "rf_grid":
        raw_starts = _load_rf_starts(n_episodes, seed)
    elif start_set == "test225":
        raw_starts = _load_test_episode_starts(DATASETS["epsilon"], n=n_episodes)
    else:
        raise ValueError(f"unknown start_set {start_set!r}")

    out = np.zeros(len(raw_starts))
    for i, raw in enumerate(raw_starts):
        s = SimState(*[float(x) for x in raw])
        # COLLISION FIX: was RandomState(9999 + seed + i)
        rng_l = np.random.RandomState(200000 + 1000 * seed + i)
        total = 0.0
        for _ in range(horizon):
            vec = np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                            s.month_sin, s.month_cos, s.extraction_rate],
                           dtype=np.float32)
            a = int(policy(vec)[0])
            s, r, done = step(s, a, config=cfg, n_actions=5, rng=rng_l)
            total += float(r)
            if done:
                break
        out[i] = total
    return out


# ------------------------------------------------------- adapter 3: train BC
def train_bc(dataset_path, out_ckpt, seed, lr=1e-3, hidden=(256, 256), steps=20000,
             policy_order=None, n_per_policy=None):
    """Train d3rlpy DiscreteBC using bc_baseline.py. Writes a .d3 + sidecar."""
    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TRAINING_SEEDS_OVERRIDE"] = str(seed)
    env["BC_LR_OVERRIDE"] = str(lr)
    env["BC_HIDDEN_OVERRIDE"] = f"{hidden[0]},{hidden[1]}"
    env["BC_STEPS_OVERRIDE"] = str(steps)

    cmd = [sys.executable, "src/rl/bc_baseline.py",
           "--dataset", str(dataset_path),
           "--model-dir", str(out_ckpt.parent),
           "--out", str(out_ckpt.with_suffix(".results.json")),
           "--overwrite"]
    if policy_order is not None:
        cmd += ["--policy-order", *policy_order]
    if n_per_policy is not None:
        cmd += ["--n-per-policy", str(n_per_policy)]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    produced      = out_ckpt.parent / f"bc_seed{seed}.d3"
    produced_meta = out_ckpt.parent / f"bc_seed{seed}.meta.json"
    if produced != out_ckpt and produced.exists():
        produced.replace(out_ckpt)
        produced_side = _sidecar_path(produced)
        if produced_side.exists():
            produced_side.replace(_sidecar_path(out_ckpt))
        if produced_meta.exists():
            produced_meta.replace(out_ckpt.with_suffix(".meta.json"))

    _ensure_sidecar(out_ckpt, dataset_path)
    return out_ckpt


# ------------------------------------------------------ adapter 4: train CQL
def train_cql(dataset_path, out_ckpt, seed, reward_weights=(1.0, 2.0, 0.5),
              lr=3e-4, alpha=1.0, steps=20000):
    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TRAINING_SEEDS_OVERRIDE"] = str(seed)
    subprocess.run(
        [sys.executable, "src/rl/train_cql.py",
         "--dataset", str(dataset_path),
         "--model-dir", str(out_ckpt.parent),
         "--out", str(out_ckpt.with_suffix(".results.json")),
         "--alpha", str(alpha),
         "--skip-sweep",
         "--learning-rate", str(lr),
         "--n-steps", str(steps),
         "--overwrite"],
        cwd=ROOT, env=env, check=True)

    produced = out_ckpt.parent / f"cql_seed{seed}.d3"
    produced_meta = out_ckpt.parent / f"cql_seed{seed}.meta.json"
    if produced != out_ckpt and produced.exists():
        produced.replace(out_ckpt)
        produced_side = _sidecar_path(produced)
        if produced_side.exists():
            produced_side.replace(_sidecar_path(out_ckpt))
        if produced_meta.exists():
            produced_meta.replace(out_ckpt.with_suffix(".meta.json"))

    _ensure_sidecar(out_ckpt, dataset_path)
    return out_ckpt


# ---------------------------------------------------------- adapter 5: ckpt
def checkpoint(family, algo, seed):
    """Return the path of a final checkpoint."""
    if family not in CKPT_FAMILIES:
        raise KeyError(f"unknown family {family!r}")
    bc_dir, cql_dir = CKPT_FAMILIES[family]
    if algo == "bc":
        return bc_dir / f"bc_seed{seed}.d3"
    elif algo == "cql":
        return cql_dir / f"cql_seed{seed}.d3"
    raise ValueError(f"unknown algo {algo!r}")