"""
ood_state_distribution_test.py
Genuine state-distribution OOD test for the Phase 5 BC and CQL policies.

Constructs ID / mild / moderate / severe OOD states using Mahalanobis distance
from the training-state distribution, queries each trained BC and CQL policy,
and compares action distributions across levels.

Reads:
  data/processed/offline_rl_dataset.h5          (per-episode keys, 6-dim obs)
  reports/paper_exp/frozen_models/bc/bc_seed{42,123,2024}.d3
  reports/paper_exp/frozen_models/cql/cql_seed{42,123,2024}.d3

Writes:
  reports/ood_state_distribution.json
  reports/ood_state_distribution.png

Read-only. Does not modify any experiment artifact.

State layout (from src/rl/validate_offline_dataset.py:309):
  0: gw_level            metres below ground (higher = deeper/worse)
  1: recent_rainfall
  2: crop_water_demand
  3: month_sin
  4: month_cos
  5: extraction_rate     0..1
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

try:
    import d3rlpy
    HAVE_D3RLPY = True
except ImportError:
    HAVE_D3RLPY = False

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

DATASET = "data/processed/offline_rl_dataset.h5"
MODEL_ROOT = Path("reports/paper_exp/frozen_models")
SEEDS = [42, 123, 2024]

# Split parameters -- match bc_baseline.py / train_cql.py
SPLIT_SEED = 1
N_TRAIN = 1050
N_VAL = 225
N_TEST = 225
N_EPISODES = N_TRAIN + N_VAL + N_TEST          # 1500

N_STATES_PER_LEVEL = 500
RNG = np.random.RandomState(2026)

DIM_NAMES = ["gw_level", "recent_rainfall", "crop_water_demand",
             "month_sin", "month_cos", "extraction_rate"]

# Bounds fixed to observed dataset ranges with margin
PHYSICAL_BOUNDS = {
    "gw_level":          (1.0,   60.0),
    "recent_rainfall":   (0.0,   400.0),
    "crop_water_demand": (0.0,   1.2),
    "month_sin":         (-1.0,  1.0),
    "month_cos":         (-1.0,  1.0),
    "extraction_rate":   (0.0,   1.0),
}


# ------------------------------------------------------------------ data
def load_dataset():
    """Return per-episode observations as a list of (20, 6) arrays, sorted."""
    with h5py.File(DATASET, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        episodes = [f[f"observations_{eid}"][:] for eid in ep_ids]
    return episodes


def split_episodes(episodes, seed=SPLIT_SEED):
    """Match the episode-level split used by the RL training scripts."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    train_idx = idx[:N_TRAIN]
    val_idx = idx[N_TRAIN:N_TRAIN + N_VAL]
    test_idx = idx[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]
    return train_idx, val_idx, test_idx


# ----------------------------------------------------- training support
def training_support(train_obs):
    """
    train_obs : (T, 6) stacked training observations.
    Returns mean (6,), cov (6,6), inv_cov (6,6), all-observation Mahalanobis^2.
    """
    mu = train_obs.mean(axis=0)
    cov = np.cov(train_obs, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-8
    inv_cov = np.linalg.inv(cov)
    diff = train_obs - mu
    m2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return mu, cov, inv_cov, m2


def band_thresholds(chi2_df):
    """Mahalanobis^2 cutoffs from chi-square with df = dim."""
    return {
        "id_max":        float(stats.chi2.ppf(0.68,  chi2_df)),
        "mild_max":      float(stats.chi2.ppf(0.95,  chi2_df)),
        "moderate_max":  float(stats.chi2.ppf(0.997, chi2_df)),
    }


# ----------------------------------------------- OOD state construction
def physical_mask(states):
    ok = np.ones(len(states), dtype=bool)
    for j, name in enumerate(DIM_NAMES):
        lo, hi = PHYSICAL_BOUNDS[name]
        ok &= (states[:, j] >= lo) & (states[:, j] <= hi)
    s = states[:, 3]
    c = states[:, 4]
    r = np.sqrt(s ** 2 + c ** 2)
    ok &= (r > 1e-6)
    return ok


def project_month_to_unit_circle(states):
    """Renormalize month_sin/month_cos to lie exactly on the unit circle."""
    s = states[:, 3]
    c = states[:, 4]
    r = np.sqrt(s ** 2 + c ** 2)
    r[r < 1e-12] = 1.0
    states[:, 3] = s / r
    states[:, 4] = c / r
    return states


def sample_states_by_band(mu, cov, inv_cov, thresholds, rng,
                          n_per_band=N_STATES_PER_LEVEL):
    proposal_cov = cov * 1.5
    want = {
        "id":        lambda m2: m2 <= thresholds["id_max"],
        "mild":      lambda m2: (m2 >  thresholds["id_max"])
                                 & (m2 <= thresholds["mild_max"]),
        "moderate":  lambda m2: (m2 >  thresholds["mild_max"])
                                 & (m2 <= thresholds["moderate_max"]),
        "severe":    lambda m2: m2 >  thresholds["moderate_max"],
    }
    collected = {k: [] for k in want}

    max_draws = 2_000_000
    drawn = 0
    while drawn < max_draws and any(len(v) < n_per_band for v in collected.values()):
        batch = rng.multivariate_normal(mu, proposal_cov, size=5000)
        drawn += len(batch)
        batch = project_month_to_unit_circle(batch)
        keep_phys = physical_mask(batch)
        batch = batch[keep_phys]
        if len(batch) == 0:
            continue
        diff = batch - mu
        m2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        for k, f in want.items():
            if len(collected[k]) >= n_per_band:
                continue
            sel = f(m2)
            take = batch[sel]
            room = n_per_band - len(collected[k])
            collected[k].extend(take[:room].tolist())

    out = {k: np.array(v[:n_per_band]) for k, v in collected.items()}
    for k, v in out.items():
        print(f"[ood] band {k:9s} sampled {len(v)} / {n_per_band}")
    return out, drawn


# ---------------------------------------------------------- policy eval
def load_policy(path: Path):
    if not HAVE_D3RLPY:
        raise RuntimeError("d3rlpy is not installed.")
    return d3rlpy.load_learnable(str(path))


def policy_actions(policy, states):
    states = states.astype(np.float32)
    actions = policy.predict(states)
    return np.asarray(actions).reshape(-1).astype(int)


# -------------------------------------------------------------- report
def summarize(name, actions, n_actions=5):
    counts = np.bincount(actions, minlength=n_actions)
    dist = counts / max(counts.sum(), 1)
    return {
        "policy": name,
        "n": int(counts.sum()),
        "counts": counts.tolist(),
        "distribution": dist.tolist(),
        "mean_action": float(actions.mean()) if len(actions) else None,
    }


def total_variation(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def main():
    print("[ood] loading dataset ...")
    episodes = load_dataset()
    train_idx, _, _ = split_episodes(episodes)
    train_obs = np.concatenate([episodes[i] for i in train_idx], axis=0)
    print(f"[ood] training observations: {train_obs.shape}")

    print("[ood] computing training support ...")
    mu, cov, inv_cov, m2_train = training_support(train_obs)
    th = band_thresholds(chi2_df=train_obs.shape[1])
    print(f"[ood] thresholds (Mahalanobis^2): {th}")

    print("[ood] sampling OOD states ...")
    bands, n_drawn = sample_states_by_band(mu, cov, inv_cov, th, RNG)

    results = {
        "setup": {
            "n_train_observations": int(len(train_obs)),
            "train_mean": mu.tolist(),
            "train_cov_diag": np.diag(cov).tolist(),
            "mahalanobis2_thresholds": th,
            "sampling_draws": int(n_drawn),
            "dim_names": DIM_NAMES,
            "bounds": PHYSICAL_BOUNDS,
        },
        "per_level": {},
        "policies": {},
    }

    all_policies = []
    for algo in ("bc", "cql"):
        for seed in SEEDS:
            p = MODEL_ROOT / algo / f"{algo}_seed{seed}.d3"
            if p.exists():
                all_policies.append((algo, seed, p))
            else:
                print(f"[ood] missing: {p}")

    print(f"[ood] loaded {len(all_policies)} policies")

    for level, states in bands.items():
        level_summary = {"n_states": int(len(states))}
        for algo, seed, path in all_policies:
            pol = load_policy(path)
            acts = policy_actions(pol, states)
            key = f"{algo}_seed{seed}"
            results["policies"].setdefault(key, {})[level] = \
                summarize(key, acts)
            level_summary[key] = results["policies"][key][level]["mean_action"]
        results["per_level"][level] = level_summary

    # ---- Cross-policy comparisons (FIXED) --------------------------------
    print("\n[ood] summary (mean action per level, mean across seeds):")
    header = f"{'level':<10} {'BC mean':>9} {'CQL mean':>9} {'d(CQL-BC)':>11} {'TV':>8}"
    print(header)
    print("-" * len(header))
    comparisons = {}
    for level in ["id", "mild", "moderate", "severe"]:
        bc_dists = []
        cql_dists = []
        for algo, seed, _ in all_policies:
            key = f"{algo}_seed{seed}"
            dist = np.asarray(results["policies"][key][level]["distribution"])
            if algo == "bc":
                bc_dists.append(dist)
            else:
                cql_dists.append(dist)

        if not bc_dists or not cql_dists:
            comparisons[level] = {
                "error": "missing distributions",
                "n_bc": len(bc_dists),
                "n_cql": len(cql_dists),
            }
            print(f"{level:<10} SKIP (bc={len(bc_dists)}, cql={len(cql_dists)})")
            continue

        bc_mean = np.mean(bc_dists, axis=0)
        cql_mean = np.mean(cql_dists, axis=0)
        bc_mean_action = float((bc_mean * np.arange(5)).sum())
        cql_mean_action = float((cql_mean * np.arange(5)).sum())
        tv = total_variation(bc_mean, cql_mean)
        comparisons[level] = {
            "bc_mean_action": bc_mean_action,
            "cql_mean_action": cql_mean_action,
            "delta_cql_minus_bc": cql_mean_action - bc_mean_action,
            "bc_distribution": bc_mean.tolist(),
            "cql_distribution": cql_mean.tolist(),
            "total_variation_bc_vs_cql": tv,
        }
        print(f"{level:<10} {bc_mean_action:>9.3f} {cql_mean_action:>9.3f} "
              f"{cql_mean_action - bc_mean_action:>11.3f} {tv:>8.3f}")
    results["comparisons"] = comparisons

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, level in zip(axes, ["id", "mild", "moderate", "severe"]):
        bc = comparisons[level]["bc_distribution"]
        cql = comparisons[level]["cql_distribution"]
        x = np.arange(5)
        w = 0.4
        ax.bar(x - w/2, bc,  width=w, label="BC",  color="steelblue")
        ax.bar(x + w/2, cql, width=w, label="CQL", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(p*100)}%" for p in np.linspace(0, 1, 5)])
        ax.set_title(f"{level.upper()}  (TV={comparisons[level]['total_variation_bc_vs_cql']:.3f})")
        ax.set_xlabel("pumping action")
        if ax is axes[0]:
            ax.set_ylabel("fraction of states")
            ax.legend()
    fig.suptitle("Action distribution: BC vs CQL across OOD severity "
                 "(mean over seeds 42/123/2024)", y=1.02)
    fig.tight_layout()
    fig.savefig(REPORTS / "ood_state_distribution.png", dpi=150,
                bbox_inches="tight")

    (REPORTS / "ood_state_distribution.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n[ood] wrote {REPORTS/'ood_state_distribution.json'}")
    print(f"[ood] wrote {REPORTS/'ood_state_distribution.png'}")


if __name__ == "__main__":
    main()