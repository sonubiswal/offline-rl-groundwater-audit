"""
Characterize the three dataset families: action distribution, reward
distribution, episode length, state coverage, and pairwise overlap.
Produces the numbers needed to explain the family x start flip.

Run from C:\\trishna-opal:
    python revision_experiments\\mech_dataset_stats.py
"""
import json
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    "base":     REPO / "data" / "processed" / "offline_rl_dataset.h5",
    "epsilon":  REPO / "data" / "processed" / "offline_rl_dataset_epsilon.h5",
    "no_oracle":REPO / "data" / "processed" / "offline_rl_dataset_no_oracle.h5",
}
N_ACTIONS = 5
OUT = REPO / "revision_experiments" / "mech_dataset_stats.json"


def load_family(path):
    """Return dict with actions, rewards, lengths, observations."""
    acts, rews, lengths, obs_list = [], [], [], []
    with h5py.File(path, "r") as f:
        ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                        if k.startswith("observations_"))
        for i in ep_ids:
            o = np.asarray(f[f"observations_{i}"], dtype=np.float32)
            a = np.asarray(f[f"actions_{i}"]).reshape(-1).astype(np.int64)
            r = np.asarray(f[f"rewards_{i}"], dtype=np.float32).reshape(-1)
            acts.append(a)
            rews.append(r)
            lengths.append(len(o))
            obs_list.append(o)
    return {
        "actions":  np.concatenate(acts),
        "rewards":  np.concatenate(rews),
        "lengths":  np.array(lengths),
        "obs":      np.concatenate(obs_list, axis=0),
        "n_episodes": len(lengths),
    }


def summarize(name, d):
    acts = d["actions"]
    rews = d["rewards"]
    lengths = d["lengths"]
    obs = d["obs"]

    action_counts = np.bincount(acts, minlength=N_ACTIONS)
    action_props = action_counts / action_counts.sum()
    entropy = -np.sum(action_props * np.log(action_props + 1e-12))

    return {
        "n_episodes":       int(d["n_episodes"]),
        "n_transitions":    int(len(acts)),
        "episode_length":   {"mean": float(lengths.mean()),
                             "min": int(lengths.min()),
                             "max": int(lengths.max()),
                             "std": float(lengths.std())},
        "action_counts":    action_counts.tolist(),
        "action_props":     action_props.tolist(),
        "action_entropy":   float(entropy),
        "reward":           {"mean": float(rews.mean()),
                             "std": float(rews.std()),
                             "min": float(rews.min()),
                             "max": float(rews.max()),
                             "p05": float(np.percentile(rews, 5)),
                             "p95": float(np.percentile(rews, 95))},
        "obs_mean":         obs.mean(axis=0).tolist(),
        "obs_std":          obs.std(axis=0).tolist(),
        "obs_p05":          np.percentile(obs, 5, axis=0).tolist(),
        "obs_p95":          np.percentile(obs, 95, axis=0).tolist(),
    }


def main():
    families = {name: load_family(p) for name, p in DATASETS.items()}
    out = {name: summarize(name, d) for name, d in families.items()}

    # Pairwise overlap in action space
    print("=== action distribution ===")
    print(f"{'family':12s} " + " ".join(f"a{i:>6d}" for i in range(N_ACTIONS))
          + "   entropy")
    for name, s in out.items():
        print(f"{name:12s} "
              + " ".join(f"{p:6.3f}" for p in s["action_props"])
              + f"   {s['action_entropy']:.3f}")

    print("\n=== reward distribution ===")
    print(f"{'family':12s} {'mean':>8s} {'std':>8s} {'p05':>8s} "
          f"{'p95':>8s} {'min':>8s} {'max':>8s}")
    for name, s in out.items():
        r = s["reward"]
        print(f"{name:12s} {r['mean']:8.3f} {r['std']:8.3f} "
              f"{r['p05']:8.3f} {r['p95']:8.3f} "
              f"{r['min']:8.3f} {r['max']:8.3f}")

    print("\n=== episode length ===")
    print(f"{'family':12s} {'mean':>8s} {'min':>5s} {'max':>5s} {'n_eps':>7s}")
    for name, s in out.items():
        L = s["episode_length"]
        print(f"{name:12s} {L['mean']:8.2f} {L['min']:5d} {L['max']:5d} "
              f"{s['n_episodes']:7d}")

    print("\n=== observation coverage (first 6 dims) ===")
    for name, s in out.items():
        print(f"{name:12s} mean={np.round(s['obs_mean'], 3).tolist()}")

    # Pairwise L1 distance between action distributions
    print("\n=== pairwise action-distribution L1 ===")
    names = list(out)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pa = np.array(out[a]["action_props"])
            pb = np.array(out[b]["action_props"])
            l1 = float(np.abs(pa - pb).sum())
            print(f"  {a:10s} vs {b:10s}  L1 = {l1:.3f}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()