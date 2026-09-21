"""
sustainability_vs_ag_tradeoff.py
Decompose the BC and CQL rollouts into their reward components.

For each (w1, w2, w3) configuration, roll the trained policy forward and
record:
  - mean crop_revenue_proxy per step (agricultural benefit)
  - mean depletion_penalty per step (groundwater stress)

Output: reports/tradeoff_decomposition.csv + .png

Read-only on all artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import d3rlpy

# Import the reward-component primitives directly. No simulator rewrite.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rl.synthetic_reward import (
    SimState, step, load_reward_config,
    crop_revenue_proxy, depletion_penalty,
)

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# --- edit these to match your actual rollout setup ---
DEFAULT_START = dict(
    gw_level=15.0,
    recent_rainfall=50.0,
    crop_water_demand=0.6,
    month_sin=0.0, month_cos=1.0,
    extraction_rate=0.3,
)
HORIZON = 20
N_EPISODES = 100
SEEDS = [42, 123, 2024]

# The seven (w1, w2, w3) configs from §47.
WEIGHT_CONFIGS = [
    (1.0, 0.0, 0.0),
    (1.0, 2.0, 0.0),
    (0.0, 2.0, 0.5),
    (1.0, 0.0, 0.5),
    (1.0, 1.0, 0.5),
    (1.0, 4.0, 0.5),
    (1.0, 2.0, 0.5),
]


def rollout_components(policy, start_state, reward_cfg, rng):
    """Roll one episode, return arrays of per-step crop_revenue and depletion_penalty."""
    s = SimState(**start_state)
    revenues, penalties = [], []
    for _ in range(HORIZON):
        obs = np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                        s.month_sin, s.month_cos, s.extraction_rate],
                       dtype=np.float32).reshape(1, -1)
        action = int(np.asarray(policy.predict(obs)).reshape(-1)[0])
        s_next, _, done = step(s, action, config=reward_cfg, n_actions=5, rng=rng)
        rev = crop_revenue_proxy(action, float(s.crop_water_demand), n_actions=5)
        pen = depletion_penalty(float(s_next.gw_level),
                                reward_cfg["sustainability_threshold_m"],
                                exponent=reward_cfg.get("depletion_penalty_exponent", 2.0))
        revenues.append(rev)
        penalties.append(pen)
        s = s_next
        if done:
            break
    return np.array(revenues), np.array(penalties)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="reports/paper_exp/frozen_models")
    ap.add_argument("--out_csv", default=str(REPORTS / "tradeoff_decomposition.csv"))
    ap.add_argument("--out_png", default=str(REPORTS / "tradeoff_decomposition.png"))
    args = ap.parse_args()

    root = Path(args.model_root)
    policies = []
    for algo in ("bc", "cql"):
        for seed in SEEDS:
            p = root / algo / f"{algo}_seed{seed}.d3"
            if p.exists():
                policies.append((algo, seed, d3rlpy.load_learnable(str(p))))
            else:
                print(f"[warn] missing {p}")

    if not policies:
        raise SystemExit("no policies found; check --model_root")

    rows = []
    for (w1, w2, w3) in WEIGHT_CONFIGS:
        cfg = load_reward_config("config/rl_config.yaml").copy()
        cfg["w1_crop_revenue"] = w1
        cfg["w2_depletion_penalty"] = w2
        cfg["w3_domestic_supply"] = w3

        for algo, seed, policy in policies:
            rng = np.random.RandomState(seed)
            all_rev, all_pen = [], []
            for ep in range(N_EPISODES):
                r, p = rollout_components(policy, DEFAULT_START, cfg, rng)
                all_rev.append(r.mean())
                all_pen.append(p.mean())
            rows.append({
                "w1": w1, "w2": w2, "w3": w3,
                "policy": algo, "seed": seed,
                "mean_crop_revenue": float(np.mean(all_rev)),
                "mean_depletion_penalty": float(np.mean(all_pen)),
            })
            print(f"[{algo} seed{seed}] w=({w1},{w2},{w3})  "
                  f"rev={np.mean(all_rev):+.3f}  pen={np.mean(all_pen):+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")

    # Plot: mean penalty (x) vs mean revenue (y), one marker per (config, policy)
    fig, ax = plt.subplots(figsize=(8, 6))
    for policy, marker, color in [("bc", "o", "steelblue"), ("cql", "s", "darkorange")]:
        sub = df[df["policy"] == policy]
        # Average across seeds within each config
        agg = sub.groupby(["w1", "w2", "w3"]).agg(
            rev=("mean_crop_revenue", "mean"),
            pen=("mean_depletion_penalty", "mean"),
        ).reset_index()
        ax.scatter(agg["pen"], agg["rev"], marker=marker, s=80, label=policy.upper(),
                   color=color, alpha=0.8)
        for _, r in agg.iterrows():
            ax.annotate(f"({r['w1']},{r['w2']},{r['w3']})",
                        (r["pen"], r["rev"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("Mean depletion penalty per step (groundwater stress)")
    ax.set_ylabel("Mean crop revenue per step (agricultural benefit)")
    ax.set_title("Sustainability vs agricultural-benefit trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()