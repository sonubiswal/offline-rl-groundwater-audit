"""
uncertainty_propagation.py
Sample from the RF's held-out error distribution, inject into the RL state's
gw_level, measure policy return degradation.

Fixes vs v1:
  - Isolated RNG per rollout (no simulator-noise confound)
  - Start states drawn from the RF seed table via sample_initial_state,
    matching the scenario eval's initial-state distribution
"""
import argparse, sys
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import d3rlpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rl.synthetic_reward import SimState, step, load_reward_config
from src.rl.generate_offline_dataset import sample_initial_state

REPORTS = Path("reports")
SEEDS = [42, 123, 2024]
HORIZON = 20
N_EPISODES = 100


def load_rf_error_pool(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["actual", "predicted"])
    med = df.groupby("well_id")["actual"].median()
    df = df.merge(med.rename("well_median"), on="well_id")
    df["ratio"] = df["actual"] / df["well_median"].clip(lower=0.5)
    df = df[df["ratio"] <= 4.0]
    return (df["predicted"] - df["actual"]).to_numpy()


def rollout_return(policy, start_state, error_pool, seed, cfg, inject):
    rng = np.random.RandomState(seed)               # simulator RNG
    err_rng = np.random.RandomState(seed + 7919)    # error-injection RNG
    s = start_state
    total = 0.0
    for _ in range(HORIZON):
        obs_gw = s.gw_level
        if inject:
            obs_gw = obs_gw + float(err_rng.choice(error_pool))
        obs = np.array([obs_gw, s.recent_rainfall, s.crop_water_demand,
                        s.month_sin, s.month_cos, s.extraction_rate],
                       dtype=np.float32).reshape(1, -1)
        action = int(np.asarray(policy.predict(obs)).reshape(-1)[0])
        s, r, done = step(s, action, config=cfg, n_actions=5, rng=rng)
        total += float(r)
        if done:
            break
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals_csv", default="reports/rf_v2_noS2_validation.csv")
    ap.add_argument("--model_root", default="reports/paper_exp/frozen_models")
    ap.add_argument("--rf_table", default="reports/rf_grid_predictions.csv",
                    help="RF seed table used for initial states (same source the "
                         "scenario eval uses).")
    ap.add_argument("--gw_target", type=float, default=15.0,
                    help="Match the 15 m scenario cell.")
    args = ap.parse_args()

    errors = load_rf_error_pool(args.residuals_csv)
    print(f"[error_pool] n={len(errors)}  mean={errors.mean():+.2f}  "
          f"std={errors.std():.2f}  p05={np.percentile(errors,5):+.2f}  "
          f"p95={np.percentile(errors,95):+.2f}")

    # ---- start states from the same source the scenario eval uses ----
    rf_df = None
    if Path(args.rf_table).exists():
        rf_df = pd.read_csv(args.rf_table)
        print(f"[init] loaded rf_table {args.rf_table}: {len(rf_df)} rows")
    else:
        print(f"[init] WARN rf_table {args.rf_table} not found; using synthetic init")

    cfg = load_reward_config("config/rl_config.yaml")

    rows = []
    for algo in ("bc", "cql"):
        for seed in SEEDS:
            p = Path(args.model_root) / algo / f"{algo}_seed{seed}.d3"
            if not p.exists():
                print(f"[skip] {p}")
                continue
            policy = d3rlpy.load_learnable(str(p))

            # Sample N start states once per (algo,seed); reuse for clean+noisy
            init_rng = np.random.RandomState(seed)
            starts = []
            for _ in range(N_EPISODES):
                st = sample_initial_state(rf_df, {}, init_rng) if rf_df is not None \
                     else SimState(gw_level=args.gw_target, recent_rainfall=50.0,
                                   crop_water_demand=0.6, month_sin=0.0,
                                   month_cos=1.0, extraction_rate=0.3)
                st = replace(st, gw_level=float(args.gw_target))
                starts.append(st)

            clean_rets, noisy_rets = [], []
            for i, st in enumerate(starts):
                r_seed = seed * 10000 + i
                clean_rets.append(rollout_return(policy, st, errors, r_seed, cfg, inject=False))
                noisy_rets.append(rollout_return(policy, st, errors, r_seed, cfg, inject=True))
            rows.append({
                "policy": algo, "seed": seed,
                "clean_return": float(np.mean(clean_rets)),
                "noisy_return": float(np.mean(noisy_rets)),
                "degradation": float(np.mean(clean_rets) - np.mean(noisy_rets)),
            })
            print(f"[{algo} seed{seed}] clean={rows[-1]['clean_return']:+.3f}  "
                  f"noisy={rows[-1]['noisy_return']:+.3f}  "
                  f"Δ={rows[-1]['degradation']:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "uncertainty_propagation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, color in [("bc", "steelblue"), ("cql", "darkorange")]:
        sub = df[df["policy"] == policy]
        if len(sub):
            ax.bar([policy.upper()], [sub["degradation"].mean()],
                   color=color, alpha=0.7,
                   yerr=[sub["degradation"].std()], capsize=6)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylabel("Mean return degradation (clean − noisy)")
    ax.set_title("Phase-2 RF error propagation through BC and CQL")
    fig.tight_layout()
    fig.savefig(REPORTS / "uncertainty_propagation.png", dpi=150)
    print(f"\nwrote {REPORTS/'uncertainty_propagation.csv'}")
    print(f"wrote {REPORTS/'uncertainty_propagation.png'}")


if __name__ == "__main__":
    main()