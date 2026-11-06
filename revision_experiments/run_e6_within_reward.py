"""
E6 — within-reward sensitivity of the CQL-BC gap at gw=25.

Purpose: the paper's A4/A5 sweep (Table XIII) crosses reward functions,
so its 227x range is not a meaning-preserving sensitivity measure. E6
holds the reward fixed (A4=15, weights (1,2,0.5)) and varies A5 only.
The resulting ratio is a real single-estimand sensitivity.

Protocol
--------
- Reward weights: (w1, w2, w3) = (1, 2, 0.5)
- A4 = 15.0 fixed
- A5 in {1.0, 1.5, 2.0, 2.5, 3.0}
- gw = 25.0, three scenarios (normal, drought, high)
- 200 episodes per seed, seeds 42/123/2024, paired bootstrap seed 2026
- Policies loaded via src.rl.policy_io.load_policy_for_eval
  (frozen_models checkpoints, dataset_normal_train.h5)

Writes
------
revision_experiments/e6_within_reward.json
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "rl"))

from src.rl.policy_io import load_policy_for_eval

import run_paper_experiments as rpe
from src.rl.synthetic_reward import SimState


FROZEN_ROOT = REPO / "reports" / "paper_exp" / "frozen_models"
BC_DIR = FROZEN_ROOT / "bc"
CQL_DIR = FROZEN_ROOT / "cql"
FROZEN_DS = REPO / "reports" / "paper_exp" / "dataset_normal_train.h5"
CONFIG_PATH = REPO / "config" / "rl_config.yaml"
RF_TABLE = REPO / "reports" / "rf_grid_predictions.csv"
OUT_JSON = REPO / "revision_experiments" / "e6_within_reward.json"

SEEDS = [42, 123, 2024]
N_EPISODES = 200
TRAJ_LEN = 20
N_ACTIONS = 5
EVAL_SEED = 9999
BOOT_SEED = 2026
N_BOOTSTRAP = 5000

A4_FIXED = 15.0
A5_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
GW = 25.0
SCENARIOS = ["normal", "drought", "high"]


def scenario_step_p(state, action, config, n_actions, rng, scn, a5_coef):
    """Identical to the A4/A5 script (see a4a5_sensitivity.py)."""
    demand_eff = float(np.clip(state.crop_water_demand * scn["demand_factor"], 0.0, 1.0))
    rain_scaled = max(0.0, state.recent_rainfall * scn["recharge_factor"])
    pumping_fraction = action / (n_actions - 1)
    drawdown = pumping_fraction * a5_coef
    recharge = 0.15 * rain_scaled / 100.0
    noise = rng.normal(0, 0.1)
    next_gw = max(0.0, state.gw_level + drawdown - recharge + noise)
    next_demand = float(np.clip(state.crop_water_demand + rng.normal(0, 0.05), 0.0, 1.0))
    next_ext = 0.7 * state.extraction_rate + 0.3 * pumping_fraction
    next_rain = max(0.0, state.recent_rainfall + rng.normal(0, 20))
    ns = SimState(gw_level=next_gw, recent_rainfall=float(next_rain),
                  crop_water_demand=next_demand,
                  month_sin=state.month_sin, month_cos=state.month_cos,
                  extraction_rate=next_ext)
    revenue = rpe.crop_revenue_proxy(action, demand_eff, n_actions)
    supply = rpe.domestic_supply_score(next_gw)
    penalty = rpe.depletion_penalty(next_gw,
                                    config["sustainability_threshold_m"],
                                    config.get("depletion_penalty_exponent", 2.0))
    reward = (config["w1_crop_revenue"] * revenue
              + config["w3_domestic_supply"] * supply
              - config["w2_depletion_penalty"] * penalty)
    return ns, float(reward), float(revenue), float(supply), float(penalty), False


def rollout(scaled_policy, init_states, reward_cfg, scn, a5_coef, seed):
    rng = np.random.RandomState(seed)
    out = np.zeros(len(init_states))
    for i, s0 in enumerate(init_states):
        s = s0
        tot = 0.0
        for _ in range(TRAJ_LEN):
            a = scaled_policy.action(s.to_array())
            ns, r, *_ = scenario_step_p(s, a, reward_cfg, N_ACTIONS, rng, scn, a5_coef)
            ang = np.arctan2(s.month_sin, s.month_cos) + (2 * np.pi / 4)
            ns = replace(ns, month_sin=float(np.sin(ang)), month_cos=float(np.cos(ang)))
            tot += r
            s = ns
        out[i] = tot
    return out


def paired_bootstrap(a, b, n=N_BOOTSTRAP, seed=BOOT_SEED):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    bd = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))


def main():
    print("[E6] loading config, RF table, frozen-model dataset")
    cfg = yaml.safe_load(open(CONFIG_PATH))
    base_reward_cfg = dict(cfg["reward"])
    # Fix reward weights to canonical.
    base_reward_cfg["w1_crop_revenue"] = 1.0
    base_reward_cfg["w2_depletion_penalty"] = 2.0
    base_reward_cfg["w3_domestic_supply"] = 0.5
    base_reward_cfg["sustainability_threshold_m"] = A4_FIXED

    rf_df = pd.read_csv(RF_TABLE)

    bc_models = {sd: load_policy_for_eval(BC_DIR / f"bc_seed{sd}.d3", FROZEN_DS)
                 for sd in SEEDS}
    cql_models = {sd: load_policy_for_eval(CQL_DIR / f"cql_seed{sd}.d3", FROZEN_DS)
                  for sd in SEEDS}
    print(f"[E6] scaler applied: mean[0]={bc_models[SEEDS[0]].obs_mean[0]:.6f}")

    results = []
    for a5, scn_name in itertools.product(A5_GRID, SCENARIOS):
        if scn_name not in rpe.SCENARIOS:
            continue
        scn = rpe.SCENARIOS[scn_name]
        reward_cfg = dict(base_reward_cfg)

        pooled_bc, pooled_cql = [], []
        for sd in SEEDS:
            rng = np.random.RandomState(EVAL_SEED + sd)
            init = rpe.make_init_states(rf_df, N_EPISODES, rng, GW)
            b = rollout(bc_models[sd], init, reward_cfg, scn, a5, EVAL_SEED + sd)
            c = rollout(cql_models[sd], init, reward_cfg, scn, a5, EVAL_SEED + sd)
            pooled_bc.append(b)
            pooled_cql.append(c)

        bc_all = np.concatenate(pooled_bc)
        cql_all = np.concatenate(pooled_cql)
        obs, lo, hi = paired_bootstrap(cql_all, bc_all)

        results.append({
            "a4": A4_FIXED, "a5": a5, "scenario": scn_name, "gw": GW,
            "bc_return": float(bc_all.mean()),
            "cql_return": float(cql_all.mean()),
            "cql_minus_bc": obs, "ci_lower": lo, "ci_upper": hi,
            "ci_contains_zero": bool(lo <= 0 <= hi),
        })
        tag = "SIG" if not (lo <= 0 <= hi) else "ns"
        print(f"  a5={a5:4.1f}  scn={scn_name:7s}  "
              f"BC={bc_all.mean():+8.3f}  CQL={cql_all.mean():+8.3f}  "
              f"CQL-BC={obs:+8.3f}  [{lo:+.3f},{hi:+.3f}]  {tag}")

    # Per-scenario min/max and ratio, holding scenario fixed.
    print()
    print("[E6] within-reward sensitivity (A5 varies, reward fixed):")
    ratio_summary = {}
    for scn_name in SCENARIOS:
        rows = [r for r in results if r["scenario"] == scn_name]
        vals = [r["cql_minus_bc"] for r in rows]
        vmin, vmax = min(vals), max(vals)
        ratio = vmax / vmin if vmin > 0 else float("inf")
        ratio_summary[scn_name] = {
            "min": vmin, "max": vmax, "ratio": ratio,
        }
        print(f"  scenario={scn_name:7s}  min={vmin:+.4f}  max={vmax:+.4f}  "
              f"ratio={ratio:.2f}x")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "protocol": {
            "a4_fixed": A4_FIXED,
            "a5_grid": A5_GRID,
            "reward_weights": {"w1": 1.0, "w2": 2.0, "w3": 0.5},
            "gw": GW,
            "scenarios": SCENARIOS,
            "n_episodes": N_EPISODES,
            "seeds": SEEDS,
        },
        "cells": results,
        "ratio_summary": ratio_summary,
    }, indent=2))
    print(f"\n[E6] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()