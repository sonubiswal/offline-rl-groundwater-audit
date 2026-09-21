"""
a4a5_sensitivity.py

3x3 sweep of A4 (sustainability_threshold_m) x A5 (drawdown coefficient).
Reuses frozen BC/CQL checkpoints at reports/paper_exp/frozen_models/.
Same eval protocol as compare_policies / mode_scenario:
    N episodes per seed, eval seed 9999 + sd, paired bootstrap seed 2026.

Answers LM §29 / VM §54.2: how stable is the ~+20 return-unit CQL-BC gap
at gw=25 under A4/A5 perturbations?

Writes: reports/a4a5_sensitivity.json
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

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "rl"))

import d3rlpy

import run_paper_experiments as rpe
from src.rl.synthetic_reward import SimState


# ------------------------------------------------ config
FROZEN_ROOT    = Path("reports/paper_exp/frozen_models")
BC_DIR         = FROZEN_ROOT / "bc"
CQL_DIR        = FROZEN_ROOT / "cql"
CONFIG_PATH    = Path("config/rl_config.yaml")
RF_TABLE       = Path("reports/rf_grid_predictions.csv")
NORM_MANIFEST  = FROZEN_ROOT / "normalization_manifest.json"
OUT_JSON       = Path("reports/a4a5_sensitivity.json")

SEEDS         = [42, 123, 2024]
N_EPISODES    = 200                 # reduced for sweep; still >> noiseless
TRAJ_LEN      = 20
N_ACTIONS     = 5
EVAL_SEED     = 9999
BOOT_SEED     = 2026
N_BOOTSTRAP   = 5000

A4_GRID       = [10.0, 15.0, 25.0]  # canonical A4 = 15.0
A5_GRID       = [1.0,  2.0,  3.0]   # canonical A5 = 2.0
GW_LEVELS     = [25.0]              # the depth the finding is about
SCENARIOS     = ["normal", "drought", "high"]


# ------------------------------------------------ parameterized scenario_step
def scenario_step_p(state, action, config, n_actions, rng, scn, a5_coef):
    """Fork of run_paper_experiments.scenario_step with A5 as a parameter."""
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
    supply  = rpe.domestic_supply_score(next_gw)
    penalty = rpe.depletion_penalty(next_gw,
                                    config["sustainability_threshold_m"],
                                    config.get("depletion_penalty_exponent", 2.0))
    reward = (config["w1_crop_revenue"] * revenue
              + config["w3_domestic_supply"] * supply
              - config["w2_depletion_penalty"] * penalty)
    return ns, float(reward), float(revenue), float(supply), float(penalty), False


# ------------------------------------------------ model → action fn
def make_action_fn(algo, obs_mean, obs_std):
    def fn(state_vec):
        x = ((np.asarray(state_vec, np.float32) - obs_mean) / obs_std).reshape(1, 6)
        return int(np.asarray(algo.predict(x)).reshape(-1)[0])
    return fn


def rollout(action_fn, init_states, reward_cfg, scn, a5_coef, seed):
    rng = np.random.RandomState(seed)
    out = np.zeros(len(init_states))
    for i, s0 in enumerate(init_states):
        s = s0
        tot = 0.0
        for _ in range(TRAJ_LEN):
            a = action_fn(s.to_array())
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


# ------------------------------------------------ main
def main():
    print("[a4a5] loading config, RF table, normalization")
    cfg = yaml.safe_load(open(CONFIG_PATH))
    base_reward_cfg = dict(cfg["reward"])
    rf_df = pd.read_csv(RF_TABLE)
    norm = json.load(open(NORM_MANIFEST))
    obs_mean = np.asarray(norm["mean"], np.float32)
    obs_std  = np.asarray(norm["std"],  np.float32)

    print(f"[a4a5] frozen BC dir:  {BC_DIR}")
    print(f"[a4a5] frozen CQL dir: {CQL_DIR}")
    bc_models  = {sd: d3rlpy.load_learnable(str(BC_DIR  / f"bc_seed{sd}.d3"),  device="cpu") for sd in SEEDS}
    cql_models = {sd: d3rlpy.load_learnable(str(CQL_DIR / f"cql_seed{sd}.d3"), device="cpu") for sd in SEEDS}
    bc_fn  = {sd: make_action_fn(bc_models[sd],  obs_mean, obs_std) for sd in SEEDS}
    cql_fn = {sd: make_action_fn(cql_models[sd], obs_mean, obs_std) for sd in SEEDS}

    missing_scn = [s for s in SCENARIOS if s not in rpe.SCENARIOS]
    if missing_scn:
        print(f"[a4a5] WARNING: scenarios not found in rpe.SCENARIOS: {missing_scn}")
        print(f"[a4a5] available: {list(rpe.SCENARIOS.keys())}")

    results = []
    for a4, a5, scn_name, gw in itertools.product(A4_GRID, A5_GRID, SCENARIOS, GW_LEVELS):
        if scn_name not in rpe.SCENARIOS:
            continue
        reward_cfg = dict(base_reward_cfg)
        reward_cfg["sustainability_threshold_m"] = a4
        scn = rpe.SCENARIOS[scn_name]

        pooled_bc, pooled_cql = [], []
        for sd in SEEDS:
            rng = np.random.RandomState(EVAL_SEED + sd)
            init = rpe.make_init_states(rf_df, N_EPISODES, rng, gw)
            b = rollout(bc_fn[sd],  init, reward_cfg, scn, a5, EVAL_SEED + sd)
            c = rollout(cql_fn[sd], init, reward_cfg, scn, a5, EVAL_SEED + sd)
            pooled_bc.append(b); pooled_cql.append(c)

        bc_all  = np.concatenate(pooled_bc)
        cql_all = np.concatenate(pooled_cql)
        obs, lo, hi = paired_bootstrap(cql_all, bc_all)

        row = {
            "a4": a4, "a5": a5, "scenario": scn_name, "gw": gw,
            "bc_return": float(bc_all.mean()),
            "cql_return": float(cql_all.mean()),
            "cql_minus_bc": obs, "ci_lower": lo, "ci_upper": hi,
            "ci_contains_zero": bool(lo <= 0 <= hi),
        }
        results.append(row)
        tag = "SIG" if not row["ci_contains_zero"] else "ns"
        print(f"  a4={a4:5.1f} a5={a5:4.1f} scn={scn_name:7s} gw={gw:4.1f}  "
              f"BC={row['bc_return']:+8.3f}  CQL={row['cql_return']:+8.3f}  "
              f"CQL-BC={obs:+8.3f}  [{lo:+.3f},{hi:+.3f}]  {tag}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n[a4a5] wrote {OUT_JSON}  ({len(results)} cells)")
    print(f"[a4a5] canonical cell (a4=15, a5=2, normal, gw=25):")
    for r in results:
        if r["a4"] == 15.0 and r["a5"] == 2.0 and r["scenario"] == "normal":
            print(f"        CQL-BC = {r['cql_minus_bc']:+.4f}  "
                  f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]")


if __name__ == "__main__":
    main()