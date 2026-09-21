"""
src/rl/synthetic_reward.py

Phase 4: water-balance simulator for synthetic offline RL data generation.
Since no real farmer pumping logs exist, this module implements
step(state, action) -> (next_state, reward, done), simulating how a
pumping decision affects groundwater level and computing a reward that
balances crop revenue, domestic supply adequacy, and a depletion penalty.

CRITICAL FRAMING: this is a SYNTHETIC ground-truth layer, separate from
and additional to the anti-circularity discipline applied to GRACE/CGWB
data (validation_methodology.md Section 3). No amount of RL training
rigor makes the reward function itself "true" -- it encodes assumptions
that must be stated explicitly, not discovered by a reader. Every
assumption below is numbered so it can be referenced directly in
reports/limitations.md's Phase 4 section.

ASSUMPTION LOG (numbered for direct citation):
  A1. crop_water_demand is a simple seasonal proxy (higher in the
      pre-monsoon/summer months), NOT derived from actual crop-type maps,
      irrigation records, or FAO crop-coefficient tables. A real
      implementation would use per-crop water requirement data; this
      proxy exists only to give the simulator a plausible seasonal signal
      to react to.
  A2. crop_revenue_proxy assumes revenue scales monotonically with
      pumping intensity up to the point demand is met, then flattens --
      i.e. it assumes farmers only benefit from water up to their crop's
      actual need, not indefinitely. No real price/yield data backs the
      specific proxy formula's magnitude.
  A3. domestic_supply_score assumes a fixed critical groundwater depth
      (independent of location) below which domestic water access is
      considered compromised. Real domestic supply adequacy depends on
      well depth, pump capacity, and local infrastructure -- none of
      which this simulator models.
  A4. depletion_penalty uses a single sustainability_threshold_m for the
      entire study region (config/rl_config.yaml). Real sustainable
      yield varies by aquifer type and geology (see
      validation_methodology.md Section 19 on the not-yet-tested
      hydrogeology covariate direction) -- this is a simplification, not
      a hydrogeologically validated threshold.
  A5. The relationship between action (pumping intensity) and next
      gw_level change is a simple linear-plus-recharge model, not a
      calibrated aquifer response function. Real aquifer drawdown/recovery
      dynamics are nonlinear and depend on transmissivity and storage
      coefficient, which are not modeled here.
  A6. Recharge from rainfall is modeled as a fixed fraction of
      recent_rainfall, not a physically calibrated infiltration model
      (which would itself depend on the soil-type covariate flagged as
      missing in limitations.md Section 9).

These assumptions do not need to be "fixed" before Phase 4 proceeds --
the roadmap's own risk register accepts this as a known, documented
limitation of using synthetic data for offline RL. The requirement is
that every assumption is numbered and stated, not that it be eliminated.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import yaml


@dataclass
class SimState:
    gw_level: float           # meters below ground (higher = deeper/worse)
    recent_rainfall: float    # mm, 3-month rolling proxy
    crop_water_demand: float  # 0-1 normalized proxy (A1)
    month_sin: float
    month_cos: float
    extraction_rate: float    # 0-1 normalized recent pumping intensity

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SimState":
        return cls(*[float(x) for x in arr])


def load_reward_config(config_path: str = "config/rl_config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["reward"]


def crop_revenue_proxy(action: int, crop_water_demand: float, n_actions: int = 5) -> float:
    """A1/A2: revenue rises with pumping up to the point demand is met,
    then flattens (extra pumping beyond crop need adds no further proxy
    revenue in this simplified model). action is 0..n_actions-1."""
    pumping_fraction = action / (n_actions - 1)  # 0.0 to 1.0
    water_supplied = min(pumping_fraction, crop_water_demand)
    return water_supplied  # normalized 0-1; scaled by w1 in the caller


def domestic_supply_score(gw_level: float, critical_depth_m: float = 25.0) -> float:
    """A3: domestic supply considered adequate (score=1) above a fixed
    critical depth, degrading linearly to 0 as depth approaches 2x that
    threshold. critical_depth_m is intentionally distinct from the RL
    sustainability_threshold_m (A4) -- domestic access failure and
    long-run aquifer depletion are different phenomena, kept as separate
    parameters even though both are simplifications."""
    if gw_level <= critical_depth_m:
        return 1.0
    score = 1.0 - (gw_level - critical_depth_m) / critical_depth_m
    return float(np.clip(score, 0.0, 1.0))


def depletion_penalty(next_gw_level: float, threshold_m: float, exponent: float = 2.0) -> float:
    """A4: zero penalty at or above the sustainability threshold (i.e.
    shallower than the threshold depth), growing as a power function of
    how far below (deeper than) the threshold the level has dropped."""
    if next_gw_level <= threshold_m:
        return 0.0
    excess = (next_gw_level - threshold_m) / threshold_m  # normalized overshoot
    return float(excess ** exponent)


def step(state: SimState, action: int, config: Optional[dict] = None,
         n_actions: int = 5, rng: Optional[np.random.RandomState] = None) -> tuple[SimState, float, bool]:
    """
    Simulates one quarterly timestep.

    Parameters
    ----------
    state : current SimState
    action : int, 0..n_actions-1, pumping intensity level
    config : reward config dict (from rl_config.yaml's "reward" section);
        loaded fresh from disk if not provided (inefficient in a tight
        loop -- callers generating many trajectories should load once
        and pass it in)
    rng : optional RandomState for reproducible stochastic recharge noise

    Returns
    -------
    (next_state, reward, done) -- done is always False here; episode
    length is controlled externally by the trajectory generator
    (config/rl_config.yaml's simulation.trajectory_length), not by any
    terminal condition in the simulator itself.
    """
    if config is None:
        config = load_reward_config()
    if rng is None:
        rng = np.random.RandomState()

    pumping_fraction = action / (n_actions - 1)

    # A5: next-level change = pumping drawdown - rainfall recharge (A6) + small noise
    drawdown = pumping_fraction * 2.0        # up to 2m drawdown per quarter at max pumping
    recharge = 0.15 * state.recent_rainfall / 100.0  # A6: crude mm->m recharge fraction
    noise = rng.normal(0, 0.1)

    next_gw_level = max(0.0, state.gw_level + drawdown - recharge + noise)

    # A1: crude seasonal demand proxy -- placeholder pattern, not derived
    # from crop calendars. Real month should be set externally by the
    # trajectory generator (see generate_offline_dataset.py), which knows
    # the actual calendar date; this fallback just perturbs demand mildly.
    next_demand = float(np.clip(state.crop_water_demand + rng.normal(0, 0.05), 0.0, 1.0))

    next_extraction_rate = 0.7 * state.extraction_rate + 0.3 * pumping_fraction

    next_state = SimState(
        gw_level=next_gw_level,
        recent_rainfall=max(0.0, state.recent_rainfall + rng.normal(0, 20)),  # crude AR(1)-ish drift
        crop_water_demand=next_demand,
        month_sin=state.month_sin,   # calendar fields are overwritten by the
        month_cos=state.month_cos,   # trajectory generator with true values -- see generate_offline_dataset.py
        extraction_rate=next_extraction_rate,
    )

    revenue = crop_revenue_proxy(action, state.crop_water_demand, n_actions)
    supply = domestic_supply_score(next_gw_level)
    penalty = depletion_penalty(next_gw_level, config["sustainability_threshold_m"],
                                 config.get("depletion_penalty_exponent", 2.0))

    reward = (
        config["w1_crop_revenue"] * revenue
        + config["w3_domestic_supply"] * supply
        - config["w2_depletion_penalty"] * penalty
    )

    done = False
    return next_state, float(reward), done