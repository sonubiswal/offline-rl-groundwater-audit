"""Translate CQL Q-values into a plain-language recommendation.

Methodology reference: reports/explainability_methodology.md
Action space source:  src/rl/synthetic_reward.py (line 148:
pumping_fraction = action / (n_actions - 1), n_actions = 5)

The five actions are discrete pumping intensities:

  action 0 -> 0.00 of maximum pumping
  action 1 -> 0.25 of maximum pumping
  action 2 -> 0.50 of maximum pumping
  action 3 -> 0.75 of maximum pumping
  action 4 -> 1.00 of maximum pumping

Crop revenue in the reward function caps at the crop's water demand, so
pumping past demand yields no extra revenue but still incurs the
depletion penalty. The plain-language text reflects that: action 4 is not
"more is better", it is "maximum extraction, justified only when demand
is high".

Confidence rule
---------------
Compute gap = Q(best_action) - Q(second_best_action).
  gap >= tau_high -> confidence = "high"
  gap >= tau_low  -> confidence = "medium"
  gap  < tau_low  -> confidence = "low"

Defaults tau_high = 0.5, tau_low = 0.1 are heuristic, calibrated against
the observed reward scale: per-episode return on the no_oracle_alpha1
family is about 9.4 over 20 steps, so per-step reward is about 0.47. A
Q-gap of 0.5 is roughly one step of reward; a gap of 0.1 is roughly one
fifth of a step. The module uses only the Q-gap, not absolute Q-values,
so it is scale-invariant to the critic's overall offset.

This module does not validate the reward function. If the critic is
miscalibrated, the advice inherits that error. See
reports/explainability_methodology.md, section "What this module cannot
do".
"""
from __future__ import annotations


ACTION_LABELS = {
    0: "no pumping this quarter",
    1: "pump 25 percent of maximum extraction",
    2: "pump 50 percent of maximum extraction",
    3: "pump 75 percent of maximum extraction",
    4: "pump 100 percent of maximum extraction (justified only when crop "
       "demand is high; otherwise the depletion penalty dominates)",
}

TAU_HIGH_DEFAULT = 0.5
TAU_LOW_DEFAULT = 0.1


def translate_to_advice(
    q_values: dict[int, float],
    tau_high: float = TAU_HIGH_DEFAULT,
    tau_low: float = TAU_LOW_DEFAULT,
) -> dict:
    """Map action -> Q-value into a plain-language recommendation.

    Parameters
    ----------
    q_values : dict[int, float]
        Mapping from action index (0..4) to the critic's Q-value.
    tau_high, tau_low : float
        Confidence thresholds. See module docstring.

    Returns
    -------
    dict with keys:
        recommended_action : int
        plain_language_text : str
        confidence_level : {"high", "medium", "low"}
        q_gap : float
        alternatives : list[dict] with keys action, q_value, delta_from_best
    """
    if not q_values:
        raise ValueError("q_values is empty")
    if len(q_values) < 2:
        raise ValueError(
            "q_values must contain at least 2 actions to compute a confidence gap"
        )

    ranked = sorted(q_values.items(), key=lambda kv: kv[1], reverse=True)
    best_action, best_q = ranked[0]
    second_action, second_q = ranked[1]
    gap = best_q - second_q

    if gap >= tau_high:
        confidence = "high"
    elif gap >= tau_low:
        confidence = "medium"
    else:
        confidence = "low"

    base = ACTION_LABELS.get(best_action, f"take action {best_action}")
    text = f"Recommended: {base}."

    if confidence == "low":
        text += (
            " Confidence is low: alternative actions score almost as well,"
            " so treat this as a guide rather than a directive."
        )
    elif confidence == "medium":
        text += " Confidence is moderate."
    else:
        text += " Confidence is high for this state."

    alternatives = [
        {
            "action": a,
            "q_value": q,
            "delta_from_best": q - best_q,
        }
        for a, q in ranked[1:4]
    ]

    return {
        "recommended_action": best_action,
        "plain_language_text": text,
        "confidence_level": confidence,
        "q_gap": gap,
        "alternatives": alternatives,
    }
