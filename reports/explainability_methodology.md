# Explainability Methodology

Module: src/explainability/qvalue_translation.py
Tests:  tests/test_qvalue_translation.py
Report section: 9 (Explainability)

## What the module does

The explainability layer translates Q-values into plain-language advice.
It does not invent advice. It reports what the critic already believes,
in English, with a stated confidence.

Given a dictionary {action: Q_value} from the trained CQL critic,
translate_to_advice returns:

  recommended_action   int, the action with the highest Q-value
  confidence_level     "high", "medium", or "low"
  plain_language_text  one sentence naming the recommended action and the
                       confidence, with an explicit caveat when confidence
                       is low
  q_gap                float, Q(best) - Q(second_best)
  alternatives         list of the next three actions with Q-value and
                       delta from best

## Why the Q-gap is the confidence signal

The confidence level is derived entirely from the Q-gap between the best
and second-best action, never from action identity or from a template.
A large gap means the critic distinguishes the best action clearly; a
small gap means it does not, and the recommendation is a coin-flip that
should not be presented as decisive.

The thresholds are:

  q_gap >= 0.5  -> high
  q_gap >= 0.1  -> medium
  q_gap  < 0.1  -> low

Calibration note: per-episode return on the no_oracle_alpha1 family is
approximately 9.4 over 20 steps, so per-step reward is approximately 0.47.
A Q-gap of 0.5 is roughly one step of reward difference; a gap of 0.1 is
roughly one fifth of a step. These are heuristic defaults. The module
takes tau_high and tau_low as keyword arguments, so a user can adjust them
after inspecting the actual distribution of Q-values on their state set.

## Why the action labels came from the code, not from a template

The five actions are pumping intensities, not arbitrary discrete choices.
The mapping from action index to pumping fraction is set by line 148 of
src/rl/synthetic_reward.py:

  pumping_fraction = action / (n_actions - 1),  n_actions = 5

So the five actions are 0.00, 0.25, 0.50, 0.75, 1.00 of maximum pumping.
The labels in ACTION_LABELS reflect this exactly. Action 4 is not "more
is better": crop revenue caps at the crop's demand, so pumping past
demand gives no extra revenue but still incurs the depletion penalty.
The label text says so.

## What this module cannot do

1. It cannot tell you whether the critic is correct. It reads Q-values.
   If the critic over-values a harmful action, the advice says so.

2. It cannot validate the reward function. The reward function is a
   stylised bucket model with assumptions A1-A6 (see
   reports/open_items.md). The advice inherits all six.

3. It cannot guarantee that the recommended action is safe for the
   aquifer. Both the reward function and the simulator are stylised.
   See reports/final_report_claims.md Claim 4 for the honest scope of
   the RL layer.

4. It cannot replace held-out validation. The advice is not validated
   against real farmer outcomes, because no such dataset exists.

## Report integration

Section 9 of the report cites this document and quotes the confidence
rule. Section 10 (Limitations) cites items 1-4 above. No claim in
the report should present the advice as more than a translation of a
stylised critic's preferences.
