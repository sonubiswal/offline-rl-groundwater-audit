"""Tests for src/explainability/qvalue_translation.py.

Coverage:
  - clear winner -> high confidence
  - near tie -> low confidence
  - mid gap -> medium confidence
  - alternatives ranked by Q
  - empty input raises
  - single-action input raises
  - text comes from ACTION_LABELS, not hardcoded
  - confidence is derived from q_gap, not from action identity
"""
import pytest

from src.explainability.qvalue_translation import (
    ACTION_LABELS,
    translate_to_advice,
)


def test_clear_winner_high_confidence():
    out = translate_to_advice({0: 1.0, 1: 1.5, 2: 3.0, 3: 2.0, 4: 0.5})
    assert out["recommended_action"] == 2
    assert out["confidence_level"] == "high"
    assert out["q_gap"] == pytest.approx(1.0)


def test_near_tie_low_confidence():
    out = translate_to_advice({0: 1.00, 1: 1.05, 2: 1.02, 3: 0.99, 4: 0.98})
    assert out["confidence_level"] == "low"
    assert out["q_gap"] == pytest.approx(0.03)
    assert "low" in out["plain_language_text"].lower()


def test_medium_confidence():
    out = translate_to_advice({0: 1.0, 1: 1.3, 2: 1.1, 3: 0.9, 4: 0.8})
    assert out["confidence_level"] == "medium"
    assert 0.1 <= out["q_gap"] < 0.5


def test_alternatives_ranked_by_q():
    out = translate_to_advice({0: 1.0, 1: 2.0, 2: 3.0, 3: 2.5, 4: 0.5})
    alt_actions = [a["action"] for a in out["alternatives"]]
    assert alt_actions[0] == 3   # highest alternative
    assert alt_actions[1] == 1
    assert alt_actions[2] == 0
    # deltas are all negative (below the best)
    for a in out["alternatives"]:
        assert a["delta_from_best"] < 0


def test_empty_q_raises():
    with pytest.raises(ValueError):
        translate_to_advice({})


def test_single_action_raises():
    with pytest.raises(ValueError):
        translate_to_advice({2: 3.0})


def test_text_uses_action_labels():
    out = translate_to_advice({0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0})
    assert out["recommended_action"] == 0
    assert ACTION_LABELS[0] in out["plain_language_text"]


def test_confidence_is_from_gap_not_template():
    hi = translate_to_advice({0: 0.0, 4: 5.0})
    lo = translate_to_advice({0: 4.99, 4: 5.0})
    assert hi["confidence_level"] == "high"
    assert lo["confidence_level"] == "low"
    assert hi["recommended_action"] == lo["recommended_action"] == 4
