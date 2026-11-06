"""Observation-standardisation regression tests.

Run from the repository root:
    pytest tests/test_scaler_pipeline.py -v -rs

Strict archive audit (fail instead of skip for missing archive/datasets):
    SCALER_AUDIT_STRICT=1 pytest tests/test_scaler_pipeline.py -v -rs

Windows PowerShell:
    $env:SCALER_AUDIT_STRICT = "1"
    pytest tests/test_scaler_pipeline.py -v -rs

Requires the REAL src.rl.policy_io implementation. These tests do not train
policies or demonstrate improved accuracy. Recorded dataset paths are resolved
against the repository root, not whichever directory pytest was launched from.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rl.policy_io import (
    ScaledPolicy, save_scaler_sidecar, read_scaler_sidecar,
)

MEAN = np.array([10.0, 20.0, 0.5, 0.0, 0.0, 0.3], dtype=np.float32)
STD = np.array([5.0, 10.0, 0.25, 1.0, 1.0, 0.2], dtype=np.float32)
RAW_LOW = np.array([3.0, 5.0, 0.1, 0.1, 0.1, 0.05], dtype=np.float32)
RAW_HIGH = np.array([15.0, 30.0, 0.7, 0.5, 0.5, 0.5], dtype=np.float32)

ARCHIVE_ROOTS = tuple(REPO / p for p in (
    'models', 'models_epsilon', 'models_no_oracle',
    'models_no_oracle_alpha1', 'models_no_oracle_alpha10',
    'reports/paper_exp/frozen_models', 'reports/paper_exp/cql_mean',
    'reports/paper_exp/cql_qr',
))
# Tight, explicit tolerances; no allclose broadcasting and no implicit rtol.
# These are test criteria, not experimentally determined accuracy thresholds.
SCALER_ATOL = 1e-6
SCALER_RTOL = 1e-6


class DummyPolicy:
    """Action 1 iff the input-row mean is negative; otherwise action 0."""
    def predict(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return (obs.mean(axis=-1, keepdims=True) < 0).astype(np.int64)


class RecordingPolicy(DummyPolicy):
    """Expose the exact numeric input received by the underlying policy."""
    def __init__(self):
        self.received = []

    def predict(self, obs):
        array = np.asarray(obs, dtype=np.float32)
        self.received.append(array.copy())
        return super().predict(array)


def require_scaler_vectors(mean, std, n_features=6):
    """Reject invalid sidecar contents before tolerant comparison."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    assert mean.shape == (n_features,), f'Unexpected mean shape: {mean.shape}'
    assert std.shape == (n_features,), f'Unexpected std shape: {std.shape}'
    assert np.isfinite(mean).all(), 'Nonfinite scaler mean'
    assert np.isfinite(std).all(), 'Nonfinite scaler std'
    assert (std > 0).all(), 'Scaler std must be strictly positive'
    return mean, std


def strict_archive_audit():
    return os.environ.get('SCALER_AUDIT_STRICT', '').strip().lower() in {'1', 'true', 'yes'}


def missing_archive_resource(reason):
    if strict_archive_audit():
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.mark.parametrize('raw, expected_action', [
    (RAW_LOW, 1), (RAW_HIGH, 0), (MEAN, 0),
], ids=['below_mean', 'above_mean', 'at_mean'])
def test_scaled_policy_standardises_before_predict(raw, expected_action):
    assert ScaledPolicy(DummyPolicy(), MEAN, STD).action(raw.copy()) == expected_action


@pytest.mark.parametrize('raw', [RAW_LOW, RAW_HIGH, MEAN],
                         ids=['below_mean', 'above_mean', 'at_mean'])
def test_exact_standardised_input_is_forwarded_once(raw):
    """Correct actions alone do not prove each field was correctly scaled."""
    recorder = RecordingPolicy()
    scaled = ScaledPolicy(recorder, MEAN, STD)
    scaled.action(raw.copy())
    assert len(recorder.received) == 1
    received = recorder.received[0]
    # The existing single-state API may forward (6,) or (1,6).
    assert received.shape in {(6,), (1, 6)}, f'Unexpected input shape: {received.shape}'
    np.testing.assert_allclose(received.reshape(-1), (raw - MEAN) / STD,
                               atol=1e-6, rtol=0)


def test_unscaled_dummy_would_give_wrong_answer():
    policy = DummyPolicy()
    assert policy.predict(RAW_LOW)[0, 0] == 0
    assert policy.predict(RAW_HIGH)[0, 0] == 0
    scaled = ScaledPolicy(policy, MEAN, STD)
    assert scaled.action(RAW_LOW.copy()) == 1
    assert scaled.action(RAW_HIGH.copy()) == 0


def test_action_does_not_mutate_raw_observation():
    raw = RAW_LOW.copy()
    original = raw.copy()
    ScaledPolicy(DummyPolicy(), MEAN, STD).action(raw)
    np.testing.assert_array_equal(raw, original)


def test_repeated_calls_do_not_double_standardise():
    raw = RAW_LOW.copy()
    recorder = RecordingPolicy()
    scaled = ScaledPolicy(recorder, MEAN, STD)
    assert scaled.action(raw) == 1
    assert scaled.action(raw) == 1
    assert len(recorder.received) == 2
    expected = (RAW_LOW - MEAN) / STD
    for received in recorder.received:
        np.testing.assert_allclose(received.reshape(-1), expected, atol=1e-6, rtol=0)


def test_scaler_sidecar_roundtrip(tmp_path):
    ckpt = tmp_path / 'dummy_seed42.d3'
    ckpt.write_bytes(b'')  # No real checkpoint deserialization in this test.
    dataset = Path('data/processed/offline_rl_dataset_epsilon.h5')
    save_scaler_sidecar(ckpt, MEAN, STD, dataset)
    sidecar = Path(str(ckpt) + '.scaler.json')
    assert sidecar.exists(), 'Expected <checkpoint>.d3.scaler.json'
    payload = read_scaler_sidecar(ckpt)
    mean, std = require_scaler_vectors(payload['mean'], payload['std'])
    np.testing.assert_allclose(mean, MEAN, atol=1e-7, rtol=0)
    np.testing.assert_allclose(std, STD, atol=1e-7, rtol=0)
    assert payload['dataset'] == str(dataset)
    assert 'created' in payload
    # Prove the persisted scaler, not just in-memory constants, gives the action.
    restored = ScaledPolicy(DummyPolicy(), mean.astype(np.float32), std.astype(np.float32))
    assert restored.action(RAW_LOW.copy()) == 1


def test_sidecar_missing_raises(tmp_path):
    ckpt = tmp_path / 'nonexistent_seed42.d3'
    ckpt.write_bytes(b'')
    with pytest.raises(FileNotFoundError):
        read_scaler_sidecar(ckpt)


@pytest.mark.parametrize('mean,std', [
    (MEAN[:-1], STD), (MEAN, STD[:-1]),
    (np.full(6, np.nan), STD), (MEAN, np.full(6, np.inf)),
    (MEAN, np.zeros(6)), (MEAN, -STD),
    (MEAN.reshape(1, -1), STD),
], ids=['wrong_mean_length', 'wrong_std_length', 'nan_mean', 'inf_std',
        'zero_std', 'negative_std', 'broadcastable_mean_shape'])
def test_audit_rejects_invalid_scaler_vectors(mean, std):
    # This tests the test-suite's audit guard, NOT production input validation.
    with pytest.raises(AssertionError):
        require_scaler_vectors(mean, std)


@pytest.fixture(scope='module')
def scaler_cache():
    return {}


def archived_sidecars():
    found = set()
    for root in ARCHIVE_ROOTS:
        if root.exists():
            found.update(root.rglob('*.d3.scaler.json'))
    return sorted(found)


_SIDECARS = archived_sidecars()


@pytest.mark.parametrize('sidecar', _SIDECARS or [None],
    ids=[str(p.relative_to(REPO)) for p in _SIDECARS] or ['no_archive_available'])
def test_real_checkpoint_sidecars_have_matching_scalers(sidecar, scaler_cache):
    """One visible test per sidecar; datasets cannot disappear silently."""
    if sidecar is None:
        missing_archive_resource('No archived *.d3.scaler.json files found in configured roots.')
    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    assert isinstance(payload, dict), f'Sidecar must contain a JSON object: {sidecar}'
    assert isinstance(payload.get('dataset'), str) and payload['dataset'], \
        f'Missing/invalid dataset provenance: {sidecar}'
    checkpoint = Path(str(sidecar).removesuffix('.scaler.json'))
    assert checkpoint.is_file(), f'Orphan sidecar: checkpoint missing: {checkpoint}'
    dataset = Path(payload['dataset']).expanduser()
    if not dataset.is_absolute():
        dataset = REPO / dataset
    dataset = dataset.resolve()
    if not dataset.is_file():
        missing_archive_resource(f'Dataset unavailable for {sidecar}: {dataset}')
    from src.rl.policy_io import compute_scaler_from_dataset
    if dataset not in scaler_cache:
        scaler_cache[dataset] = compute_scaler_from_dataset(dataset)
    expected_mean, expected_std = scaler_cache[dataset]
    expected_mean, expected_std = require_scaler_vectors(expected_mean, expected_std)
    actual_mean, actual_std = require_scaler_vectors(payload['mean'], payload['std'])
    np.testing.assert_allclose(actual_mean, expected_mean, atol=SCALER_ATOL,
        rtol=SCALER_RTOL, err_msg=f'Mean mismatch: {sidecar}')
    np.testing.assert_allclose(actual_std, expected_std, atol=SCALER_ATOL,
        rtol=SCALER_RTOL, err_msg=f'Std mismatch: {sidecar}')


def test_strict_archive_audit_requires_sidecars_for_all_checkpoints():
    """Optional coverage check: no checkpoint may be missing its sidecar."""
    if not strict_archive_audit():
        pytest.skip('Set SCALER_AUDIT_STRICT=1 for complete archive coverage.')
    checkpoints = sorted({p for root in ARCHIVE_ROOTS if root.exists()
                          for p in root.rglob('*.d3')})
    assert checkpoints, 'Strict archive audit requested but no checkpoints found.'
    missing = [str(p) for p in checkpoints if not Path(str(p)+'.scaler.json').is_file()]
    assert not missing, 'Checkpoints missing sidecars:\n' + '\n'.join(missing)
