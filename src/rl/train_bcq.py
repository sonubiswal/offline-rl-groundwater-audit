"""Discrete BCQ: return optimization, NOT action-match optimization.

Discrete BCQ trains a Q-function alongside a behaviour-cloning head over
the same discrete action set. The imitation loss is weighted by ``beta``;
``action_flexibility`` is the probability-ratio threshold below which an
action is masked at evaluation. There is no perturbation network; that
belongs to continuous-action BCQ. The config exposes
``target_update_interval`` (hard updates) and has no ``tau``.

Same split, scaler, oversampling, provenance and diagnostic action-match
reporting as train_cql.py. Action-match will sit near the BC baseline by
construction, because discrete BCQ restricts the greedy action to ones the
behaviour policy plausibly takes. Distinguishing BCQ from BC requires a
return-based comparison, not action-match.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import yaml
import d3rlpy
from d3rlpy.algos import DiscreteBCQConfig
from d3rlpy.optimizers import AdamFactory

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.rl import bc_baseline as bc
from src.rl.bc_baseline import (
    SPLIT_SEED, TRAINING_SEEDS, POLICY_ORDER, N_PER_POLICY,
    build_dataset, action_accuracy, write_json,
    load_split_and_scaler, record_split, refuse_existing_outputs,
)

N_STEPS, N_STEPS_PER_EPOCH = 20_000, 2_000
CROP_DEMAND_IDX, NEGATIVE_TAIL_REWARD_CUTOFF = 2, -1.90
HIGH_DEMAND_QUANTILE, DEFAULT_OVERSAMPLE_FACTOR = 0.75, 1
DEFAULT_N_QUANTILES = 200

DEFAULT_ACTION_FLEXIBILITY = 0.3
DEFAULT_BETA = 0.5

_ALLOWED_KEYS = {
    'model_dir', 'qfunc', 'oversample_factor', 'target_update_interval',
    'n_steps', 'n_steps_per_epoch', 'learning_rate', 'n_quantiles',
    'action_flexibility', 'beta', 'batch_size', 'gamma', 'n_critics',
    'clip_grad_norm', 'show_progress', 'device',
}


# --------------------------------------------------------------- validation
def _config_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(DiscreteBCQConfig)}


def _as_int(cfg, key, default):
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f'{key} must be a positive integer, got bool {raw!r}.')
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be numeric, got {raw!r}.')
    if not f.is_integer():
        raise ValueError(f'{key} must be an integer, got {raw!r}.')
    if f <= 0:
        raise ValueError(f'{key} must be positive, got {raw!r}.')
    return int(f)


def _as_float(cfg, key, default, lo=None, hi=None):
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f'{key} must be numeric, got bool {raw!r}.')
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be numeric, got {raw!r}.')
    if not np.isfinite(f):
        raise ValueError(f'{key} must be finite, got {raw!r}.')
    if lo is not None and f < lo:
        raise ValueError(f'{key} must be >= {lo}, got {f}.')
    if hi is not None and f > hi:
        raise ValueError(f'{key} must be <= {hi}, got {f}.')
    return f


def validate_bcq_cfg(cfg):
    unknown = set(cfg) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in bcq config: {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_KEYS)}"
        )
    steps = _as_int(cfg, 'n_steps', N_STEPS)
    epoch = _as_int(cfg, 'n_steps_per_epoch', N_STEPS_PER_EPOCH)
    if steps % epoch:
        raise ValueError('n_steps must be a positive multiple of '
                         'n_steps_per_epoch.')
    if cfg.get('qfunc', 'qr') not in ('mean', 'qr'):
        raise ValueError('qfunc must be mean or qr.')
    _as_float(cfg, 'learning_rate', 6.25e-5, lo=1e-9)
    _as_float(cfg, 'gamma', 0.99, lo=0.0, hi=1.0)
    _as_float(cfg, 'action_flexibility', DEFAULT_ACTION_FLEXIBILITY,
              lo=0.0, hi=1.0)
    _as_float(cfg, 'beta', DEFAULT_BETA, lo=1e-9)
    _as_float(cfg, 'clip_grad_norm', 1.0, lo=1e-9)
    for key, default in [('batch_size', 100),
                         ('target_update_interval', 8000),
                         ('n_critics', 1),
                         ('n_quantiles', DEFAULT_N_QUANTILES),
                         ('oversample_factor', DEFAULT_OVERSAMPLE_FACTOR)]:
        _as_int(cfg, key, default)
    return steps, epoch


# ------------------------------------------------------------ algorithm build
def make_bcq(cfg):
    validate_bcq_cfg(cfg)
    factory = (d3rlpy.models.MeanQFunctionFactory()
               if cfg.get('qfunc', 'qr') == 'mean'
               else d3rlpy.models.QRQFunctionFactory(
                   n_quantiles=int(cfg.get('n_quantiles',
                                           DEFAULT_N_QUANTILES))))
    clip = float(cfg.get('clip_grad_norm', 1.0))
    kwargs = dict(
        batch_size=int(cfg.get('batch_size', 100)),
        gamma=float(cfg.get('gamma', 0.99)),
        learning_rate=float(cfg.get('learning_rate', 6.25e-5)),
        target_update_interval=int(cfg.get('target_update_interval', 8000)),
        action_flexibility=float(cfg.get('action_flexibility',
                                         DEFAULT_ACTION_FLEXIBILITY)),
        beta=float(cfg.get('beta', DEFAULT_BETA)),
        n_critics=int(cfg.get('n_critics', 1)),
        q_func_factory=factory,
        optim_factory=AdamFactory(clip_grad_norm=clip),
    )
    fields = _config_fields()
    unknown = set(kwargs) - fields
    if unknown:
        raise RuntimeError(
            f"DiscreteBCQConfig in this d3rlpy build does not accept "
            f"{sorted(unknown)}. Available fields: {sorted(fields)}"
        )
    return DiscreteBCQConfig(**kwargs).create(device=cfg.get('device', False))


# ------------------------------------------------------------- oversampling
def oversample_negative_tail_episodes(
        episodes, oversample_factor=DEFAULT_OVERSAMPLE_FACTOR,
        demand_idx=CROP_DEMAND_IDX, reward_cutoff=NEGATIVE_TAIL_REWARD_CUTOFF,
        demand_quantile=HIGH_DEMAND_QUANTILE):
    """Same policy as train_cql.py, kept identical for comparability."""
    if not episodes:
        raise ValueError('No training episodes.')
    factor = _as_int({'x': oversample_factor}, 'x',
                     DEFAULT_OVERSAMPLE_FACTOR)
    if not 0 <= demand_quantile <= 1:
        raise ValueError('demand_quantile must lie in [0,1].')
    all_obs = np.concatenate([ep.observations for ep in episodes])
    if demand_idx < 0 or demand_idx >= all_obs.shape[1]:
        raise ValueError('Demand index outside observation vector.')
    cutoff = np.quantile(all_obs[:, demand_idx], demand_quantile)
    matching = []
    for ep in episodes:
        hits = ((ep.observations[:, demand_idx] >= cutoff)
                & (ep.actions.reshape(-1) == 3)
                & (ep.rewards.reshape(-1) <= reward_cutoff))
        if hits.any():
            matching.append(ep)
    print(f'[train_bcq] {len(matching)}/{len(episodes)} episodes match '
          f'negative tail; factor={factor}, demand_cutoff={cutoff:.4f}')
    return list(episodes) + matching * (factor - 1)


# ---------------------------------------------------------------- history
def _serializable_history(history):
    """d3rlpy fit() returns [(epoch, {metric: value}), ...].

    Non-finite scalars are recorded as None so the output JSON is strict.
    Non-scalar entries are coerced to strings rather than passed through.
    """
    out = []
    for entry in history or []:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            out.append({'epoch': None, 'raw': str(entry)})
            continue
        epoch, metrics = entry
        clean = {}
        for k, v in dict(metrics).items():
            if isinstance(v, bool):
                clean[str(k)] = int(v)
            elif isinstance(v, (int, np.integer)):
                clean[str(k)] = int(v)
            elif isinstance(v, (float, np.floating)):
                clean[str(k)] = float(v) if np.isfinite(v) else None
            else:
                clean[str(k)] = str(v)
        out.append({'epoch': int(epoch), 'metrics': clean})
    return out


# ------------------------------------------------------------------- save
def _atomic_save_pair(algo, model_path, obs_mean, obs_std, dataset_path):
    """Write model and sidecar to their final names; clean up both on failure.

    Writes directly rather than to temp names, because save_scaler_sidecar
    derives its own filename from the model path it receives. Passing a
    ``.tmp`` path would produce ``<name>.d3.d3.scaler.json`` rather than
    ``<name>.d3.scaler.json``. Any exception deletes both files so the
    directory is left clean for the next run.
    """
    from src.rl.policy_io import save_scaler_sidecar, scaler_sidecar_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_side = scaler_sidecar_path(model_path)

    try:
        save_scaler_sidecar(model_path, obs_mean, obs_std, Path(dataset_path))
        algo.save(str(model_path))
    except Exception:
        for p in (final_side, model_path):
            if p.exists():
                p.unlink()
        raise

    if not final_side.exists():
        raise RuntimeError(
            f"Sidecar was not written at the expected path {final_side}. "
            f"Check src/rl/policy_io.save_scaler_sidecar's naming convention."
        )


# ---------------------------------------------------------------- one seed
def run_one_seed(seed, train_ds, val_episodes, val_indices,
                 test_episodes, test_indices, obs_mean, obs_std,
                 cfg, dataset_path):
    steps, epoch = validate_bcq_cfg(cfg)
    d3rlpy.seed(seed)
    algo = make_bcq(cfg)
    history = algo.fit(train_ds, n_steps=steps, n_steps_per_epoch=epoch,
                       show_progress=bool(cfg.get('show_progress', True)))
    val = action_accuracy(algo, val_episodes, obs_mean, obs_std, val_indices)
    test = action_accuracy(algo, test_episodes, obs_mean, obs_std, test_indices)

    model_path = (Path(cfg.get('model_dir', 'models/bcq'))
                  / f'bcq_seed{seed}.d3')
    _atomic_save_pair(algo, model_path, obs_mean, obs_std, dataset_path)

    write_json(model_path.with_suffix('.meta.json'), {
        'seed': seed,
        'algo': 'DiscreteBCQ',
        'dataset': str(dataset_path),
        'config_used': cfg,
        'qfunc': cfg.get('qfunc', 'qr'),
        'learning_rate': cfg.get('learning_rate', 6.25e-5),
        'target_update_interval': cfg.get('target_update_interval', 8000),
        'gamma': cfg.get('gamma', 0.99),
        'batch_size': cfg.get('batch_size', 100),
        'action_flexibility': cfg.get('action_flexibility',
                                      DEFAULT_ACTION_FLEXIBILITY),
        'beta': cfg.get('beta', DEFAULT_BETA),
        'n_steps': steps,
        'n_steps_per_epoch': epoch,
        'n_critics': cfg.get('n_critics', 1),
        'clip_grad_norm': cfg.get('clip_grad_norm', 1.0),
        'oversample_factor': cfg.get('oversample_factor',
                                     DEFAULT_OVERSAMPLE_FACTOR),
        'n_quantiles': (cfg.get('n_quantiles', DEFAULT_N_QUANTILES)
                        if cfg.get('qfunc', 'qr') == 'qr' else None),
        'training_history': _serializable_history(history),
        'split_seed': SPLIT_SEED,
        'val_action_match': val,
        'test_action_match': test,
        'checkpoint_selection': ('fixed training budget; action-match '
                                 'diagnostic only'),
        'seeded_via': 'd3rlpy.seed before create',
        'action_match_caveat': (
            'BCQ restricts the greedy action to ones the behaviour head '
            'finds likely, so action-match sits near the BC baseline by '
            'construction; it is not a return proxy.'),
    })
    print(f'[train_bcq] seed={seed}, val match={val["overall"]:.4f}, '
          f'test match={test["overall"]:.4f} (diagnostic only)')
    return {'seed': seed, 'val_action_match': val,
            'test_action_match': test, 'model_path': str(model_path)}


# --------------------------------------------------------------- helpers
def _safe_mean(xs):
    if not xs:
        return None
    m = float(np.mean(xs))
    return m if np.isfinite(m) else None


def _safe_std(xs):
    if not xs:
        return None
    s = float(np.std(xs))
    return s if np.isfinite(s) else None


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/rl_config.yaml')
    ap.add_argument('--dataset',
                    default='data/processed/offline_rl_dataset.h5')
    ap.add_argument('--out', default='reports/bcq_results.json')
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--qfunc', choices=['qr', 'mean'], default=None)
    ap.add_argument('--oversample-factor', type=int, default=None)
    ap.add_argument('--target-update-interval', type=int, default=None)
    ap.add_argument('--n-steps', type=int, default=None)
    ap.add_argument('--n-steps-per-epoch', type=int, default=None)
    ap.add_argument('--learning-rate', type=float, default=None)
    ap.add_argument('--n-quantiles', type=int, default=None)
    ap.add_argument('--action-flexibility', type=float, default=None)
    ap.add_argument('--beta', type=float, default=None)
    ap.add_argument('--policy-order', nargs='+',
                    default=POLICY_ORDER.copy())
    ap.add_argument('--n-per-policy', type=int, default=N_PER_POLICY)
    ap.add_argument('--seeds', default=None,
                    help='Comma-separated seed override; default uses '
                         'TRAINING_SEEDS')
    return ap


# ------------------------------------------------------------------- main
def main():
    args = build_argparser().parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f) or {}
    cfg = dict(full_cfg.get('bcq', {}))

    for key in ['model_dir', 'qfunc', 'oversample_factor',
                'target_update_interval', 'n_steps', 'n_steps_per_epoch',
                'learning_rate', 'n_quantiles', 'action_flexibility',
                'beta']:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    validate_bcq_cfg(cfg)
    refuse_existing_outputs(cfg.get('model_dir', 'models/bcq'),
                            'bcq', args.out, args.overwrite)

    from src.rl.policy_io import save_scaler_sidecar  # noqa: F401

    episodes, indices, groups, mean, std = load_split_and_scaler(
        args.dataset, args.policy_order, args.n_per_policy)
    train_idx, val_idx, test_idx = indices
    train_eps, val_eps, test_eps = groups

    factor = cfg.get('oversample_factor', DEFAULT_OVERSAMPLE_FACTOR)
    fit_eps = oversample_negative_tail_episodes(train_eps, factor)
    train_ds = build_dataset(fit_eps, mean, std)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = TRAINING_SEEDS

    if int(cfg.get('n_steps', N_STEPS)) < N_STEPS:
        print('[train_bcq] SHORT PROBE: not comparable to full-budget runs.')

    results = [run_one_seed(seed, train_ds, val_eps, val_idx,
                            test_eps, test_idx, mean, std,
                            cfg, Path(args.dataset))
               for seed in seeds]

    vals = [r['val_action_match']['overall'] for r in results]
    tests = [r['test_action_match']['overall'] for r in results]

    write_json(args.out, {
        'per_seed': results,
        'config_used': cfg,
        'val_action_match_mean': _safe_mean(vals),
        'val_action_match_std': _safe_std(vals),
        'test_action_match_mean': _safe_mean(tests),
        'test_action_match_std': _safe_std(tests),
        'std_convention': 'population standard deviation, ddof=0',
        'n_train_episodes': len(train_idx),
        'n_train_episodes_after_oversampling': len(fit_eps),
        'n_val_episodes': len(val_idx),
        'n_test_episodes': len(test_idx),
        'split': record_split(indices, mean, std),
        'split_seed': SPLIT_SEED,
        'training_seeds': seeds,
        'policy_order_used': list(args.policy_order),
        'n_per_policy_used': int(args.n_per_policy),
        'dataset': args.dataset,
        'model_dir': cfg.get('model_dir', 'models/bcq'),
        'is_short_probe': int(cfg.get('n_steps', N_STEPS)) < N_STEPS,
        'important': ('Action-match is diagnostic, not policy-return '
                      'performance.'),
    })


if __name__ == '__main__':
    main()