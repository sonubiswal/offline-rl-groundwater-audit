"""Discrete BCQ: return optimization, NOT action-match optimization.

Structural twin of train_cql.py. Replaces the CQL conservatism coefficient
with the BCQ perturbation network and action-flexibility threshold. Same
split, scaler, oversampling, provenance, and diagnostic action-match
reporting. Requires the repository's existing src.rl.policy_io.
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
HIGH_DEMAND_QUANTILE, DEFAULT_OVERSAMPLE_FACTOR = 0.75, 3
DEFAULT_N_QUANTILES = 200

# DiscreteBCQ defaults, matching the BCQ paper for discrete actions and
# the same fixed training budget as CQL for comparability.
DEFAULT_ACTION_FLEXIBILITY = 0.3
DEFAULT_BETA = 0.5


def _config_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(DiscreteBCQConfig)}


def validate_bcq_cfg(cfg):
    steps = int(cfg.get('n_steps', N_STEPS))
    epoch = int(cfg.get('n_steps_per_epoch', N_STEPS_PER_EPOCH))
    if steps <= 0 or epoch <= 0 or steps % epoch:
        raise ValueError('n_steps must be a positive multiple of n_steps_per_epoch.')
    if cfg.get('qfunc', 'qr') not in ('mean', 'qr'):
        raise ValueError('qfunc must be mean or qr.')
    if float(cfg.get('learning_rate', 6.25e-5)) <= 0:
        raise ValueError('learning_rate must be positive.')
    if not 0 <= float(cfg.get('gamma', 0.99)) <= 1:
        raise ValueError('gamma must lie in [0, 1].')
    if not 0 <= float(cfg.get('tau', 0.005)) <= 1:
        raise ValueError('tau must lie in [0, 1].')
    af = float(cfg.get('action_flexibility', DEFAULT_ACTION_FLEXIBILITY))
    if not 0 <= af <= 1:
        raise ValueError('action_flexibility must lie in [0, 1].')
    beta = float(cfg.get('beta', DEFAULT_BETA))
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive.')
    for key, default in [('batch_size', 100), ('target_update_interval', 8000),
                         ('n_critics', 1), ('n_quantiles', DEFAULT_N_QUANTILES)]:
        try:
            value = int(cfg.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f'{key} must be an integer.')
        if value <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    return steps, epoch


def make_bcq(cfg):
    validate_bcq_cfg(cfg)
    factory = (d3rlpy.models.MeanQFunctionFactory()
               if cfg.get('qfunc', 'qr') == 'mean'
               else d3rlpy.models.QRQFunctionFactory(
                   n_quantiles=int(cfg.get('n_quantiles', DEFAULT_N_QUANTILES))))
    clip = float(cfg.get('clip_grad_norm', 1.0))
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('clip_grad_norm must be finite and positive.')

    kwargs = dict(
        batch_size=int(cfg.get('batch_size', 100)),
        gamma=float(cfg.get('gamma', 0.99)),
        learning_rate=float(cfg.get('learning_rate', 6.25e-5)),
        tau=float(cfg.get('tau', 0.005)),
        target_update_interval=int(cfg.get('target_update_interval', 8000)),
        action_flexibility=float(cfg.get('action_flexibility',
                                         DEFAULT_ACTION_FLEXIBILITY)),
        beta=float(cfg.get('beta', DEFAULT_BETA)),
        n_critics=int(cfg.get('n_critics', 1)),
        q_func_factory=factory,
        optim_factory=AdamFactory(clip_grad_norm=clip),
    )

    # Fail loudly if a field is missing or renamed in this d3rlpy build.
    fields = _config_fields()
    unknown = set(kwargs) - fields
    if unknown:
        raise RuntimeError(
            f"DiscreteBCQConfig in this d3rlpy build does not accept "
            f"{sorted(unknown)}. Available fields: {sorted(fields)}"
        )
    return DiscreteBCQConfig(**kwargs).create(device=cfg.get('device', False))


def oversample_negative_tail_episodes(
        episodes, oversample_factor=DEFAULT_OVERSAMPLE_FACTOR,
        demand_idx=CROP_DEMAND_IDX, reward_cutoff=NEGATIVE_TAIL_REWARD_CUTOFF,
        demand_quantile=HIGH_DEMAND_QUANTILE):
    """Same policy as train_cql.py, kept identical for comparability."""
    if not episodes:
        raise ValueError('No training episodes.')
    if (isinstance(oversample_factor, bool)
            or int(oversample_factor) != oversample_factor
            or oversample_factor < 1):
        raise ValueError('oversample_factor must be an integer >=1.')
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
          f'negative tail; factor={oversample_factor}, '
          f'demand_cutoff={cutoff:.4f}')
    return list(episodes) + matching * (int(oversample_factor) - 1)


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

    model_path = Path(cfg.get('model_dir', 'models/bcq')) / f'bcq_seed{seed}.d3'
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Write scaler sidecar first, then the model. If sidecar write fails, no
    # orphan .d3 is left for refuse_existing_outputs to trip on.
    from src.rl.policy_io import save_scaler_sidecar
    save_scaler_sidecar(model_path, obs_mean, obs_std, Path(dataset_path))
    algo.save(str(model_path))

    write_json(model_path.with_suffix('.meta.json'), {
        'seed': seed,
        'algo': 'DiscreteBCQ',
        'dataset': str(dataset_path),
        'resolved_config': cfg,
        'qfunc': cfg.get('qfunc', 'qr'),
        'learning_rate': cfg.get('learning_rate', 6.25e-5),
        'target_update_interval': cfg.get('target_update_interval', 8000),
        'tau': cfg.get('tau', 0.005),
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
    })
    print(f'[train_bcq] seed={seed}, val match={val["overall"]:.4f}, '
          f'test match={test["overall"]:.4f} (diagnostic only)')
    return {'seed': seed, 'val_action_match': val,
            'test_action_match': test, 'model_path': str(model_path)}


def _serializable_history(history):
    """d3rlpy fit() returns [(epoch, {metric: value}), ...] with numpy floats."""
    out = []
    for entry in history or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            epoch, metrics = entry
            clean = {str(k): float(v) for k, v in dict(metrics).items()
                     if np.isscalar(v) or isinstance(v, (int, float, np.floating))}
            out.append((int(epoch), clean))
        else:
            out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/rl_config.yaml')
    ap.add_argument('--dataset', default='data/processed/offline_rl_dataset.h5')
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
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--policy-order', nargs='+', default=POLICY_ORDER.copy())
    ap.add_argument('--n-per-policy', type=int, default=N_PER_POLICY)
    ap.add_argument('--seeds', default=None,
                    help='Comma-separated seed override; default uses TRAINING_SEEDS')
    args = ap.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f) or {}
    cfg = dict(full_cfg.get('bcq', {}))

    for key in ['model_dir', 'qfunc', 'oversample_factor',
                'target_update_interval', 'n_steps', 'n_steps_per_epoch',
                'learning_rate', 'n_quantiles', 'action_flexibility',
                'beta', 'tau']:
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

    # Provenance records what was actually used, not the module defaults.
    write_json(args.out, {
        'per_seed': results,
        'resolved_config': cfg,
        'val_action_match_mean': float(np.mean(vals)),
        'val_action_match_std': float(np.std(vals)),
        'test_action_match_mean': float(np.mean(tests)),
        'test_action_match_std': float(np.std(tests)),
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
        'important': 'Action-match is diagnostic, not policy-return performance.',
    })


if __name__ == '__main__':
    main()