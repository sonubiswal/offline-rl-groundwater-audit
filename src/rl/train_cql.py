"""Discrete CQL: return optimization, NOT action-match optimization.

Retains explicit alpha, original episode split, train-only scaler, seed calls,
scaler sidecars, mean/QR options, negative-tail oversampling, and diagnostic
alpha sweep. No checkpoint or alpha selection by action-match or own Q score.
Existing defaults are preserved; proposed settings are hypotheses to evaluate.
Requires the repository's existing src.rl.policy_io.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml
import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.optimizers import AdamFactory

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.rl import bc_baseline as bc
from src.rl.bc_baseline import (
    SPLIT_SEED, TRAINING_SEEDS, POLICY_ORDER, N_PER_POLICY,
    split_episode_indices, episodes_to_arrays, build_dataset, action_accuracy,
    write_json, load_split_and_scaler, record_split, refuse_existing_outputs,
)
N_STEPS, N_STEPS_PER_EPOCH = 20_000, 2_000
ALPHA_GRID, SWEEP_N_STEPS = [0.5, 1.0, 4.0, 10.0], 5_000
CROP_DEMAND_IDX, NEGATIVE_TAIL_REWARD_CUTOFF = 2, -1.90
HIGH_DEMAND_QUANTILE, DEFAULT_OVERSAMPLE_FACTOR = 0.75, 3
DEFAULT_N_QUANTILES = 200


def validate_cql_cfg(cfg, alpha):
    steps = int(cfg.get('n_steps', N_STEPS))
    epoch = int(cfg.get('n_steps_per_epoch', N_STEPS_PER_EPOCH))
    if steps <= 0 or epoch <= 0 or steps % epoch:
        raise ValueError('n_steps must be a positive multiple of n_steps_per_epoch.')
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative.')
    if cfg.get('qfunc', 'qr') not in ('mean', 'qr'):
        raise ValueError('qfunc must be mean or qr; invalid values cannot silently select QR.')
    if float(cfg.get('learning_rate', 6.25e-5)) <= 0:
        raise ValueError('learning_rate must be positive.')
    if not 0 <= float(cfg.get('gamma', 0.99)) <= 1:
        raise ValueError('gamma must lie in [0, 1].')
    for key, default in [('batch_size', 100), ('target_update_interval', 8000),
                         ('n_critics', 1), ('n_quantiles', DEFAULT_N_QUANTILES)]:
        value = cfg.get(key, default)
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    return steps, epoch


def make_cql(alpha: float, cfg: dict):
    validate_cql_cfg(cfg, alpha)
    factory = (d3rlpy.models.MeanQFunctionFactory()
        if cfg.get('qfunc', 'qr') == 'mean'
        else d3rlpy.models.QRQFunctionFactory(
            n_quantiles=int(cfg.get('n_quantiles', DEFAULT_N_QUANTILES))))
    clip = float(cfg.get('clip_grad_norm', 1.0))
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError('clip_grad_norm must be finite and positive.')
    return DiscreteCQLConfig(
        batch_size=int(cfg.get('batch_size', 100)),
        gamma=float(cfg.get('gamma', 0.99)),
        learning_rate=float(cfg.get('learning_rate', 6.25e-5)),
        alpha=float(alpha),
        target_update_interval=int(cfg.get('target_update_interval', 8000)),
        n_critics=int(cfg.get('n_critics', 1)), q_func_factory=factory,
        optim_factory=AdamFactory(clip_grad_norm=clip),
    ).create(device=cfg.get('device', False))


def oversample_negative_tail_episodes(episodes, oversample_factor=DEFAULT_OVERSAMPLE_FACTOR,
        demand_idx=CROP_DEMAND_IDX, reward_cutoff=NEGATIVE_TAIL_REWARD_CUTOFF,
        demand_quantile=HIGH_DEMAND_QUANTILE):
    if not episodes:
        raise ValueError('No training episodes.')
    if isinstance(oversample_factor, bool) or int(oversample_factor) != oversample_factor or oversample_factor < 1:
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
    print(f'[train_cql] {len(matching)}/{len(episodes)} episodes match negative tail; '
          f'factor={oversample_factor}, demand_cutoff={cutoff:.4f}')
    return list(episodes)+matching*(int(oversample_factor)-1)


def score_init_state_value(algo, dataset, val_episodes_std):
    return float(InitialStateValueEstimationEvaluator(episodes=val_episodes_std)(algo, dataset=dataset))


def run_alpha_sweep_diagnostic(train_ds, val_ds_std, val_episodes_std, cfg):
    steps = int(cfg.get('sweep_n_steps', SWEEP_N_STEPS))
    epoch_steps = min(int(cfg.get('n_steps_per_epoch', N_STEPS_PER_EPOCH)), steps)
    # 5,000 is not divisible by 2,000. Use a divisor so requested probe steps
    # are honored, without relying on fitter's treatment of partial epochs.
    while steps > 0 and steps % epoch_steps:
        epoch_steps -= 1
    if steps <= 0:
        raise ValueError('sweep_n_steps must be positive.')
    rows = []
    for alpha in cfg.get('alpha_grid', ALPHA_GRID):
        probe_cfg = {**cfg, 'n_steps': steps, 'n_steps_per_epoch': epoch_steps}
        d3rlpy.seed(TRAINING_SEEDS[0])
        algo = make_cql(float(alpha), probe_cfg)
        algo.fit(train_ds, n_steps=steps, n_steps_per_epoch=epoch_steps, show_progress=False)
        score = score_init_state_value(algo, val_ds_std, val_episodes_std)
        rows.append({'alpha': float(alpha),
                     'init_state_value': score if np.isfinite(score) else None,
                     'finite_score': bool(np.isfinite(score))})
    return {'grid_results': rows, 'sweep_seed': TRAINING_SEEDS[0],
            'selection_disabled_reason': 'Own estimated Q is affected by alpha; diagnostic only.'}


def run_one_seed(seed, alpha, train_ds, val_episodes, val_indices,
                 test_episodes, test_indices, obs_mean, obs_std, cfg, dataset_path):
    steps, epoch = validate_cql_cfg(cfg, alpha)
    d3rlpy.seed(seed)
    algo = make_cql(alpha, cfg)
    history = algo.fit(train_ds, n_steps=steps, n_steps_per_epoch=epoch,
                      show_progress=bool(cfg.get('show_progress', True)))
    # Fixed training budget; NEVER select this checkpoint by action-match.
    val = action_accuracy(algo, val_episodes, obs_mean, obs_std, val_indices)
    test = action_accuracy(algo, test_episodes, obs_mean, obs_std, test_indices)
    model_path = Path(cfg.get('model_dir', 'models/cql'))/f'cql_seed{seed}.d3'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(model_path))
    from src.rl.policy_io import save_scaler_sidecar
    save_scaler_sidecar(model_path, obs_mean, obs_std, Path(dataset_path))
    write_json(model_path.with_suffix('.meta.json'), {
        'seed': seed, 'alpha': alpha, 'algo': 'DiscreteCQL', 'dataset': str(dataset_path),
        'resolved_config': cfg, 'qfunc': cfg.get('qfunc', 'qr'),
        'learning_rate': cfg.get('learning_rate', 6.25e-5),
        'target_update_interval': cfg.get('target_update_interval', 8000),
        'gamma': cfg.get('gamma', 0.99), 'batch_size': cfg.get('batch_size', 100),
        'n_steps': steps, 'n_steps_per_epoch': epoch,
        'n_critics': cfg.get('n_critics', 1), 'clip_grad_norm': cfg.get('clip_grad_norm', 1.0),
        'oversample_factor': cfg.get('oversample_factor', DEFAULT_OVERSAMPLE_FACTOR),
        'n_quantiles': cfg.get('n_quantiles', DEFAULT_N_QUANTILES) if cfg.get('qfunc', 'qr') == 'qr' else None,
        'training_history': history, 'split_seed': SPLIT_SEED,
        'val_action_match': val, 'test_action_match': test,
        'alpha_selected_via': 'explicit CLI/config; no automatic selection',
        'checkpoint_selection': 'fixed training budget; action-match diagnostic only',
        'seeded_via': 'd3rlpy.seed before create'})
    print(f'[train_cql] seed={seed}, val match={val["overall"]:.4f}, '
          f'test match={test["overall"]:.4f} (diagnostic only)')
    return {'seed': seed, 'alpha': alpha, 'val_action_match': val,
            'test_action_match': test, 'model_path': str(model_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/rl_config.yaml')
    ap.add_argument('--dataset', default='data/processed/offline_rl_dataset.h5')
    ap.add_argument('--out', default='reports/cql_results.json')
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--overwrite', action='store_true', help='Explicitly permit replacing existing checkpoints/reports.')
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--qfunc', choices=['qr', 'mean'], default=None)
    ap.add_argument('--oversample-factor', type=int, default=None)
    ap.add_argument('--target-update-interval', type=int, default=None)
    ap.add_argument('--n-steps', type=int, default=None)
    ap.add_argument('--n-steps-per-epoch', type=int, default=None)
    ap.add_argument('--learning-rate', type=float, default=None)
    ap.add_argument('--n-quantiles', type=int, default=None)
    ap.add_argument('--policy-order', nargs='+', default=POLICY_ORDER.copy())
    ap.add_argument('--n-per-policy', type=int, default=N_PER_POLICY)
    args = ap.parse_args()
    with open(args.config) as f:
        full_cfg = yaml.safe_load(f) or {}
    cfg = dict(full_cfg.get('cql', {}))
    for key in ['model_dir', 'qfunc', 'oversample_factor', 'target_update_interval',
                'n_steps', 'n_steps_per_epoch', 'learning_rate', 'n_quantiles']:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    alpha = args.alpha if args.alpha is not None else cfg.get('alpha')
    if alpha is None:
        ap.error('Supply --alpha or cql.alpha. Own-Q alpha selection is disabled.')
    alpha = float(alpha)
    validate_cql_cfg(cfg, alpha)
    refuse_existing_outputs(cfg.get('model_dir', 'models/cql'), 'cql', args.out, args.overwrite)
    from src.rl.policy_io import save_scaler_sidecar  # Fail before training if absent.
    episodes, indices, groups, mean, std = load_split_and_scaler(
        args.dataset, args.policy_order, args.n_per_policy)
    train_idx, val_idx, test_idx = indices
    train_eps, val_eps, test_eps = groups
    factor = cfg.get('oversample_factor', DEFAULT_OVERSAMPLE_FACTOR)
    fit_eps = oversample_negative_tail_episodes(train_eps, factor)
    train_ds = build_dataset(fit_eps, mean, std)
    val_ds = build_dataset(val_eps, mean, std)
    sweep = ({'skipped': True} if args.skip_sweep else
             run_alpha_sweep_diagnostic(train_ds, val_ds, list(val_ds.episodes), cfg))
    if int(cfg.get('n_steps', N_STEPS)) < N_STEPS:
        print('[train_cql] SHORT PROBE: not comparable to final full-budget runs.')
    results = [run_one_seed(seed, alpha, train_ds, val_eps, val_idx,
                           test_eps, test_idx, mean, std, cfg, Path(args.dataset))
               for seed in TRAINING_SEEDS]
    vals = [r['val_action_match']['overall'] for r in results]
    tests = [r['test_action_match']['overall'] for r in results]
    write_json(args.out, {
        'per_seed': results, 'resolved_config': cfg, 'alpha_used': alpha,
        'alpha_source': '--alpha' if args.alpha is not None else 'cql.alpha',
        'alpha_sweep_diagnostic': sweep,
        'val_action_match_mean': float(np.mean(vals)), 'val_action_match_std': float(np.std(vals)),
        'test_action_match_mean': float(np.mean(tests)), 'test_action_match_std': float(np.std(tests)),
        'std_convention': 'population standard deviation, ddof=0',
        'n_train_episodes': len(train_idx), 'n_train_episodes_after_oversampling': len(fit_eps),
        'n_val_episodes': len(val_idx), 'n_test_episodes': len(test_idx),
        'split': record_split(indices, mean, std), 'split_seed': SPLIT_SEED,
        'training_seeds': TRAINING_SEEDS,
        'policy_order_assumption': POLICY_ORDER.copy(), 'n_per_policy_assumption': bc.N_PER_POLICY,
        'dataset': args.dataset, 'model_dir': cfg.get('model_dir', 'models/cql'),
        'is_short_probe': int(cfg.get('n_steps', N_STEPS)) < N_STEPS,
        'important': 'Action-match is diagnostic, not policy-return performance.'})


if __name__ == '__main__':
    main()
