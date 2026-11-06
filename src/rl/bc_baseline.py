"""Discrete BC with validation-only tuning and best-validation checkpoints.

Requires d3rlpy==2.8.1 and the repository's existing src.rl.policy_io.
The episode-order labels remain a declared assumption, NOT inferred truth.
Default: all training policies. --exclude-random is a separately labelled
filtered-data baseline, not a fair drop-in replacement for full-data BC.
No test metric is evaluated during tuning or checkpoint selection.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import yaml
import d3rlpy
from d3rlpy.algos import DiscreteBCConfig
from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SPLIT_SEED = 1
import os as _os
_ENV_SEEDS = _os.environ.get("TRAINING_SEEDS_OVERRIDE", "").strip()
if _ENV_SEEDS:
    TRAINING_SEEDS = [int(s.strip()) for s in _ENV_SEEDS.split(",")]
else:
    TRAINING_SEEDS = [42, 123, 2024]
N_STEPS, N_STEPS_PER_EPOCH = 20_000, 2_000
N_PER_POLICY = 500
POLICY_ORDER = ["random", "greedy_extraction", "conservative"]
N_ACTIONS = 5


def refuse_existing_outputs(model_dir, prefix, report_path, overwrite=False):
    if overwrite:
        return
    paths = [Path(report_path)]
    for seed in TRAINING_SEEDS:
        model = Path(model_dir)/f'{prefix}_seed{seed}.d3'
        paths.extend([model, model.with_suffix('.meta.json'),
                      Path(str(model)+'.scaler.json'), model.with_suffix('.scaler.json')])
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite prior results: '+', '.join(existing)+
                              '. Choose new output paths or explicitly pass --overwrite.')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def policy_for_episode_index(idx: int) -> str:
    idx = int(idx)
    if idx < 0 or idx >= N_PER_POLICY * len(POLICY_ORDER):
        raise ValueError('Episode index outside declared policy-order blocks.')
    return POLICY_ORDER[idx // N_PER_POLICY]


def configure_policy_layout(order, count, n_episodes):
    global N_PER_POLICY
    if count <= 0 or len(order) != len(set(order)) or not order:
        raise ValueError('Policy order must be nonempty/unique; block size positive.')
    if len(order) * count != n_episodes:
        raise ValueError(f'{n_episodes} episodes do not match policy-order assumption: '
                         f'{len(order)} blocks x {count}. Supply the correct '
                         '--policy-order and --n-per-policy; never guess labels.')
    POLICY_ORDER[:] = order
    N_PER_POLICY = count


def split_episode_indices(n_episodes: int, seed: int = SPLIT_SEED):
    idx = np.random.RandomState(seed).permutation(n_episodes)
    n_train, n_val = int(n_episodes * TRAIN_FRAC), int(n_episodes * VAL_FRAC)
    if min(n_train, n_val, n_episodes - n_train - n_val) <= 0:
        raise ValueError('Dataset too small for nonempty episode splits.')
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def episodes_to_arrays(episodes, obs_mean=None, obs_std=None):
    if not episodes:
        raise ValueError('No episodes supplied.')
    if (obs_mean is None) != (obs_std is None):
        raise ValueError('Supply both scaler mean and std, or neither.')
    if obs_std is not None and (not np.isfinite(obs_std).all() or (np.asarray(obs_std) <= 0).any()):
        raise ValueError('Scaler standard deviations must be finite and positive.')
    obs_list, act_list, rew_list, term_list, timeout_list = [], [], [], [], []
    for ep in episodes:
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions).reshape(-1)
        rew = np.asarray(ep.rewards, dtype=np.float32).reshape(-1)
        if obs.ndim != 2 or not len(obs) or not (len(obs) == len(act) == len(rew)):
            raise ValueError('Invalid episode shapes or empty episode.')
        if not np.isfinite(obs).all() or not np.isfinite(rew).all() or not np.isfinite(act).all():
            raise ValueError('Nonfinite dataset values.')
        if not np.equal(act, np.floor(act)).all() or ((act < 0) | (act >= N_ACTIONS)).any():
            raise ValueError('Expected discrete actions 0..4.')
        if obs_mean is not None:
            obs = ((obs - obs_mean) / obs_std).astype(np.float32)
            if not np.isfinite(obs).all():
                raise ValueError('Nonfinite standardized observations.')
        obs_list.append(obs)
        act_list.append(act.astype(np.int64))
        rew_list.append(rew)
        terminals, timeouts = np.zeros(len(obs), np.float32), np.zeros(len(obs), np.float32)
        if ep.terminated:
            terminals[-1] = 1.0
        else:
            timeouts[-1] = 1.0  # Continuing-process collection cutoff, NOT terminal.
        term_list.append(terminals)
        timeout_list.append(timeouts)
    return tuple(np.concatenate(v, axis=0) for v in
                 (obs_list, act_list, rew_list, term_list, timeout_list))


def build_dataset(episodes, obs_mean=None, obs_std=None) -> MDPDataset:
    obs, act, rew, term, timeout = episodes_to_arrays(episodes, obs_mean, obs_std)
    return MDPDataset(observations=obs, actions=act, rewards=rew,
                      terminals=term, timeouts=timeout)


def action_accuracy(algo, episodes, obs_mean, obs_std, episode_indices=None) -> dict:
    if episode_indices is not None and len(episodes) != len(episode_indices):
        raise ValueError('Episode indices must align with episode list.')
    correct = total = 0
    pc, pt = dict.fromkeys(POLICY_ORDER, 0), dict.fromkeys(POLICY_ORDER, 0)
    hist = np.zeros(N_ACTIONS, dtype=np.int64)
    nonrandom_correct = nonrandom_total = 0
    for i, ep in enumerate(episodes):
        obs = ((np.asarray(ep.observations, np.float32)-obs_mean)/obs_std).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError('Nonfinite evaluation observations.')
        truth = np.asarray(ep.actions).reshape(-1).astype(np.int64)
        pred = np.asarray(algo.predict(obs)).reshape(-1)
        if pred.shape != truth.shape:
            raise ValueError(f'Prediction shape {pred.shape} != target {truth.shape}.')
        if not np.isfinite(pred).all() or not np.equal(pred, np.floor(pred)).all():
            raise ValueError('Predictions are not finite discrete actions.')
        pred = pred.astype(np.int64)
        if ((pred < 0) | (pred >= N_ACTIONS)).any():
            raise ValueError('Predicted action outside 0..4.')
        nc = int(np.sum(pred == truth))
        correct += nc; total += len(truth)
        hist += np.bincount(pred, minlength=N_ACTIONS)
        if episode_indices is not None:
            p = policy_for_episode_index(episode_indices[i])
            pc[p] += nc; pt[p] += len(truth)
            if p != 'random':
                nonrandom_correct += nc; nonrandom_total += len(truth)
    result = {'overall': correct/total if total else None,
              'prediction_counts': hist.tolist(), 'n_transitions': total}
    if episode_indices is not None:
        result.update({p: pc[p]/pt[p] if pt[p] else None for p in POLICY_ORDER})
        result['policy_transition_counts'] = pt
        result['nonrandom'] = nonrandom_correct/nonrandom_total if nonrandom_total else None
    return result


def validate_training_cfg(cfg):
    steps = int(cfg.get('n_steps', N_STEPS))
    epoch = int(cfg.get('n_steps_per_epoch', N_STEPS_PER_EPOCH))
    if steps <= 0 or epoch <= 0 or steps % epoch:
        raise ValueError('n_steps must be a positive multiple of n_steps_per_epoch.')
    if float(cfg.get('learning_rate', 0.001)) <= 0 or int(cfg.get('batch_size', 100)) <= 0:
        raise ValueError('Positive learning rate and batch size required.')
    if float(cfg.get('beta', 0.5)) < 0:
        raise ValueError('BC beta must be nonnegative.')
    return steps, epoch


def make_bc(cfg):
    validate_training_cfg(cfg)
    return DiscreteBCConfig(
        batch_size=int(cfg.get('batch_size', 100)),
        learning_rate=float(cfg.get('learning_rate', 0.001)),
        beta=float(cfg.get('beta', 0.5)),
        encoder_factory=d3rlpy.models.VectorEncoderFactory(
            hidden_units=list(cfg.get('hidden_units', [256, 256])), activation='relu'),
    ).create(device=cfg.get('device', False))


def validation_selected_fit(seed, train_ds, val_episodes, val_indices,
                            obs_mean, obs_std, cfg, best_path, metric='overall'):
    steps, epoch_steps = validate_training_cfg(cfg)
    d3rlpy.seed(seed)  # Before construction; independent seeds for every candidate.
    algo = make_bc(cfg)
    best_score, best_epoch, best_val = -np.inf, None, None
    history, stale = [], 0
    patience = int(cfg.get('patience', 0))  # 0 disables early stopping.
    if patience < 0:
        raise ValueError('patience must be nonnegative.')
    for epoch, losses in algo.fitter(train_ds, n_steps=steps,
            n_steps_per_epoch=epoch_steps, show_progress=bool(cfg.get('show_progress', True))):
        val = action_accuracy(algo, val_episodes, obs_mean, obs_std, val_indices)
        score = val.get(metric)
        if score is None or not np.isfinite(score):
            raise ValueError(f'No finite validation score for {metric}.')
        history.append({'epoch': int(epoch), 'validation_score': float(score), 'losses': losses})
        if score > best_score:
            best_score, best_epoch, best_val = float(score), int(epoch), val
            algo.save(str(best_path)); stale = 0
        else:
            stale += 1
        if patience and stale >= patience:
            break
    if best_epoch is None:
        raise RuntimeError('Training yielded no validation checkpoint.')
    best_algo = d3rlpy.load_learnable(str(best_path), device=cfg.get('device', False))
    return best_algo, best_val, {'best_epoch': best_epoch,
        'best_validation_score': best_score, 'selection_metric': metric,
        'trained_epochs': len(history), 'history': history}


def tune_bc(train_ds, val_episodes, val_indices, obs_mean, obs_std, cfg, metric):
    grid = cfg.get('tuning_grid', {
        'learning_rate': [0.001, 0.0003], 'beta': [0.0, 0.5],
        'hidden_units': [[128, 128], [256, 256]], 'n_steps': [20_000, 40_000]})
    keys = list(grid)
    if not keys or any(not grid[k] for k in keys):
        raise ValueError('Tuning grid cannot be empty.')
    if not set(keys) <= {'learning_rate', 'beta', 'hidden_units', 'n_steps'}:
        raise ValueError('Unsupported BC tuning key.')
    rows = []
    with tempfile.TemporaryDirectory(prefix='bc_validation_tuning_') as folder:
        for candidate, values in enumerate(itertools.product(*(grid[k] for k in keys))):
            trial = {**cfg, **dict(zip(keys, values))}
            best_path = Path(folder)/f'candidate{candidate}.d3'
            _, _, fit_info = validation_selected_fit(TRAINING_SEEDS[0], train_ds,
                val_episodes, val_indices, obs_mean, obs_std, trial, best_path, metric)
            rows.append({'candidate': candidate, 'settings': dict(zip(keys, values)),
                         'validation_score': fit_info['best_validation_score'],
                         'best_epoch': fit_info['best_epoch']})
            print('[bc_baseline] Tuning:', rows[-1])
    winner = max(rows, key=lambda r: r['validation_score'])
    return {**cfg, **winner['settings']}, {'seed': TRAINING_SEEDS[0], 'metric': metric,
         'selection': 'validation only; test never accessed', 'candidates': rows, 'winner': winner}


def run_one_seed(seed: int, train_ds, val_episodes, val_indices,
                 test_episodes, test_indices, obs_mean, obs_std,
                 cfg: dict, dataset_path: Path, model_dir='models/bc') -> dict:
    out_dir = Path(model_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir/f'bc_seed{seed}.d3'
    algo, val_acc, fit_info = validation_selected_fit(seed, train_ds, val_episodes,
        val_indices, obs_mean, obs_std, cfg, model_path, cfg.get('selection_metric', 'overall'))
    # Test is assessed ONLY after the checkpoint has been chosen on validation.
    test_acc = action_accuracy(algo, test_episodes, obs_mean, obs_std, test_indices)
    from src.rl.policy_io import save_scaler_sidecar  # Deliberately local.
    save_scaler_sidecar(model_path, obs_mean, obs_std, Path(dataset_path))
    meta = {'seed': seed, 'algo': 'DiscreteBC', 'dataset': str(dataset_path),
        'resolved_config': cfg, 'selection': fit_info,
        'n_steps': cfg.get('n_steps', N_STEPS),
        'n_steps_per_epoch': cfg.get('n_steps_per_epoch', N_STEPS_PER_EPOCH),
        'batch_size': cfg.get('batch_size', 100),
        'learning_rate': cfg.get('learning_rate', 0.001), 'beta': cfg.get('beta', 0.5),
        'val_accuracy': val_acc, 'test_accuracy': test_acc,
        'split_seed': SPLIT_SEED, 'seeded_via': 'd3rlpy.seed before create',
        'filtered_training': bool(cfg.get('exclude_random', False))}
    write_json(model_path.with_suffix('.meta.json'), meta)
    print(f'[bc_baseline] seed={seed} validation={val_acc["overall"]:.4f} '
          f'test={test_acc["overall"]:.4f} best_epoch={fit_info["best_epoch"]}')
    return {'seed': seed, 'val_accuracy': val_acc, 'test_accuracy': test_acc,
            'model_path': str(model_path), 'best_epoch': fit_info['best_epoch']}


def load_split_and_scaler(dataset_path, order, n_per_policy):
    with open(dataset_path, 'rb') as f:
        episodes = list(ReplayBuffer.load(f, buffer=InfiniteBuffer()).episodes)
    configure_policy_layout(order, n_per_policy, len(episodes))
    indices = split_episode_indices(len(episodes))
    groups = [[episodes[int(i)] for i in part] for part in indices]
    obs, *_ = episodes_to_arrays(groups[0])
    mean, std = obs.mean(axis=0), obs.std(axis=0)
    std[std < 1e-6] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Invalid train-only scaler.')
    return episodes, indices, groups, mean, std


def record_split(indices, mean, std):
    return {'split_seed': SPLIT_SEED, 'train_indices': indices[0].tolist(),
        'val_indices': indices[1].tolist(), 'test_indices': indices[2].tolist(),
        'obs_mean': mean.tolist(), 'obs_std': std.tolist(),
        'train_index_sha256': hashlib.sha256(indices[0].astype('<i8').tobytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/rl_config.yaml')
    ap.add_argument('--dataset', default='data/processed/offline_rl_dataset.h5')
    ap.add_argument('--out', default='reports/bc_baseline_results.json')
    ap.add_argument('--model-dir', default='models/bc')
    ap.add_argument('--overwrite', action='store_true', help='Explicitly permit replacing existing checkpoints/reports.')
    ap.add_argument('--tune', action='store_true', help='16 validation-only candidates by default.')
    ap.add_argument('--exclude-random', action='store_true', help='Separate filtered-data BC baseline.')
    ap.add_argument('--selection-metric', choices=['overall', 'nonrandom'], default=None)
    ap.add_argument('--policy-order', nargs='+', default=POLICY_ORDER.copy())
    ap.add_argument('--n-per-policy', type=int, default=N_PER_POLICY)
    args = ap.parse_args()
    with open(args.config) as f:
        full_cfg = yaml.safe_load(f) or {}
    cfg = dict(full_cfg.get('bc', {}))
    refuse_existing_outputs(args.model_dir, 'bc', args.out, args.overwrite)
    # Preflight sidecar dependency before spending time on training.
    from src.rl.policy_io import save_scaler_sidecar
    episodes, indices, groups, mean, std = load_split_and_scaler(
        args.dataset, args.policy_order, args.n_per_policy)
    train_idx, val_idx, test_idx = indices
    train_eps, val_eps, test_eps = groups
    exclude = args.exclude_random or bool(cfg.get('exclude_random', False))
    if exclude:
        if 'random' not in POLICY_ORDER:
            raise ValueError('--exclude-random requested without a random policy label.')
        train_eps = [ep for ep, i in zip(train_eps, train_idx)
                     if policy_for_episode_index(i) != 'random']
    metric = args.selection_metric or cfg.get('selection_metric', 'nonrandom' if exclude else 'overall')
    cfg.update({'exclude_random': exclude, 'selection_metric': metric})

    # ---- env-var overrides injected by scripts/adapters.py ----
    # bc_baseline.py has no CLI flag for lr / hidden_units / n_steps; the
    # sweep adapter communicates them through the environment. Applied
    # here, after the YAML-derived cfg is final and before any training
    # call reads it.
    import os as _os
    _lr     = _os.environ.get("BC_LR_OVERRIDE")
    _steps  = _os.environ.get("BC_STEPS_OVERRIDE")
    _hidden = _os.environ.get("BC_HIDDEN_OVERRIDE")
    if _lr or _steps or _hidden:
        if _lr:
            cfg['learning_rate'] = float(_lr)
        if _steps:
            cfg['n_steps'] = int(_steps)
        if _hidden:
            cfg['hidden_units'] = [int(x) for x in _hidden.split(",")]
        print(f"[bc_baseline] env overrides applied: "
              f"learning_rate={cfg.get('learning_rate')} "
              f"n_steps={cfg.get('n_steps')} "
              f"hidden_units={cfg.get('hidden_units')}")
    # -----------------------------------------------------------

    train_ds = build_dataset(train_eps, mean, std)
    tuning = {'enabled': False}
    if args.tune:
        cfg, tuning = tune_bc(train_ds, val_eps, val_idx, mean, std, cfg, metric)
    rows = [run_one_seed(seed, train_ds, val_eps, val_idx, test_eps, test_idx,
             mean, std, cfg, Path(args.dataset), args.model_dir) for seed in TRAINING_SEEDS]
    vals = [r['val_accuracy']['overall'] for r in rows]
    tests = [r['test_accuracy']['overall'] for r in rows]
    summary = {'per_seed': rows, 'resolved_config': cfg, 'tuning': tuning,
        'val_accuracy_mean': float(np.mean(vals)), 'val_accuracy_std': float(np.std(vals)),
        'test_accuracy_mean': float(np.mean(tests)), 'test_accuracy_std': float(np.std(tests)),
        'std_convention': 'population standard deviation, ddof=0',
        'n_train_episodes': len(train_idx), 'n_train_episodes_for_fit': len(train_eps),
        'n_val_episodes': len(val_idx), 'n_test_episodes': len(test_idx),
        'split': record_split(indices, mean, std), 'split_seed': SPLIT_SEED,
        'training_seeds': TRAINING_SEEDS,
        'policy_order_assumption': POLICY_ORDER.copy(), 'n_per_policy_assumption': N_PER_POLICY,
        'dataset': args.dataset, 'model_dir': args.model_dir}
    write_json(args.out, summary)
    print(f'[bc_baseline] Test mean={np.mean(tests):.4f}; std={np.std(tests):.4f}')


if __name__ == '__main__':
    main()