"""
src/rl/train_cql.py

Phase 5: Discrete Conservative Q-Learning (CQL), trained on the same
Phase 4 offline dataset and the same episode-level split as
bc_baseline.py, for comparison against the BC baseline.

VERIFIED API (checked against d3rlpy 2.8.1 docs this session, not
assumed):
  - d3rlpy.algos.DiscreteCQLConfig(batch_size=32, gamma=0.99,
    observation_scaler=None, action_scaler=None, reward_scaler=None,
    compile_graph=False, learning_rate=6.25e-05, optim_factory=...,
    encoder_factory=..., q_func_factory=..., n_critics=1,
    target_update_interval=8000, alpha=1.0).
    IMPORTANT DIVERGENCE FROM bc_baseline.py: DiscreteCQL is a
    DoubleDQN-based value algorithm, not an actor-critic BC-style
    algorithm. Its conservative-penalty weight is the constructor
    argument `alpha` (default 1.0) -- NOT `beta` (that name belongs
    to DiscreteBC's regularization weight; see bc_baseline.py). Do
    not copy a `beta` key out of a `bc:` config section into `cql:`;
    read `alpha` from rl_config.yaml instead.
  - DiscreteCQLConfig(...).create(device=...) -> DiscreteCQL, with the
    same .fit(dataset, n_steps, n_steps_per_epoch, evaluators=..., ...)
    signature used by DiscreteBC in bc_baseline.py.
  - d3rlpy.metrics.InitialStateValueEstimationEvaluator(episodes=...)
    is a real, documented evaluator that can be passed inside the
    `evaluators` dict of .fit() to log the mean estimated Q-value at
    each episode's initial state over the course of training.
  - algo.save(path) / d3rlpy.load_learnable(path) is the documented
    save/load pair (confirmed via a working example this session);
    used here for saving each seed's model, NOT yet used for loading
    since that happens in fqe_eval.py / compare_policies.py.
  - d3rlpy.seed(seed) is the library's documented reproducibility entry
    point: it seeds Python's random, NumPy's global RNG, and PyTorch
    (CPU + CUDA) in one call.
  - d3rlpy.models.QRQFunctionFactory(n_quantiles=...) is the documented
    quantile-regression distributional Q-function factory, confirmed
    against the v2.8.1 "Use Distributional Q-Function" tutorial page
    this session. Passed as q_func_factory to DiscreteCQLConfig, same
    constructor slot the mean Q-function already used implicitly.

NOT independently verified in this session (flagged, not assumed
away -- CHECK THESE WHEN YOU FIRST RUN THIS):
  - The exact return type/shape of algo.fit(). This script does NOT
    rely on the return value at all -- it queries the evaluator
    results via a fresh, explicit call after training instead (see
    score_init_state_value() below).
  - d3rlpy.metrics.TDErrorEvaluator was NOT checked this session and
    is deliberately not used here.
  - Whether predict_value() on a QRQFunctionFactory-based algo returns
    the distribution mean by default (assumed, based on d3rlpy's
    general API consistency across mean/distributional Q-functions,
    but not spot-checked against a trained QR checkpoint this session).
    If downstream scripts (run_fqe_check.py, compare_policies.py) that
    call predict_value() on a QR-trained CQL checkpoint return
    nonsensical values, check this first.

BUGFIX 1 (earlier revision): d3rlpy.seed(seed) added at the start of
run_one_seed(), before the algo is constructed, and before each alpha
sweep candidate, so seeds/candidates no longer share one continuous,
unseeded RNG stream.

BUGFIX 2 (earlier revision) -- ALPHA-SELECTION BIAS:
  run_alpha_sweep() previously picked the "best" alpha by taking
  argmax over InitialStateValueEstimationEvaluator's score. This is
  WRONG, not just noisy -- see full reasoning below, unchanged from
  the prior revision. Alpha must be chosen explicitly (--alpha or
  cql.alpha), never auto-selected from this metric.

FIX 3 (THIS revision) -- DISTRIBUTIONAL CRITIC:
  Confirmed via a separate FQE-based investigation (run_fqe_check.py,
  seed-filtered against quintile_characterization_all_seeds.csv) that
  CQL's overvaluation of the high-crop_water_demand / action-3 bucket
  is NOT a CQL-training-specific bug: an independently trained FQE
  critic on the same data reaches the same conclusion (same-sign gap,
  z=2.03 for FQE vs z=2.21 for CQL). Root cause: a standard
  expected-value TD objective, applied to this bucket's bimodal reward
  distribution (near-ceiling most of the time, heavy negative tail --
  p10=-1.90, p1=-4.70, min=-7.48), averages toward optimism regardless
  of which value-based algorithm does the averaging.

  Fix: swap the default mean Q-function for a quantile-regression
  (QRQFunctionFactory) distributional Q-function, so the critic learns
  the full return distribution per state-action instead of collapsing
  it to a mean. This does not, by itself, change WHICH action CQL
  picks (predict() still uses the mean/expected value by default) --
  it makes the LOW-quantile estimate available for downstream risk-
  aware action selection or evaluation, which the mean-only critic
  could not provide at all. n_quantiles=200 matches the value used in
  d3rlpy's own v2.8.1 distributional-Q tutorial.

FIX 4 (THIS revision) -- NEGATIVE-TAIL OVERSAMPLING:
  Complementary to FIX 3, not a replacement: duplicates whole training
  episodes that contain at least one transition in the confirmed
  high-crop_water_demand / action==3 / severe-negative-reward region,
  so the critic sees more gradient signal from the rare catastrophic
  outcomes relative to the frequent near-ceiling ones. Operates on
  whole Episode objects (not flattened transition arrays) so episode
  boundaries / terminal-timeout structure stay intact under
  duplication. obs_mean/obs_std are computed from the ORIGINAL
  (pre-oversampling) train_episodes and only the oversampled list is
  passed to build_dataset() -- this keeps standardization statistics
  identical to what run_fqe_check.py independently reconstructs via
  load_training_standardization() (which also reads the ORIGINAL,
  non-oversampled dataset off disk), avoiding a repeat of the
  observation-scaling mismatch bug found earlier in this investigation.

REUSED FROM bc_baseline.py (same repo, same session -- deliberately
NOT reimplemented here, so the split/standardization logic can't
silently drift out of sync between BC and CQL):
  - SPLIT_SEED, TRAINING_SEEDS, split_episode_indices(): same fixed
    episode-level train/val/test partition BC used, so BC and CQL are
    compared on identical episodes.
  - episodes_to_arrays(), build_dataset(): same standardization
    discipline (mean/std fit on TRAIN observations only, applied
    unchanged to val/test).
  - action_accuracy(), policy_for_episode_index(), POLICY_ORDER,
    N_PER_POLICY: same ordering ASSUMPTION as bc_baseline.py (see that
    file's module docstring) -- used here only for a diagnostic
    action-match breakdown, never as a CQL selection criterion.

WHY ACTION-MATCH ACCURACY IS NOT THE CQL SUCCESS METRIC:
  Unlike BC, CQL is not trying to imitate the logged actions -- it is
  trying to find a policy that OUTPERFORMS them. A CQL policy that
  scores low on action-match against random-policy episodes is not a
  failure; it may correctly be refusing to imitate a bad action.
  action_accuracy() is computed and logged below purely as a
  diagnostic (e.g. to catch a degenerate policy that always picks one
  action), never as a selection or comparison criterion. The real
  comparison against BC happens via FQE + direct rollout, not here.

FIX 7 (THIS revision) -- TARGET-UPDATE-INTERVAL ABLATION HOOK:
  Added --target-update-interval as an explicit CLI override (mirrors
  the existing --qfunc / --oversample-factor override pattern) so the
  target-network-refresh-rate ablation can be run holding every other
  knob byte-for-byte identical to the current default run. No default
  behavior changes: omitting the flag still falls back to
  cql.target_update_interval in the config, or 8000 if that's absent
  too, exactly as before. Motivation: with target_update_interval=8000
  and n_steps=20000, the target network only refreshes ~2 times over
  the whole run, which is a plausible bottleneck for propagating a
  rare, backloaded negative-reward signal (concentrated in the final
  few of 20 timesteps, per reward_timing_profile.csv) back to the
  initial-state value -- independent of anything about reward shaping.
  This flag exists to test that hypothesis in isolation.

Usage:
    python src/rl/train_cql.py --config config/rl_config.yaml --alpha 4.0
    python src/rl/train_cql.py --config config/rl_config.yaml \
        --skip-sweep --alpha 4.0
    python src/rl/train_cql.py --config config/rl_config.yaml \
        --alpha 4.0 --target-update-interval 2000
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
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.optimizers import AdamFactory

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rl.bc_baseline import (
    SPLIT_SEED,
    TRAINING_SEEDS,
    POLICY_ORDER,
    N_PER_POLICY,
    split_episode_indices,
    episodes_to_arrays,
    build_dataset,
    action_accuracy,
)

N_STEPS = 20_000
N_STEPS_PER_EPOCH = 2_000

ALPHA_GRID = [0.5, 1.0, 4.0, 10.0]
SWEEP_N_STEPS = 5_000

# FIX 4: oversampling constants. See module docstring for provenance
# of these specific values (carried over from the FQE investigation).
CROP_DEMAND_IDX = 2  # gw_level, recent_rainfall, crop_water_demand, ...
NEGATIVE_TAIL_REWARD_CUTOFF = -1.90  # p10 of reward dist -- NOT recomputed
                                      # against this specific train split,
                                      # see oversample_negative_tail_episodes()
                                      # docstring
HIGH_DEMAND_QUANTILE = 0.75
DEFAULT_OVERSAMPLE_FACTOR = 3
DEFAULT_N_QUANTILES = 200  # FIX 3: matches d3rlpy's own tutorial default


def make_cql(alpha: float, cfg: dict) -> DiscreteCQLConfig:
    # FIX 3: distributional (quantile) critic instead of the default
    # mean Q-function. See module docstring FIX 3 for the full
    # reasoning -- confirmed via run_fqe_check.py that CQL's
    # overvaluation of the high-demand/action-3 bucket is inherent to
    # a mean-based TD objective on this bimodal reward, not a
    # CQL-training-specific bug.
    # FIX 6 / isolation toggle: allow falling back to the original
    # mean Q-function to test whether a training divergence is caused
    # by the QR critic itself vs. something else (target update,
    # alpha, data). See --qfunc CLI flag.
    qfunc_type = cfg.get("qfunc", "qr")
    if qfunc_type == "mean":
        q_func_factory = d3rlpy.models.MeanQFunctionFactory()
    else:
        q_func_factory = d3rlpy.models.QRQFunctionFactory(
            n_quantiles=cfg.get("n_quantiles", DEFAULT_N_QUANTILES)
        )
    # FIX 5: explicit gradient clipping. The step-8000 divergence
    # (td_loss ~24x jump right after the first target-network sync,
    # observed this session with QR critic + oversampling both on)
    # is consistent with an unclipped gradient spike from a few
    # extreme-reward transitions getting copied into the target
    # network. clip_grad_norm=1.0 caps that spike; this is a general
    # stability fix, not conditional on oversampling being on.
    optim_factory = AdamFactory(clip_grad_norm=cfg.get("clip_grad_norm", 1.0))
    return DiscreteCQLConfig(
        batch_size=cfg.get("batch_size", 100),
        gamma=cfg.get("gamma", 0.99),
        learning_rate=cfg.get("learning_rate", 6.25e-5),
        alpha=alpha,
        target_update_interval=cfg.get("target_update_interval", 8000),
        n_critics=cfg.get("n_critics", 1),
        q_func_factory=q_func_factory,
        optim_factory=optim_factory,
    ).create(device=False)


def oversample_negative_tail_episodes(episodes, oversample_factor=DEFAULT_OVERSAMPLE_FACTOR,
                                       demand_idx=CROP_DEMAND_IDX,
                                       reward_cutoff=NEGATIVE_TAIL_REWARD_CUTOFF,
                                       demand_quantile=HIGH_DEMAND_QUANTILE):
    """FIX 4: duplicates whole episodes that contain at least one
    transition in the high-crop_water_demand / action==3 /
    severe-negative-reward region -- the same region run_fqe_check.py
    confirmed CQL and an independent FQE critic both overvalue.

    Operates on whole Episode objects, not flattened arrays, so each
    duplicated episode keeps its own internal terminal/timeout
    structure intact -- oversampling flattened transition rows
    directly would risk breaking episode boundaries that MDPDataset
    relies on.

    NOT YET INDEPENDENTLY VERIFIED THIS SESSION: reward_cutoff=-1.90
    is carried over from the p10 figure reported earlier in this
    investigation, computed against the FULL dataset's reward
    distribution -- not recomputed here against this specific train
    split. If train/val/test splits differ meaningfully in their
    reward tails, recompute this cutoff from train_episodes directly
    before trusting it.
    """
    all_obs = np.concatenate([ep.observations for ep in episodes], axis=0)
    demand_cutoff = np.quantile(all_obs[:, demand_idx], demand_quantile)

    matching_episodes = []
    for ep in episodes:
        obs = ep.observations
        actions = ep.actions.reshape(-1)
        rewards = ep.rewards.reshape(-1)
        hits = (
            (obs[:, demand_idx] >= demand_cutoff)
            & (actions == 3)
            & (rewards <= reward_cutoff)
        )
        if hits.any():
            matching_episodes.append(ep)

    print(f"[train_cql] Oversampling: {len(matching_episodes)} of "
          f"{len(episodes)} train episodes contain a negative-tail "
          f"high-demand/action-3 transition (demand_cutoff={demand_cutoff:.4f}, "
          f"reward_cutoff={reward_cutoff}). Duplicating each "
          f"{oversample_factor}x.")

    return list(episodes) + matching_episodes * (oversample_factor - 1)


def score_init_state_value(algo, dataset, val_episodes_std: list) -> float:
    evaluator = InitialStateValueEstimationEvaluator(episodes=val_episodes_std)
    return float(evaluator(algo, dataset=dataset))


def run_alpha_sweep_diagnostic(train_ds, val_ds_std, val_episodes_std,
                                cfg: dict) -> dict:
    """Logs per-alpha InitialStateValueEstimationEvaluator scores as a
    DIAGNOSTIC ONLY. Does NOT select an alpha.

    WHY: this evaluator reports the network's own estimated Q-value at
    the initial state. CQL's alpha term exists to suppress exactly this
    quantity for actions it isn't confident are in-distribution, so
    argmax-ing over it is biased toward the least-conservative (smallest
    alpha) candidate by construction -- it does not tell you which
    alpha produces the best real policy. See BUGFIX 2 in the module
    docstring for the full explanation and what this replaced.

    Use this output only to sanity-check that training is proceeding
    (e.g. catch a candidate whose value estimate is diverging/NaN), not
    to rank candidates against each other.
    """
    grid = cfg.get("alpha_grid", ALPHA_GRID)
    sweep_seed = TRAINING_SEEDS[0]
    print(f"\n[train_cql] === Alpha sweep DIAGNOSTIC ONLY (seed={sweep_seed}, "
          f"{cfg.get('sweep_n_steps', SWEEP_N_STEPS)} steps/candidate) ===")
    print("[train_cql] NOTE: InitialStateValueEstimationEvaluator is "
          "structurally biased toward low alpha (it scores the network's "
          "own Q-estimate, which CQL's alpha term is designed to "
          "suppress). These numbers are logged for visibility only and "
          "are NOT used to pick alpha. See module docstring, BUGFIX 2.")

    sweep_results = []
    for alpha in grid:
        d3rlpy.seed(sweep_seed)
        algo = make_cql(alpha, cfg)
        algo.fit(
            train_ds,
            n_steps=cfg.get("sweep_n_steps", SWEEP_N_STEPS),
            n_steps_per_epoch=cfg.get("n_steps_per_epoch", N_STEPS_PER_EPOCH),
            show_progress=False,
        )
        score = score_init_state_value(algo, val_ds_std, val_episodes_std)
        print(f"[train_cql]   alpha={alpha:<6} init_state_value={score:.4f}  "
              f"(diagnostic only -- do not use to rank alpha)")
        sweep_results.append({"alpha": alpha, "init_state_value": score})

    return {"grid_results": sweep_results, "sweep_seed": sweep_seed,
            "selection_disabled_reason": (
                "InitialStateValueEstimationEvaluator is biased toward low "
                "alpha; auto-selection from this metric was removed. See "
                "module docstring BUGFIX 2. Alpha must be chosen via "
                "--alpha / cql.alpha in config, ideally validated by the "
                "full paired-bootstrap comparison in compare_policies.py.")}


def run_one_seed(seed: int, alpha: float, train_ds, val_episodes, val_indices,
                  test_episodes, test_indices, obs_mean, obs_std,
                  cfg: dict) -> dict:
    print(f"\n[train_cql] === Seed {seed} (alpha={alpha}) ===")

    d3rlpy.seed(seed)

    algo = make_cql(alpha, cfg)
    algo.fit(
        train_ds,
        n_steps=cfg.get("n_steps", N_STEPS),
        n_steps_per_epoch=cfg.get("n_steps_per_epoch", N_STEPS_PER_EPOCH),
        show_progress=True,
    )

    val_acc = action_accuracy(algo, val_episodes, obs_mean, obs_std, val_indices)
    test_acc = action_accuracy(algo, test_episodes, obs_mean, obs_std, test_indices)

    print(f"[train_cql] Seed {seed}: val_action_match={val_acc['overall']:.4f}, "
          f"test_action_match={test_acc['overall']:.4f}  "
          f"(diagnostic only -- see module docstring)")
    print(f"[train_cql] Seed {seed} val by policy: "
          + ", ".join(f"{p}={val_acc[p]:.4f}" for p in POLICY_ORDER))
    print(f"[train_cql] Seed {seed} test by policy: "
          + ", ".join(f"{p}={test_acc[p]:.4f}" for p in POLICY_ORDER))

    model_path = Path(cfg.get("model_dir", "models/cql")) / f"cql_seed{seed}.d3"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    algo.save(str(model_path))

    # BUGFIX (this revision) -- METADATA/CRITIC MISMATCH: this used to
    # hardcode "q_func_factory": "QRQFunctionFactory" regardless of the
    # --qfunc/cql.qfunc value actually passed to make_cql(). That meant
    # any run using --qfunc mean (e.g. probes B/D) had a saved sidecar
    # that silently lied about which critic trained the checkpoint.
    # Compute both fields from the same qfunc_used value make_cql() saw.
    qfunc_used = cfg.get("qfunc", "qr")
    meta_path = model_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "seed": seed,
            "alpha": alpha,
            "algo": "DiscreteCQL",
            "qfunc": qfunc_used,
            "q_func_factory": (
                "MeanQFunctionFactory" if qfunc_used == "mean"
                else "QRQFunctionFactory"
            ),
            "n_quantiles": (
                cfg.get("n_quantiles", DEFAULT_N_QUANTILES)
                if qfunc_used != "mean" else None
            ),
            "oversample_factor": cfg.get("oversample_factor", DEFAULT_OVERSAMPLE_FACTOR),
            "n_steps": cfg.get("n_steps", N_STEPS),
            "n_steps_per_epoch": cfg.get("n_steps_per_epoch", N_STEPS_PER_EPOCH),
            "batch_size": cfg.get("batch_size", 100),
            "gamma": cfg.get("gamma", 0.99),
            "learning_rate": cfg.get("learning_rate", 6.25e-5),
            "target_update_interval": cfg.get("target_update_interval", 8000),
            "n_critics": cfg.get("n_critics", 1),
            "clip_grad_norm": cfg.get("clip_grad_norm", 1.0),
            "val_action_match_overall": val_acc["overall"],
            "test_action_match_overall": test_acc["overall"],
            "seeded_via": "d3rlpy.seed(seed), called before .create()",
            "alpha_selected_via": "explicit --alpha / cql.alpha config value "
                                   "(NOT the init-state-value sweep -- see "
                                   "BUGFIX 2 in train_cql.py docstring)",
        }, f, indent=2)

    print(f"[train_cql] Seed {seed}: saved -> {model_path}")

    return {
        "seed": seed,
        "alpha": alpha,
        "val_action_match": val_acc,
        "test_action_match": test_acc,
        "model_path": str(model_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/rl_config.yaml")
    ap.add_argument("--dataset", default="data/processed/offline_rl_dataset.h5")
    ap.add_argument("--out", default="reports/cql_results.json")
    ap.add_argument("--model-dir", default=None,
                     help="Directory to save each seed's DiscreteCQL model + "
                          "metadata sidecar. Overrides cql.model_dir in the "
                          "config if given. Useful for running multiple "
                          "alpha candidates side by side without overwriting "
                          "each other, e.g. --model-dir models_alpha10.")
    ap.add_argument("--skip-sweep", action="store_true",
                     help="Skip the diagnostic sweep entirely (it no longer "
                          "selects alpha either way).")
    ap.add_argument("--alpha", type=float, default=None,
                     help="REQUIRED (here or via cql.alpha in config). Alpha "
                          "is no longer auto-selected -- see module "
                          "docstring BUGFIX 2.")
    ap.add_argument("--qfunc", choices=["qr", "mean"], default=None,
                     help="Override cql.qfunc from config. 'mean' reverts "
                          "to the original non-distributional critic, for "
                          "isolating whether a training divergence is "
                          "caused by the QR critic (FIX 3) itself.")
    ap.add_argument("--oversample-factor", type=int, default=None,
                     help="Override cql.oversample_factor from the config. "
                          "Pass 1 to disable oversampling entirely (FIX 4 "
                          "off, FIX 3 QR critic still on) -- useful for "
                          "isolating which fix caused a training "
                          "divergence.")
    ap.add_argument("--target-update-interval", type=int, default=None,
                     help="FIX 7: override cql.target_update_interval from "
                          "the config (default 8000 if neither is given). "
                          "Everything else in the run (alpha, qfunc, "
                          "oversample-factor, seeds, split) stays whatever "
                          "it would otherwise be, so this flag alone is "
                          "enough to run a target-update-interval ablation "
                          "with every other knob held fixed.")
    ap.add_argument("--n-steps", type=int, default=None,
                     help="FIX 8: override cql.n_steps (default 20000 if "
                          "neither is given) for BOTH the alpha-sweep-"
                          "skipped seed-training loop's total step count. "
                          "Does NOT touch cql.sweep_n_steps (the alpha "
                          "sweep's own step count, default 5000) -- use "
                          "--skip-sweep to bypass that phase entirely if "
                          "you only want a short probe of the main "
                          "training loop. Intended for quickly screening "
                          "whether a config (e.g. a target_update_interval "
                          "/ qfunc / oversample-factor combination) is "
                          "diverging before committing the full 20k-step, "
                          "3-seed budget to it -- divergence in this repo's "
                          "observed cases has shown up within the first "
                          "4-8k steps (2-4 epochs at the default "
                          "n_steps_per_epoch), well before 20k.")
    ap.add_argument("--n-steps-per-epoch", type=int, default=None,
                     help="FIX 8: override cql.n_steps_per_epoch (default "
                          "2000 if neither is given). Only changes the "
                          "number of training steps represented by each "
                          "logged epoch (i.e. how often progress/metrics "
                          "are printed) -- it does not change the total "
                          "training length by itself, and it does NOT "
                          "checkpoint the model per epoch: the only model "
                          "save is the single algo.save() call after "
                          "training completes. Combine with --n-steps for "
                          "a short probe, e.g. --n-steps 8000 "
                          "--n-steps-per-epoch 2000 for 4 logged epochs.")
    args = ap.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    cql_cfg = full_cfg.get("cql", {})

    if args.model_dir is not None:
        cql_cfg["model_dir"] = args.model_dir
    if args.oversample_factor is not None:
        cql_cfg["oversample_factor"] = args.oversample_factor
    if args.qfunc is not None:
        cql_cfg["qfunc"] = args.qfunc
    if args.target_update_interval is not None:
        cql_cfg["target_update_interval"] = args.target_update_interval
    if args.n_steps is not None:
        cql_cfg["n_steps"] = args.n_steps
    if args.n_steps_per_epoch is not None:
        cql_cfg["n_steps_per_epoch"] = args.n_steps_per_epoch

    alpha = args.alpha if args.alpha is not None else cql_cfg.get("alpha")
    if alpha is None:
        raise SystemExit(
            "[train_cql] ERROR: no alpha specified. Pass --alpha, or set "
            "cql.alpha in the config file. Auto-selection via the init-"
            "state-value sweep was removed because it is structurally "
            "biased toward low (less conservative) alpha values -- see "
            "BUGFIX 2 in this file's module docstring for why. If you "
            "need to choose among ALPHA_GRID candidates, train a full "
            "seed set for each candidate and compare them with the "
            "paired-bootstrap evaluation in compare_policies.py instead "
            "of a cheap proxy."
        )

    print(f"[train_cql] Loading dataset from {args.dataset}...")
    with open(args.dataset, "rb") as f:
        ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(ds.episodes)
    n_episodes = len(episodes)
    print(f"[train_cql] Loaded {n_episodes} episodes.")

    train_idx, val_idx, test_idx = split_episode_indices(n_episodes)
    print(f"[train_cql] Episode split (fixed seed={SPLIT_SEED}, shared "
          f"with bc_baseline.py): {len(train_idx)} train / "
          f"{len(val_idx)} val / {len(test_idx)} test")

    train_episodes = [episodes[i] for i in train_idx]
    val_episodes = [episodes[i] for i in val_idx]
    test_episodes = [episodes[i] for i in test_idx]

    # obs_mean/obs_std computed from the ORIGINAL train_episodes, BEFORE
    # any oversampling -- see FIX 4 in the module docstring for why this
    # ordering matters (keeps standardization in sync with what
    # run_fqe_check.py independently reconstructs from the non-
    # oversampled dataset on disk).
    train_obs_raw, _, _, _, _ = episodes_to_arrays(train_episodes)
    obs_mean = train_obs_raw.mean(axis=0)
    obs_std = train_obs_raw.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0
    print(f"[train_cql] Train obs mean: {obs_mean}")
    print(f"[train_cql] Train obs std:  {obs_std}")

    # FIX 4: oversample negative-tail high-demand/action-3 episodes for
    # TRAINING ONLY. val_episodes/test_episodes and val_ds_std below all
    # continue to use the original, non-oversampled episode lists, so
    # evaluation metrics stay comparable to prior runs.
    train_episodes_for_fit = oversample_negative_tail_episodes(
        train_episodes, oversample_factor=cql_cfg.get("oversample_factor", DEFAULT_OVERSAMPLE_FACTOR)
    )
    train_ds = build_dataset(train_episodes_for_fit, obs_mean, obs_std)
    val_ds_std = build_dataset(val_episodes, obs_mean, obs_std)
    val_episodes_std = list(val_ds_std.episodes)

    print(f"[train_cql] Using alpha={alpha} "
          f"(source: {'--alpha flag' if args.alpha is not None else 'cql.alpha config'})")
    print(f"[train_cql] Using target_update_interval="
          f"{cql_cfg.get('target_update_interval', 8000)} "
          f"(source: {'--target-update-interval flag' if args.target_update_interval is not None else 'cql.target_update_interval config / default'})")
    print(f"[train_cql] Using n_steps={cql_cfg.get('n_steps', N_STEPS)}, "
          f"n_steps_per_epoch={cql_cfg.get('n_steps_per_epoch', N_STEPS_PER_EPOCH)} "
          f"(source: {'--n-steps/--n-steps-per-epoch flag(s)' if (args.n_steps is not None or args.n_steps_per_epoch is not None) else 'cql.n_steps/n_steps_per_epoch config / default'})")
    if cql_cfg.get('n_steps', N_STEPS) < N_STEPS:
        print(f"[train_cql] NOTE: n_steps ({cql_cfg.get('n_steps', N_STEPS)}) is below the "
              f"default full-training length ({N_STEPS}). This run is a SHORT PROBE -- "
              f"treat its val/test action-match and saved checkpoint as a divergence/"
              f"stability screen only, not as a final comparable result against full "
              f"20k-step runs.")

    if args.skip_sweep:
        sweep_summary = {"skipped": True}
    else:
        sweep_summary = run_alpha_sweep_diagnostic(
            train_ds, val_ds_std, val_episodes_std, cql_cfg)

    results = []
    for seed in TRAINING_SEEDS:
        result = run_one_seed(seed, alpha, train_ds, val_episodes, val_idx,
                               test_episodes, test_idx, obs_mean, obs_std, cql_cfg)
        results.append(result)

    val_overall = [r["val_action_match"]["overall"] for r in results]
    test_overall = [r["test_action_match"]["overall"] for r in results]
    summary = {
        "alpha_sweep_diagnostic": sweep_summary,
        "alpha_used": alpha,
        "alpha_source": "--alpha flag" if args.alpha is not None else "cql.alpha config",
        "target_update_interval_used": cql_cfg.get("target_update_interval", 8000),
        "target_update_interval_source": (
            "--target-update-interval flag" if args.target_update_interval is not None
            else "cql.target_update_interval config / default"),
        "n_steps_used": cql_cfg.get("n_steps", N_STEPS),
        "n_steps_per_epoch_used": cql_cfg.get("n_steps_per_epoch", N_STEPS_PER_EPOCH),
        "is_short_probe": cql_cfg.get("n_steps", N_STEPS) < N_STEPS,
        "qfunc": cql_cfg.get("qfunc", "qr"),
        "q_func_factory": (
            "MeanQFunctionFactory" if cql_cfg.get("qfunc", "qr") == "mean"
            else "QRQFunctionFactory"
        ),
        "n_quantiles": (
            cql_cfg.get("n_quantiles", DEFAULT_N_QUANTILES)
            if cql_cfg.get("qfunc", "qr") != "mean" else None
        ),
        "clip_grad_norm": cql_cfg.get("clip_grad_norm", 1.0),
        "oversample_factor": cql_cfg.get("oversample_factor", DEFAULT_OVERSAMPLE_FACTOR),
        "model_dir": cql_cfg.get("model_dir", "models/cql"),
        "per_seed": results,
        "val_action_match_mean": float(np.mean(val_overall)),
        "val_action_match_std": float(np.std(val_overall)),
        "test_action_match_mean": float(np.mean(test_overall)),
        "test_action_match_std": float(np.std(test_overall)),
        "n_train_episodes": len(train_idx),
        "n_train_episodes_after_oversampling": len(train_episodes_for_fit),
        "n_val_episodes": len(val_idx),
        "n_test_episodes": len(test_idx),
        "split_seed": SPLIT_SEED,
        "training_seeds": TRAINING_SEEDS,
        "policy_order_assumption": POLICY_ORDER,
        "n_per_policy_assumption": N_PER_POLICY,
    }

    print(f"\n[train_cql] === Summary across 3 seeds (alpha={alpha}, "
          f"target_update_interval={cql_cfg.get('target_update_interval', 8000)}) ===")
    print(f"  Val action-match:  {summary['val_action_match_mean']:.4f} "
          f"± {summary['val_action_match_std']:.4f}  (diagnostic only)")
    print(f"  Test action-match: {summary['test_action_match_mean']:.4f} "
          f"± {summary['test_action_match_std']:.4f}  (diagnostic only)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train_cql] Saved -> {args.out}")


if __name__ == "__main__":
    main()