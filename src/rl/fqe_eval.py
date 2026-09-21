"""
src/rl/fqe_eval.py

Phase 5b: Fitted Q Evaluation (FQE) -- independent validation of the
direct-rollout comparison in compare_policies.py.

WHY THIS EXISTS
----------------
compare_policies.py estimates BC/CQL policy value by rolling both
policies out through the exact simulator (synthetic_reward.step()).
That is a strong evaluation *of the simulator's response to each
policy*, but it says nothing about whether CQL's own Q-function is
trustworthy off the training distribution, and it can't be run at all
in a real deployment setting where you don't have a differentiable/
exact simulator to roll out against -- only the logged offline dataset.

FQE answers a different question: "using ONLY the offline dataset
(no simulator, no environment interaction), what does an independently
trained critic estimate this policy's value to be?" It fits a Q-function
via iterated Bellman backups where the BOOTSTRAPPED ACTION at each next-
state comes from the policy being evaluated (BC or CQL), not from the
dataset's logged action -- this is what distinguishes FQE from ordinary
TD-learning on the dataset.

If FQE's off-policy value estimates agree in direction/magnitude with
compare_policies.py's direct-rollout results, that's independent
validation the Stage 3 (epsilon-coverage) CQL>BC finding is not a
simulator/rollout-only artifact. If they disagree, that's a signal to
investigate further (e.g. CQL Q-overestimation despite the alpha
penalty, or insufficient offline coverage even after epsilon-noise).

============================================================================
API VERIFICATION STATUS -- READ BEFORE RUNNING
============================================================================
VERIFIED against a live d3rlpy 2.8.1 install this session (via
inspect.signature() on d3rlpy.ope.FQEConfig.__init__/.create,
d3rlpy.ope.DiscreteFQE.__init__/.fit -- not assumed):
  - There is NO DiscreteFQEConfig class. The real names are
    d3rlpy.ope.FQEConfig (a single config class, hyperparameters only:
    batch_size, gamma, learning_rate, target_update_interval, n_critics,
    etc. -- same hyperparameter shape as DiscreteCQLConfig, but with NO
    target-policy argument anywhere in the config) and
    d3rlpy.ope.DiscreteFQE / d3rlpy.ope.FQE (the actual algo classes for
    discrete/continuous action spaces respectively).
  - `n_critics` IS a real, live field on FQEConfig -- confirmed directly
    from the printed `params` dict of an actual run this session
    (params={'observation_shape': [6], ..., 'config': {'type': 'fqe',
    'params': {..., 'n_critics': 1, 'target_update_interval': 2000}}}),
    not just from a docs guess. It was previously left at its default
    (1) because the script didn't expose it as a knob -- see
    INVESTIGATION NOTE below for why that's now being tested.
  - FQEConfig(...).create(device=..., enable_ddp=...) exists but takes
    NO algo argument -- it is not how the target policy gets attached.
  - The target policy is passed directly to DiscreteFQE's constructor:
    DiscreteFQE(algo=target_algo, config=fqe_config, device=...).
    This is the one genuinely non-obvious part of the real API: FQE is
    instantiated directly (not via its config's .create()), with the
    already-trained BC/CQL algo as its first argument.
  - DiscreteFQE.fit(dataset, n_steps, n_steps_per_epoch=...,
    evaluators=..., show_progress=..., ...) matches the same call
    convention as DiscreteCQL.fit()/DiscreteBC.fit() used elsewhere in
    this repo.
  - The dataset action space in this project's offline dataset is
    confirmed DISCRETE (action_size=5, auto-detected by d3rlpy from
    the .h5 file at load time), so DiscreteFQE (not the continuous
    FQE class) is the correct one to import and use here.
  - d3rlpy.optimizers.AdamFactory accepts a clip_grad_norm kwarg (visible
    directly in the printed params dict's optim_factory block:
    {'type': 'adam', 'params': {'clip_grad_norm': None, ...}}) and is
    passed into FQEConfig via its optim_factory argument -- this is how
    gradient clipping gets applied to the FQE critic's optimizer (see
    INVESTIGATION NOTE 2 below).

STILL NOT independently verified this session (only confirmed the
signature exists -- not that the underlying computation is correct for
this project's use case; sanity-check numbers before scaling up):
  - InitialStateValueEstimationEvaluator(episodes=...), when called on
    the TRAINED FQE CRITIC (not the original BC/CQL algo) via
    evaluator(fqe, dataset=train_ds_std), is assumed to return the
    critic's mean value estimate at each episode's t=0 observation
    under target_algo's action -- not re-verified beyond the signature
    existing. If the fqe_init_state_value numbers look implausible
    relative to compare_policies.py's return_mean values for the same
    policies, suspect this evaluator call first.

INVESTIGATION NOTE 1 -- divergence, n_critics=1 vs n_critics=2:
  A seed-42-only run at target_update_interval=2000 (n_steps=100000,
  stopped early once the pattern was clear) showed init_state_value
  climbing ~linearly (+0.8/epoch: 0.80, 1.66, 2.50, 3.30, 4.12, 4.91,
  5.64...) with LOSS ALSO GROWING each epoch (0.011, 0.015, 0.025,
  0.041, 0.064, 0.095, 0.142...) rather than shrinking. This ruled out
  "just needs more target refreshes" -- a shorter interval (more
  refreshes within the same step budget) produced the same steady
  per-refresh growth rate as the original long-interval run, not a
  shrinking one.

  n_critics=2 was tested as a targeted, well-established mitigation for
  single-critic Q-overestimation (double-Q-style pessimistic
  bootstrapping), DISTINCT from CQL's alpha-based conservatism. It made
  essentially no difference (n_critics=1 vs n_critics=2 trajectories
  were nearly identical epoch-for-epoch, with n_critics=2 loss even
  slightly higher) -- this ruled out single-critic overestimation as
  the driver.

INVESTIGATION NOTE 2 -- overshoot-then-decline, and the grad-clip/LR fix:
  Extending the seed-42 (n_critics=1) run to 40000 steps / 20 epochs
  revealed the full shape the earlier short runs couldn't show: value
  climbs smoothly through epoch 17 (BC: 10.10, CQL: 9.78), THEN REVERSES
  and declines for the remaining epochs (BC ended at 9.47, CQL at 9.27,
  both still falling at epoch 20). BC and CQL peaked at the SAME epoch
  and declined at similar rates despite being independent critics fit to
  different target policies -- this rules out "specific to one policy's
  action distribution" and points at the FQE fitting setup itself
  (learning rate / gradient scale / target_update_interval interaction).

  Critically, loss climbed EVERY epoch for both fits, all the way
  through 40000 steps (BC: 0.029 -> 1.567; CQL: 0.035 -> 1.251), with no
  sign of turning over even as the value estimate was already reversing
  direction. Growing loss throughout is a standard signature of
  gradient steps that are too large relative to how fast the target
  moves -- each update overshoots, the next target is fit from an
  already-overshot online network, and the fit never settles.

  This is also why the crude "last-3-epoch std" convergence heuristic
  gave a FALSE POSITIVE: it flagged the seed-42 BC fit "CONVERGED" at
  epoch 20 because the std of (9.94, 9.73, 9.47) is small in absolute
  terms, even though those three points are monotonically decreasing
  after a clear peak three epochs earlier -- i.e. still moving, just
  slowly enough over 3 points to look flat. Fixed below by additionally
  checking for a sign reversal among the last few deltas.

  Two changes applied to test the gradient-scale hypothesis directly:
    1. optim_factory=AdamFactory(clip_grad_norm=1.0) on FQEConfig
       (previously unclipped -- 'clip_grad_norm': None in every run's
       printed params).
    2. learning_rate lowered from d3rlpy's Atari-derived default
       (6.25e-5) to 1e-5, both now overridable via --grad-clip and
       --learning-rate.
  If this tames the loss growth and produces a value curve that
  flattens instead of overshooting, that confirms a gradient-scale
  issue. If the same overshoot-then-decline shape persists even with
  clipping and a lower LR, that argues instead for a more structural
  fix (e.g. decoupling target_update_interval from n_steps_per_epoch so
  the critic gets more gradient steps to actually settle onto each
  target before it moves), and the overshoot itself becomes a finding
  about FQE's applicability here rather than something to keep tuning
  around.

REUSED FROM bc_baseline.py / compare_policies.py (NOT reimplemented,
same discipline as train_cql.py's own docstring insists on -- so the
split/standardization logic cannot silently drift out of sync across
BC, CQL, and now FQE):
  - SPLIT_SEED, TRAINING_SEEDS, split_episode_indices()
  - episodes_to_arrays() for recovering train-only obs mean/std
  - load_d3rlpy_model() / model path convention (--bc-dir, --cql-dir,
    same flags and defaults as compare_policies.py)

WHAT THIS SCRIPT DOES NOT DO
  - It does not re-derive alpha, retrain BC/CQL, or touch the
    simulator. It only reads already-saved .d3 model files and the
    already-generated offline dataset.
  - It does not replace compare_policies.py's bootstrap CI -- it
    produces an independent, offline-only estimate to compare
    against that CI's observed_difference, using the SAME paired-
    bootstrap machinery for apples-to-apples uncertainty bars.

Usage:
    # Smoke test first (fast, verifies the API assumptions above run
    # at all before spending real compute):
    python -m src.rl.fqe_eval --n-steps 200 --n-steps-per-epoch 100 \
        --bc-dir models_epsilon/bc --cql-dir models_epsilon/cql \
        --dataset data/processed/offline_rl_dataset_epsilon.h5 \
        --out reports/fqe_epsilon_SMOKETEST.json

    # Quick single-seed convergence/divergence check (no report written):
    python -m src.rl.fqe_eval --bc-dir models_epsilon/bc --cql-dir models_epsilon/cql \
        --dataset data/processed/offline_rl_dataset_epsilon.h5 \
        --only-seed 42 --n-steps 40000 --n-steps-per-epoch 2000 \
        --target-update-interval 2000 --n-critics 1 \
        --learning-rate 1e-5 --grad-clip 1.0

    # Real run, matching the Stage 3 (epsilon-coverage) comparison in
    # run_phase5_experiments.ps1:
    python -m src.rl.fqe_eval \
        --bc-dir models_epsilon/bc --cql-dir models_epsilon/cql \
        --dataset data/processed/offline_rl_dataset_epsilon.h5 \
        --out reports/fqe_epsilon.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import d3rlpy
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rl.bc_baseline import (
    SPLIT_SEED,
    TRAINING_SEEDS,
    split_episode_indices,
    episodes_to_arrays,
)

DEFAULT_MODEL_DIR = _REPO_ROOT / "models"
DEFAULT_DATASET = _REPO_ROOT / "data" / "processed" / "offline_rl_dataset.h5"
DEFAULT_OUT = _REPO_ROOT / "reports" / "fqe_results.json"

FQE_N_STEPS = 20_000
FQE_N_STEPS_PER_EPOCH = 2_000
DEFAULT_N_BOOTSTRAP = 5000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_CRITICS = 1  # d3rlpy's own default; kept explicit here so the
                        # --n-critics flag's default is visible in one place
DEFAULT_LEARNING_RATE = 1e-5  # lowered from d3rlpy's Atari-derived default
                               # (6.25e-5) after the seed-42/40k-step run
                               # showed overshoot-then-decline with loss
                               # climbing every epoch at the higher LR --
                               # see module docstring INVESTIGATION NOTE 2.
DEFAULT_GRAD_CLIP = 1.0  # d3rlpy's own default is unclipped (None); added
                          # alongside the lower LR as the first thing to
                          # test against that same overshoot pattern.


def load_dataset_and_split(dataset_path: Path):
    with dataset_path.open("rb") as f:
        ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())
    episodes = list(ds.episodes)
    train_idx, val_idx, test_idx = split_episode_indices(len(episodes), seed=SPLIT_SEED)
    return ds, episodes, train_idx, val_idx, test_idx


def recover_standardization(episodes, train_idx):
    train_episodes = [episodes[i] for i in train_idx]
    train_obs_raw, _, _, _, _ = episodes_to_arrays(train_episodes)
    obs_mean = train_obs_raw.mean(axis=0).astype(np.float32)
    obs_std = train_obs_raw.std(axis=0).astype(np.float32)
    obs_std[obs_std < 1e-6] = 1.0
    return obs_mean, obs_std


def build_standardized_replay_buffer(episodes_subset, obs_mean, obs_std):
    """Rebuilds a ReplayBuffer of standardized episodes, matching the
    exact approach build_dataset() uses in bc_baseline.py/train_cql.py
    (train-only mean/std applied unchanged here) -- reimplemented
    minimally rather than imported because build_dataset() there is
    scoped to the train split only; this needs it for arbitrary splits
    (test, for FQE's initial-state evaluation set).
    """
    from src.rl.bc_baseline import build_dataset  # local import: avoid
    # circular import at module load time, since build_dataset() itself
    # imports d3rlpy dataset internals lazily too.
    return build_dataset(episodes_subset, obs_mean, obs_std)


def action_agreement_diagnostic(algo_a, algo_b, episodes_std: list) -> Dict[str, Any]:
    """DIAGNOSTIC added after FQE (20k steps) showed CQL-BC ~ 0 while
    compare_policies.py's direct rollout showed CQL-BC = +0.11. Checks a
    much cheaper, more direct question before trusting/distrusting FQE's
    convergence: at each test episode's INITIAL observation, do BC and
    CQL even choose different actions?

    If agreement is high (e.g. >80%), a near-zero FQE value gap at t=0 is
    at least partly EXPECTED regardless of FQE's convergence quality --
    two policies that act identically at t=0 will only diverge in value
    to the extent their actions differ LATER in the trajectory, and a
    correctly-fit Q(s0, a0) should already price that in, but identical
    a0 is still consistent with a smaller true gap than the rollout
    aggregate suggests. If agreement is LOW and FQE still shows ~0 gap,
    that points more strongly at an FQE fitting/convergence problem
    rather than a genuine null result.

    This does not resolve the disagreement by itself -- it narrows which
    explanation to chase first.
    """
    agree = 0
    total = 0
    for ep in episodes_std:
        obs0 = np.asarray(ep.observations[0], dtype=np.float32).reshape(1, -1)
        a_a = int(np.asarray(algo_a.predict(obs0)).reshape(-1)[0])
        a_b = int(np.asarray(algo_b.predict(obs0)).reshape(-1)[0])
        agree += int(a_a == a_b)
        total += 1
    rate = agree / total if total else float("nan")
    return {"n_episodes": total, "n_agree": agree, "agreement_rate": rate}


def load_d3rlpy_model(model_path: Path, label: str):
    if not model_path.exists():
        raise FileNotFoundError(f"{label} model does not exist:\n{model_path}")
    print(f"[fqe_eval] Loading {label}: {model_path}")
    return d3rlpy.load_learnable(str(model_path), device="cpu")


def fit_fqe_for_policy(
    target_algo: Any,
    train_ds_std,
    eval_episodes_std: list,
    cfg: Dict[str, Any],
    label: str,
) -> Dict[str, Any]:
    """Fits an independent FQE critic for `target_algo`'s policy using
    ONLY the offline dataset (train_ds_std), then reads off the
    critic's mean estimated value at eval_episodes_std's initial
    states.

    n_critics defaults to 1 (d3rlpy's own default) unless cfg overrides
    it -- n_critics=2 was tested and ruled out as a fix for the
    overshoot/instability seen here (see module docstring INVESTIGATION
    NOTE 1). The optimizer now defaults to a lower learning rate and
    clipped gradients (see INVESTIGATION NOTE 2) as the current best
    guess at the actual fix; both remain overridable via cfg so this
    can still be swept.
    """
    from d3rlpy.ope import DiscreteFQE, FQEConfig
    from d3rlpy.metrics import InitialStateValueEstimationEvaluator
    from d3rlpy.optimizers import AdamFactory

    print(f"\n[fqe_eval] === Fitting FQE critic for {label} "
          f"(n_critics={cfg.get('n_critics', DEFAULT_N_CRITICS)}, "
          f"lr={cfg.get('learning_rate', DEFAULT_LEARNING_RATE)}, "
          f"grad_clip={cfg.get('grad_clip', DEFAULT_GRAD_CLIP)}) ===")

    fqe_config = FQEConfig(
        batch_size=cfg.get("batch_size", 100),
        gamma=cfg.get("gamma", 0.99),
        learning_rate=cfg.get("learning_rate", DEFAULT_LEARNING_RATE),
        target_update_interval=cfg.get("target_update_interval", 8000),
        n_critics=cfg.get("n_critics", DEFAULT_N_CRITICS),
        optim_factory=AdamFactory(
            clip_grad_norm=cfg.get("grad_clip", DEFAULT_GRAD_CLIP)
        ),
    )
    # target_algo's actions (not the dataset's logged actions) define the
    # bootstrap target at each next-state -- this is what makes it FQE
    # rather than plain TD-learning on the dataset.
    fqe = DiscreteFQE(algo=target_algo, config=fqe_config, device="cpu")

    # CONVERGENCE DIAGNOSTIC (added after the epsilon-coverage run showed
    # FQE disagreeing with compare_policies.py's rollout result -- see
    # conversation notes / reports/fqe_epsilon.json vs
    # reports/epsilon/policy_comparison.json). Passing the evaluator INTO
    # .fit() via evaluators=... logs the init-state-value estimate at
    # EVERY epoch, not just at the end -- this tells us whether the final
    # number is a converged fixed point or still drifting when training
    # stopped. A single post-hoc evaluator call (the previous approach)
    # cannot distinguish those two cases.
    evaluator = InitialStateValueEstimationEvaluator(episodes=eval_episodes_std)
    fit_log = fqe.fit(
        train_ds_std,
        n_steps=cfg.get("n_steps", FQE_N_STEPS),
        n_steps_per_epoch=cfg.get("n_steps_per_epoch", FQE_N_STEPS_PER_EPOCH),
        evaluators={"init_state_value": evaluator},
        show_progress=True,
    )
    # NOTE: the debug print below was previously (incorrectly) placed
    # INSIDE the .fit(...) call above, as if it were one of its
    # arguments -- that's a syntax error, and even if it somehow parsed
    # it would have referenced fit_log before the assignment that
    # creates it existed. It must go here, as its own statement, AFTER
    # fit_log is actually assigned.
    print(f"[DEBUG] fit_log type={type(fit_log)}, "
          f"first entry={fit_log[0] if fit_log else 'EMPTY'}")

    # fit_log is a list of (epoch_step, {metric_name: value}) tuples per
    # d3rlpy's documented .fit() return type -- extract the per-epoch
    # trajectory of the value estimate.
    value_trajectory = [
        (step, metrics.get("init_state_value")) for step, metrics in fit_log
    ]
    init_state_value = float(value_trajectory[-1][1]) if value_trajectory else float("nan")

    # Convergence check: compare the last few epochs' values for both
    # (a) small absolute movement AND (b) no sign reversal among the
    # recent deltas.
    #
    # Originally this only checked (a) via the std of the last 3 values
    # against a relative/absolute threshold. That gave a FALSE POSITIVE
    # on the seed-42/40k-step BC run: init_state_value peaked at epoch 17
    # (10.10) then declined for 3 straight epochs to 9.47, and the std of
    # those last 3 declining values (9.94, 9.73, 9.47) was small enough
    # in absolute terms to trip the old threshold and get flagged
    # "CONVERGED" -- even though the trajectory had clearly not settled,
    # it was still moving, just slowly enough over a 3-point window to
    # look flat. See module docstring INVESTIGATION NOTE 2.
    #
    # Widened to the last 4 values (3 deltas) so a genuine sign reversal
    # can be distinguished from noise -- 2 deltas can't tell "still
    # turning" from "just noisy".
    last_vals = [v for _, v in value_trajectory[-4:] if v is not None]
    if len(last_vals) >= 2:
        deltas = [b - a for a, b in zip(last_vals, last_vals[1:])]
        drift = float(np.std(last_vals))
        small_and_flat = drift < max(0.02 * abs(np.mean(last_vals)), 0.01)
        signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in deltas if d != 0}
        no_reversal = len(signs) <= 1
        converged = small_and_flat and no_reversal
    else:
        drift, converged = float("nan"), False

    print(f"[fqe_eval] {label}: value trajectory (epoch_step, estimate) = "
          f"{value_trajectory}")
    print(f"[fqe_eval] {label}: FQE initial-state value estimate = "
          f"{init_state_value:.4f}  [last-4-epoch std={drift:.4f}, "
          f"{'CONVERGED' if converged else 'NOT CONVERGED -- treat with suspicion'}]")

    return {
        "label": label,
        "fqe_init_state_value": init_state_value,
        "value_trajectory": value_trajectory,
        "last_4_epoch_std": drift,
        "converged_heuristic": converged,
        "n_steps": cfg.get("n_steps", FQE_N_STEPS),
        "n_steps_per_epoch": cfg.get("n_steps_per_epoch", FQE_N_STEPS_PER_EPOCH),
        "gamma": cfg.get("gamma", 0.99),
        "n_critics": cfg.get("n_critics", DEFAULT_N_CRITICS),
        "learning_rate": cfg.get("learning_rate", DEFAULT_LEARNING_RATE),
        "grad_clip": cfg.get("grad_clip", DEFAULT_GRAD_CLIP),
    }


def paired_bootstrap_scalar_difference(
    values_a: List[float], values_b: List[float], n_bootstrap: int, seed: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Dict[str, Any]:
    """Same paired-bootstrap logic as compare_policies.py's
    paired_bootstrap_difference(), applied here to the small
    per-seed FQE value-estimate arrays (one scalar per seed, e.g. 3
    BC-vs-CQL differences) rather than per-episode rollout returns.
    Kept as a separate, minimal reimplementation rather than importing
    compare_policies.py, since that module has an unrelated,
    heavier import chain (matplotlib, the real simulator, RF table
    loading) that this script does not need.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires equal-length arrays.")
    if len(a) < 2:
        raise ValueError("At least two paired values (e.g. seeds) are required.")

    diff = a - b
    observed = float(np.mean(diff))

    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_bootstrap, len(diff)))
    boot_diffs = np.mean(diff[idx], axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "observed_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_contains_zero": bool(lower <= 0.0 <= upper),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "n_paired_values": int(len(diff)),
        "caveat": (
            "This bootstrap resamples across only len(values) paired points "
            "(typically 3 training seeds) -- treat the resulting CI as "
            "illustrative of seed variability, not a substitute for the "
            "much larger per-episode bootstrap in compare_policies.py."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FQE-based off-policy validation of BC vs CQL.")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                     help="Parent dir with bc/ and cql/ subfolders. Only used "
                          "to derive --bc-dir/--cql-dir when not given "
                          "explicitly (same convention as compare_policies.py).")
    ap.add_argument("--bc-dir", default=None)
    ap.add_argument("--cql-dir", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-steps", type=int, default=FQE_N_STEPS)
    ap.add_argument("--n-steps-per-epoch", type=int, default=FQE_N_STEPS_PER_EPOCH)
    ap.add_argument("--target-update-interval", type=int, default=8000,
                     help="Steps between FQE target-network refreshes. The "
                          "epsilon-run diagnostic showed the value estimate "
                          "jumping by ~0.9 at each refresh with the default "
                          "8000 (only 2 refreshes across 20k steps) and NOT "
                          "settling -- try something like 1000-2000 so there "
                          "are 10-20 refreshes to actually observe whether "
                          "the jump size shrinks toward a fixed point.")
    ap.add_argument("--n-critics", type=int, default=DEFAULT_N_CRITICS,
                     help="Number of critics FQE fits (d3rlpy default: 1). "
                          "Tested at n_critics=2 as a fix for suspected "
                          "single-critic Q-overestimation -- made "
                          "essentially no difference (near-identical "
                          "trajectories to n_critics=1, loss slightly "
                          "higher), which ruled that hypothesis out. See "
                          "module docstring INVESTIGATION NOTE 1.")
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                     help="FQE critic learning rate (d3rlpy default: 6.25e-5, "
                          "tuned for Atari-scale DQN). The seed-42/40k-step "
                          "run at 6.25e-5 showed init_state_value overshoot "
                          "past a peak around epoch 17 with loss climbing "
                          "every epoch (0.03 -> 1.57), never turning over -- "
                          "lowered the default here to test whether that's "
                          "a gradient-scale instability rather than a "
                          "structural fitting problem. See module docstring "
                          "INVESTIGATION NOTE 2.")
    ap.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
                     help="Max gradient norm for the FQE critic's optimizer "
                          "(d3rlpy default: None, i.e. unclipped). Paired "
                          "with --learning-rate as the first thing tried "
                          "against the overshoot-then-decline pattern seen "
                          "at the old default LR with no clipping. Pass a "
                          "very large value (e.g. 1e9) to effectively "
                          "disable clipping while keeping the flag wired "
                          "through, if you need to isolate the LR change "
                          "on its own.")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    ap.add_argument("--bootstrap-seed", type=int, default=2026)
    ap.add_argument("--only-seed", type=int, default=None,
                     help="Fit FQE for only this one training seed (both BC "
                          "and CQL for it) instead of all of TRAINING_SEEDS. "
                          "For quickly checking whether a new "
                          "--target-update-interval / --n-critics / "
                          "--learning-rate / --grad-clip combination "
                          "actually converges before committing to the "
                          "full 6-fit run. No bootstrap/JSON report is "
                          "produced in this mode -- it just prints the two "
                          "value trajectories.")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = _REPO_ROOT / model_dir
    bc_dir = Path(args.bc_dir) if args.bc_dir else model_dir / "bc"
    cql_dir = Path(args.cql_dir) if args.cql_dir else model_dir / "cql"
    if not bc_dir.is_absolute():
        bc_dir = _REPO_ROOT / bc_dir
    if not cql_dir.is_absolute():
        cql_dir = _REPO_ROOT / cql_dir
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    print("=" * 80)
    print("PHASE 5b -- FQE OFF-POLICY VALIDATION (offline-only, no simulator)")
    print("=" * 80)
    print(f"Dataset : {dataset_path}")
    print(f"BC dir  : {bc_dir}")
    print(f"CQL dir : {cql_dir}")
    print(f"n_steps={args.n_steps}, n_steps_per_epoch={args.n_steps_per_epoch}, "
          f"gamma={args.gamma}, n_critics={args.n_critics}, "
          f"target_update_interval={args.target_update_interval}, "
          f"learning_rate={args.learning_rate}, grad_clip={args.grad_clip}")

    print("\n[fqe_eval] Loading dataset and recovering the SAME episode split "
          "used by bc_baseline.py / train_cql.py / compare_policies.py...")
    ds, episodes, train_idx, val_idx, test_idx = load_dataset_and_split(dataset_path)
    print(f"[fqe_eval] {len(episodes)} episodes -> {len(train_idx)} train / "
          f"{len(val_idx)} val / {len(test_idx)} test")

    obs_mean, obs_std = recover_standardization(episodes, train_idx)
    print(f"[fqe_eval] Train obs mean: {obs_mean}")
    print(f"[fqe_eval] Train obs std : {obs_std}")

    train_episodes = [episodes[i] for i in train_idx]
    test_episodes = [episodes[i] for i in test_idx]
    train_ds_std = build_standardized_replay_buffer(train_episodes, obs_mean, obs_std)
    test_ds_std = build_standardized_replay_buffer(test_episodes, obs_mean, obs_std)
    test_episodes_std = list(test_ds_std.episodes)
    print(f"[fqe_eval] FQE will be evaluated at the {len(test_episodes_std)} "
          f"TEST-split episodes' initial states (held out from FQE's own "
          f"critic training, consistent with val/test discipline elsewhere "
          f"in this repo).")

    fqe_cfg = {
        "n_steps": args.n_steps,
        "n_steps_per_epoch": args.n_steps_per_epoch,
        "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "n_critics": args.n_critics,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
    }

    bc_paths = {seed: bc_dir / f"bc_seed{seed}.d3" for seed in TRAINING_SEEDS}
    cql_paths = {seed: cql_dir / f"cql_seed{seed}.d3" for seed in TRAINING_SEEDS}

    seeds_to_run = [args.only_seed] if args.only_seed is not None else TRAINING_SEEDS
    if args.only_seed is not None:
        print(f"\n[fqe_eval] --only-seed {args.only_seed}: quick single-seed "
              f"convergence check, no bootstrap/report will be written.")

    bc_fqe_results = {}
    cql_fqe_results = {}
    action_agreement_results = {}
    for seed in seeds_to_run:
        bc_algo = load_d3rlpy_model(bc_paths[seed], f"BC seed {seed}")
        cql_algo = load_d3rlpy_model(cql_paths[seed], f"CQL seed {seed}")

        agreement = action_agreement_diagnostic(bc_algo, cql_algo, test_episodes_std)
        action_agreement_results[seed] = agreement
        print(f"\n[fqe_eval] Seed {seed}: BC/CQL agree on the FIRST action in "
              f"{agreement['n_agree']}/{agreement['n_episodes']} test episodes "
              f"({agreement['agreement_rate']:.1%}).")

        bc_fqe_results[seed] = fit_fqe_for_policy(
            bc_algo, train_ds_std, test_episodes_std, fqe_cfg, f"BC_seed{seed}")
        cql_fqe_results[seed] = fit_fqe_for_policy(
            cql_algo, train_ds_std, test_episodes_std, fqe_cfg, f"CQL_seed{seed}")

    if args.only_seed is not None:
        print("\n[fqe_eval] --only-seed run complete. Inspect the "
              "value_trajectory / NOT CONVERGED flags above for both fits. "
              "If the value estimate is still climbing/declining without "
              "flattening, and loss is still growing (not shrinking) "
              "epoch-over-epoch, that is evidence against convergence "
              "regardless of what the flag says. If this run already used "
              "the lowered --learning-rate / --grad-clip defaults and still "
              "shows the same overshoot-then-decline shape as the "
              "unclipped/higher-LR run, that argues against a "
              "gradient-scale explanation and toward decoupling "
              "--target-update-interval from --n-steps-per-epoch instead.")
        return

    bc_values = [bc_fqe_results[s]["fqe_init_state_value"] for s in TRAINING_SEEDS]
    cql_values = [cql_fqe_results[s]["fqe_init_state_value"] for s in TRAINING_SEEDS]

    bootstrap_result = paired_bootstrap_scalar_difference(
        cql_values, bc_values, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)

    all_converged = all(
        bc_fqe_results[s].get("converged_heuristic") for s in TRAINING_SEEDS
    ) and all(
        cql_fqe_results[s].get("converged_heuristic") for s in TRAINING_SEEDS
    )
    mean_agreement = float(np.mean(
        [action_agreement_results[s]["agreement_rate"] for s in TRAINING_SEEDS]
    ))

    print("\n" + "=" * 80)
    print("FQE CQL - BC (offline-only estimate)")
    print("=" * 80)
    print(f"BC  FQE values : {bc_values}")
    print(f"CQL FQE values : {cql_values}")
    print(f"Observed diff  : {bootstrap_result['observed_difference']:.4f}")
    print(f"95% CI         : [{bootstrap_result['ci_lower']:.4f}, "
          f"{bootstrap_result['ci_upper']:.4f}]")
    print(f"CI contains 0  : {bootstrap_result['ci_contains_zero']}")
    print(f"\nConvergence heuristic (all 6 fits): "
          f"{'ALL CONVERGED' if all_converged else 'AT LEAST ONE NOT CONVERGED -- see per-seed value_trajectory in the JSON'}")
    print(f"Mean BC/CQL first-action agreement across seeds: {mean_agreement:.1%}")
    print("\nCompare this against compare_policies.py's "
          "cql_minus_bc_paired_bootstrap.observed_difference / CI in the "
          "corresponding policy_comparison.json. If this CI now excludes "
          "zero AND agrees in sign with the rollout result, that is "
          "validation. If it contains zero (as seen in the epsilon run "
          "this diagnostic was added after), do NOT read that as "
          "'FQE disproves the rollout finding' or vice versa -- read the "
          "convergence heuristic and action-agreement rate above first: a "
          "low agreement rate + NOT CONVERGED strongly suggests the FQE "
          "critic simply hasn't reached a reliable estimate yet; a high "
          "agreement rate is a separate, legitimate reason the two "
          "policies could have similar t=0 value even if their aggregate "
          "rollout returns differ.")

    report = {
        "experiment": "Phase 5b FQE-based offline validation of compare_policies.py",
        "config": {
            "dataset": str(dataset_path),
            "bc_dir": str(bc_dir),
            "cql_dir": str(cql_dir),
            "training_seeds": TRAINING_SEEDS,
            "split_seed": SPLIT_SEED,
            "n_train_episodes": len(train_idx),
            "n_val_episodes": len(val_idx),
            "n_test_episodes": len(test_idx),
            "fqe_n_steps": args.n_steps,
            "fqe_n_steps_per_epoch": args.n_steps_per_epoch,
            "gamma": args.gamma,
            "n_critics": args.n_critics,
            "target_update_interval": args.target_update_interval,
            "learning_rate": args.learning_rate,
            "grad_clip": args.grad_clip,
        },
        "per_seed_bc_fqe": bc_fqe_results,
        "per_seed_cql_fqe": cql_fqe_results,
        "per_seed_action_agreement": action_agreement_results,
        "all_fqe_fits_converged_heuristic": all_converged,
        "mean_bc_cql_first_action_agreement": mean_agreement,
        "bc_fqe_values": bc_values,
        "cql_fqe_values": cql_values,
        "cql_minus_bc_fqe_bootstrap": bootstrap_result,
        "how_to_cross_check": (
            "Load the corresponding compare_policies.py output "
            "(reports/<run>/policy_comparison.json) and compare "
            "cql_minus_bc_paired_bootstrap.observed_difference / ci_lower / "
            "ci_upper there against cql_minus_bc_fqe_bootstrap above. FQE "
            "and direct rollout measure value via completely different "
            "mechanisms (offline critic fit vs exact simulator rollout) so "
            "exact numeric agreement is not expected -- sign agreement and "
            "non-overlapping conclusions (both exclude zero in the same "
            "direction, or both contain zero) is the meaningful check."
        ),
        "api_verification_caveat": (
            "d3rlpy.ope.FQEConfig / DiscreteFQE usage verified against a "
            "live d3rlpy 2.8.1 install this session (see module docstring "
            "'API VERIFICATION STATUS'). InitialStateValueEstimationEvaluator's "
            "exact semantics when applied to a trained FQE critic are still "
            "only signature-confirmed, not independently re-derived -- if "
            "numbers look implausible relative to compare_policies.py's "
            "return_mean values, suspect that evaluator call first."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[fqe_eval] Saved -> {out_path}")


if __name__ == "__main__":
    main()