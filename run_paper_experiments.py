"""
run_paper_experiments.py — closes the 5 remaining Trishna-OPAL gaps.

Modes: scenario | adapt | ood | weights | rfsens | qrvsmean | selftest

Design (current implementation):
  * scenario_step() is a CONTROLLED FORK of synthetic_reward.step() that
    applies scenario factors as per-step INTENSITY, never as a state
    multiplier that compounds. Under f=1 it is NUMERICALLY EQUIVALENT to
    step() within specified tolerances (selftest 1/1b: reward < 1e-12,
    state < 1e-9) and non-compounding under f!=1 (selftest 7/8). It is
    NOT bit-for-bit equivalent.
  * Common random numbers: paired BC/CQL rollouts share init states and
    an identical RNG stream (RNG reset per policy / per k).
  * Perturbation = SIGNED DETERMINISTIC observation-error sensitivity (not
    uncertainty propagation): policy input gw_level perturbed by
    k*RMSE(depth bin) for signed k, clipped >=0; simulator state/reward
    untouched.
  * OOD support: cKDTree, leave-one-out self-distance threshold,
    train-split-only reference, provenance manifests. The OOD fraction is
    the fraction of POLICY-ROLLOUT STATE OCCUPANCY outside the
    training-state support; it is NOT a claim that the scenario itself is
    statistically outside the training distribution.
  * Manifests: normalization_manifest.json (mean/std + dataset sha1 +
    config_sha1 + script_sha1 + norm_fingerprint + a SELECTED subset of
    training hparams) and per-variant training_manifest.json gate all
    resume paths; all must agree on dataset_sha1 and norm_fingerprint
    before a resume is honored.
    - Training manifests record BOTH "variant" (the on-disk directory
      suffix, e.g. "frozen", "qr", "mean") AND "qfunc" (the actual CQL
      critic type used during training, e.g. "qr", "mean"). These are
      deliberately separate keys: for scenario/adapt/ood the directory is
      "frozen" but qfunc is the CLI-selected critic; for qrvsmean the
      directory suffix is the qfunc itself.
    - Scenario resume validates the SELECTED subset via check_norm_hparams
      AND the FULL CQL + BC hyperparameter set via the MANDATORY training
      manifest (see _full_training_cfg).
    - The training manifest written by scenario/adapt/ood/weights records
      the FULL CQL config (batch_size, learning_rate, gamma,
      target_update_interval, n_critics, clip_grad_norm,
      n_steps_per_epoch) plus the FULL BC config (bc_batch_size,
      bc_learning_rate, bc_beta, bc_n_steps, bc_n_steps_per_epoch).
  * Dataset sidecars record dataset sha1 + config_sha1 + script_sha1;
    required-key presence is enforced strictly (missing key = rejection).
  * Training-normalization link: split_and_std() accepts a norm_override,
    and train_pair()/train_cql_variant() accept a norm= argument; the
    values used for training are exactly the values returned and
    fingerprinted (no re-derivation between training and manifest).
  * scenario mode refuses to overwrite an existing validated frozen_models
    directory; --resume reuses it after manifest validation. adapt, ood,
    and weights modes also refuse to overwrite existing model directories
    and write their own normalization + training manifests (matching
    scenario).
  * qrvsmean --resume requires the directory to exist.
  * load_rf_table validates the seed-table schema before returning.
  * CQL learning rate is a CLI parameter (--cql-lr) and is recorded in the
    training manifest. Default 6.25e-5; the earlier canonical Phase-5
    baseline used 3e-4 -- pass --cql-lr 3e-4 to reproduce it exactly.
  * d3rlpy-dependent imports are LAZY; selftest and pure-numpy paths run
    without d3rlpy.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, sys, tempfile, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "src" / "rl"), str(ROOT / "src" / "downscaling")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- pure-numpy imports only at module level (lazy everything d3rlpy) ----
from src.rl.synthetic_reward import (SimState, step, crop_revenue_proxy,
    domestic_supply_score, depletion_penalty)
from src.rl.generate_offline_dataset import (policy_random,
    policy_greedy_extraction, sample_initial_state)

SEEDS = [42, 123, 2024]
SPLIT_SEED = 1
_EPS = 0.1  # epsilon-greedy exploration fraction (coverage ablation)
EXPECTED_STATE_DIM = 6  # gw_level, recent_rainfall, crop_water_demand,
                        # month_sin, month_cos, extraction_rate
DEFAULT_CQL_LR = 6.25e-5  # paper-experiment default; earlier canonical Phase-5
                          # baseline used 3e-4 -- override with --cql-lr 3e-4

# Fixed CQL / BC training constants used by train_pair() / train_cql_variant().
# Recorded verbatim in the training manifest via _full_training_cfg().
CQL_BATCH_SIZE = 100
CQL_GAMMA = 0.99
CQL_TARGET_UPDATE_INTERVAL = 8000
CQL_N_CRITICS = 1
CQL_CLIP_GRAD_NORM = 1.0
CQL_N_STEPS_PER_EPOCH = 2000
BC_BATCH_SIZE = 100
BC_LEARNING_RATE = 0.001
BC_BETA = 0.5
BC_N_STEPS_PER_EPOCH = 2000

SCENARIOS = {
    "normal":  dict(recharge_factor=1.0, demand_factor=1.0),
    "drought": dict(recharge_factor=0.5, demand_factor=1.2),
    "high":    dict(recharge_factor=1.4, demand_factor=0.9),
    "extreme": dict(recharge_factor=0.25, demand_factor=1.5),
}
GW_LEVELS = [5.0, 15.0, 25.0]
WEIGHT_VECTORS = [  # (w1 crop revenue, w2 depletion penalty, w3 domestic supply)
    (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.5), (1.0, 0.0, 0.5),
    (1.0, 1.0, 0.5), (1.0, 4.0, 0.5), (1.0, 2.0, 0.5),  # last = baseline
]
# limitations.md Section 1 depth-stratified RF RMSE (m) -- fallback; use --rf-rmse-json
RF_RMSE_BINS = [(0, 10, 3.10), (10, 20, 6.17), (20, 30, 14.39),
                (30, 50, 26.78), (50, np.inf, 51.49)]
RMSE_TABLE = RF_RMSE_BINS

# RF seed-table schema: at least one depth-like column must be present, else
# sample_initial_state() would fail later with a less actionable message.
RF_DEPTH_COL_CANDIDATES = ("depth_m", "predicted_depth_m", "predicted_depth",
                           "gw_level", "groundwater_depth", "gw_level_pred", "depth")

_WARNED_SYNTHETIC = [False]


def file_sha1(path, chunk_size=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _arrays_sha1(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        h.update(arr.tobytes())
    return h.hexdigest()


def _script_sha1():
    return file_sha1(Path(__file__).resolve())


def norm_manifest_path(model_root):
    return Path(model_root) / "normalization_manifest.json"


def training_manifest_path(root, variant):
    return Path(root) / f"cql_{variant}" / "training_manifest.json"


def ensure_eps_synced(args):
    args_eps = getattr(args, "eps", None)
    if args_eps is None:
        return float(_EPS)
    if not np.isclose(float(_EPS), float(args_eps), atol=0.0, rtol=0.0):
        raise RuntimeError(
            f"[eps] global _EPS={_EPS} != args.eps={args_eps}; synchronize "
            f"_EPS before calling mode helpers directly, or invoke main().")
    return float(_EPS)


def _resolve_cql_lr(args):
    return float(getattr(args, "cql_lr", DEFAULT_CQL_LR))


def _full_training_cfg(h5, args, mean, std, cql_lr, include_bc=True):
    """Build the full manifest configuration for a training run, matching
    exactly what train_pair() / train_cql_variant() pass to the trainers.
    Used by scenario/adapt/ood/weights (paired BC+CQL, include_bc=True) and
    by qrvsmean (CQL-only variants, include_bc=False).

    The manifest written with this cfg gives check_training_manifest() the
    full CQL hyperparameter set (batch_size, learning_rate, gamma,
    target_update_interval, n_critics, clip_grad_norm, n_steps_per_epoch)
    plus, when include_bc is True, the full BC set (bc_batch_size,
    bc_learning_rate, bc_beta, bc_n_steps, bc_n_steps_per_epoch). This
    closes the previous provenance gap where the header claimed the
    mandatory training manifest validated those fields but they were
    absent from the manifest itself.

    The 'qfunc' key here is the CLI-selected critic type (args.qfunc),
    which for scenario/adapt/ood/weights is the actual CQL critic used.
    For qrvsmean, callers must override qfunc per-variant before writing
    the manifest, because both QR and Mean variants are trained in a
    single mode invocation."""
    cfg = {
        "dataset": str(h5),
        "config": str(args.config),
        "config_sha1": file_sha1(args.config),
        "split_seed": SPLIT_SEED,
        "alpha": args.alpha,
        "qfunc": args.qfunc,
        "n_quantiles": args.n_quantiles,
        "oversample": args.oversample,
        "n_steps": args.n_steps,
        # CQL hparams -- must match train_pair / train_cql_variant exactly.
        "batch_size": CQL_BATCH_SIZE,
        "learning_rate": float(cql_lr),
        "gamma": CQL_GAMMA,
        "target_update_interval": CQL_TARGET_UPDATE_INTERVAL,
        "n_critics": CQL_N_CRITICS,
        "clip_grad_norm": CQL_CLIP_GRAD_NORM,
        "n_steps_per_epoch": CQL_N_STEPS_PER_EPOCH,
        "norm_fingerprint": _arrays_sha1(mean, std),
    }
    if include_bc:
        cfg.update({
            "bc_batch_size": BC_BATCH_SIZE,
            "bc_learning_rate": BC_LEARNING_RATE,
            "bc_beta": BC_BETA,
            "bc_n_steps": args.n_steps,
            "bc_n_steps_per_epoch": BC_N_STEPS_PER_EPOCH,
        })
    return cfg


def check_norm_hparams(data, args, cql_lr):
    """Validate the SELECTED subset of training hyperparameters recorded in
    a loaded normalization manifest against the current invocation. Closes
    the scenario --resume hole for these fields: a frozen model trained
    with one config for these hparams cannot be silently resumed under a
    different one. The remaining training hyperparameters (batch_size,
    gamma, target_update_interval, n_critics, clip_grad_norm,
    n_steps_per_epoch, plus the BC set) are validated via the mandatory
    training manifest, not here."""
    required = ("alpha", "qfunc", "n_quantiles", "oversample", "n_steps", "cql_lr")
    for k in required:
        if k not in data:
            raise SystemExit(
                f"[norm] manifest missing training-hparam key: {k} "
                f"(regenerate the frozen models with the current script)")
    mismatches = []
    if not np.isclose(float(data["alpha"]), float(args.alpha)):
        mismatches.append(f"alpha {data['alpha']} != {args.alpha}")
    if not np.isclose(float(data["cql_lr"]), float(cql_lr)):
        mismatches.append(f"cql_lr {data['cql_lr']} != {cql_lr}")
    if str(data["qfunc"]) != str(args.qfunc):
        mismatches.append(f"qfunc {data['qfunc']} != {args.qfunc}")
    if int(data["n_quantiles"]) != int(args.n_quantiles):
        mismatches.append(f"n_quantiles {data['n_quantiles']} != {args.n_quantiles}")
    if int(data["oversample"]) != int(args.oversample):
        mismatches.append(f"oversample {data['oversample']} != {args.oversample}")
    if int(data["n_steps"]) != int(args.n_steps):
        mismatches.append(f"n_steps {data['n_steps']} != {args.n_steps}")
    if mismatches:
        raise SystemExit("[norm] training-hparam mismatch: " + "; ".join(mismatches))


# ------------------------------------------------------------------ helpers
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _validate_rf_table_schema(df, path):
    """Fail fast if the CSV does not look like an RF seed table."""
    if df is None or df.empty:
        raise SystemExit(f"[data] RF seed table {path} is empty")
    cols = set(map(str, df.columns))
    if not (cols & set(RF_DEPTH_COL_CANDIDATES)):
        raise SystemExit(
            f"[data] RF seed table {path} lacks a depth-like column; "
            f"expected one of {sorted(RF_DEPTH_COL_CANDIDATES)}, "
            f"got columns={sorted(cols)}")


def load_rf_table(args):
    for p in (getattr(args, "rf_table", "") or "",
              "reports/rf_grid_predictions.csv",
              "reports/rf_downscale_validation.csv"):
        p = Path(p)
        if p.exists():
            df = pd.read_csv(p)
            _validate_rf_table_schema(df, p)
            print(f"[data] RF seed table: {p} ({len(df)} rows)")
            return df
    print("[data] WARNING: no RF table found -> fully synthetic initial states")
    return None


def synthetic_init(rng, gw_target=None):
    month = rng.randint(1, 13)
    st = SimState(gw_level=float(rng.uniform(2, 30)),
                  recent_rainfall=float(rng.uniform(20, 300)),
                  crop_water_demand=float(rng.uniform(0.2, 0.8)),
                  month_sin=float(np.sin(2 * np.pi * month / 12)),
                  month_cos=float(np.cos(2 * np.pi * month / 12)),
                  extraction_rate=float(rng.uniform(0.1, 0.5)))
    if gw_target is not None:
        st = replace(st, gw_level=float(gw_target))
    return st


def make_init_states(rf_df, n, rng, gw_target=None):
    """FAIL LOUDLY when the RF path is requested but sample_initial_state fails
    (no except-swallow). Falls to synthetic ONLY when no RF table was found /
    provided at all, and announces it once."""
    out = []
    if rf_df is not None:
        for _ in range(n):
            st = sample_initial_state(rf_df, {}, rng)   # let it raise
            if gw_target is not None:
                st = replace(st, gw_level=float(gw_target))
            out.append(st)
        return out
    if not _WARNED_SYNTHETIC[0]:
        print("[data] NOTE: fully synthetic initial states (no RF table) — "
              "paper claim 'RF-initialized rollouts' does NOT apply to this run")
        _WARNED_SYNTHETIC[0] = True
    for _ in range(n):
        st = synthetic_init(rng)
        if gw_target is not None:
            st = replace(st, gw_level=float(gw_target))
        out.append(st)
    return out


def scenario_step(state, action, config, n_actions, rng, scn):
    """Controlled FORK of synthetic_reward.step(): reimplements the transition
    and reward equations so scenario factors apply as per-step INTENSITY
    without compounding across steps. It does NOT call step() at runtime,
    despite what older docstrings claimed; numerical equivalence under f=1
    within specified tolerances is established by selftest 1/1b.

    Single application point for BOTH scenario knobs (never at init):
      * demand_factor: the revenue term sees effective demand =
        clip(state.crop_water_demand * f, 0, 1) EVERY step (regime
        intensity), while the STORED demand AR(1) walks on the UNSCALED
        base -- so the factor NEVER compounds across steps
        (demand_t ~ base_t, revenue sees f*base_t). `f=1` recovers
        synthetic_reward.step() dynamics numerically. Do not re-apply at
        init.
      * recharge_factor: the A6 recharge term sees rainfall*f EVERY
        transition, but the STORED rainfall AR(1) random walk stays on
        the UNSCALED base scale, so the factor NEVER compounds across
        steps (rain_t ~ f * base_walk_t, NOT f**t * base). Verified
        empirically in selftest 7 (multi-timestep rainfall trace).
    """
    demand_eff = float(np.clip(state.crop_water_demand * scn["demand_factor"], 0.0, 1.0))
    rain_scaled = max(0.0, state.recent_rainfall * scn["recharge_factor"])
    pumping_fraction = action / (n_actions - 1)
    drawdown = pumping_fraction * 2.0                      # A5
    recharge = 0.15 * rain_scaled / 100.0                  # A6, regime-scaled
    noise = rng.normal(0, 0.1)
    next_gw_level = max(0.0, state.gw_level + drawdown - recharge + noise)
    next_demand = float(np.clip(state.crop_water_demand + rng.normal(0, 0.05), 0.0, 1.0))  # base AR, unscaled
    next_extraction_rate = 0.7 * state.extraction_rate + 0.3 * pumping_fraction
    next_rainfall = max(0.0, state.recent_rainfall + rng.normal(0, 20))  # base AR, unscaled
    next_state = SimState(gw_level=next_gw_level,
                          recent_rainfall=float(next_rainfall),
                          crop_water_demand=next_demand,
                          month_sin=state.month_sin,
                          month_cos=state.month_cos,
                          extraction_rate=next_extraction_rate)
    revenue = crop_revenue_proxy(action, demand_eff, n_actions)
    supply = domestic_supply_score(next_gw_level)
    penalty = depletion_penalty(next_gw_level,
                                config["sustainability_threshold_m"],
                                config.get("depletion_penalty_exponent", 2.0))
    reward = (config["w1_crop_revenue"] * revenue
              + config["w3_domestic_supply"] * supply
              - config["w2_depletion_penalty"] * penalty)
    done = False
    return next_state, float(reward), float(revenue), float(supply), float(penalty), done


def _eps_policy(state, n_actions, rng):
    if rng.rand() < _EPS:
        return int(rng.randint(0, n_actions))
    return policy_greedy_extraction(state, n_actions, rng)


POLICY_MAP = {"random": policy_random,
              "greedy_extraction": policy_greedy_extraction,
              "epsilon_greedy_extraction": _eps_policy}


def make_arrays(rf_df, cfg, reward_cfg, n_actions, scn, gw_target, rng, behavior,
                expected_eps=None):
    behavior = [str(pol).strip() for pol in behavior]
    unknown = [pol for pol in behavior if pol not in POLICY_MAP]
    if unknown:
        raise KeyError(f"Unknown behavior policy/policies: {unknown}")
    if "epsilon_greedy_extraction" in behavior and expected_eps is not None and \
       not np.isclose(float(_EPS), float(expected_eps), atol=0.0, rtol=0.0):
        raise RuntimeError(
            f"[eps] make_arrays using _EPS={_EPS} but caller expects {expected_eps}; "
            f"this would make behavior generation and recorded provenance diverge.")
    obs, act, rew, to = [], [], [], []
    n_traj = cfg["simulation"]["n_trajectories_per_policy"]
    traj_len = cfg["simulation"]["trajectory_length"]
    init = make_init_states(rf_df, n_traj * len(behavior), rng, gw_target)
    k = 0
    for pol in behavior:
        fn = POLICY_MAP[pol]
        for _ in range(n_traj):
            st = init[k]; k += 1
            for t in range(traj_len):
                a = int(fn(st, n_actions, rng))
                ns, r, *_ = scenario_step(st, a, reward_cfg, n_actions, rng, scn)
                ang = np.arctan2(st.month_sin, st.month_cos) + (2 * np.pi / 4)  # quarterly
                ns = replace(ns, month_sin=float(np.sin(ang)),
                             month_cos=float(np.cos(ang)))
                obs.append(st.to_array()); act.append(a); rew.append(r)
                to.append(1 if t == traj_len - 1 else 0)
                st = ns
    return (np.array(obs, np.float32), np.array(act, np.int64),
            np.array(rew, np.float32), np.zeros(len(obs), np.float32),
            np.array(to, np.float32))


def dump_dataset(h5_path, obs, act, rew, term, timeout, provenance):
    from d3rlpy.dataset import MDPDataset                          # lazy
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    MDPDataset(observations=obs, actions=act, rewards=rew,
               terminals=term, timeouts=timeout).dump(str(h5_path))
    sidecar = h5_path.with_suffix(".json")
    sidecar.write_text(json.dumps({**provenance,
        "n_transitions": int(len(obs)),
        "sha1": file_sha1(h5_path),
        "script_sha1": _script_sha1(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2))
    print(f"[data] dataset + provenance sidecar -> {h5_path} / {sidecar.name}")


def load_episodes_raw(h5):
    from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer          # lazy
    with open(h5, "rb") as f:
        return list(ReplayBuffer.load(f, buffer=InfiniteBuffer()).episodes)


def split_and_std(episodes, split_seed=SPLIT_SEED, norm_override=None):
    """Compute the train/val/test split and the normalization statistics.

    If norm_override=(mean, std) is given, those exact values are used for
    normalizing the training dataset (and returned unchanged). This lets a
    caller pre-compute the normalization once and force training to use it,
    so the statistics used during training and the statistics recorded in
    the manifest are provably identical."""
    from src.rl.bc_baseline import (split_episode_indices,            # lazy
        episodes_to_arrays, build_dataset)
    n = len(episodes)
    tr, va, te = split_episode_indices(n, seed=split_seed)
    tr_ep, va_ep, te_ep = ([episodes[i] for i in ix] for ix in (tr, va, te))
    if norm_override is None:
        raw, *_ = episodes_to_arrays(tr_ep)
        mean, std = raw.mean(0), raw.std(0); std[std < 1e-6] = 1.0
        mean = np.asarray(mean, np.float64)
        std = np.asarray(std, np.float64)
    else:
        mean = np.asarray(norm_override[0], np.float64)
        std = np.asarray(norm_override[1], np.float64)
    tr_ds = build_dataset(tr_ep, mean, std)
    return tr, va, te, tr_ep, va_ep, te_ep, tr_ds, mean, std


def train_pair(h5, out_dir, alpha, qfunc, n_quantiles, oversample, n_steps, seeds,
               cql_lr=DEFAULT_CQL_LR, norm=None):
    """Train BC + CQL under the given normalization. If norm=(mean, std) is
    supplied, the training uses exactly those statistics; the returned
    (mean, std) are the ones actually used, which the caller should fingerprint.

    The CQL and BC hyperparameters here MUST match the constants recorded by
    _full_training_cfg() in the training manifest."""
    from src.rl.bc_baseline import run_one_seed as run_bc_seed      # lazy
    from src.rl.train_cql import run_one_seed as run_cql_seed        # lazy
    tr, va, te, tr_ep, va_ep, te_ep, tr_ds, mean, std = split_and_std(
        load_episodes_raw(h5), norm_override=norm)
    bc_dir = Path(out_dir) / "bc"; bc_dir.mkdir(parents=True, exist_ok=True)
    cql_dir = Path(out_dir) / "cql"; cql_dir.mkdir(parents=True, exist_ok=True)
    bc_cfg = {"batch_size": BC_BATCH_SIZE, "learning_rate": BC_LEARNING_RATE,
              "beta": BC_BETA, "n_steps": n_steps,
              "n_steps_per_epoch": BC_N_STEPS_PER_EPOCH}
    cql_cfg = {"model_dir": str(cql_dir), "qfunc": qfunc,
               "n_quantiles": n_quantiles, "oversample_factor": oversample,
               "n_steps": n_steps, "n_steps_per_epoch": CQL_N_STEPS_PER_EPOCH,
               "batch_size": CQL_BATCH_SIZE, "gamma": CQL_GAMMA,
               "learning_rate": float(cql_lr),
               "target_update_interval": CQL_TARGET_UPDATE_INTERVAL,
               "n_critics": CQL_N_CRITICS, "clip_grad_norm": CQL_CLIP_GRAD_NORM}
    for s in seeds:
        run_bc_seed(s, tr_ds, va_ep, va, te_ep, te, mean, std, bc_cfg, str(bc_dir))
        run_cql_seed(s, alpha, tr_ds, va_ep, va, te_ep, te, mean, std, cql_cfg)
    return bc_dir, cql_dir, mean, std


def train_cql_variant(h5, out_dir, qfunc, alpha, n_quantiles, oversample,
                      n_steps, seeds, cql_lr=DEFAULT_CQL_LR, norm=None):
    """CQL-only training into out_dir/cql_<qfunc> (canonical dirs: cql_qr,
    cql_mean). Same dataset/split/seeds/alpha/steps/hparams as train_pair's
    CQL leg, only the critic type differs (C19). If norm=(mean, std) is
    supplied, training uses exactly those statistics and returns them."""
    from src.rl.train_cql import run_one_seed as run_cql_seed        # lazy
    tr, va, te, tr_ep, va_ep, te_ep, tr_ds, mean, std = split_and_std(
        load_episodes_raw(h5), norm_override=norm)
    cql_dir = Path(out_dir) / f"cql_{qfunc}"; cql_dir.mkdir(parents=True, exist_ok=True)
    cql_cfg = {"model_dir": str(cql_dir), "qfunc": qfunc,
               "n_quantiles": n_quantiles, "oversample_factor": oversample,
               "n_steps": n_steps, "n_steps_per_epoch": CQL_N_STEPS_PER_EPOCH,
               "batch_size": CQL_BATCH_SIZE, "gamma": CQL_GAMMA,
               "learning_rate": float(cql_lr),
               "target_update_interval": CQL_TARGET_UPDATE_INTERVAL,
               "n_critics": CQL_N_CRITICS, "clip_grad_norm": CQL_CLIP_GRAD_NORM}
    for s in seeds:
        run_cql_seed(s, alpha, tr_ds, va_ep, va, te_ep, te, mean, std, cql_cfg)
    return cql_dir, mean, std


def load_model_fn(path, mean, std):
    import d3rlpy                                                # lazy
    from src.rl.compare_policies import make_model_action_fn      # lazy
    return make_model_action_fn(d3rlpy.load_learnable(str(path)), mean, std)


def paired_bootstrap_difference(returns_a, returns_b, n_bootstrap, seed,
                                confidence_level=0.95):
    """Faithful inline copy of src/rl/compare_policies.py:392-430 (same keys).
    observed = mean(a - b); pass a=CQL, b=BC for cql_minus_bc."""
    returns_a = np.asarray(returns_a, dtype=np.float64)
    returns_b = np.asarray(returns_b, dtype=np.float64)
    if returns_a.shape != returns_b.shape:
        raise ValueError("Paired bootstrap requires equal-length return arrays.")
    if len(returns_a) < 2:
        raise ValueError("At least two episodes are required.")
    if not np.all(np.isfinite(returns_a)) or not np.all(np.isfinite(returns_b)):
        raise ValueError("Inputs contain non-finite values.")
    paired_difference = returns_a - returns_b
    observed_difference = float(np.mean(paired_difference))
    rng = np.random.RandomState(seed)
    sampled = rng.randint(0, len(paired_difference),
                          size=(n_bootstrap, len(paired_difference)))
    boot = np.mean(paired_difference[sampled], axis=1)
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {"observed_difference": observed_difference,
            "confidence_level": confidence_level, "ci_lower": lower,
            "ci_upper": upper, "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(seed),
            "probability_bootstrap_difference_positive": float(np.mean(boot > 0.0)),
            "ci_contains_zero": bool(lower <= 0.0 <= upper)}


def bootstrap_row(name, a_ret, b_ret, seed, n=5000):
    r = paired_bootstrap_difference(a_ret, b_ret, n, seed)
    return {"comparison": name, "observed_difference": r["observed_difference"],
            "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
            "ci_contains_zero": r["ci_contains_zero"],
            "p_positive": r["probability_bootstrap_difference_positive"]}


def validate_rewards(rew, tag, w):
    """Finiteness + exact upper envelope (revenue, supply each in [0,1]):
    reward_max <= w1 + w3. min unbounded (penalty grows quadratically below
    threshold); always reported."""
    rew = np.asarray(rew, np.float64)
    if not np.all(np.isfinite(rew)):
        raise ValueError(f"{tag}: non-finite rewards")
    cap = w[0] + w[2]
    if rew.max() > cap + 1e-6:
        raise ValueError(f"{tag}: reward max {rew.max():.4f} > envelope w1+w3={cap}")
    return {"kind": "reward_stats", "cell": tag, "n": int(len(rew)),
            "min": float(rew.min()), "max": float(rew.max()),
            "mean": float(rew.mean()), "envelope_w1_plus_w3": cap, "finite": True}


def action_hist(actions, n_actions=None):
    flat = np.asarray(actions, dtype=np.int64).ravel()
    if flat.size == 0:
        n = int(n_actions or 0)
        return [0] * n
    if np.any(flat < 0):
        raise ValueError("action_hist received negative action ids")
    inferred = int(flat.max()) + 1
    if n_actions is None:
        n_actions = inferred
    else:
        n_actions = int(n_actions)
        if inferred > n_actions:
            raise ValueError(
                f"action_hist saw action id {inferred - 1} but n_actions={n_actions}")
    counts = np.bincount(flat, minlength=n_actions)
    return [int(x) for x in counts[:n_actions]]


def build_support(ref_states, mean, std):
    from scipy.spatial import cKDTree                              # lazy
    z = (ref_states - mean) / std
    tree = cKDTree(z)
    _, self_dist = tree.query(z, k=2)   # leave-one-out (k=2 -> nearest other)
    return tree, self_dist[:, 1]


def ood_frac(states, tree, self_dist, mean, std, q=0.95):
    """Fraction of POLICY-ROLLOUT STATE OCCUPANCY outside the training-state
    support. This measures state-occupancy shift under the scenario, NOT
    whether the scenario dynamics themselves are out-of-distribution."""
    if len(states) == 0:
        return float("nan")
    z = (np.asarray(states, np.float64) - mean) / std
    d, _ = tree.query(z, k=1)
    thr = np.quantile(self_dist, q)
    return float((d > thr).mean())


def rmse_for_depth(gw):
    for lo, hi, rmse in RMSE_TABLE:
        if lo <= gw < hi:
            return rmse
    return RMSE_TABLE[-1][2]


def _validate_rmse_bins(raw):
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("RMSE table must be a non-empty list of [lo, hi, rmse] bins")
    bins = []
    prev_hi = None
    for i, row in enumerate(raw):
        if isinstance(row, dict):
            lo = row.get("lo", row.get("min"))
            hi = row.get("hi", row.get("max"))
            rmse = row.get("rmse")
        elif isinstance(row, (list, tuple)) and len(row) == 3:
            lo, hi, rmse = row
        else:
            raise ValueError(
                f"RMSE bin #{i} must be a dict or length-3 sequence, got {row!r}")
        lo = float(lo); hi = float(hi); rmse = float(rmse)
        if not np.isfinite(lo):
            raise ValueError(f"RMSE bin #{i} has non-finite lower bound: {lo}")
        if not (np.isfinite(hi) or np.isinf(hi)):
            raise ValueError(f"RMSE bin #{i} has invalid upper bound: {hi}")
        if hi <= lo:
            raise ValueError(f"RMSE bin #{i} must satisfy hi > lo, got [{lo}, {hi})")
        if rmse < 0 or not np.isfinite(rmse):
            raise ValueError(f"RMSE bin #{i} has invalid rmse: {rmse}")
        if prev_hi is not None and lo < prev_hi:
            raise ValueError(
                f"RMSE bin #{i} overlaps previous bin: previous hi={prev_hi}, lo={lo}")
        bins.append((lo, hi, rmse))
        prev_hi = hi
    # Full depth-domain coverage: must start at (or below) 0 and extend to +inf.
    if bins[0][0] > 0.0:
        raise ValueError(
            f"RMSE table must start at 0 (or below) for full coverage; "
            f"first bin lo={bins[0][0]}")
    if not np.isinf(bins[-1][1]):
        raise ValueError(
            f"RMSE table must extend to +inf for full coverage; "
            f"last bin hi={bins[-1][1]}")
    return bins


def save_norm_manifest(model_root, h5, mean, std, args):
    manifest_path = norm_manifest_path(model_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    eps = ensure_eps_synced(args)
    cql_lr = _resolve_cql_lr(args)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError("Normalization mean/std must be same-shape 1D vectors")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Normalization manifest contains invalid mean/std values")
    payload = {
        "dataset": str(Path(h5).resolve()),
        "dataset_sha1": file_sha1(h5),
        "config": str(args.config),
        "config_sha1": file_sha1(args.config),
        "script_sha1": _script_sha1(),
        "split_seed": SPLIT_SEED,
        "eps": eps,
        "cql_lr": cql_lr,
        # SELECTED training hyperparameters (used by check_norm_hparams on
        # scenario resume). The remaining hparams are validated via the
        # mandatory training manifest.
        "alpha": float(args.alpha),
        "qfunc": str(args.qfunc),
        "n_quantiles": int(args.n_quantiles),
        "oversample": int(args.oversample),
        "n_steps": int(args.n_steps),
        "n_features": int(mean.size),
        "norm_fingerprint": _arrays_sha1(mean, std),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    print(f"[data] normalization manifest -> {manifest_path}")


def load_norm_manifest(model_root, h5, args):
    manifest_path = norm_manifest_path(model_root)
    if not manifest_path.exists():
        raise SystemExit(f"[norm] missing normalization manifest: {manifest_path}")
    data = json.load(open(manifest_path))
    expected = {
        "dataset": str(Path(h5).resolve()),
        "dataset_sha1": file_sha1(h5),
        "config": str(args.config),
        "config_sha1": file_sha1(args.config),
        "script_sha1": _script_sha1(),
        "split_seed": SPLIT_SEED,
        "eps": ensure_eps_synced(args),
    }
    mismatches = []
    for k, v in expected.items():
        if k not in data:
            mismatches.append(f"missing required key: {k}")
        elif data[k] != v:
            mismatches.append(f"{k}={data[k]!r} != {v!r}")
    mean = np.asarray(data.get("mean"), dtype=np.float64)
    std = np.asarray(data.get("std"), dtype=np.float64)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape or mean.size == 0:
        mismatches.append("mean/std missing or shape mismatch")
    elif mean.size != EXPECTED_STATE_DIM:
        mismatches.append(
            f"mean.size={mean.size} != expected state dim {EXPECTED_STATE_DIM}")
    elif not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        mismatches.append("mean/std contain non-finite or non-positive values")
    if "n_features" not in data:
        mismatches.append("missing required key: n_features")
    else:
        try:
            nf = int(data["n_features"])
        except (TypeError, ValueError):
            mismatches.append(f"n_features not an int: {data['n_features']!r}")
        else:
            if mean.size > 0 and nf != int(mean.size):
                mismatches.append(
                    f"n_features={nf} != mean.size={mean.size}")
            if nf != EXPECTED_STATE_DIM:
                mismatches.append(
                    f"n_features={nf} != expected state dim {EXPECTED_STATE_DIM}")
    if "norm_fingerprint" not in data:
        mismatches.append("missing required key: norm_fingerprint")
    elif mean.size > 0:
        recomputed = _arrays_sha1(mean, std)
        if data["norm_fingerprint"] != recomputed:
            mismatches.append(
                f"norm_fingerprint={data['norm_fingerprint']!r} != "
                f"recomputed={recomputed!r}")
    if mismatches:
        raise SystemExit("[norm] REFUSING manifest: " + "; ".join(mismatches))
    return mean, std


def write_training_manifest(root, variant, expected_cfg):
    path = training_manifest_path(root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Path(expected_cfg["dataset"]).resolve()
    payload = dict(expected_cfg)
    # Two SEPARATE keys are recorded:
    #   "variant" = the on-disk directory suffix (e.g. "frozen", "qr", "mean")
    #   "qfunc"   = the actual CQL critic type used during training
    # They coincide for qrvsmean ("qr" / "mean") but differ for
    # scenario/adapt/ood/weights, where the directory is "frozen" but the
    # qfunc is the CLI-selected critic (args.qfunc, default "qr"). Recording
    # both closes the earlier bug where the manifest's qfunc was set to the
    # directory suffix and therefore never validated the real critic type.
    actual_qfunc = expected_cfg.get("qfunc", variant)
    payload.update({
        "dataset": str(dataset),
        "dataset_sha1": file_sha1(dataset),
        "script_sha1": _script_sha1(),
        "variant": str(variant),
        "qfunc": str(actual_qfunc),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    path.write_text(json.dumps(payload, indent=2))
    print(f"[data] training manifest -> {path}")


def check_training_manifest(root, variant, expected_cfg):
    path = training_manifest_path(root, variant)
    if not path.exists():
        raise SystemExit(f"[qrvsmean] missing training manifest: {path}")
    got = json.load(open(path))
    dataset = Path(expected_cfg["dataset"]).resolve()
    expected = dict(expected_cfg)
    actual_qfunc = expected_cfg.get("qfunc", variant)
    expected.update({
        "dataset": str(dataset),
        "dataset_sha1": file_sha1(dataset),
        "script_sha1": _script_sha1(),
        "variant": str(variant),
        "qfunc": str(actual_qfunc),
    })
    mismatches = []
    for k, v in expected.items():
        if k not in got:
            mismatches.append(f"missing required key: {k}")
        elif got.get(k) != v:
            mismatches.append(f"{k}={got.get(k)!r} != {v!r}")
    if mismatches:
        raise SystemExit(
            f"[qrvsmean] REFUSING resume for {variant}: " + "; ".join(mismatches))
    return got


def load_rmse_table(path):
    global RMSE_TABLE
    if path and Path(path).exists():
        d = json.load(open(path))
        raw = d["bins"] if isinstance(d, dict) and "bins" in d else d
        RMSE_TABLE = _validate_rmse_bins(raw)
        print(f"[rfsens] RMSE table loaded from {path}")
    else:
        print("[rfsens] WARNING: no --rf-rmse-json; using documented "
              "limitations.md §1 bins")


def save_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[out] {path} ({len(rows)} rows)")


# ------------------------------------------------------------------- modes
def mode_selftest(args):
    cfg = load_config(args.config)
    reward_cfg = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    ok = True
    # 1) scenario_step(normal) == step() (numerical equivalence within tol)
    st = synthetic_init(np.random.RandomState(0))
    a = 2
    ns1, r1, rev1, sup1, pen1, _ = scenario_step(
        st, a, reward_cfg, n_actions, np.random.RandomState(7), SCENARIOS["normal"])
    ns2, r2, _ = step(st, a, reward_cfg, n_actions, np.random.RandomState(7))
    eq = (abs(r1 - r2) < 1e-12) and \
         (abs(ns1.recent_rainfall - ns2.recent_rainfall) < 1e-9)
    ok &= eq
    print(f"[selftest 1] scenario_step(normal)==step(): {'OK' if eq else 'FAIL'} "
          f"(r={r1:.6f} vs {r2:.6f}, rain={ns1.recent_rainfall:.4f} vs "
          f"{ns2.recent_rainfall:.4f})")
    # 1b) multi-step numerical equivalence under normal (draw-order check)
    st1 = synthetic_init(np.random.RandomState(0))
    st2 = synthetic_init(np.random.RandomState(0))
    rng1, rng2 = np.random.RandomState(11), np.random.RandomState(11)
    fields = ["gw_level", "recent_rainfall", "crop_water_demand",
              "month_sin", "month_cos", "extraction_rate"]
    ok1b = True
    for t in range(5):
        ns1, r1, *_ = scenario_step(st1, 2, reward_cfg, n_actions, rng1, SCENARIOS["normal"])
        ns2, r2, _ = step(st2, 2, reward_cfg, n_actions, rng2)
        same = (abs(r1 - r2) < 1e-12) and all(
            abs(getattr(ns1, f) - getattr(ns2, f)) < 1e-9 for f in fields)
        if not same:
            ok1b = False
            print(f"[selftest 1b] mismatch at t={t}: r={r1:.6f} vs {r2:.6f}")
            break
        st1, st2 = ns1, ns2
    ok &= ok1b
    print(f"[selftest 1b] 5-step numerical equivalence under normal: "
          f"{'OK' if ok1b else 'FAIL'}")
    # 2) revenue curve probe
    print("[selftest 2] crop_revenue_proxy(action, demand) over d in [0,1]:")
    for act in range(n_actions):
        pf = act / (n_actions - 1)
        dgrid = np.linspace(0, 1, 11)
        rs = [crop_revenue_proxy(act, d, n_actions) for d in dgrid]
        monotone = all(b >= a for a, b in zip(rs, rs[1:]))
        plateau_ok = abs(rs[-1] - pf) < 1e-12
        ok &= monotone and plateau_ok
        print(f"    action {act} pf={pf:.2f}: {[round(float(r), 3) for r in rs]} "
              f"| monotone={monotone} plateau_at_pf={plateau_ok}")
    # 3) deterministic split with explicit seed
    def _split_replica(n_episodes, seed):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n_episodes)
        n_train = int(n_episodes * 0.70)
        n_val = int(n_episodes * 0.15)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    try:
        from src.rl.bc_baseline import split_episode_indices
        real = split_episode_indices(1500, seed=SPLIT_SEED)
        rep = _split_replica(1500, SPLIT_SEED)
        match = all((np.array(a) == np.array(b)).all() for a, b in zip(real, rep))
        det = all((np.array(a) == np.array(b)).all()
                  for a, b in zip(real, _split_replica(1500, SPLIT_SEED)))
        ok &= match and det
        print(f"[selftest 3] split deterministic + replica==real "
              f"(seed={SPLIT_SEED}): {'OK' if (match and det) else 'FAIL'}")
    except ImportError:
        s1 = _split_replica(1500, SPLIT_SEED)
        s2 = _split_replica(1500, SPLIT_SEED)
        sizes_ok = (len(s1[0]) == 1050 and len(s1[1]) == 225 and len(s1[2]) == 225)
        det = all((np.array(a) == np.array(b)).all() for a, b in zip(s1, s2))
        ok &= det and sizes_ok
        print(f"[selftest 3] split replica deterministic + sizes 1050/225/225 "
              f"(seed={SPLIT_SEED}, d3rlpy absent): "
              f"{'OK' if (det and sizes_ok) else 'FAIL'}")
    # 4) tiny pure-numpy rollout pipeline
    def dummy(state_vec):
        return int(np.clip(round(state_vec[0] / 10.0), 0, n_actions - 1))
    init = make_init_states(None, 25, np.random.RandomState(3))
    res = rollout_policy(dummy, init, cfg, reward_cfg, n_actions,
                         SCENARIOS["normal"],
                         np.random.RandomState(5), 6, 0.99)
    ret_mean = float(res["returns"].mean())
    ok &= np.isfinite(ret_mean)
    print(f"[selftest 4] dummy rollout (25 eps x 6 steps): "
          f"return_mean={ret_mean:.3f}, finite={np.isfinite(ret_mean)}, "
          f"action_hist={action_hist(res['actions'], n_actions)}")
    # 5) bootstrap inline copy smoke
    b = paired_bootstrap_difference(np.zeros(20), np.ones(20), 200, 1)
    ok &= abs(b["observed_difference"] + 1.0) < 1e-12 and b["ci_contains_zero"] is False
    print(f"[selftest 5] inline bootstrap: obs_diff={b['observed_difference']:.3f} "
          f"(expect -1.0), contains0={b['ci_contains_zero']} -> "
          f"{'OK' if ok else 'FAIL'}")
    # 6) validate_rewards envelope
    vr = validate_rewards(np.array([0.0, 0.5, 1.5]), "t", (1.0, 2.0, 0.5))
    print(f"[selftest 6] validate_rewards envelope max={vr['max']} cap={vr['envelope_w1_plus_w3']}: OK")
    # 7) rainfall non-compounding regression check under drought
    st = SimState(gw_level=10.0, recent_rainfall=100.0, crop_water_demand=0.5,
                  month_sin=0.0, month_cos=1.0, extraction_rate=0.3)
    trace, s, rng = [], st, np.random.RandomState(11)
    for _ in range(8):
        ns, r, _, _, _, _ = scenario_step(s, 1, reward_cfg, n_actions, rng,
                                          SCENARIOS["drought"])
        trace.append(round(float(ns.recent_rainfall), 2))
        s = ns
    noncomp = all(v > 40.0 for v in trace)
    ok &= noncomp
    print(f"[selftest 7] drought 8-step rainfall trace (base=100, f=0.5): "
          f"{trace} -> mean={np.mean(trace):.1f} "
          f"non-compounding={'OK' if noncomp else 'FAIL'}")
    # 8) drought demand does NOT compound (regression check)
    st = SimState(gw_level=10.0, recent_rainfall=100.0, crop_water_demand=0.5,
                  month_sin=0.0, month_cos=1.0, extraction_rate=0.3)
    trace_d, s8, rng8 = [], st, np.random.RandomState(11)
    for _ in range(8):
        ns, r, _, _, _, _ = scenario_step(s8, 0, reward_cfg, n_actions, rng8,
                                          SCENARIOS["drought"])
        trace_d.append(round(float(ns.crop_water_demand), 4))
        s8 = ns
    noncomp_d = all(v < 0.99 for v in trace_d)
    ok &= noncomp_d
    print(f"[selftest 8] drought 8-step demand trace (base=0.5, f=1.2): "
          f"{trace_d} -> non-compounding={'OK' if noncomp_d else 'FAIL'}")
    # 9) manifest roundtrip
    tmp_root = Path(tempfile.mkdtemp(prefix="run_paper_selftest_"))
    try:
        tmp_h5 = tmp_root / "dummy_dataset.h5"
        tmp_h5.write_bytes(b"dummy")
        class _A:
            config = "config/rl_config.yaml"
            eps = _EPS
            alpha = 1.0
            qfunc = "qr"
            n_quantiles = 32
            oversample = 3
            n_steps = 10
            cql_lr = DEFAULT_CQL_LR
        mean0 = np.ones(EXPECTED_STATE_DIM)
        std0 = np.full(EXPECTED_STATE_DIM, 0.5)
        save_norm_manifest(tmp_root, tmp_h5, mean0, std0, _A)
        m, s = load_norm_manifest(tmp_root, tmp_h5, _A)
        expected_cfg = _full_training_cfg(tmp_h5, _A, mean0, std0,
                                          DEFAULT_CQL_LR, include_bc=True)
        write_training_manifest(tmp_root, "qr", expected_cfg)
        check_training_manifest(tmp_root, "qr", expected_cfg)
        ok9 = np.allclose(m, mean0) and np.allclose(s, std0)
        ok &= ok9
        print(f"[selftest 9] manifest roundtrip: {'OK' if ok9 else 'FAIL'}")
    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
    # 10) negative provenance tests
    tmp_root = Path(tempfile.mkdtemp(prefix="run_paper_selftest_neg_"))
    try:
        tmp_h5 = tmp_root / "dummy_dataset.h5"
        tmp_h5.write_bytes(b"dummy")
        class _A2:
            config = "config/rl_config.yaml"
            eps = _EPS
            alpha = 1.0
            qfunc = "qr"
            n_quantiles = 32
            oversample = 3
            n_steps = 10
            cql_lr = DEFAULT_CQL_LR
        mean0 = np.ones(EXPECTED_STATE_DIM)
        std0 = np.full(EXPECTED_STATE_DIM, 0.5)
        save_norm_manifest(tmp_root, tmp_h5, mean0, std0, _A2)
        p = norm_manifest_path(tmp_root)
        good = p.read_text()
        ok10 = True
        # 10a) missing required key -> reject
        d = json.loads(good); d.pop("dataset_sha1", None)
        p.write_text(json.dumps(d))
        try:
            load_norm_manifest(tmp_root, tmp_h5, _A2)
            ok10 = False; print("[selftest 10a] missing dataset_sha1 not rejected: FAIL")
        except SystemExit:
            pass
        p.write_text(good)
        # 10b) tampered dataset bytes -> sha1 mismatch -> reject
        tmp_h5.write_bytes(b"tampered")
        try:
            load_norm_manifest(tmp_root, tmp_h5, _A2)
            ok10 = False; print("[selftest 10b] tampered dataset not rejected: FAIL")
        except SystemExit:
            pass
        tmp_h5.write_bytes(b"dummy")
        # 10c) tampered norm_fingerprint -> reject
        d = json.loads(good); d["norm_fingerprint"] = "0" * 40
        p.write_text(json.dumps(d))
        try:
            load_norm_manifest(tmp_root, tmp_h5, _A2)
            ok10 = False; print("[selftest 10c] tampered norm_fingerprint not rejected: FAIL")
        except SystemExit:
            pass
        p.write_text(good)
        # 10d) wrong training hyperparameter -> reject
        cfg_ok = _full_training_cfg(tmp_h5, _A2, mean0, std0,
                                    DEFAULT_CQL_LR, include_bc=True)
        write_training_manifest(tmp_root, "qr", cfg_ok)
        cfg_bad = dict(cfg_ok); cfg_bad["alpha"] = 2.0
        try:
            check_training_manifest(tmp_root, "qr", cfg_bad)
            ok10 = False; print("[selftest 10d] wrong hyperparameter not rejected: FAIL")
        except SystemExit:
            pass
        # 10e) missing required key in training manifest -> reject
        tp = training_manifest_path(tmp_root, "qr")
        good_t = tp.read_text()
        dt = json.loads(good_t); dt.pop("qfunc", None)
        tp.write_text(json.dumps(dt))
        try:
            check_training_manifest(tmp_root, "qr", cfg_ok)
            ok10 = False; print("[selftest 10e] missing qfunc in training manifest not rejected: FAIL")
        except SystemExit:
            pass
        tp.write_text(good_t)
        # 10f) scenario-resume hparam check: mismatch must be rejected
        check_norm_hparams(json.loads(good), _A2, DEFAULT_CQL_LR)
        class _ABad:
            alpha = 2.0
            qfunc = "qr"
            n_quantiles = 32
            oversample = 3
            n_steps = 10
            cql_lr = DEFAULT_CQL_LR
        try:
            check_norm_hparams(json.loads(good), _ABad, DEFAULT_CQL_LR)
            ok10 = False; print("[selftest 10f] scenario hparam mismatch not rejected: FAIL")
        except SystemExit:
            pass
        # 10g) missing gamma in training manifest -> reject
        dt2 = json.loads(good_t); dt2.pop("gamma", None)
        tp.write_text(json.dumps(dt2))
        try:
            check_training_manifest(tmp_root, "qr", cfg_ok)
            ok10 = False; print("[selftest 10g] missing gamma not rejected: FAIL")
        except SystemExit:
            pass
        # 10h) missing bc_beta in training manifest -> reject
        dt3 = json.loads(good_t); dt3.pop("bc_beta", None)
        tp.write_text(json.dumps(dt3))
        try:
            check_training_manifest(tmp_root, "qr", cfg_ok)
            ok10 = False; print("[selftest 10h] missing bc_beta not rejected: FAIL")
        except SystemExit:
            pass
        tp.write_text(good_t)
        # 10i) qfunc in manifest disagrees with expected qfunc -> reject
        #      (validates the new variant/qfunc split actually distinguishes
        #      the on-disk directory suffix from the trained critic type)
        dt4 = json.loads(good_t); dt4["qfunc"] = "mean"
        tp.write_text(json.dumps(dt4))
        try:
            check_training_manifest(tmp_root, "qr", cfg_ok)   # cfg_ok.qfunc = "qr"
            ok10 = False; print("[selftest 10i] qfunc mismatch not rejected: FAIL")
        except SystemExit:
            pass
        tp.write_text(good_t)
        ok &= ok10
        print(f"[selftest 10] negative provenance tests: {'OK' if ok10 else 'FAIL'}")
    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
    try:
        import d3rlpy
        print("[selftest] d3rlpy present -> training paths runnable")
    except Exception:
        print("[selftest] d3rlpy NOT installed -> training/load paths not "
              "smoke-tested here (expected in this sandbox)")
    print(f"\n[selftest] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


def rollout_policy(action_fn, init_states, cfg, reward_cfg, n_actions, scn, rng,
                   traj_len, gamma=0.99, perturb_k=0.0, track_components=True,
                   record_actions=True):
    returns, drets, comps, actions, states = [], [], [], [], []
    for st0 in init_states:
        st = st0; tot = d = 0.0; disc = 1.0; c = np.zeros(3); acts = []
        for _ in range(traj_len):
            vec = st.to_array()
            if perturb_k:  # signed deterministic observation-error sensitivity only
                vec = vec.copy()
                vec[0] = float(np.clip(st.gw_level +
                                       perturb_k * rmse_for_depth(st.gw_level),
                                       0.0, None))
            a = int(action_fn(vec))
            acts.append(a)
            ns, r, rev, sup, pen, _ = scenario_step(st, a, reward_cfg, n_actions, rng, scn)
            ang = np.arctan2(st.month_sin, st.month_cos) + (2 * np.pi / 4)
            ns = replace(ns, month_sin=float(np.sin(ang)),
                         month_cos=float(np.cos(ang)))
            if track_components:
                c += np.array([rev, sup, pen])
            tot += r; d += disc * r; disc *= gamma
            states.append(st.to_array())
            st = ns
        returns.append(tot); drets.append(d); comps.append(c); actions.append(acts)
    return {"returns": np.array(returns), "discounted": np.array(drets),
            "components": np.array(comps), "actions": np.array(actions),
            "states": np.array(states)}


def paired_rollout(bc_fn, cql_fn, init_states, cfg, reward_cfg, n_actions, scn,
                   traj_len, gamma, eval_seed, track_components=False):
    # Common random numbers: identical init states and identical RNG stream
    b = rollout_policy(bc_fn, init_states, cfg, reward_cfg, n_actions, scn,
                       np.random.RandomState(eval_seed), traj_len, gamma,
                       track_components=track_components)
    c = rollout_policy(cql_fn, init_states, cfg, reward_cfg, n_actions, scn,
                       np.random.RandomState(eval_seed), traj_len, gamma,
                       track_components=track_components)
    return b, c


def mode_adapt(args):
    """ADAPTATION experiment (secondary): per-scenario retraining --
    train BC/CQL on each (scenario x gw) dataset, then evaluate under the
    SAME scenario. Answers 'how does a policy trained per-scenario perform
    there?'. For the robustness/generalization claim use the `scenario` mode
    (train-on-normal, frozen policy). Refuses to overwrite an existing
    model directory. Writes a normalization manifest and a training
    manifest for each cell."""
    eps = ensure_eps_synced(args)
    cql_lr = _resolve_cql_lr(args)
    rf_df = load_rf_table(args)
    cfg = load_config(args.config)
    reward_cfg = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    scn_names = [s.strip() for s in args.scenarios.split(",")]
    gws = [float(x) for x in args.gw_levels.split(",")]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scn, gw in itertools.product(scn_names, gws):
        s = SCENARIOS[scn]; tag = f"adapt_{scn}_gw{int(gw)}"
        cell_root = out / tag
        cell_norm = norm_manifest_path(cell_root)
        if cell_norm.exists():
            raise SystemExit(
                f"[adapt] refusing to overwrite existing validated models at "
                f"{cell_root} (remove it, or choose a different --out-dir)")
        rng = np.random.RandomState(cfg["simulation"]["random_seed"])
        obs, act, rew, term, to = make_arrays(rf_df, cfg, reward_cfg, n_actions,
                                              s, gw, rng, args.behavior.split(","),
                                              expected_eps=args.eps)
        rows.append(validate_rewards(rew, tag, (reward_cfg["w1_crop_revenue"],
                                                reward_cfg["w2_depletion_penalty"],
                                                reward_cfg["w3_domestic_supply"])))
        h5 = out / f"dataset_{tag}.h5"
        dump_dataset(h5, obs, act, rew, term, to,
                     {"mode": "adapt", "scenario": scn, "gw_level": gw,
                      "behavior": args.behavior.split(","), "eps": eps,
                      "alpha": args.alpha, "cql_lr": cql_lr,
                      "config": str(args.config),
                      "config_sha1": file_sha1(args.config)})
        bc_dir, cql_dir, mean, std = train_pair(h5, cell_root, args.alpha,
            args.qfunc, args.n_quantiles, args.oversample, args.n_steps, SEEDS,
            cql_lr=cql_lr)
        save_norm_manifest(cell_root, h5, mean, std, args)
        expected_cfg = _full_training_cfg(h5, args, mean, std, cql_lr,
                                          include_bc=True)
        write_training_manifest(cell_root, "frozen", expected_cfg)
        bc_all, cql_all = [], []
        for sd in SEEDS:
            init = make_init_states(rf_df, args.n_episodes,
                np.random.RandomState(args.seed_eval + sd), gw)
            b, c = paired_rollout(load_model_fn(bc_dir / f"bc_seed{sd}.d3", mean, std),
                load_model_fn(cql_dir / f"cql_seed{sd}.d3", mean, std), init,
                cfg, reward_cfg, n_actions, s, traj_len, args.gamma,
                args.seed_eval + sd)
            bc_all.append(b["returns"]); cql_all.append(c["returns"])
            rows.append({"kind": "per_seed", "cell": tag, "seed": sd,
                "bc_return": float(b["returns"].mean()),
                "cql_return": float(c["returns"].mean()),
                "bc_actions": action_hist(b["actions"], n_actions),
                "cql_actions": action_hist(c["actions"], n_actions),
                **bootstrap_row(f"{tag}_s{sd}", c["returns"], b["returns"],
                                args.seed_bootstrap)})
        rows.append({"kind": "pooled", "cell": tag,
            "bc_return": float(np.concatenate(bc_all).mean()),
            "cql_return": float(np.concatenate(cql_all).mean()),
            **bootstrap_row(f"{tag}_pooled", np.concatenate(cql_all),
                            np.concatenate(bc_all), args.seed_bootstrap)})
    # per-seed + pooled delta_from_normal
    norm = {}
    for r in rows:
        if r["kind"] == "per_seed" and r["cell"].startswith("adapt_normal_"):
            gw = int(r["cell"].split("_gw")[1])
            norm.setdefault((gw, r["seed"]), (r["bc_return"], r["cql_return"]))
    for r in rows:
        if r["kind"] == "per_seed":
            gw = int(r["cell"].split("_gw")[1])
            refb, refc = norm.get((gw, r["seed"]), (float("nan"), float("nan")))
            r["bc_delta_from_normal"] = float(r["bc_return"] - refb)
            r["cql_delta_from_normal"] = float(r["cql_return"] - refc)
        elif r["kind"] == "pooled":
            gw = int(r["cell"].split("_gw")[1])
            nb = np.mean([norm[(gw, s)][0] for s in SEEDS if (gw, s) in norm])
            nc = np.mean([norm[(gw, s)][1] for s in SEEDS if (gw, s) in norm])
            r["bc_delta_from_normal"] = float(r["bc_return"] - nb)
            r["cql_delta_from_normal"] = float(r["cql_return"] - nc)
    save_rows(out / "scenario_adaptation_summary.csv", rows)
    print("[ok] ADAPTATION (per-scenario retraining) done -> scenario_adaptation_summary.csv")


def mode_scenario(args):
    """FROZEN-POLICY robustness (MAIN result): train BC+CQL ONCE on normal
    conditions, freeze the models, then evaluate them across the full
    scenario x GW-depth grid. Answers 'how robust is a fixed policy trained
    under normal conditions when the environment changes?'. (Per-scenario
    retraining lives in `adapt` mode.)

    Training is guarded: if a validated frozen_models directory already
    exists, this mode refuses to overwrite it unless --resume is passed
    (which reuses the existing models after MANDATORY manifest + hparam
    validation)."""
    eps = ensure_eps_synced(args)
    cql_lr = _resolve_cql_lr(args)
    rf_df = load_rf_table(args)
    cfg = load_config(args.config)
    reward_cfg = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    scn_names = [s.strip() for s in args.scenarios.split(",")]
    gws = [float(x) for x in args.gw_levels.split(",")]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frozen_root = out / "frozen_models"
    frozen_norm = norm_manifest_path(frozen_root)
    # ---- decide between resume / fresh-train / refuse-overwrite ----
    if args.resume:
        if not frozen_norm.exists():
            raise SystemExit(
                f"[scenario] --resume requires an existing validated "
                f"frozen_models directory with a normalization manifest: "
                f"{frozen_norm} not found.")
        man_data = json.loads(frozen_norm.read_text())
        h5 = Path(man_data["dataset"])
        if not h5.exists():
            raise SystemExit(
                f"[scenario] --resume: dataset referenced by manifest missing: {h5}")
        if args.dataset and Path(args.dataset).resolve() != h5.resolve():
            raise SystemExit(
                f"[scenario] --resume dataset mismatch: manifest={h5}, "
                f"--dataset={args.dataset}")
        # Re-validate normalization + config/script/provenance
        mean, std = load_norm_manifest(frozen_root, h5, args)
        # Validate SELECTED training hyperparameters (closes the resume-mislabel
        # hole for these fields)
        check_norm_hparams(man_data, args, cql_lr)
        # Validate the frozen-model training manifest -- MANDATORY on resume:
        # the full CQL + BC hyperparameter set is checked there (see
        # _full_training_cfg).
        fm_path = training_manifest_path(frozen_root, "frozen")
        if not fm_path.exists():
            raise SystemExit(
                f"[scenario] --resume requires a training manifest at "
                f"{fm_path}; the frozen models are not fully validated "
                f"without it (regenerate with the current script).")
        expected_cfg = _full_training_cfg(h5, args, mean, std, cql_lr,
                                          include_bc=True)
        check_training_manifest(frozen_root, "frozen", expected_cfg)
        bc_dir = frozen_root / "bc"
        cql_dir = frozen_root / "cql"
        if not bc_dir.exists() or not cql_dir.exists():
            raise SystemExit(
                f"[scenario] --resume: model dirs missing under {frozen_root}")
        print(f"[scenario] resuming frozen models from {frozen_root}")
    else:
        if frozen_norm.exists():
            raise SystemExit(
                f"[scenario] refusing to overwrite existing validated models "
                f"({frozen_norm}); pass --resume to reuse them, or remove "
                f"{frozen_root} to retrain from scratch.")
        # ---- train once on NORMAL, freeze ----
        if args.dataset and Path(args.dataset).exists():
            h5 = Path(args.dataset)
            side = h5.with_suffix(".json")
            if side.exists():
                prov = json.load(open(side))
                actual_sha1 = file_sha1(h5)
                mismatches = []
                for req in ("mode", "scenario", "behavior", "eps", "alpha",
                            "config", "config_sha1", "script_sha1", "sha1"):
                    if req not in prov:
                        mismatches.append(f"missing required key: {req}")
                if "mode" in prov and prov["mode"] != "scenario_frozen":
                    mismatches.append(f"mode {prov['mode']!r} != 'scenario_frozen'")
                if "scenario" in prov and prov["scenario"] != "normal":
                    mismatches.append(f"scenario {prov['scenario']!r} != 'normal'")
                expected_behavior = sorted(args.behavior.split(","))
                if "behavior" in prov and sorted(prov["behavior"]) != expected_behavior:
                    mismatches.append(
                        f"behavior {prov['behavior']} != {expected_behavior}")
                if "eps" in prov and not np.isclose(float(prov["eps"]), float(args.eps)):
                    mismatches.append(f"eps {prov['eps']} != {args.eps}")
                if "alpha" in prov and not np.isclose(float(prov["alpha"]), float(args.alpha)):
                    mismatches.append(f"alpha {prov['alpha']} != {args.alpha}")
                if "config" in prov and str(prov["config"]) != str(args.config):
                    mismatches.append(f"config {prov['config']} != {args.config}")
                if "config_sha1" in prov and prov["config_sha1"] != file_sha1(args.config):
                    mismatches.append(
                        f"config_sha1 {prov['config_sha1']} != {file_sha1(args.config)}")
                if "script_sha1" in prov and prov["script_sha1"] != _script_sha1():
                    mismatches.append(
                        f"script_sha1 {prov['script_sha1']} != {_script_sha1()}")
                if "sha1" in prov and prov["sha1"] != actual_sha1:
                    mismatches.append(f"sha1 {prov['sha1']} != {actual_sha1}")
                if mismatches:
                    raise SystemExit(
                        "[scenario] REFUSING dataset: provenance mismatch(es): "
                        + "; ".join(mismatches))
                print(f"[scenario] reusing normal train dataset {h5} "
                      f"(scenario={prov.get('scenario')}, sha1={actual_sha1})")
            elif args.allow_missing_provenance:
                print(f"[scenario] --allow-missing-provenance: no sidecar for {h5}; "
                      f"trusting that it is normal-condition training data "
                      f"(NOT recommended for paper results)")
            else:
                raise SystemExit(
                    f"[scenario] REFUSING dataset: provenance sidecar missing for "
                    f"{h5}; pass --allow-missing-provenance to override")
        else:
            rng = np.random.RandomState(cfg["simulation"]["random_seed"])
            obs, act, rew, term, to = make_arrays(rf_df, cfg, reward_cfg, n_actions,
                SCENARIOS["normal"], None, rng, args.behavior.split(","),
                expected_eps=args.eps)
            h5 = out / "dataset_normal_train.h5"
            dump_dataset(h5, obs, act, rew, term, to,
                         {"mode": "scenario_frozen", "scenario": "normal",
                          "behavior": args.behavior.split(","), "eps": eps,
                          "alpha": args.alpha, "cql_lr": cql_lr,
                          "config": str(args.config),
                          "config_sha1": file_sha1(args.config)})
        bc_dir, cql_dir, mean, std = train_pair(h5, frozen_root, args.alpha,
            args.qfunc, args.n_quantiles, args.oversample, args.n_steps, SEEDS,
            cql_lr=cql_lr)
        save_norm_manifest(frozen_root, h5, mean, std, args)
        expected_cfg = _full_training_cfg(h5, args, mean, std, cql_lr,
                                          include_bc=True)
        write_training_manifest(frozen_root, "frozen", expected_cfg)
    # ---- evaluate FROZEN models across scenario x gw grid ----
    rows = []
    for scn, gw in itertools.product(scn_names, gws):
        s = SCENARIOS[scn]; tag = f"{scn}_gw{int(gw)}"
        bc_all, cql_all = [], []
        for sd in SEEDS:
            init = make_init_states(rf_df, args.n_episodes,
                np.random.RandomState(args.seed_eval + sd), gw)
            b, c = paired_rollout(load_model_fn(bc_dir / f"bc_seed{sd}.d3", mean, std),
                load_model_fn(cql_dir / f"cql_seed{sd}.d3", mean, std), init,
                cfg, reward_cfg, n_actions, s, traj_len, args.gamma,
                args.seed_eval + sd)
            bc_all.append(b["returns"]); cql_all.append(c["returns"])
            rows.append({"kind": "per_seed", "cell": tag, "scenario": scn,
                "gw_level": gw, "seed": sd,
                "bc_return": float(b["returns"].mean()),
                "cql_return": float(c["returns"].mean()),
                "bc_actions": action_hist(b["actions"], n_actions),
                "cql_actions": action_hist(c["actions"], n_actions),
                **bootstrap_row(f"{tag}_s{sd}", c["returns"], b["returns"],
                                args.seed_bootstrap)})
        rows.append({"kind": "pooled", "cell": tag, "scenario": scn,
            "gw_level": gw,
            "bc_return": float(np.concatenate(bc_all).mean()),
            "cql_return": float(np.concatenate(cql_all).mean()),
            **bootstrap_row(f"{tag}_pooled", np.concatenate(cql_all),
                            np.concatenate(bc_all), args.seed_bootstrap)})
    # per-seed + pooled degradation vs normal at the SAME gw depth
    norm = {}
    for r in rows:
        if r["kind"] == "per_seed" and r["scenario"] == "normal":
            norm[(int(r["gw_level"]), r["seed"])] = (r["bc_return"], r["cql_return"])
    for r in rows:
        if r["kind"] == "per_seed" and r["scenario"] != "normal":
            ref = norm.get((int(r["gw_level"]), r["seed"]), (float("nan"), float("nan")))
            r["bc_delta_from_normal"] = float(r["bc_return"] - ref[0])
            r["cql_delta_from_normal"] = float(r["cql_return"] - ref[1])
        elif r["kind"] == "pooled" and r["scenario"] != "normal":
            gw = int(r["gw_level"])
            nb = np.mean([norm[(gw, s)][0] for s in SEEDS if (gw, s) in norm])
            nc = np.mean([norm[(gw, s)][1] for s in SEEDS if (gw, s) in norm])
            r["bc_delta_from_normal"] = float(r["bc_return"] - nb)
            r["cql_delta_from_normal"] = float(r["cql_return"] - nc)
        else:
            r["bc_delta_from_normal"] = 0.0
            r["cql_delta_from_normal"] = 0.0
    save_rows(out / "scenario_summary.csv", rows)
    print("[ok] FROZEN-POLICY scenario robustness done -> scenario_summary.csv")


def mode_ood(args):
    eps = ensure_eps_synced(args)
    cql_lr = _resolve_cql_lr(args)
    rf_df = load_rf_table(args)
    cfg = load_config(args.config)
    reward_cfg = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ood_models_root = out / "ood_models"
    ood_norm = norm_manifest_path(ood_models_root)
    if ood_norm.exists():
        raise SystemExit(
            f"[ood] refusing to overwrite existing validated models at "
            f"{ood_models_root} (remove it, or choose a different --out-dir)")
    test_scenarios = [s.strip() for s in args.test_scenarios.split(",") if s.strip()]
    dup = [s for s in test_scenarios if s == args.train_scenario]
    if dup:
        raise SystemExit(
            f"[ood] --test-scenarios includes the training scenario "
            f"'{args.train_scenario}'; remove it (the train scenario is "
            f"always evaluated automatically as the reference row)")
    if args.dataset:
        h5 = Path(args.dataset)
        if not h5.exists():
            raise SystemExit(f"[ood] --dataset does not exist: {h5}")
        side = h5.with_suffix(".json")
        if side.exists():
            prov = json.load(open(side))
            actual_sha1 = file_sha1(h5)
            print(f"[ood] dataset provenance sidecar: {prov.get('scenario')} "
                  f"({prov.get('n_transitions')} transitions, sha1={prov.get('sha1')})")
            mismatches = []
            for req in ("mode", "scenario", "behavior", "eps", "alpha",
                        "config", "config_sha1", "script_sha1", "sha1"):
                if req not in prov:
                    mismatches.append(f"missing required key: {req}")
            if "mode" in prov and prov["mode"] != "ood":
                mismatches.append(f"mode {prov['mode']!r} != 'ood'")
            if "scenario" in prov and prov["scenario"] != args.train_scenario:
                mismatches.append(
                    f"scenario '{prov['scenario']}' != "
                    f"'{args.train_scenario}'")
            expected_behavior = sorted(args.behavior.split(","))
            if "behavior" in prov and sorted(prov["behavior"]) != expected_behavior:
                mismatches.append(
                    f"behavior {prov['behavior']} != {expected_behavior}")
            if "eps" in prov and not np.isclose(float(prov["eps"]), float(args.eps)):
                mismatches.append(f"eps {prov['eps']} != {args.eps}")
            if "alpha" in prov and not np.isclose(float(prov["alpha"]), float(args.alpha)):
                mismatches.append(f"alpha {prov['alpha']} != {args.alpha}")
            if "config" in prov and str(prov["config"]) != str(args.config):
                mismatches.append(f"config {prov['config']} != {args.config}")
            if "config_sha1" in prov and prov["config_sha1"] != file_sha1(args.config):
                mismatches.append(
                    f"config_sha1 {prov['config_sha1']} != {file_sha1(args.config)}")
            if "script_sha1" in prov and prov["script_sha1"] != _script_sha1():
                mismatches.append(
                    f"script_sha1 {prov['script_sha1']} != {_script_sha1()}")
            if "sha1" in prov and prov["sha1"] != actual_sha1:
                mismatches.append(f"sha1 {prov['sha1']} != {actual_sha1}")
            if mismatches:
                raise SystemExit(
                    "[ood] REFUSING dataset: provenance mismatch(es): "
                    + "; ".join(mismatches))
        else:
            if args.allow_missing_provenance:
                print(f"[ood] --allow-missing-provenance: no sidecar for {h5}; "
                      f"trusting --train-scenario='{args.train_scenario}'")
            else:
                raise SystemExit(
                    f"[ood] REFUSING dataset: provenance sidecar missing for {h5}; "
                    f"pass --allow-missing-provenance to override")
    else:
        rng = np.random.RandomState(cfg["simulation"]["random_seed"])
        obs, act, rew, term, to = make_arrays(rf_df, cfg, reward_cfg, n_actions,
            SCENARIOS[args.train_scenario], None, rng, args.behavior.split(","),
            expected_eps=args.eps)
        h5 = out / f"dataset_{args.train_scenario}_train.h5"
        dump_dataset(h5, obs, act, rew, term, to,
                     {"mode": "ood", "scenario": args.train_scenario,
                      "behavior": args.behavior.split(","), "eps": eps,
                      "alpha": args.alpha, "cql_lr": cql_lr,
                      "config": str(args.config),
                      "config_sha1": file_sha1(args.config)})
    bc_dir, cql_dir, mean, std = train_pair(h5, ood_models_root, args.alpha,
        args.qfunc, args.n_quantiles, args.oversample, args.n_steps, SEEDS,
        cql_lr=cql_lr)
    save_norm_manifest(ood_models_root, h5, mean, std, args)
    expected_cfg = _full_training_cfg(h5, args, mean, std, cql_lr,
                                      include_bc=True)
    write_training_manifest(ood_models_root, "frozen", expected_cfg)
    # train-split-only support reference -- single load
    ep = load_episodes_raw(h5)
    tr = split_and_std(ep)[0]
    ref_raw = np.concatenate([ep[i].observations for i in tr], 0)
    if len(ref_raw) > 5000:
        ref_raw = ref_raw[np.random.RandomState(0).choice(len(ref_raw), 5000,
                                                          replace=False)]
    tree, self_dist = build_support(ref_raw, mean, std)
    rows = []
    for scn in [args.train_scenario] + test_scenarios:
        s = SCENARIOS[scn]
        bc_all, cql_all = [], []
        for sd in SEEDS:
            init = make_init_states(rf_df, args.n_episodes,
                np.random.RandomState(args.seed_eval + sd), None)
            b, c = paired_rollout(load_model_fn(bc_dir / f"bc_seed{sd}.d3", mean, std),
                load_model_fn(cql_dir / f"cql_seed{sd}.d3", mean, std), init,
                cfg, reward_cfg, n_actions, s, traj_len, args.gamma,
                args.seed_eval + sd)
            bc_all.append(b["returns"]); cql_all.append(c["returns"])
            rows.append({"kind": "per_seed", "scenario": scn, "seed": sd,
                "bc_return": float(b["returns"].mean()),
                "cql_return": float(c["returns"].mean()),
                "bc_ood_frac": ood_frac(b["states"], tree, self_dist, mean, std),
                "cql_ood_frac": ood_frac(c["states"], tree, self_dist, mean, std),
                **bootstrap_row(f"ood_{scn}_s{sd}", c["returns"], b["returns"],
                                args.seed_bootstrap)})
        rows.append({"kind": "pooled", "scenario": scn,
            "bc_return": float(np.concatenate(bc_all).mean()),
            "cql_return": float(np.concatenate(cql_all).mean()),
            **bootstrap_row(f"ood_{scn}_pooled", np.concatenate(cql_all),
                            np.concatenate(bc_all), args.seed_bootstrap)})
    # per-seed + pooled degradation vs TRAIN scenario
    norm_seed = {}
    for r in rows:
        if r["kind"] == "per_seed" and r["scenario"] == args.train_scenario:
            norm_seed[r["seed"]] = (r["bc_return"], r["cql_return"])
    for r in rows:
        if r["kind"] == "per_seed" and r["scenario"] != args.train_scenario:
            refb, refc = norm_seed.get(r["seed"], (float("nan"), float("nan")))
            r["bc_delta_from_train"] = float(r["bc_return"] - refb)
            r["cql_delta_from_train"] = float(r["cql_return"] - refc)
    pooled_norm = {r["scenario"]: (r["bc_return"], r["cql_return"]) for r in rows
                   if r["kind"] == "pooled" and r["scenario"] == args.train_scenario}
    for r in rows:
        if r["kind"] == "pooled" and r["scenario"] != args.train_scenario:
            nb, nc = pooled_norm.get(args.train_scenario,
                                     (float("nan"), float("nan")))
            r["bc_delta_from_train"] = float(r["bc_return"] - nb)
            r["cql_delta_from_train"] = float(r["cql_return"] - nc)
    save_rows(out / "ood_summary.csv", rows)
    print("[ok] OOD done -> ood_summary.csv")


def mode_weights(args):
    eps = ensure_eps_synced(args)
    cql_lr = _resolve_cql_lr(args)
    rf_df = load_rf_table(args)
    cfg = load_config(args.config)
    base = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vecs = [tuple(float(x) for x in v.split(",")) for v in args.weights.split(";")]
    rows = []
    for w in vecs:
        rc = dict(base)
        rc.update(w1_crop_revenue=w[0], w2_depletion_penalty=w[1], w3_domestic_supply=w[2])
        tag = f"w{w[0]:.1f}_{w[1]:.1f}_{w[2]:.1f}"
        cell_root = out / tag
        # Overwrite guard: refuse if a previously trained model directory is
        # already present under this weight cell (mirrors scenario/ood/adapt).
        if (cell_root / "bc").exists() or (cell_root / "cql").exists() \
           or norm_manifest_path(cell_root).exists():
            raise SystemExit(
                f"[weights] refusing to overwrite existing model directory at "
                f"{cell_root} (remove it, or choose a different --out-dir)")
        rng = np.random.RandomState(cfg["simulation"]["random_seed"])
        obs, act, rew, term, to = make_arrays(rf_df, cfg, rc, n_actions,
            SCENARIOS["normal"], None, rng, args.behavior.split(","),
            expected_eps=args.eps)
        rows.append(validate_rewards(rew, tag, w))
        h5 = out / f"dataset_{tag}.h5"
        dump_dataset(h5, obs, act, rew, term, to,
                     {"mode": "weights", "w": list(w), "behavior": args.behavior.split(","),
                      "eps": eps, "alpha": args.alpha, "cql_lr": cql_lr,
                      "config": str(args.config),
                      "config_sha1": file_sha1(args.config)})
        bc_dir, cql_dir, mean, std = train_pair(h5, cell_root, args.alpha,
            args.qfunc, args.n_quantiles, args.oversample, args.n_steps, SEEDS,
            cql_lr=cql_lr)
        save_norm_manifest(cell_root, h5, mean, std, args)
        expected_cfg = _full_training_cfg(h5, args, mean, std, cql_lr,
                                          include_bc=True)
        write_training_manifest(cell_root, "frozen", expected_cfg)
        bc_all, cql_all = [], []
        for sd in SEEDS:
            init = make_init_states(rf_df, args.n_episodes,
                np.random.RandomState(args.seed_eval + sd), None)
            b, c = paired_rollout(load_model_fn(bc_dir / f"bc_seed{sd}.d3", mean, std),
                load_model_fn(cql_dir / f"cql_seed{sd}.d3", mean, std), init,
                cfg, rc, n_actions, SCENARIOS["normal"], traj_len, args.gamma,
                args.seed_eval + sd, track_components=True)
            bc_all.append(b["returns"]); cql_all.append(c["returns"])
            bc_comp = b["components"].mean(0); cql_comp = c["components"].mean(0)
            bc_recon = w[0] * bc_comp[0] + w[2] * bc_comp[1] - w[1] * bc_comp[2]
            cql_recon = w[0] * cql_comp[0] + w[2] * cql_comp[1] - w[1] * cql_comp[2]
            bc_ret_mean = float(b["returns"].mean())
            cql_ret_mean = float(c["returns"].mean())
            if not np.isclose(bc_recon, bc_ret_mean, atol=1e-6):
                raise ValueError(
                    f"[weights] {tag} seed{sd}: BC reconstructed return "
                    f"{bc_recon:.6f} != actual return {bc_ret_mean:.6f}")
            if not np.isclose(cql_recon, cql_ret_mean, atol=1e-6):
                raise ValueError(
                    f"[weights] {tag} seed{sd}: CQL reconstructed return "
                    f"{cql_recon:.6f} != actual return {cql_ret_mean:.6f}")
            rows.append({"kind": "per_seed", "cell": tag, "seed": sd,
                "bc_return": bc_ret_mean,
                "cql_return": cql_ret_mean,
                "bc_crop": float(bc_comp[0]), "bc_supply": float(bc_comp[1]),
                "bc_penalty": float(bc_comp[2]), "bc_reconstructed": float(bc_recon),
                "cql_crop": float(cql_comp[0]), "cql_supply": float(cql_comp[1]),
                "cql_penalty": float(cql_comp[2]), "cql_reconstructed": float(cql_recon),
                "bc_actions": action_hist(b["actions"], n_actions),
                "cql_actions": action_hist(c["actions"], n_actions),
                **bootstrap_row(f"{tag}_s{sd}", c["returns"], b["returns"],
                                args.seed_bootstrap)})
        rows.append({"kind": "pooled", "cell": tag,
            "bc_return": float(np.concatenate(bc_all).mean()),
            "cql_return": float(np.concatenate(cql_all).mean()),
            **bootstrap_row(f"{tag}_pooled", np.concatenate(cql_all),
                            np.concatenate(bc_all), args.seed_bootstrap)})
    save_rows(out / "weight_ablation_summary.csv", rows)
    print("[ok] reward-weight ablation done -> weight_ablation_summary.csv")


def mode_rfsens(args):
    ensure_eps_synced(args)
    rf_df = load_rf_table(args)
    cfg = load_config(args.config)
    reward_cfg = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    h5 = Path(args.dataset)
    if not h5.exists():
        raise SystemExit(f"[rfsens] dataset not found: {h5}")
    load_rmse_table(args.rf_rmse_json)
    bc_dir, cql_dir = Path(args.bc_dir), Path(args.cql_dir)
    norm_root = None
    for cand in (bc_dir.parent, cql_dir.parent, bc_dir, cql_dir):
        if norm_manifest_path(cand).exists():
            norm_root = cand
            break
    if norm_root is None:
        raise SystemExit(
            f"[rfsens] no normalization_manifest.json found near --bc-dir="
            f"{bc_dir} or --cql-dir={cql_dir}; tried: "
            f"{[str(norm_manifest_path(c)) for c in (bc_dir.parent, cql_dir.parent, bc_dir, cql_dir)]}")
    mean, std = load_norm_manifest(norm_root, h5, args)
    ks = [float(x) for x in args.ks.split(",")]
    if 0.0 not in ks:
        ks = [0.0] + ks
    rows = []
    for sd in SEEDS:
        init = make_init_states(rf_df, args.n_episodes,
            np.random.RandomState(args.seed_eval + sd), None)
        for pol, mdir in (("BC", bc_dir), ("CQL", cql_dir)):
            fn = load_model_fn(mdir / f"{pol.lower()}_seed{sd}.d3", mean, std)
            base = None
            for k in ks:
                res = rollout_policy(fn, init, cfg, reward_cfg, n_actions,
                    SCENARIOS["normal"],
                    np.random.RandomState(args.seed_eval + sd + 1000),
                    traj_len, args.gamma, perturb_k=k)
                if k == 0:
                    base = res
                flip = float((res["actions"] != base["actions"]).mean()) if k != 0 else 0.0
                rows.append({"policy": pol, "seed": sd, "k": k,
                    "return": float(res["returns"].mean()),
                    "delta_return": float(res["returns"].mean()
                                          - base["returns"].mean()) if k != 0 else 0.0,
                    "action_flip_rate": flip})
    save_rows(out / "rf_sensitivity.csv", rows)
    print("[ok] RF-error sensitivity done -> rf_sensitivity.csv")


def mode_qrvsmean(args):
    ensure_eps_synced(args)
    cql_lr = _resolve_cql_lr(args)
    rf_df = load_rf_table(args)
    cfg = load_config(args.config)
    reward_cfg = cfg["reward"]; n_actions = cfg["action_space"]["n_actions"]
    traj_len = cfg["simulation"]["trajectory_length"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    h5 = Path(args.dataset)
    if not h5.exists():
        raise SystemExit(f"[qrvsmean] dataset not found: {h5}")
    episodes = load_episodes_raw(h5)
    resume = Path(args.resume) if args.resume else None
    if resume and not resume.exists():
        raise SystemExit(f"[qrvsmean] --resume path does not exist: {resume}")
    run_root = resume or out
    qr_dir = run_root / "cql_qr"
    mq_dir = run_root / "cql_mean"
    if resume:
        if not qr_dir.exists() or not mq_dir.exists():
            raise SystemExit(
                f"[qrvsmean] --resume={resume} requires both {qr_dir.name}/ "
                f"and {mq_dir.name}/ to exist (qr exists={qr_dir.exists()}, "
                f"mean exists={mq_dir.exists()}); partial resume is not "
                f"supported (either retrain fresh into --out-dir, or ensure "
                f"both dirs are present).")
        mean, std = load_norm_manifest(resume, h5, args)
        # qrvsmean trains BOTH QR and Mean CQL variants in one mode call.
        # Override qfunc per-variant so each training manifest records the
        # actual critic type it was trained with (args.qfunc may be "qr"
        # by default and is not the right value for the "mean" variant).
        cfg_base = _full_training_cfg(h5, args, mean, std, cql_lr, include_bc=False)
        cfg_qr = dict(cfg_base); cfg_qr["qfunc"] = "qr"
        cfg_mean = dict(cfg_base); cfg_mean["qfunc"] = "mean"
        check_training_manifest(resume, "qr", cfg_qr)
        check_training_manifest(resume, "mean", cfg_mean)
        print(f"[qrvsmean] resuming from {resume} (manifests verified)")
    else:
        _, _, _, _, _, _, _, split_mean, split_std = split_and_std(episodes)
        qr_dir, mean_qr, std_qr = train_cql_variant(h5, out, "qr", args.alpha,
            args.n_quantiles, args.oversample, args.n_steps, SEEDS,
            cql_lr=cql_lr, norm=(split_mean, split_std))
        mq_dir, mean_mq, std_mq = train_cql_variant(h5, out, "mean", args.alpha,
            args.n_quantiles, args.oversample, args.n_steps, SEEDS,
            cql_lr=cql_lr, norm=(split_mean, split_std))
        if not (np.allclose(mean_qr, split_mean) and np.allclose(std_qr, split_std)):
            raise RuntimeError(
                "[qrvsmean] QR training returned normalization different from "
                "the caller-supplied split_and_std statistics")
        if not (np.allclose(mean_mq, split_mean) and np.allclose(std_mq, split_std)):
            raise RuntimeError(
                "[qrvsmean] Mean-Q training returned normalization different "
                "from the caller-supplied split_and_std statistics")
        mean, std = split_mean, split_std
        save_norm_manifest(out, h5, mean, std, args)
        cfg_base = _full_training_cfg(h5, args, mean, std, cql_lr, include_bc=False)
        cfg_qr = dict(cfg_base); cfg_qr["qfunc"] = "qr"
        cfg_mean = dict(cfg_base); cfg_mean["qfunc"] = "mean"
        write_training_manifest(out, "qr", cfg_qr)
        write_training_manifest(out, "mean", cfg_mean)
    norm_man = json.loads(norm_manifest_path(run_root).read_text())
    qr_man = json.loads(training_manifest_path(run_root, "qr").read_text())
    mq_man = json.loads(training_manifest_path(run_root, "mean").read_text())
    shas = {norm_man.get("dataset_sha1"),
            qr_man.get("dataset_sha1"),
            mq_man.get("dataset_sha1")}
    if None in shas or len(shas) != 1:
        raise SystemExit(
            f"[qrvsmean] REFUSING: dataset_sha1 disagreement across "
            f"normalization/training manifests: {shas}")
    fps = {norm_man.get("norm_fingerprint"),
           qr_man.get("norm_fingerprint"),
           mq_man.get("norm_fingerprint")}
    if None in fps or len(fps) != 1:
        raise SystemExit(
            f"[qrvsmean] REFUSING: norm_fingerprint disagreement across "
            f"normalization/training manifests: {fps}")
    _, _, _, _, _, _, _, check_mean, check_std = split_and_std(episodes)
    if not (np.allclose(mean, check_mean) and np.allclose(std, check_std)):
        raise RuntimeError(
            "[qrvsmean] stored normalization does not match split_and_std normalization")
    rows = []
    for sd in SEEDS:
        init = make_init_states(rf_df, args.n_episodes,
            np.random.RandomState(args.seed_eval + sd), None)
        qr_fn = load_model_fn(qr_dir / f"cql_seed{sd}.d3", mean, std)
        mq_fn = load_model_fn(mq_dir / f"cql_seed{sd}.d3", mean, std)
        qr = rollout_policy(qr_fn, init, cfg, reward_cfg, n_actions,
            SCENARIOS["normal"], np.random.RandomState(args.seed_eval + sd),
            traj_len, args.gamma)
        mq = rollout_policy(mq_fn, init, cfg, reward_cfg, n_actions,
            SCENARIOS["normal"], np.random.RandomState(args.seed_eval + sd),
            traj_len, args.gamma)
        rows.append({"kind": "per_seed", "seed": sd,
            "qr_return": float(qr["returns"].mean()),
            "meanq_return": float(mq["returns"].mean()),
            "qr_minus_meanq": float(qr["returns"].mean() - mq["returns"].mean()),
            **bootstrap_row(f"qr_vs_mean_s{sd}", qr["returns"], mq["returns"],
                            args.seed_bootstrap)})
    save_rows(out / "qr_vs_mean_returns.csv", rows)
    print("[ok] QR-vs-Mean-Q return-level done -> qr_vs_mean_returns.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    def add_common(p):
        p.add_argument("--config", default="config/rl_config.yaml")
        p.add_argument("--rf-table", default="")
        p.add_argument("--out-dir", default="reports/paper_exp")
        p.add_argument("--behavior", default="random,greedy_extraction,epsilon_greedy_extraction")
        p.add_argument("--eps", type=float, default=0.1)
        p.add_argument("--alpha", type=float, default=1.0)
        p.add_argument("--qfunc", default="qr", choices=["qr", "mean"])
        p.add_argument("--n-quantiles", type=int, default=32)
        p.add_argument("--oversample", type=int, default=3)
        p.add_argument("--n-steps", type=int, default=20_000)
        p.add_argument("--n-episodes", type=int, default=500)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--cql-lr", type=float, default=DEFAULT_CQL_LR,
                       help=f"CQL learning rate (default {DEFAULT_CQL_LR}); "
                            f"the earlier canonical Phase-5 baseline used 3e-4.")
        p.add_argument("--seed-eval", type=int, default=9999)
        p.add_argument("--seed-bootstrap", type=int, default=2026)
        return p
    p = add_common(sub.add_parser("selftest"))
    p = add_common(sub.add_parser("scenario"))
    p.add_argument("--scenarios", default="drought,normal,high")
    p.add_argument("--gw-levels", default=",".join(str(int(g)) for g in GW_LEVELS))
    p.add_argument("--dataset", default="",
                   help="Optional existing normal-condition dataset (must have "
                        "provenance sidecar; else generated fresh)")
    p.add_argument("--allow-missing-provenance", action="store_true",
                   help="Override the hard provenance check for datasets without "
                        "a sidecar (NOT recommended for paper results)")
    p.add_argument("--resume", action="store_true",
                   help="Reuse an existing validated frozen_models directory "
                        "(refuses if missing); without --resume, an existing "
                        "validated frozen_models dir is not overwritten.")
    p = add_common(sub.add_parser("adapt"))
    p.add_argument("--scenarios", default="drought,normal,high")
    p.add_argument("--gw-levels", default=",".join(str(int(g)) for g in GW_LEVELS))
    p = add_common(sub.add_parser("ood"))
    p.add_argument("--dataset", default="")
    p.add_argument("--train-scenario", default="normal")
    p.add_argument("--test-scenarios", default="drought,extreme")
    p.add_argument("--allow-missing-provenance", action="store_true",
                   help="Override the hard provenance check for datasets without "
                        "a sidecar (NOT recommended for paper results)")
    p = add_common(sub.add_parser("weights"))
    p.add_argument("--weights", default=";".join(
        ",".join(str(int(x) if float(x).is_integer() else x) for x in v)
        for v in WEIGHT_VECTORS))
    p = add_common(sub.add_parser("rfsens"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--bc-dir", required=True)
    p.add_argument("--cql-dir", required=True)
    p.add_argument("--ks", default="-2,-1,-0.5,0.5,1,2")
    p.add_argument("--rf-rmse-json", default="")
    p = add_common(sub.add_parser("qrvsmean"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--resume", default="")
    args = ap.parse_args()
    global _EPS
    _EPS = args.eps
    {"selftest": mode_selftest, "scenario": mode_scenario, "adapt": mode_adapt,
     "ood": mode_ood, "weights": mode_weights, "rfsens": mode_rfsens,
     "qrvsmean": mode_qrvsmean}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())