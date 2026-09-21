"""
inspect_cql_scaler.py -- check whether the saved CQL (and BC, for
comparison) checkpoint carries its OWN internal observation scaler, and
if so, whether its parameters match the --obs-mean/--obs-std values
used everywhere in this diagnostics pipeline (from
policy_comparison.json). d3rlpy algorithms can be configured with an
observation_scaler that gets saved/restored inside the .d3 checkpoint
via load_learnable -- if that scaler's mean/std differ from what we've
been manually applying via standardize(), every predict_value() and
predict() call in this whole investigation has been feeding the critic
inputs it wasn't actually trained to see in that form, which would
explain confident-but-wrong extrapolation localized to specific input
regions (like the high crop_water_demand tail) without needing any
coverage or reward-function problem at all.

This script does NOT retrain or change anything -- it only reports.

USAGE
-----
python inspect_cql_scaler.py `
    --cql-dir models_epsilon/cql `
    --bc-dir models_epsilon/bc `
    --seeds 42 123 2024 `
    --obs-mean 15.075033 36.681023 0.391654 -0.000000000002838 0.000000000002838 0.423686 `
    --obs-std 7.286923 41.271557 0.253303 0.707107 0.707107 0.194976
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import d3rlpy
except ImportError:
    raise ImportError("d3rlpy is required: pip install d3rlpy")


def describe_scaler(policy, label: str):
    print(f"\n--- {label} ---")

    scaler = getattr(policy, "observation_scaler", None)
    if scaler is None:
        print("  observation_scaler: None (no internal scaler saved -- "
              "this policy expects RAW, unstandardized observations at "
              "predict()/predict_value() time, UNLESS it was trained on "
              "pre-standardized data with scaling done outside d3rlpy "
              "entirely, in which case manual standardization before "
              "calling predict() is correct and expected.)")
    else:
        print(f"  observation_scaler type: {type(scaler).__name__}")
        # Try the common d3rlpy attribute names across versions.
        for attr in ("mean", "std", "minimum", "maximum", "eps"):
            val = getattr(scaler, attr, None)
            if val is not None:
                val_arr = np.asarray(val).ravel()
                print(f"    {attr}: {val_arr}")

    action_scaler = getattr(policy, "action_scaler", None)
    print(f"  action_scaler: {type(action_scaler).__name__ if action_scaler is not None else None}")

    reward_scaler = getattr(policy, "reward_scaler", None)
    print(f"  reward_scaler: {type(reward_scaler).__name__ if reward_scaler is not None else None}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cql-dir", required=True)
    ap.add_argument("--bc-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--obs-mean", nargs=6, type=float, required=True)
    ap.add_argument("--obs-std", nargs=6, type=float, required=True)
    args = ap.parse_args()

    obs_mean = np.array(args.obs_mean, dtype=float)
    obs_std = np.array(args.obs_std, dtype=float)

    print("=" * 70)
    print("MANUALLY-SUPPLIED --obs-mean / --obs-std (from policy_comparison.json)")
    print("=" * 70)
    print(f"  obs_mean: {obs_mean}")
    print(f"  obs_std:  {obs_std}")

    for seed in args.seeds:
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        cql_ckpt = Path(args.cql_dir) / f"cql_seed{seed}.d3"
        bc_ckpt = Path(args.bc_dir) / f"bc_seed{seed}.d3"

        if cql_ckpt.exists():
            cql = d3rlpy.load_learnable(str(cql_ckpt))
            describe_scaler(cql, f"CQL seed {seed}")

            scaler = getattr(cql, "observation_scaler", None)
            if scaler is not None:
                ckpt_mean = getattr(scaler, "mean", None)
                ckpt_std = getattr(scaler, "std", None)
                if ckpt_mean is not None and ckpt_std is not None:
                    ckpt_mean = np.asarray(ckpt_mean).ravel()
                    ckpt_std = np.asarray(ckpt_std).ravel()
                    if ckpt_mean.shape == obs_mean.shape:
                        mean_diff = np.abs(ckpt_mean - obs_mean)
                        std_diff = np.abs(ckpt_std - obs_std)
                        print(f"\n  *** COMPARISON vs manually-supplied obs_mean/obs_std ***")
                        print(f"  max |mean difference| per dim: {mean_diff}")
                        print(f"  max |std difference| per dim:  {std_diff}")
                        if mean_diff.max() > 1e-3 or std_diff.max() > 1e-3:
                            print(
                                "  *** MISMATCH DETECTED: checkpoint's internal scaler "
                                "differs from the manually-applied obs_mean/obs_std used "
                                "throughout diagnostics_distribution_shift*.py. This means "
                                "every predict_value() call in this investigation has been "
                                "feeding CQL double-standardized or mis-standardized inputs. "
                                "This is a strong, directly actionable candidate root cause "
                                "for the observed critic miscalibration. ***"
                            )
                        else:
                            print("  Scalers match closely -- this is not the issue.")
        else:
            print(f"CQL checkpoint not found: {cql_ckpt}")

        if bc_ckpt.exists():
            bc = d3rlpy.load_learnable(str(bc_ckpt))
            describe_scaler(bc, f"BC seed {seed}")
        else:
            print(f"BC checkpoint not found: {bc_ckpt}")

    print("\n" + "=" * 70)
    print("READ THIS")
    print("=" * 70)
    print(
        "If observation_scaler is NOT None on the CQL checkpoint AND its "
        "mean/std differ from the manually-supplied --obs-mean/--obs-std, "
        "then every place in this pipeline that calls "
        "standardize(obs, obs_mean, obs_std) before predict()/predict_value() "
        "is EITHER (a) double-standardizing (scaling already-scaled data "
        "again, since d3rlpy would apply its own internal scaler on top), "
        "or (b) applying the wrong scale entirely if d3rlpy's scaler "
        "differs materially. Either way, the fix is: DO NOT manually "
        "standardize before calling predict()/predict_value() if the "
        "checkpoint has its own observation_scaler -- pass raw "
        "observations directly and let d3rlpy's internal scaler handle it. "
        "This would require updating make_model_action_fn (in "
        "rollout_eval.py) and every standardize(...) call before "
        "predict_value in the diagnostics scripts, THEN rerunning the "
        "full diagnostics suite from scratch -- no retraining needed, "
        "since the checkpoints themselves may already be fine."
    )


if __name__ == "__main__":
    main()