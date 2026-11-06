"""
run_ood_w3_zero.py

Falsification test for the paper's candidate mechanism: does the monotone
widening of the banded OOD diagnostic survive removal of the domestic
supply term (w3 = 0)?

Variants:
    eval_only: use existing canonical checkpoints, evaluate under w3 = 0
    retrain  : retrain BC and CQL on a w3 = 0 reward-conditioned dataset

Adds a paired endpoint CI: band4 - band1 gets a proper seed-level
t-interval in addition to the pre-registered point-estimate rule.

Usage:
    python revision_experiments\run_ood_w3_zero.py --variant eval_only --w3 0.0 --tag w3_zero_eval_only
    python revision_experiments\run_ood_w3_zero.py --variant eval_only --w3 0.5 --tag w3_full_eval_only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A
from stats_utils import hierarchical_bootstrap, t_interval


BANDS = ["in_dist", "mild", "moderate", "severe"]
N_PER_BAND = 200
SEEDS = A.SEEDS   # all 10 seeds


def sample_band_states(band_index, n, seed):
    """Reuse the Mahalanobis band sampler from ood_return_test.py."""
    sys.path.insert(0, str(REPO))
    from src.rl.policy_io import compute_scaler_from_dataset
    from ood_return_test import (sample_bands, training_support,
                                 band_thresholds, load_dataset,
                                 split_episodes, PHYSICAL_BOUNDS)

    episodes = load_dataset()
    train_idx, _, _ = split_episodes(episodes)
    train_obs = np.concatenate([episodes[i] for i in train_idx], axis=0)
    mu, cov, inv_cov = training_support(train_obs)
    th = band_thresholds(df=train_obs.shape[1])
    rng = np.random.RandomState(2026 + seed)
    bands, _ = sample_bands(mu, cov, inv_cov, th, rng,
                            n_per_band=N_PER_BAND)
    canonical = BANDS[band_index]
    for key in (canonical, "id" if canonical == "in_dist" else canonical):
        if key in bands:
            return bands[key]
    raise KeyError(
        f"band {canonical!r} not found; sampler returned keys {list(bands)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["eval_only", "retrain"],
                    default="eval_only")
    ap.add_argument("--w3", type=float, default=0.0,
                    help="Domestic-supply reward weight used at evaluation.")
    ap.add_argument("--tag", default=None,
                    help="Output filename tag; default is w3_zero_<variant>.")
    args = ap.parse_args()

    weights = (1.0, 2.0, args.w3)
    out = {"variant": args.variant, "weights": list(weights), "bands": {}}

    diffs_by_band = {}

    for idx, band_name in enumerate(BANDS):
        diffs_by_seed = []
        for s in SEEDS:
            starts = sample_band_states(idx, N_PER_BAND, s)

            if args.variant == "retrain":
                ds = REPO / "reports" / "paper_exp" / \
                    f"dataset_w{weights[0]}_{weights[1]}_{weights[2]}.h5"
                if not ds.exists():
                    raise FileNotFoundError(
                        f"w3={args.w3} dataset not found: {ds}. "
                        "Generate via run_paper_experiments.py with matching weights.")
                bc_ck = REPO / "reports" / "w3_zero" / "bc" / f"bc_seed{s}.d3"
                cq_ck = REPO / "reports" / "w3_zero" / "cql" / f"cql_seed{s}.d3"
                bc_ck.parent.mkdir(parents=True, exist_ok=True)
                cq_ck.parent.mkdir(parents=True, exist_ok=True)
                if not bc_ck.exists():
                    A.train_bc(ds, bc_ck, s)
                if not cq_ck.exists():
                    A.train_cql(ds, cq_ck, s, reward_weights=weights)
            else:
                ds = A.DATASETS["epsilon"]
                bc_ck = A.checkpoint("epsilon", "bc", s)
                cq_ck = A.checkpoint("epsilon", "cql", s)

            bc = A.load_policy(bc_ck, ds)
            cq = A.load_policy(cq_ck, ds)
            bc_r = A.rollout_returns(bc, N_PER_BAND, s,
                                     start_set=starts, reward_weights=weights)
            cq_r = A.rollout_returns(cq, N_PER_BAND, s,
                                     start_set=starts, reward_weights=weights)
            diffs_by_seed.append(cq_r - bc_r)

        m, lo, hi = hierarchical_bootstrap(diffs_by_seed)
        out["bands"][band_name] = {"obs": m, "lo": lo, "hi": hi}
        diffs_by_band[band_name] = diffs_by_seed
        print(f"[w3={args.w3}] band {band_name:9s} CQL-BC {m:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    # ---- Per-seed mean per band (for downstream band-contrast analysis) ----
    out["band_per_seed"] = {
        b: [float(x.mean()) for x in diffs_by_band[b]] for b in BANDS
    }

    # ---- Paired endpoint contrast: band4 - band1 -------------------------
    # Per seed: mean(band4 diffs) - mean(band1 diffs). Then seed-level
    # t-interval on the 10 deltas.
    b1 = diffs_by_band["in_dist"]
    b4 = diffs_by_band["severe"]
    seed_deltas = [float(b4[i].mean() - b1[i].mean()) for i in range(len(b1))]
    delta_m, delta_lo, delta_hi = t_interval(seed_deltas)

    print(f"\nEndpoint contrast (severe - in_dist), n={len(seed_deltas)} seeds:")
    print(f"  point estimate: {delta_m:+.4f}")
    print(f"  95% paired CI:  [{delta_lo:+.4f}, {delta_hi:+.4f}]")
    excludes = delta_lo > 0 or delta_hi < 0
    print(f"  excludes 0:     {'YES' if excludes else 'NO'}")

    out["endpoint_contrast"] = {
        "obs": delta_m, "lo": delta_lo, "hi": delta_hi,
        "n_seeds": len(seed_deltas),
        "per_seed_delta": seed_deltas,
        "construction": "seed t-interval on per-seed band4 - band1",
    }

    # Keep the pre-registered point-estimate rule for continuity
    band_1 = out["bands"]["in_dist"]["obs"]
    band_4 = out["bands"]["severe"]["obs"]
    survives_pt = band_4 > band_1
    print(f"\nPre-registered point-estimate rule: band4 - band1 > 0")
    print(f"  band1 {band_1:+.4f}  band4 {band_4:+.4f}  "
          f"delta {band_4 - band_1:+.4f}  ->  "
          f"{'SURVIVES' if survives_pt else 'DISAPPEARS'}")
    out["survives"] = bool(survives_pt)
    out["survives_paired_ci"] = bool(excludes)

    # ---- Tag-aware output filename ---------------------------------------
    tag = args.tag if args.tag else f"w3_zero_{args.variant}"
    out_path = REPO / "revision_experiments" / f"ood_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[w3] wrote {out_path.name}")


if __name__ == "__main__":
    main()