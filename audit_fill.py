#!/usr/bin/env python3
"""
audit_fill.py — fill the eight missing values for the RNG rerun and
robustness table.

Usage:
    python audit_fill.py check   --arrays results/per_seed.npz
    python audit_fill.py robust  --arrays results/per_seed.npz
    python audit_fill.py rerun   --arrays results/per_seed.npz \
                                 --rerun  results/per_seed_rerun.npz
    python audit_fill.py report  --arrays results/per_seed.npz \
                                 --rerun  results/per_seed_rerun.npz

The .npz must contain one of:
    D            shape (3, 10, 225)  paired difference, interval x seed x start
  or
    fqe, rollout shape (3, 10, 225)  marginals; D is computed as fqe - rollout.

Seed order must match the manuscript:
    42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999
Interval order must match the manuscript: 100, 200, 500.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
INTERVALS = [100, 200, 500]

# Published values from Tables VII and VIII, used as sanity checks.
PUBLISHED = {
    100: dict(fqe=+0.8984, contrast=+0.9879, ci=(-0.9955, +2.9713),
              seeds_pos=6, sd=2.773, median=+0.16),
    200: dict(fqe=+0.3478, contrast=+0.4373, ci=(+0.1131, +0.7615),
              seeds_pos=9, sd=0.453, median=+0.28),
    500: dict(fqe=+0.8226, contrast=+0.9121, ci=(-0.1006, +1.9248),
              seeds_pos=8, sd=1.416, median=+0.47),
}
PUBLISHED_LOO = {100: ("123", +0.14), 500: ("99999", +0.52)}


def load_D(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    if "D" in z:
        D = z["D"]
    elif "fqe" in z and "rollout" in z:
        D = z["fqe"] - z["rollout"]
    else:
        sys.exit(f"{path}: need key 'D' or pair ('fqe','rollout').")
    if D.shape != (3, 10, 225):
        sys.exit(f"{path}: D has shape {D.shape}, expected (3, 10, 225). "
                 f"Check axis order: interval x seed x start.")
    return D


def seed_means(D: np.ndarray) -> np.ndarray:
    """(3, 10) mean over starts."""
    return D.mean(axis=2)


def t_interval(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(1 - alpha / 2, df=n - 1) * se
    return m - h, m + h


def trimmed_mean(x: np.ndarray) -> float:
    """Drop one min and one max, average the rest."""
    return np.sort(x)[1:-1].mean()


def loo_means(x: np.ndarray) -> tuple[int, np.ndarray]:
    """Return (index of most influential seed, leave-one-out means)."""
    loo = np.array([np.delete(x, i).mean() for i in range(len(x))])
    full = x.mean()
    # Most influential = the one whose removal moves the mean most.
    influence = np.abs(loo - full)
    return int(np.argmax(influence)), loo


def check(D: np.ndarray) -> bool:
    sm = seed_means(D)
    ok = True
    print("Sanity check against Tables VII and VIII\n")
    header = f"{'int':>5} {'fqe_gap':>9} {'contrast':>9} {'ci_low':>9} {'ci_hi':>9} {'+':>3} {'sd':>7} {'med':>7}"
    print(header)
    print("-" * len(header))
    for k, iv in enumerate(INTERVALS):
        pub = PUBLISHED[iv]
        c = sm[k].mean()
        lo, hi = t_interval(sm[k])
        pos = int((sm[k] > 0).sum())
        sd = sm[k].std(ddof=1)
        med = float(np.median(sm[k]))
        print(f"{iv:>5} {pub['fqe']:>+9.4f} {c:>+9.4f} {lo:>+9.4f} "
              f"{hi:>+9.4f} {pos:>3} {sd:>7.3f} {med:>+7.2f}")
        # Tolerance is loose because the published values are rounded.
        if abs(c - pub["contrast"]) > 1e-3:
            print(f"  !! contrast mismatch at {iv}: "
                  f"{c:+.4f} vs published {pub['contrast']:+.4f}")
            ok = False
        if (pos, round(sd, 3), round(med, 2)) != (pub["seeds_pos"],
                                                  round(pub["sd"], 3),
                                                  round(pub["median"], 2)):
            print(f"  !! summary mismatch at {iv}: "
                  f"seeds+={pos} sd={sd:.3f} median={med:+.2f}")
            ok = False
    return ok


def robust(D: np.ndarray) -> None:
    sm = seed_means(D)
    print("\nRobustness statistics\n")
    header = (f"{'int':>5} {'mean':>9} {'trimmed':>9} "
              f"{'influential':>11} {'loo_mean':>9} {'loo_ci':>22}")
    print(header)
    print("-" * len(header))
    for k, iv in enumerate(INTERVALS):
        x = sm[k]
        tm = trimmed_mean(x)
        idx, loo = loo_means(x)
        lo, hi = t_interval(np.delete(x, idx))
        name = str(SEEDS[idx])
        print(f"{iv:>5} {x.mean():>+9.4f} {tm:>+9.4f} "
              f"{name:>11} {loo[idx]:>+9.4f} "
              f"  [{lo:+.4f}, {hi:+.4f}]")
        if iv in PUBLISHED_LOO:
            pub_seed, pub_val = PUBLISHED_LOO[iv]
            if name != pub_seed or abs(loo[idx] - pub_val) > 0.02:
                print(f"  !! LOO mismatch at {iv}: script says "
                      f"drop {name} -> {loo[idx]:+.4f}, Section VII says "
                      f"drop {pub_seed} -> {pub_val:+.4f}")
    # Diagnostic: does the interval-200 leave-one-out still exclude zero?
    x200 = sm[1]
    idx200, loo200 = loo_means(x200)
    lo, hi = t_interval(np.delete(x200, idx200))
    print("\nInterval-200 leave-one-out check (drives the caption sentence):")
    print(f"  drop seed {SEEDS[idx200]} -> mean {loo200[idx200]:+.4f}, "
          f"CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("  -> still excludes zero. Caption sentence stands.")
    else:
        print("  -> spans zero. Caption sentence must be rewritten; "
              "the interval-200 result is seed-driven.")


def compare_rerun(orig: np.ndarray, rerun: np.ndarray) -> None:
    so, sr = seed_means(orig), seed_means(rerun)
    print("\nRerun comparison (disjoint RNG streams)\n")
    header = (f"{'int':>5} {'orig':>9} {'orig_ci':>22} "
              f"{'rerun':>9} {'rerun_ci':>22} {'delta':>8}")
    print(header)
    print("-" * len(header))
    for k, iv in enumerate(INTERVALS):
        co = so[k].mean(); lo_o, hi_o = t_interval(so[k])
        cr = sr[k].mean(); lo_r, hi_r = t_interval(sr[k])
        print(f"{iv:>5} {co:>+9.4f}   [{lo_o:+.4f}, {hi_o:+.4f}] "
              f"{cr:>+9.4f}   [{lo_r:+.4f}, {hi_r:+.4f}] "
              f"{cr - co:>+8.4f}")
    # Branch selection.
    x = sr[1]
    lo, hi = t_interval(x)
    print("\nBranch for item 2 of the revision:")
    if lo > 0 or hi < 0:
        print("  Branch A — interval-200 rerun still excludes zero.")
        print("  Fill: '[+%.4f, +%.4f]' and 'exclusion of zero is unchanged'."
              % (lo, hi))
    else:
        print("  Branch B — interval-200 rerun spans zero.")
        print("  Fill: '[+%.4f, +%.4f]' and 'the original exclusion at "
              "interval 200 was partly an artifact of the seed collision'."
              % (lo, hi))
        print("  IMPORTANT: abstract, contribution (i), conclusion and "
              "cover note must all be rewritten.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "robust", "rerun", "report"])
    p.add_argument("--arrays", type=Path, required=True)
    p.add_argument("--rerun", type=Path)
    a = p.parse_args()

    D = load_D(a.arrays)

    if a.mode == "check":
        sys.exit(0 if check(D) else 1)

    if a.mode == "robust":
        if not check(D):
            print("\nRefusing to compute robustness stats: sanity check failed.")
            sys.exit(1)
        robust(D)
        return

    if a.mode in ("rerun", "report"):
        if a.rerun is None:
            sys.exit("--rerun is required for mode 'rerun' and 'report'.")
        if not check(D):
            print("\nRefusing to compare: sanity check failed.")
            sys.exit(1)
        Dr = load_D(a.rerun)
        compare_rerun(D, Dr)
        robust(D)
        return


if __name__ == "__main__":
    main()