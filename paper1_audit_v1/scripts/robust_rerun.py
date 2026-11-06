#!/usr/bin/env python3
"""robust_rerun.py — Table VII after the collision-free rerun.

FQE source     : revision_experiments/E2/seeds/tui{100,200,500}/seed_*.npz
Rollout source : revision_experiments/e3c_test225_constructions_rerun.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(r"C:\trishna-opal")
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
INTERVALS = [100, 200, 500]

# Colliding-scheme references from the manuscript, for comparison.
COLLIDING = {
    100: dict(contrast=+0.9879, ci=(-0.9955, +2.9713), sd=2.773, pos=6),
    200: dict(contrast=+0.4373, ci=(+0.1131, +0.7615), sd=0.453, pos=9),
    500: dict(contrast=+0.9121, ci=(-0.1006, +1.9248), sd=1.416, pos=8),
}

# --- sources ----------------------------------------------------------------
rerun = json.loads((REPO / "revision_experiments"
                    / "e3c_test225_constructions_rerun.json").read_text())
roll = {int(r["seed"]): float(r["mean_gap"]) for r in rerun["per_seed"]}

def fqe_gaps(tui: int) -> dict[int, float]:
    out = {}
    for s in SEEDS:
        p = (REPO / "revision_experiments" / "E2" / "seeds"
             / f"tui{tui}" / f"seed_{s}.npz")
        z = np.load(p)
        out[s] = float((z["cql_q_starts"] - z["bc_q_starts"]).mean())
    return out

# --- statistics -------------------------------------------------------------
def t_interval(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, df=n - 1) * se
    return float(x.mean() - h), float(x.mean() + h)

def trimmed_mean(x: np.ndarray) -> float:
    return float(np.sort(x)[1:-1].mean())

def loo(x: np.ndarray) -> tuple[int, float, tuple[float, float]]:
    full = x.mean()
    means = np.array([np.delete(x, i).mean() for i in range(len(x))])
    idx = int(np.argmax(np.abs(means - full)))
    lo, hi = t_interval(np.delete(x, idx))
    return idx, float(means[idx]), (lo, hi)

# --- report -----------------------------------------------------------------
print("After collision-free rerun\n")
hdr = (f"{'interval':>8} {'contrast':>9} {'sd':>7} {'median':>8} "
       f"{'seeds+':>6} {'trimmed':>8} {'drop':>6} {'loo_mean':>9} {'loo_ci':>22}")
print(hdr); print("-" * len(hdr))

rerun_rows = {}
for tui in INTERVALS:
    fqe = fqe_gaps(tui)
    d = np.array([fqe[s] - roll[s] for s in SEEDS])
    m = d.mean(); sd = d.std(ddof=1); med = float(np.median(d))
    pos = int((d > 0).sum()); tm = trimmed_mean(d)
    idx, lm, (lo, hi) = loo(d)
    rerun_rows[tui] = dict(contrast=float(m), sd=float(sd), median=med,
                           pos=pos, trimmed=float(tm),
                           drop=int(SEEDS[idx]), loo_mean=float(lm),
                           loo_ci=(lo, hi))
    print(f"{tui:>8} {m:>+9.4f} {sd:>7.3f} {med:>+8.4f} {pos:>4}/10 "
          f"{tm:>+8.4f} {str(SEEDS[idx]):>6} {lm:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]")

print("\nBefore (colliding scheme, from manuscript)\n")
print(hdr); print("-" * len(hdr))
for tui in INTERVALS:
    c = COLLIDING[tui]
    print(f"{tui:>8} {c['contrast']:>+9.4f} {c['sd']:>7.3f} "
          f"{'—':>8} {c['pos']:>4}/10 {'—':>8} {'—':>6} {'—':>9} "
          f"   [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")

print("\nDelta (rerun − colliding)\n")
for tui in INTERVALS:
    r = rerun_rows[tui]; c = COLLIDING[tui]
    print(f"  interval {tui:>3}: contrast {r['contrast'] - c['contrast']:+.4f}  "
          f"sd {r['sd'] - c['sd']:+.3f}  seeds+ {r['pos'] - c['pos']:+d}")

print("\nJSON for the manuscript build:")
print(json.dumps(rerun_rows, indent=2))