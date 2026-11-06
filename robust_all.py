#!/usr/bin/env python3
"""robust_all.py — all Table VII cells from the manuscript's actual sources.

Rollout source : revision_experiments/e3c_test225_constructions.json
FQE source     : revision_experiments/E2/seeds/tui{100,200,500}/seed_*.npz
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(r"C:\trishna-opal")
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
INTERVALS = [100, 200, 500]

PUBLISHED = {
    100: dict(contrast=+0.9879, sd=2.773, median=+0.16),
    200: dict(contrast=+0.4373, sd=0.453, median=+0.28),
    500: dict(contrast=+0.9121, sd=1.416, median=+0.47),
}

# --- sources ----------------------------------------------------------------
e3c = json.loads((REPO / "revision_experiments"
                  / "e3c_test225_constructions.json").read_text())
roll = {int(r["seed"]): float(r["mean_gap"]) for r in e3c["per_seed"]}

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
hdr = (f"{'interval':>8} {'contrast':>9} {'sd':>7} {'median':>8} "
       f"{'seeds+':>6} {'trimmed':>8} {'drop':>6} {'loo_mean':>9} {'loo_ci':>22}")
print(hdr); print("-" * len(hdr))

for tui in INTERVALS:
    fqe = fqe_gaps(tui)
    d = np.array([fqe[s] - roll[s] for s in SEEDS])
    m = d.mean()
    sd = d.std(ddof=1)
    med = float(np.median(d))
    pos = int((d > 0).sum())
    tm = trimmed_mean(d)
    idx, lm, (lo, hi) = loo(d)
    name = str(SEEDS[idx])
    print(f"{tui:>8} {m:>+9.4f} {sd:>7.3f} {med:>+8.4f} {pos:>4}/10 "
          f"{tm:>+8.4f} {name:>6} {lm:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]")
    pub = PUBLISHED[tui]
    assert abs(m - pub["contrast"]) < 5e-4, \
        f"contrast mismatch at {tui}: {m:+.4f} vs {pub['contrast']:+.4f}"
    assert pos == (9 if tui == 200 else (6 if tui == 100 else 8)), \
        f"seeds+ mismatch at {tui}: {pos}"

print("\nAll published contrast and seeds+ values reproduced.")
print("\nReference (Table VII):")
print(f"{100:>8} {'+0.9879':>9} {'2.773':>7} {'+0.16':>8}")
print(f"{200:>8} {'+0.4373':>9} {'0.453':>7} {'+0.28':>8}")
print(f"{500:>8} {'+0.9121':>9} {'1.416':>7} {'+0.47':>8}")