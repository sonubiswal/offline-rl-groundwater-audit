"""Per-pairing protocol-shift statistic: (common-epsilon gap) - (RF-grid gap)
with a seed-level 95% CI. Continuous replacement for the '9 of 9 flip' count."""
import numpy as np
from scipy import stats

z = np.load(r"C:\trishna-opal\revision_experiments\family_start_rerun.npz")
FAMS = ["base", "epsilon", "no_oracle"]
N_SEEDS = 10

def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, N_SEEDS - 1) * x.std(ddof=1) / np.sqrt(N_SEEDS)
    return float(m), float(m - h), float(m + h)

header = f"{'pairing':>22} {'shift':>9} {'95% CI':>24} {'excl 0':>7}"
print(header)
print("-" * len(header))
for bc in FAMS:
    for cql in FAMS:
        rf = z[f"{bc}__{cql}__rf_grid"].mean(axis=1)
        ce = z[f"{bc}__{cql}__common_eps"].mean(axis=1)
        shift = ce - rf
        m, lo, hi = ci(shift)
        ex = "yes" if (lo > 0 or hi < 0) else "no"
        name = f"{bc}/{cql}"
        print(f"{name:>22} {m:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]   {ex:>4}")