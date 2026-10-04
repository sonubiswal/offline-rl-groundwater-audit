"""FQE gap seed-level CIs at intervals 100, 200, 500, for direct
comparison with the paired contrasts in Table VII."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(r"C:\trishna-opal")
REV = REPO / "revision_experiments"
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]


def t_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h)


# Load per-seed FQE gaps at each interval
fqe = {}
fqe[200] = np.array(json.loads(
    (REV / "E2" / "e2_fqe.json").read_text()
)["primary_episode_starts"]["per_seed_diffs"])

for tui in (100, 500):
    out = []
    for s in SEEDS:
        z = np.load(REV / "E2" / "seeds" / f"tui{tui}" / f"seed_{s}.npz")
        out.append(float((z["cql_q_starts"] - z["bc_q_starts"]).mean()))
    fqe[tui] = np.array(out)

# Also load rollout gap for the paired contrast
roll = np.array([
    r["mean_gap"]
    for r in json.loads(
        (REV / "e3c_test225_constructions_rerun.json").read_text()
    )["per_seed"]
])

print("FQE gap seed-level 95% CIs (ten seeds)")
print(f"{'tui':>6} {'mean':>9} {'lo':>9} {'hi':>9} {'excl 0':>7}")
print("-" * 45)
for tui in (100, 200, 500):
    m, lo, hi = t_ci(fqe[tui])
    ex = "yes" if (lo > 0 or hi < 0) else "no"
    print(f"{tui:>6} {m:>+9.4f} {lo:>+9.4f} {hi:>+9.4f} {ex:>7}")

print()
print("Rollout gap seed-level 95% CI (same ten seeds)")
m, lo, hi = t_ci(roll)
ex = "yes" if (lo > 0 or hi < 0) else "no"
print(f"{'roll':>6} {m:>+9.4f} {lo:>+9.4f} {hi:>+9.4f} {ex:>7}")

print()
print("Paired contrast (FQE gap - rollout gap) per interval")
print(f"{'tui':>6} {'mean':>9} {'lo':>9} {'hi':>9} {'excl 0':>7}")
print("-" * 45)
for tui in (100, 200, 500):
    d = fqe[tui] - roll
    m, lo, hi = t_ci(d)
    ex = "yes" if (lo > 0 or hi < 0) else "no"
    print(f"{tui:>6} {m:>+9.4f} {lo:>+9.4f} {hi:>+9.4f} {ex:>7}")