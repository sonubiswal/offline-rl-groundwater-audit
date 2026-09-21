import json
from pathlib import Path
import numpy as np

CONFIGS = [
    "reports/epsilon",
    "reports/no_oracle",
    "reports/no_oracle_alpha1",
    "reports/no_oracle_alpha10",
    "reports",
]
SEEDS = (42, 123, 2024)


def paired_bootstrap(a, b, n_bootstrap=5000, seed=2026, confidence=0.95):
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_bootstrap, len(diff)))
    boots = diff[idx].mean(axis=1)
    alpha = 1.0 - confidence
    return (float(diff.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


print(f"{'config':28s} {'observed':>11s} {'CI lo':>11s} {'CI hi':>11s}  match")
print("-" * 76)
for cfg in CONFIGS:
    d = Path(cfg)
    arr = d / "returns"
    jp = d / "policy_comparison.json"
    if not arr.exists() or not jp.exists():
        print(f"{cfg:28s} (no returns/ — not re-run yet)")
        continue
    bc  = np.vstack([np.load(arr / f"returns_bc_seed{s}.npy")  for s in SEEDS])
    cql = np.vstack([np.load(arr / f"returns_cql_seed{s}.npy") for s in SEEDS])
    obs, lo, hi = paired_bootstrap(cql.mean(axis=0), bc.mean(axis=0))
    s = json.load(open(jp))["cql_minus_bc_paired_bootstrap"]
    ok = (abs(obs - s["observed_difference"]) < 1e-4
          and abs(lo - s["ci_lower"]) < 1e-4
          and abs(hi - s["ci_upper"]) < 1e-4)
    print(f"{cfg:28s} {obs:+11.4f} {lo:+11.4f} {hi:+11.4f}  {'OK' if ok else 'DIFF'}")
    if not ok:
        print(f"  stored:                    {s['observed_difference']:+11.4f} "
              f"{s['ci_lower']:+11.4f} {s['ci_upper']:+11.4f}")