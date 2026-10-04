"""Non-parametric tests on the interval-200 paired contrast."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

REV = Path(r"C:\trishna-opal\revision_experiments")

# Rebuild the ten paired contrasts at interval 200
fqe200 = np.array(json.loads(
    (REV / "E2" / "e2_fqe.json").read_text()
)["primary_episode_starts"]["per_seed_diffs"])

roll = np.array([
    r["mean_gap"]
    for r in json.loads(
        (REV / "e3c_test225_constructions_rerun.json").read_text()
    )["per_seed"]
])

contrast = fqe200 - roll
print("per-seed contrasts at interval 200:")
print("  " + "  ".join(f"{c:+.4f}" for c in contrast))
print(f"  mean = {contrast.mean():+.4f}")

# Wilcoxon signed-rank, two-sided
w, p = stats.wilcoxon(contrast, alternative="two-sided")
print(f"\nWilcoxon signed-rank: W = {w:.1f}, p = {p:.4f}")

# Sign-flip permutation, 10000 reps
rng = np.random.default_rng(2026)
obs = contrast.mean()
flips = rng.choice([-1.0, 1.0], size=(10000, len(contrast)))
null = (contrast * flips).mean(axis=1)
p_flip = float((np.abs(null) >= np.abs(obs)).mean())
print(f"Sign-flip permutation (10000 reps): p = {p_flip:.4f}")

# BCa interval via scipy bootstrap
res = stats.bootstrap(
    (contrast,), np.mean, n_resamples=10000,
    method="BCa", random_state=2026, confidence_level=0.95,
)
lo, hi = res.confidence_interval
print(f"\nBCa 95% interval: [{lo:+.4f}, {hi:+.4f}]")
print(f"  excludes zero: {'yes' if (lo > 0 or hi < 0) else 'no'}")