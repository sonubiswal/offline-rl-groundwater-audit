"""Bound on the maximum feasible 20-step return in the simulator, then
compare against the FQE BCQ estimates."""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\trishna-opal\scripts")
sys.path.insert(0, r"C:\trishna-opal")
import adapters as A

starts = A._load_test_episode_starts(A.DATASETS["epsilon"], n=225)

# Strategy 1: analytic-ish upper bound.
# Reward = w1*rev + w3*sup - w2*pen, w = (1.0, 2.0, 0.5).
# If rev <= 1 and sup <= 1 and pen >= 0, per-step upper bound is 1.5.
# 20-step = 30. Use this as the analytic cap.
analytic_cap = 1.5 * 20
print(f"analytic upper bound (rev<=1, sup<=1, pen>=0): {analytic_cap:+.4f}")

# Strategy 2: empirical max over 10000 rollouts with a greedy max-action policy
def always_max(_obs):
    return np.array([4])

max_emp = -np.inf
for i in range(225):
    start_i = starts[i:i + 1]
    for seed in range(1):
        r = A.rollout_returns(always_max, 1, seed=i * 10 + seed,
                              start_set=start_i)
        max_emp = max(max_emp, float(r.max()))
print(f"empirical max (always action 4, 225 starts): {max_emp:+.4f}")

# Strategy 3: sample random policies
rng = np.random.default_rng(0)
max_rand = -np.inf
for trial in range(2000):
    def rand_pol(_obs, _rng=rng):
        return np.array([_rng.integers(0, 5)])
    i = int(rng.integers(0, 225))
    r = A.rollout_returns(rand_pol, 1, seed=trial * 7 + i,
                          start_set=starts[i:i + 1])
    max_rand = max(max_rand, float(r.max()))
print(f"empirical max over 2000 random rollouts: {max_rand:+.4f}")

overall_max = max(analytic_cap, max_emp, max_rand)
print(f"\noverall feasibility ceiling for a 20-step return: {overall_max:+.4f}")

# Compare to BCQ FQE values
d = json.loads(Path(r"C:\trishna-opal\revision_experiments\E4\e4_results.json").read_text())
bcq_fqe = [r["bcq_q_mean"] for r in d["per_seed"]]
print(f"\nFQE BCQ values (ten seeds): " + ", ".join(f"{v:+.2f}" for v in bcq_fqe))
print(f"max {max(bcq_fqe):+.2f}  min {min(bcq_fqe):+.2f}")

n_over = sum(1 for v in bcq_fqe if v > overall_max)
if n_over == len(bcq_fqe):
    print("=> ALL ten BCQ FQE values exceed the ceiling. FQE has diverged.")
elif n_over > 0:
    print(f"=> {n_over} of 10 BCQ FQE values exceed the ceiling.")
else:
    print("=> All BCQ FQE values stay under the ceiling.")