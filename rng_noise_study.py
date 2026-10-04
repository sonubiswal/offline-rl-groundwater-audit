"""RNG noise study for the rollout-collision claim.

The paper asserts that the collision materially changed the rollout gap
(from -0.0895 to +0.0141). This test asks: does -0.0895 fall inside the
Monte Carlo noise distribution of the repaired scheme itself?

For each of N_OFFSETS independent rollout seed offsets (all using the
disjoint-stream scheme with different bases), compute the ten-seed mean
rollout gap. Report the distribution. Then place the colliding value and
the reported value on it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\trishna-opal\scripts")
sys.path.insert(0, r"C:\trishna-opal")
import adapters as A

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
N_OFFSETS = 12          # use 12 to keep runtime under ~30 min
OFFSET_STEP = 500_000   # large separation between bases
BASE = 200_000


def rollout_with_base(policy, starts, seed, base):
    """Same as adapters.rollout_returns but with configurable base seed."""
    cfg = A.load_reward_config(str(A.CFG_PATH))
    cfg["w1_crop_revenue"] = 1.0
    cfg["w2_depletion_penalty"] = 2.0
    cfg["w3_domestic_supply"] = 0.5
    out = np.zeros(len(starts))
    for i, raw in enumerate(starts):
        s = A.SimState(*[float(x) for x in raw])
        rng_l = np.random.RandomState(base + 1000 * seed + i)
        total = 0.0
        for _ in range(20):
            vec = np.array([s.gw_level, s.recent_rainfall, s.crop_water_demand,
                            s.month_sin, s.month_cos, s.extraction_rate],
                           dtype=np.float32)
            a = int(policy(vec)[0])
            s, r, done = A.step(s, a, config=cfg, n_actions=5, rng=rng_l)
            total += float(r)
            if done:
                break
        out[i] = total
    return out


def gap_for_base(starts, base, load_cache):
    gaps = []
    ds = A.DATASETS["epsilon"]
    for s in SEEDS:
        bc = load_cache[f"bc_{s}"]
        cq = load_cache[f"cql_{s}"]
        bc_r = rollout_with_base(bc, starts, s, base)
        cq_r = rollout_with_base(cq, starts, s, base)
        gaps.append(float((cq_r - bc_r).mean()))
    return float(np.mean(gaps))


def main():
    print("loading 225 epsilon starts ...")
    starts = A._load_test_episode_starts(A.DATASETS["epsilon"], n=225)

    print("preloading 20 checkpoints (10 BC + 10 CQL) ...")
    load_cache = {}
    ds = A.DATASETS["epsilon"]
    for s in SEEDS:
        load_cache[f"bc_{s}"] = A.load_policy(
            A.checkpoint("epsilon", "bc", s), ds)
        load_cache[f"cql_{s}"] = A.load_policy(
            A.checkpoint("epsilon", "cql", s), ds)
    print("checkpoints loaded.")

    print(f"\nrunning {N_OFFSETS} noise replicates ...")
    t0 = time.time()
    gaps = []
    for k in range(N_OFFSETS):
        base = BASE + k * OFFSET_STEP
        g = gap_for_base(starts, base, load_cache)
        gaps.append(g)
        print(f"  offset {k+1:>2}/{N_OFFSETS}  base={base:>10}  "
              f"rollout gap = {g:+.4f}  ({time.time()-t0:.1f}s elapsed)")

    gaps = np.array(gaps)
    print("\n" + "=" * 60)
    print("Noise distribution of the rollout gap under the repaired scheme")
    print(f"  mean  {gaps.mean():+.4f}")
    print(f"  sd    {gaps.std(ddof=1):.4f}")
    print(f"  min   {gaps.min():+.4f}")
    print(f"  max   {gaps.max():+.4f}")
    print()
    print("Reference values:")
    print(f"  colliding-scheme rollout gap:  -0.0895")
    print(f"  reported repaired rollout gap: +0.0141")

    inside = gaps.min() <= -0.0895 <= gaps.max()
    print()
    if inside:
        print("=> -0.0895 lies INSIDE the noise range.")
        print("   The collision effect is within Monte Carlo noise.")
    else:
        z = (-0.0895 - gaps.mean()) / gaps.std(ddof=1)
        print(f"=> -0.0895 lies OUTSIDE the noise range (z = {z:+.2f}).")
        print("   The collision effect exceeds Monte Carlo noise.")

    # Save
    out = Path(r"C:\trishna-opal\revision_experiments\rng_noise_study.json")
    import json
    out.write_text(json.dumps({
        "n_offsets": N_OFFSETS,
        "base": BASE,
        "offset_step": OFFSET_STEP,
        "gaps": gaps.tolist(),
        "mean": float(gaps.mean()),
        "sd": float(gaps.std(ddof=1)),
        "min": float(gaps.min()),
        "max": float(gaps.max()),
        "colliding_value": -0.0895,
        "reported_value": 0.0141,
        "colliding_inside_range": bool(inside),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()