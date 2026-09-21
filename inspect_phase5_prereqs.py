"""
inspect_phase5_prereqs.py

Phase 5 prerequisite check, per this project's established discipline
(the same instinct that caught the FEATURE_COLUMNS/kriged_dir/max_depth
drift before Phase 4): inspect what's actually installed and actually in
the dataset before writing BC/CQL/FQE code against assumed APIs/shapes.

Checks:
  1. d3rlpy version and which BC/DiscreteCQL/FQE classes/signatures it
     actually exposes (these differ meaningfully across d3rlpy versions
     -- v1 and v2 have different constructor patterns).
  2. The actual saved MDPDataset: episode count, transition count,
     observation/action shapes, terminal flag handling, reward range.
  3. rl_config.yaml's actual contents (in case it's drifted from what
     generate_offline_dataset.py was run with).

Run this BEFORE writing bc_baseline.py / train_cql.py / fqe_eval.py.

Usage:
    python inspect_phase5_prereqs.py
"""

import sys

import numpy as np
import yaml


def check_d3rlpy():
    print("=" * 70)
    print("1. d3rlpy VERSION AND API SHAPE")
    print("=" * 70)
    try:
        import d3rlpy
        print(f"d3rlpy version: {d3rlpy.__version__}")
    except ImportError:
        print("d3rlpy NOT INSTALLED. Install with: pip install d3rlpy --break-system-packages")
        return False

    # Check which algorithm classes/configs actually exist -- d3rlpy v2's
    # API (Config + create() pattern) differs substantially from v1's
    # (direct constructor with hyperparameters as kwargs).
    print("\nChecking algorithm class availability and construction pattern...")
    try:
        from d3rlpy.algos import DiscreteCQLConfig
        print("  Found DiscreteCQLConfig -- this is the v2-style Config/create() API.")
        import inspect
        sig = inspect.signature(DiscreteCQLConfig.__init__)
        print(f"  DiscreteCQLConfig params: {list(sig.parameters.keys())}")
        v2_api = True
    except ImportError:
        try:
            from d3rlpy.algos import DiscreteCQL
            print("  Found DiscreteCQL directly -- this is the v1-style direct-constructor API.")
            import inspect
            sig = inspect.signature(DiscreteCQL.__init__)
            print(f"  DiscreteCQL params: {list(sig.parameters.keys())}")
            v2_api = False
        except ImportError:
            print("  ERROR: could not find DiscreteCQL or DiscreteCQLConfig. "
                  "Check d3rlpy installation.")
            return False

    try:
        if v2_api:
            from d3rlpy.algos import DiscreteBCConfig
            print("  Found DiscreteBCConfig (v2-style).")
        else:
            from d3rlpy.algos import DiscreteBC
            print("  Found DiscreteBC (v1-style).")
    except ImportError:
        print("  WARNING: could not find a discrete BC class under the expected name -- "
              "check d3rlpy.algos namespace manually (`dir(d3rlpy.algos)`).")

    try:
        from d3rlpy.ope import FQE, FQEConfig
        print("  Found FQE and FQEConfig (v2-style OPE module).")
    except ImportError:
        try:
            from d3rlpy.ope import FQE
            print("  Found FQE directly (v1-style).")
        except ImportError:
            print("  WARNING: could not find FQE under d3rlpy.ope -- check installed version's "
                  "off-policy evaluation module location.")

    print(f"\n  --> API STYLE DETECTED: {'v2 (Config + .create())' if v2_api else 'v1 (direct constructor)'}")
    print("  All Phase 5 scripts must be written against this actual API, not assumed.")
    return True


def check_dataset(path: str = "data/processed/offline_rl_dataset.h5"):
    print()
    print("=" * 70)
    print("2. OFFLINE RL DATASET")
    print("=" * 70)
    try:
        from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
    except ImportError:
        print("Cannot import ReplayBuffer/InfiniteBuffer -- d3rlpy not installed, skipping.")
        return

    try:
        with open(path, "rb") as f:
            dataset = ReplayBuffer.load(f, InfiniteBuffer())
        print(f"Loaded via d3rlpy 2.x API: ReplayBuffer.load(file_handle, InfiniteBuffer())")
    except Exception as e:
        print(f"Could not load {path} even with the corrected v2 API: {e}")
        print("This needs manual investigation -- check d3rlpy 2.8.1's actual "
              "ReplayBuffer.load/dump docs, or print dir(ReplayBuffer) to see available methods.")
        return

    episodes = dataset.episodes
    n_episodes = len(episodes)
    n_transitions = sum(len(ep) for ep in episodes)
    ep_lengths = [len(ep) for ep in episodes]

    print(f"Episodes: {n_episodes}")
    print(f"Total transitions: {n_transitions}")
    print(f"Episode length: min={min(ep_lengths)}, max={max(ep_lengths)}, "
          f"mean={np.mean(ep_lengths):.1f}")

    first_ep = episodes[0]
    print(f"\nFirst episode inspection:")
    print(f"  observation shape: {np.array(first_ep.observations).shape}")
    print(f"  action shape: {np.array(first_ep.actions).shape}, "
          f"dtype: {np.array(first_ep.actions).dtype}")
    print(f"  reward shape: {np.array(first_ep.rewards).shape}")
    print(f"  terminated: {getattr(first_ep, 'terminated', 'ATTRIBUTE NOT FOUND')}")

    all_rewards = np.concatenate([np.array(ep.rewards).flatten() for ep in episodes])
    print(f"\nReward stats across all episodes: "
          f"mean={all_rewards.mean():.3f}, std={all_rewards.std():.3f}, "
          f"min={all_rewards.min():.3f}, max={all_rewards.max():.3f}")

    all_actions = np.concatenate([np.array(ep.actions).flatten() for ep in episodes])
    print(f"Action distribution: {np.bincount(all_actions.astype(int))}")

    n_terminated = sum(1 for ep in episodes if getattr(ep, "terminated", False))
    print(f"\nEpisodes with terminated=True: {n_terminated}/{n_episodes}")
    print("IMPORTANT: check above whether 'terminated' is ever True. Phase 4's "
          "generate_offline_dataset.py always sets done=False from synthetic_reward.step() "
          "-- if terminated is False for every episode, every trajectory is being treated "
          "as a FIXED-LENGTH, NON-TERMINATING sequence artificially cut at 20 steps, not a "
          "true episodic MDP. This affects how CQL/FQE should discount the final step and "
          "should be stated explicitly in the methodology, not silently assumed either way.")


def check_config(path: str = "config/rl_config.yaml"):
    print()
    print("=" * 70)
    print("3. rl_config.yaml CONTENTS")
    print("=" * 70)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    print(yaml.dump(cfg, default_flow_style=False))


if __name__ == "__main__":
    ok = check_d3rlpy()
    if ok:
        check_dataset()
    check_config()