"""
check_d3rlpy_terminal_handling.py

Follow-up to inspect_phase5_prereqs.py's finding: Phase 4's dataset marks
EVERY trajectory's 20th step as terminated=True, even though
synthetic_reward.step() never actually terminates -- this is a time-limit
cutoff being mismarked as a true terminal state (the well-known "time
limits in RL" issue, Pardo et al. 2018).

This script checks whether d3rlpy 2.8.1 exposes a way to build a dataset
that distinguishes "true terminal" from "timeout/episode-boundary-only"
(mirroring Gymnasium's terminated/truncated split), before deciding how
to fix generate_offline_dataset.py.

Usage:
    python check_d3rlpy_terminal_handling.py
"""

import inspect

import d3rlpy
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer, Episode

print(f"d3rlpy version: {d3rlpy.__version__}\n")

print("=" * 70)
print("Episode dataclass fields (does it separate terminated vs truncated?)")
print("=" * 70)
try:
    print(f"Episode fields: {list(inspect.signature(Episode.__init__).parameters.keys())}")
except Exception as e:
    print(f"Could not inspect Episode: {e}")

print()
print("=" * 70)
print("Checking for a 'timeout' or 'truncated' concept anywhere in d3rlpy.dataset")
print("=" * 70)
from d3rlpy import dataset as d3_dataset_module
names = [n for n in dir(d3_dataset_module) if not n.startswith("_")]
timeout_related = [n for n in names if "trunc" in n.lower() or "timeout" in n.lower()]
print(f"All public names in d3rlpy.dataset: {names}")
print(f"\nNames matching 'trunc'/'timeout': {timeout_related if timeout_related else 'NONE FOUND'}")

print()
print("=" * 70)
print("Checking ReplayBuffer.append_episode / add methods for a timeout param")
print("=" * 70)
for method_name in ["append_episode", "add_episode", "append", "clip_episode"]:
    method = getattr(ReplayBuffer, method_name, None)
    if method:
        try:
            print(f"{method_name}{inspect.signature(method)}")
        except Exception:
            print(f"{method_name}: found but signature not inspectable")
    else:
        print(f"{method_name}: not found on ReplayBuffer")

print()
print("=" * 70)
print("Checking Episode's own fields for a distinct truncation signal")
print("=" * 70)
import numpy as np
try:
    ep = Episode(
        observations=np.zeros((2, 6), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.zeros((2, 1), dtype=np.float32),
        terminated=False,
    )
    print(f"Episode attributes: {[a for a in dir(ep) if not a.startswith('_')]}")
except Exception as e:
    print(f"Could not construct a test Episode: {e}")