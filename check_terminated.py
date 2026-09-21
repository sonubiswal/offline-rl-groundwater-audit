"""Quick diagnostic: inspect what d3rlpy's Episode objects actually
carry for the 'terminated' field once loaded from the .h5 dataset,
and whether episodes are flagged terminal or timeout. Run this
directly against your dataset -- no training involved, should take
a few seconds."""
from pathlib import Path
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer

dataset_path = Path("data/processed/offline_rl_dataset_epsilon.h5")
with dataset_path.open("rb") as f:
    ds = ReplayBuffer.load(f, buffer=InfiniteBuffer())

episodes = list(ds.episodes)
print(f"Total episodes: {len(episodes)}")

ep = episodes[0]
print(f"\nFirst episode attributes: {[a for a in dir(ep) if not a.startswith('_')]}")
print(f"observations.shape: {ep.observations.shape}")
print(f"actions.shape: {ep.actions.shape}")
print(f"rewards.shape: {ep.rewards.shape}")

# The two fields that matter most for bootstrapping behavior:
if hasattr(ep, "terminated"):
    print(f"terminated: {ep.terminated}")
if hasattr(ep, "is_timeout"):
    print(f"is_timeout: {ep.is_timeout}")

# Check across ALL episodes whether terminated is ever True
if hasattr(episodes[0], "terminated"):
    terminated_values = [bool(e.terminated) for e in episodes]
    print(f"\nAcross all {len(episodes)} episodes: "
          f"{sum(terminated_values)} have terminated=True, "
          f"{len(terminated_values) - sum(terminated_values)} have terminated=False")

# Check how many transitions this episode contributes to training --
# if d3rlpy treats it as non-terminal, it will attempt to bootstrap at
# every one of the 19 transitions including the last (obs[18]->obs[19]),
# using obs[19] as a valid non-terminal next-state.
print(f"\nEpisode length (size): {ep.size() if hasattr(ep, 'size') else len(ep.observations)}")