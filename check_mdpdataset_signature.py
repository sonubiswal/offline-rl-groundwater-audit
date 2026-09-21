import inspect
from d3rlpy.dataset import MDPDataset, EpisodeGenerator

print("MDPDataset.__init__ signature:")
print(inspect.signature(MDPDataset.__init__))
print()
print("MDPDataset docstring:")
print(MDPDataset.__init__.__doc__)
print()
print("=" * 70)
print("EpisodeGenerator signature (this is likely what actually splits")
print("flat arrays into Episode objects based on the terminals array):")
print("=" * 70)
print(inspect.signature(EpisodeGenerator.__init__))
print(EpisodeGenerator.__init__.__doc__)