import json
from pathlib import Path
d = json.loads(Path("revision_experiments/e9_three_construction_table.json").read_text())
print("=== pooled only ===")
for k, v in d.items():
    if v["class"] == "pooled only":
        print(f"  {k:50s}  "
              f"pooled [{v['pooled']['lo']:+.3f},{v['pooled']['hi']:+.3f}]  "
              f"seed_t [{v['seed_t']['lo']:+.3f},{v['seed_t']['hi']:+.3f}]  "
              f"hier [{v['hierarchical']['lo']:+.3f},{v['hierarchical']['hi']:+.3f}]")
print()
print("=== pooled=True seed=False hier=True ===")
for k, v in d.items():
    if "seed=False hier=True" in v["class"]:
        print(f"  {k:50s}")
        print(f"    pooled       [{v['pooled']['lo']:+.3f}, {v['pooled']['hi']:+.3f}]  mean {v['pooled']['mean']:+.3f}")
        print(f"    seed_t       [{v['seed_t']['lo']:+.3f}, {v['seed_t']['hi']:+.3f}]  mean {v['seed_t']['mean']:+.3f}")
        print(f"    hierarchical [{v['hierarchical']['lo']:+.3f}, {v['hierarchical']['hi']:+.3f}]  mean {v['hierarchical']['mean']:+.3f}")
print()
print("=== none ===")
for k, v in d.items():
    if v["class"] == "none":
        print(f"  {k:50s}  mean {v['pooled']['mean']:+.3f}")
