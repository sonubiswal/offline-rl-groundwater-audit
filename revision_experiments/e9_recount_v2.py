"""
E9 recount v2: count only cells where the three constructions actually
disagree on whether zero is excluded. The "none" class is not a
disagreement — all three constructions span zero there.
"""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
d = json.loads((REPO / "revision_experiments"
                / "e9_three_construction_table.json").read_text())


def sig(v):
    return (
        (round(v["pooled"]["mean"], 6), round(v["pooled"]["lo"], 6),
         round(v["pooled"]["hi"], 6)),
        (round(v["seed_t"]["mean"], 6), round(v["seed_t"]["lo"], 6),
         round(v["seed_t"]["hi"], 6)),
        (round(v["hierarchical"]["mean"], 6), round(v["hierarchical"]["lo"], 6),
         round(v["hierarchical"]["hi"], 6)),
    )


unique = {}
for k, v in d.items():
    s = sig(v)
    if s not in unique:
        unique[s] = k

counts = Counter(d[k]["class"] for k in unique.values())
print(f"27 rows -> {len(unique)} distinct cells\n")
print("class counts (distinct cells):")
for cls, n in counts.most_common():
    print(f"  {cls:35s} {n}")

# Disagreement = any class other than 'all' or 'none'
disagree = [k for k in unique.values()
            if d[k]["class"] not in ("all", "none")]
print(f"\ndisagreement cells (constructions differ on zero): "
      f"{len(disagree)} of {len(unique)}  "
      f"({100*len(disagree)/len(unique):.0f}%)")
for k in disagree:
    print(f"  {d[k]['class']:35s} {k}")

# Cross-family vs matched among the disagreement cells
def is_cross(k):
    parts = k.split("|")
    return parts[1].split("=")[1] != parts[2].split("=")[1]

cross = sum(1 for k in disagree if is_cross(k))
print(f"\nof {len(disagree)} disagreement cells: "
      f"{cross} cross-family, {len(disagree)-cross} matched")