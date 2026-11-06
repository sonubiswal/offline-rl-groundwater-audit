"""
E9 recount: collapse the 27 rows into distinct cells and report per-class
counts. Also flag the matched-family outlier (no_oracle/no_oracle on
common epsilon test) separately.

Run from C:\\trishna-opal:
    python revision_experiments\\e9_recount.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
d = json.loads((REPO / "revision_experiments"
                / "e9_three_construction_table.json").read_text())

# Collapse exact duplicates by the tuple (mean, lo, hi) of all three
# constructions. This identifies own_family==common for epsilon-BC
# without hard-coding the pairing.
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

print(f"27 rows -> {len(unique)} distinct cells\n")

# Count by class
counts = Counter()
for s, k in unique.items():
    v = d[k]
    counts[v["class"]] += 1

print("class counts (distinct cells):")
for cls, n in counts.most_common():
    print(f"  {cls:35s} {n}")

# Cross-family vs matched
def is_cross(k):
    parts = k.split("|")
    bc = parts[1].split("=")[1]
    cql = parts[2].split("=")[1]
    return bc != cql

cross = sum(1 for k in unique.values() if is_cross(k))
matched = len(unique) - cross
print(f"\ncross-family: {cross}   matched-family: {matched}")

# pooled-only cells
print("\npooled-only cells (distinct):")
for s, k in unique.items():
    v = d[k]
    if v["class"] == "pooled only":
        cf = "cross" if is_cross(k) else "matched"
        print(f"  [{cf:7s}] {k}")

# non-"all" cells (construction disagreement anywhere)
disagree = [k for k in unique.values() if d[k]["class"] != "all"]
print(f"\nconstruction disagreement (any class != 'all'): "
      f"{len(disagree)} of {len(unique)}  "
      f"({100*len(disagree)/len(unique):.0f}%)")

# In no cell should seed-aware exclude zero while pooled doesn't
print("\nchecking for reverse-direction disagreements...")
reverse = []
for k in unique.values():
    v = d[k]
    p = v["pooled"]["lo"] > 0 or v["pooled"]["hi"] < 0
    s = v["seed_t"]["lo"] > 0 or v["seed_t"]["hi"] < 0
    h = v["hierarchical"]["lo"] > 0 or v["hierarchical"]["hi"] < 0
    if (s or h) and not p:
        reverse.append(k)
if reverse:
    print(f"  found {len(reverse)} reverse-direction cells (unexpected):")
    for k in reverse:
        print(f"    {k}")
else:
    print("  none — no cell shows a seed-aware interval excluding zero "
          "while the pooled interval spans it.")