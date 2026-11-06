"""
E2 dump: print the full 27-pairing table from e2_family_specific_starts.json.

Run from C:\\trishna-opal:
    python revision_experiments\\e2_dump.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
path = REPO / "revision_experiments" / "e2_family_specific_starts.json"
d = json.loads(path.read_text())

print(f"total keys: {len(d)}\n")

# Group by start set for a clean table
by_kind = {}
for k, v in d.items():
    kind, pairing = k.split("|", 1)
    by_kind.setdefault(kind, {})[pairing] = v

for kind in ["rf_grid", "common_epsilon_test", "own_family_test"]:
    if kind not in by_kind:
        continue
    print(f"=== {kind} ===")
    for pairing, v in sorted(by_kind[kind].items()):
        flag = "excl 0" if v["excludes_zero"] else "spans 0"
        print(f"  {pairing:35s} {v['mean']:+.3f} "
              f"[{v['lo']:+.3f},{v['hi']:+.3f}] n={v['n_starts']} {flag}")
    print()