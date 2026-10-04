"""Compare hierarchical and crossed bootstraps cell by cell; recompute the
construction-dependent count using (pooled, seed_t, crossed)."""
import json
from pathlib import Path
import numpy as np

REV = Path(r"C:\trishna-opal\revision_experiments")

# Pooled and seed_t results, plus the old hierarchical classification
old = json.loads((REV / "table_x_rerun.json").read_text())
# Crossed results, from the run just completed
crossed = json.loads((REV / "table_x_crossed.json").read_text())

# Index crossed by (bc, cql, protocol)
crossed_idx = {(r["bc"], r["cql"], r["protocol"]): r for r in crossed}

print(f"{'cell':>30} {'pooled':>8} {'seed_t':>8} {'hier':>8} {'crossed':>8} "
      f"{'hier!=crossed':>15}")
print("-" * 85)

cells = []
flips = []
for r in old:
    key = (r["bc"], r["cql"], r["protocol"])
    if key not in crossed_idx:
        continue
    cr = crossed_idx[key]

    p_ex = r["pooled_excludes"]
    t_ex = r["seed_t_excludes"]
    h_ex = r["hier_excludes"]
    c_ex = cr["excludes"]

    if h_ex != c_ex:
        flips.append(key)

    label = f"{r['bc']}/{r['cql']} {r['protocol']}"
    print(f"{label:>30} {str(p_ex):>8} {str(t_ex):>8} {str(h_ex):>8} "
          f"{str(c_ex):>8} {str(h_ex != c_ex):>15}")

    cells.append({
        "bc": r["bc"], "cql": r["cql"], "protocol": r["protocol"],
        "pooled": p_ex, "seed_t": t_ex, "crossed": c_ex,
        "pattern": ("all_exclude" if (p_ex and t_ex and c_ex) else
                    "none_exclude" if not (p_ex or t_ex or c_ex) else
                    "dependent"),
    })

# Collapse the three epsilon-BC own-family rows
distinct = [c for c in cells
            if not (c["bc"] == "epsilon" and c["protocol"] == "own_family")]

print()
print(f"cells where hierarchical and crossed disagree on excluding zero: "
      f"{len(flips)}")
for f in flips:
    print(f"  {f[0]}/{f[1]} {f[2]}")

n_all = sum(1 for c in distinct if c["pattern"] == "all_exclude")
n_none = sum(1 for c in distinct if c["pattern"] == "none_exclude")
n_dep = sum(1 for c in distinct if c["pattern"] == "dependent")
print()
print(f"Under (pooled, seed_t, crossed):")
print(f"  all three exclude: {n_all}")
print(f"  none exclude:      {n_none}")
print(f"  construction-dependent: {n_dep} of {len(distinct)}")
print(f"  paper's hierarchical count: 7 of 24")