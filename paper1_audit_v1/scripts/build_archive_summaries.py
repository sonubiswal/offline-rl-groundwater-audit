#!/usr/bin/env python3
"""build_archive_summaries.py — dump every table's summary as JSON.

Reads source JSONs from C:\\trishna-opal\\revision_experiments.
Writes into ../summaries/ relative to this script.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
OUT = ARCHIVE / "summaries"
OUT.mkdir(parents=True, exist_ok=True)

REV = Path(r"C:\trishna-opal\revision_experiments")

# --- Table VII: hand-transcribed from robust_rerun.py output ---
table_vii = {
    "100": {"contrast": 0.8843, "ci": [-1.1760, 2.9443], "seeds_pos": 6,
            "sd": 2.880, "median": 0.13,  "trimmed": 0.1379,
            "loo": -0.0055, "loo_drop": 123, "loo_ci": [-0.5040, 0.4929]},
    "200": {"contrast": 0.3338, "ci": [ 0.0981, 0.5695], "seeds_pos": 8,
            "sd": 0.329, "median": 0.32,  "trimmed": 0.3224,
            "loo":  0.2774, "loo_drop": 7,   "loo_ci": [ 0.0515, 0.5033]},
    "500": {"contrast": 0.8086, "ci": [-0.1500, 1.7673], "seeds_pos": 8,
            "sd": 1.340, "median": 0.39,  "trimmed": 0.5191,
            "loo":  0.4257, "loo_drop": 99999, "loo_ci": [-0.0427, 0.8941]},
}
(OUT / "table_vii.json").write_text(json.dumps(table_vii, indent=2))

# --- Table VIII: matched-start margins ---
table_viii = {
    "fqe_100": {"bc": 11.600, "cql": 12.498, "gap": 0.898},
    "fqe_200": {"bc": 12.918, "cql": 13.266, "gap": 0.348},
    "fqe_500": {"bc":  9.666, "cql": 10.489, "gap": 0.823},
    "rollout": {"bc":  9.857, "cql":  9.871, "gap": 0.0141},
}
(OUT / "table_viii.json").write_text(json.dumps(table_viii, indent=2))

# --- Table XI: reward decomposition ---
table_xi = {
    "endpoint": {"removed": 4.8383, "full": 5.0929,
                 "removed_minus_full": -0.2546,
                 "ci": [-0.4110, -0.0982]},
    "bands": {
        "in_dist":  {"removed": 1.1675, "full": 1.2265, "diff": -0.0590},
        "mild":     {"removed": 3.3187, "full": 3.4770, "diff": -0.1582},
        "moderate": {"removed": 6.7263, "full": 7.0554, "diff": -0.3291},
        "severe":   {"removed": 6.0059, "full": 6.3194, "diff": -0.3136},
    },
}
(OUT / "table_xi.json").write_text(json.dumps(table_xi, indent=2))

# --- Tables IX, X, XII: copy straight from source JSONs ---
for src_name, dst_name in [
    ("family_start_rerun.json", "table_ix.json"),
    ("table_x_rerun.json",      "table_x.json"),
    ("bc_sweep.json",           "table_xii.json"),
]:
    src = REV / src_name
    if src.exists():
        (OUT / dst_name).write_text(src.read_text())
        print(f"copied {src_name} -> {dst_name}")
    else:
        print(f"MISSING {src}")

print(f"\nsummaries written to {OUT}")