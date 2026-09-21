"""
transcribe_3seed.py -- extract per-seed BC/CQL metrics from the report JSONs
into a single table for the paper. Read-only.
"""
import json
from pathlib import Path
import pandas as pd

REPORTS = Path("reports")

# Adjust this list to whatever run dirs exist for your 3-seed evaluation.
# The script prints what it finds so you can correct the paths.
CANDIDATE_DIRS = [
    REPORTS / "epsilon",
    REPORTS / "no_oracle",
    REPORTS / "no_oracle_alpha1",
    REPORTS / "no_oracle_alpha10",
]


def main():
    rows = []
    for d in CANDIDATE_DIRS:
        pc = d / "policy_comparison.json"
        if not pc.exists():
            print(f"[skip] {pc} not found")
            continue
        with open(pc) as f:
            data = json.load(f)
        # data may be a dict or list; print structure the first time
        if not rows:
            print(f"[structure] {pc} top-level keys/types:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {type(v).__name__}")
        # Try to pull per-seed and pooled fields. The exact key names depend
        # on your schema; print everything on first run so it can be adjusted.
        rows.append({"source": str(pc), "raw_keys": list(data.keys())})

    print()
    print("--- first pass: raw key inventory ---")
    for r in rows:
        print(f"{r['source']}:")
        for k in r["raw_keys"]:
            print(f"  - {k}")


if __name__ == "__main__":
    main()