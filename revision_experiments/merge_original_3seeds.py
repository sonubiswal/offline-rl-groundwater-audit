"""
Copy the original 3-seed checkpoints (seeds 42, 123, 2024) from the
original per-config directories into models_10seed/<config>/{bc,cql}/,
so that each config has all 10 seeds (original 3 + 7 new).

Idempotent: skips files that already exist in the destination.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ORIGINAL_SEEDS = [42, 123, 2024]

# (config_name, source_bc_dir, source_cql_dir)
SOURCES = [
    ("base",          "models/bc",                   "models/cql"),
    ("epsilon",       "models_epsilon/bc",           "models_epsilon/cql"),
    ("no_oracle",     "models_no_oracle/bc",         "models_no_oracle/cql"),
    ("no_oracle_a1",  "models_no_oracle/bc",         "models_no_oracle_alpha1"),
    ("no_oracle_a10", "models_no_oracle/bc",         "models_no_oracle_alpha10"),
]


def copy_three(prefix, seed, src_dir, dst_dir):
    """Copy <prefix>_seed<seed>.d3 and .d3.scaler.json from src to dst.
    Also copy .meta.json if present. Return count of files copied."""
    copied = 0
    for ext in (".d3", ".d3.scaler.json", ".meta.json"):
        name = f"{prefix}_seed{seed}{ext}"
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            continue
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main():
    total_copied = 0
    for config, src_bc, src_cql in SOURCES:
        bc_dst = REPO / "models_10seed" / config / "bc"
        cql_dst = REPO / "models_10seed" / config / "cql"
        bc_src = REPO / src_bc
        cql_src = REPO / src_cql

        if not bc_src.exists():
            print(f"[merge] WARNING: BC source missing {bc_src}")
            continue
        if not cql_src.exists():
            print(f"[merge] WARNING: CQL source missing {cql_src}")
            continue

        print(f"\n[merge] config={config}")
        print(f"  BC  from {src_bc}  ->  {bc_dst.relative_to(REPO)}")
        print(f"  CQL from {src_cql}  ->  {cql_dst.relative_to(REPO)}")

        for seed in ORIGINAL_SEEDS:
            n = copy_three("bc", seed, bc_src, bc_dst)
            total_copied += n
            n = copy_three("cql", seed, cql_src, cql_dst)
            total_copied += n

    print(f"\n[merge] copied {total_copied} files total")


if __name__ == "__main__":
    main()