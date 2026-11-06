"""
train_missing_seeds.py

Trains the canonical checkpoints for any seed in A.SEEDS that doesn't
already exist on disk. Existing seeds are skipped, so this is resumable.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import adapters as A

FAMILIES = ["base", "no_oracle"]
EXISTING = {42, 123, 2024}
MISSING = [s for s in A.SEEDS if s not in EXISTING]


def main():
    print(f"missing seeds: {MISSING}")
    for fam in FAMILIES:
        ds = A.DATASETS[fam]
        for algo in ("bc", "cql"):
            for s in MISSING:
                ck = A.checkpoint(fam, algo, s)
                if ck.exists():
                    print(f"[skip ] {ck}")
                    continue
                print(f"[train] {ck}")
                if algo == "bc":
                    A.train_bc(ds, ck, s)
                else:
                    A.train_cql(ds, ck, s)
    print("done.")


if __name__ == "__main__":
    main()