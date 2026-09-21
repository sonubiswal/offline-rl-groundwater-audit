"""
inspect_h5_summary.py -- summarize an HDF5 file's schema by key PREFIX
instead of printing every individual key. Your file has ~1000 episodes,
each stored as its own dataset (e.g. observations_0, observations_1, ...,
terminated_0, terminated_1, ...), so a raw per-key dump scrolls off the
terminal. This groups keys like 'observations_0', 'observations_1', ...
under the prefix 'observations_N' and reports count/shape/dtype once.

USAGE
-----
python inspect_h5_summary.py --file data/processed/offline_rl_dataset_epsilon.h5
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")


def prefix_of(name: str) -> str:
    """Collapse trailing _<int> into _N, and root-level attrs stay as-is."""
    m = re.match(r"^(.*)_(\d+)$", name)
    if m:
        return f"{m.group(1)}_N"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    groups = defaultdict(list)  # prefix -> list of (name, shape, dtype)

    with h5py.File(p, "r") as f:
        print(f"File: {p}")
        print(f"Root attrs: {dict(f.attrs)}")
        print("-" * 60)

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                groups[prefix_of(name)].append((name, obj.shape, obj.dtype))

        f.visititems(visit)

        for prefix in sorted(groups):
            entries = groups[prefix]
            shapes = {e[1] for e in entries}
            dtypes = {str(e[2]) for e in entries}
            example = sorted(e[0] for e in entries)[:3]
            print(f"[{len(entries):>5} keys]  prefix={prefix!r}")
            print(f"           shapes={shapes}  dtypes={dtypes}")
            print(f"           example keys: {example}")
            print()

    print("-" * 60)
    print(
        "If you see prefixes like 'observations_N', 'actions_N', "
        "'rewards_N', 'terminated_N' (each with hundreds/thousands of "
        "numbered keys, one per episode), your file is d3rlpy's "
        "PER-EPISODE HDF5 export format, not the flat "
        "observations/actions/rewards/terminals array format the "
        "original diagnostics script assumed. Use the accompanying "
        "load_episodic_h5_dataset() replacement instead of load_h5_dataset()."
    )


if __name__ == "__main__":
    main()