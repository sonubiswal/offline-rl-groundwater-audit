"""
inspect_h5.py -- standalone HDF5 structure inspector.

Prints every group/dataset in the file, with shape/dtype for datasets,
so you can see the ACTUAL key names instead of assuming the d3rlpy
MDPDataset schema (observations/actions/rewards/terminals).

USAGE
-----
python inspect_h5.py --file data/processed/offline_rl_dataset_epsilon.h5
"""

import argparse
from pathlib import Path

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")


def walk(name, obj, indent=0):
    pad = "  " * indent

    if isinstance(obj, h5py.Dataset):
        print(f"{pad}[dataset] {name!r}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{pad}[group]   {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to .h5 file to inspect")
    ap.add_argument(
        "--attrs",
        action="store_true",
        help="Also print any HDF5 attributes attached to root/groups/datasets",
    )
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with h5py.File(p, "r") as f:
        print(f"File: {p}")
        print(f"Top-level keys: {list(f.keys())}")
        print("-" * 60)

        # Recursively visit every group/dataset in the file.
        f.visititems(walk)

        if args.attrs:
            print("-" * 60)
            print("Root attrs:", dict(f.attrs))
            for name, obj in f.items():
                if obj.attrs:
                    print(f"{name} attrs:", dict(obj.attrs))

    print("-" * 60)
    print(
        "If 'observations'/'actions'/'rewards'/'terminals' are NOT shown "
        "above as top-level dataset names, update load_h5_dataset() in "
        "diagnostics_distribution_shift.py to use the real key names "
        "printed here (e.g. a nested path like 'data/observations', or "
        "differently-named keys like 'obs'/'act'/'reward'/'done')."
    )


if __name__ == "__main__":
    main()