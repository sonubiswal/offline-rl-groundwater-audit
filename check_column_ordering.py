"""
check_column_ordering.py -- verify that the offline dataset's declared
observation column order matches the ACTUAL order SimState.to_array()
produces during rollout. A mismatch here would silently feed CQL/BC a
different feature ordering at eval time than they were trained on,
which is a plausible root cause for a systematic negative correlation
between predicted Q-values and realized returns.

WHAT THIS CHECKS
----------------
1. Reads the h5 file's 'columns' dataset (declared obs column names)
   and compares its length against 'observations' actual dimensionality.
2. Imports SimState from src/rl/synthetic_reward.py and introspects its
   dataclass fields (in declaration order) -- this is what
   state.to_array() presumably serializes, assuming to_array() just
   returns the dataclass fields in order. This script does NOT assume
   that; it also inspects to_array()'s source if possible.
3. Prints both orderings side by side so you can see immediately if the
   h5 columns names line up with the dataclass field names/order, or if
   they diverge -- WITHOUT guessing or "fixing" anything itself.
4. Also spot-checks a handful of REAL rows from the dataset (raw,
   unstandardized) against the --obs-mean values you supplied, since if
   the dataset's actual column order differs from what obs_mean/obs_std
   assume, per-column raw means printed here should look implausible
   for known fields (e.g. a sin/cos column should have mean near 0 and
   values in [-1, 1]; this script prints per-column min/max/mean so you
   can eyeball that directly rather than trusting obs_mean blindly).

USAGE
-----
python check_column_ordering.py `
    --dataset data/processed/offline_rl_dataset_epsilon.h5 `
    --obs-mean 15.075033 36.681023 0.391654 -0.000000000002838 0.000000000002838 0.423686 `
    --obs-std 7.286923 41.271557 0.253303 0.707107 0.707107 0.194976
"""

import argparse
import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "rl"))

try:
    from synthetic_reward import SimState
except ImportError as e:
    raise ImportError(
        "Could not import SimState from src/rl/synthetic_reward.py. "
        "Run this script from the same location diagnostics_distribution_shift.py "
        "runs from, or adjust the sys.path.insert above."
    ) from e


def load_columns_and_observations(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        columns = None
        if "columns" in f:
            raw = f["columns"][()]
            columns = [
                c.decode() if isinstance(c, bytes) else str(c)
                for c in raw
            ]

        # Reuse the same episode-discovery logic as the diagnostics script.
        import re
        ep_indices = sorted(
            int(m.group(1))
            for k in f.keys()
            if (m := re.match(r"^observations_(\d+)$", k))
        )
        if not ep_indices:
            raise KeyError("No observations_N keys found -- unexpected schema.")

        # Just load a handful of episodes' worth of raw observations for
        # the min/max/mean spot-check -- no need to load all 1500.
        sample_eps = ep_indices[:50]
        obs_chunks = [np.array(f[f"observations_{i}"]) for i in sample_eps]
        observations_sample = np.concatenate(obs_chunks, axis=0)

    return columns, observations_sample


def dataclass_field_order(cls):
    if not dataclasses.is_dataclass(cls):
        return None
    return [f.name for f in dataclasses.fields(cls)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--obs-mean", nargs="+", type=float, required=True)
    ap.add_argument("--obs-std", nargs="+", type=float, required=True)
    args = ap.parse_args()

    obs_mean = np.array(args.obs_mean, dtype=float)
    obs_std = np.array(args.obs_std, dtype=float)

    columns, obs_sample = load_columns_and_observations(args.dataset)

    print("=" * 70)
    print("1. DECLARED COLUMN METADATA IN THE H5 FILE")
    print("=" * 70)
    if columns is None:
        print("No 'columns' key found in the file.")
    else:
        print(f"columns dataset ({len(columns)} entries): {columns}")

    print(f"\nActual observations dimensionality: {obs_sample.shape[1]}")

    if columns is not None and len(columns) != obs_sample.shape[1]:
        print(
            f"\n*** MISMATCH: 'columns' has {len(columns)} names but "
            f"observations have {obs_sample.shape[1]} dimensions. "
            f"Either 'columns' doesn't describe the full observation "
            f"vector (e.g. only labels a subset, or labels something "
            f"else entirely like the action-space or metadata schema), "
            f"or the file is internally inconsistent. Do not assume "
            f"columns[i] corresponds to observations[:, i] without "
            f"resolving this discrepancy first. ***"
        )

    print()
    print("=" * 70)
    print("2. SimState DATACLASS FIELD ORDER (from synthetic_reward.py)")
    print("=" * 70)
    fields = dataclass_field_order(SimState)
    if fields is None:
        print("SimState is not a dataclass (or dataclasses.fields() failed) -- "
              "inspect it manually.")
    else:
        print(f"SimState fields, in declaration order ({len(fields)}): {fields}")

    if hasattr(SimState, "to_array"):
        try:
            src = inspect.getsource(SimState.to_array)
            print("\nSimState.to_array() source:")
            print(src)
        except (OSError, TypeError):
            print("\n(Could not retrieve to_array() source -- inspect it manually "
                  "in synthetic_reward.py.)")
    else:
        print("\nSimState has no to_array() method -- check the actual method name "
              "used to serialize state to a vector.")

    print()
    print("=" * 70)
    print("3. PER-COLUMN RAW STATS FROM THE DATASET (first 50 episodes)")
    print("=" * 70)
    print(f"{'idx':<4}{'declared name':<20}{'min':>12}{'max':>12}{'mean':>14}{'--obs-mean arg':>18}")
    for i in range(obs_sample.shape[1]):
        name = columns[i] if (columns is not None and i < len(columns)) else "(unlabeled)"
        col = obs_sample[:, i]
        mean_arg = obs_mean[i] if i < len(obs_mean) else float("nan")
        print(f"{i:<4}{name:<20}{col.min():>12.4f}{col.max():>12.4f}{col.mean():>14.6f}{mean_arg:>18.6f}")

    print()
    print("=" * 70)
    print("READ THIS")
    print("=" * 70)
    print(
        "Compare the SimState field list in section 2 against the 'columns' "
        "list in section 1 -- same names, same order? If SimState has a "
        "field not present in 'columns' (or vice versa), or the orders "
        "differ, that's a strong candidate root cause for the negative "
        "Q-vs-return correlation: rollout observations (built from "
        "SimState.to_array()) would be feeding the model features in a "
        "different order than the offline training data actually used, "
        "even though both are 'the same 6 numbers'.\n\n"
        "Also sanity-check section 3 against what each column SHOULD look "
        "like physically -- e.g. a month_sin/month_cos pair should have "
        "min/max close to [-1, 1] and mean close to 0; if a column you "
        "expect to be sin/cos instead has a min/max like [0, 100], the "
        "column at that index is not what obs_mean/obs_std assumed."
    )


if __name__ == "__main__":
    main()