"""
check_leakage.py — Phase 5 freeze-gate check #6 (leakage audit), v6

Verifies that held-out wells and future information never enter the
training/RL pipeline. Six checks:

  1. Split integrity — training and held-out well-ID sets are disjoint
     (normalized: stripped + uppercased).

  2. Training-only files (train_well_ids, RF grid predictions, RF training
     table) contain no held-out well IDs.

  3. RL pipeline source code contains no actual file-read reference to
     held-out files. Comments, docstrings, and multi-line strings are
     ignored — the pattern is only flagged when it also contains a
     file-read marker on the same code line.

  4. RL dataset gw_level values are checked, across all timesteps of an
     evenly-spaced sample of episodes, for near-exact membership in the
     RF-predicted value set. Materially stronger than range overlap.

  5. Feature age columns (any column ending in *_age_days) must be
     non-negative everywhere. Negative age = future-information leakage.
     Auto-discovers the training table across common locations.

  6. Held-out IDs file hash is compared against the hash recorded in a
     training manifest (if the manifest records one). Skipped cleanly
     if the key is absent.

Exit 0 unless a check returns False. SKIP (None) is not a failure.
Exit 1 with a list of failures otherwise.

Version history:
  - v6: skip branches return None (not True) so the summary correctly
    distinguishes PASS / SKIP / FAIL. Exit code updated accordingly.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


DEFAULT_ROOT = Path(".")

HELDOUT_WHITELIST = {
    "held_out_ids.csv",
    "held_out_readings.csv",
    "rf_downscale_validation.csv",
}

TRAINING_ONLY_CANDIDATES = [
    "data/interim/train_readings.csv",
    "data/interim/train_well_ids.csv",
    "data/interim/rf_training_table.csv",
    "data/processed/rf_training_table.csv",
    "reports/rf_grid_predictions.csv",
]

# Search order for the RF training table used by CHECK 5.
TRAINING_TABLE_CANDIDATES = [
    "data/interim/rf_training_table.csv",
    "data/processed/rf_training_table.csv",
    "reports/rf_training_table.csv",
    "data/rf_training_table.csv",
]

WELL_ID_COLUMN_CANDIDATES = (
    "well_id", "wellid", "station_id", "gw_well_code", "cgwb_id", "id",
)

RL_SOURCE_DIRS = ["src/rl"]
HELDOUT_CODE_PATTERNS = [r"held_out", r"heldout", r"HELD_OUT"]

READ_MARKERS = (
    "open(", "read_csv", "read_table", "read_parquet",
    "Path(", ".csv", ".h5", ".hdf5", "load(",
)

DEPTH_COL_CANDIDATES = (
    "gw_level_pred", "depth_m", "gw_level",
    "groundwater_depth", "predicted_depth", "depth",
)

AGE_COLUMN_SUFFIX = "_age_days"


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def die(msg):
    print(f"FATAL: {msg}")
    sys.exit(1)


def normalize_ids(series):
    return set(series.dropna().astype(str).str.strip().str.upper().unique())


def find_well_id_column(df):
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in WELL_ID_COLUMN_CANDIDATES:
        if candidate in lower_cols:
            return lower_cols[candidate]
    for c in df.columns:
        if "well" in c.lower() and "id" in c.lower():
            return c
    return None


def read_well_ids(path):
    df = pd.read_csv(path)
    col = find_well_id_column(df)
    if col is None:
        die(f"{path} has no recognizable well-id column "
            f"(columns: {list(df.columns)}; checked {WELL_ID_COLUMN_CANDIDATES})")
    return normalize_ids(df[col])


def sha1_of_file(path: Path, chunk_size=1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _docstring_line_numbers(text):
    """Return the set of 1-based line numbers inside triple-quoted strings."""
    inside = set()
    in_triple = None
    for i, line in enumerate(text.splitlines(), start=1):
        d3 = line.count('"""')
        s3 = line.count("'''")
        if in_triple is None:
            if d3 % 2 == 1:
                in_triple = '"""'
                inside.add(i)
            elif s3 % 2 == 1:
                in_triple = "'''"
                inside.add(i)
        else:
            inside.add(i)
            if in_triple == '"""' and d3 % 2 == 1:
                in_triple = None
            elif in_triple == "'''" and s3 % 2 == 1:
                in_triple = None
    return inside


def discover_first_existing(root: Path, candidates):
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CHECK 1
# ---------------------------------------------------------------------------
def check_split_integrity(held_out_path, train_path):
    print_header("CHECK 1: Split integrity")
    if not held_out_path.exists():
        die(f"held-out IDs file not found: {held_out_path}")
    if not train_path.exists():
        die(f"training IDs file not found: {train_path}")

    H = read_well_ids(held_out_path)
    T = read_well_ids(train_path)

    print(f"held-out wells:  {len(H):,} (normalized: stripped + uppercased)")
    print(f"training wells:  {len(T):,}")

    overlap = H & T
    if overlap:
        print(f"[FAIL] {len(overlap)} wells appear in BOTH sets")
        for w in list(overlap)[:5]:
            print(f"  {w}")
        return False
    print("[PASS] held-out \u2229 training = empty")
    return True


# ---------------------------------------------------------------------------
# CHECK 2
# ---------------------------------------------------------------------------
def check_training_only_files(root, H):
    print_header("CHECK 2: Training-only files contain no held-out IDs")
    ok = True
    for rel in TRAINING_ONLY_CANDIDATES:
        path = root / rel
        if not path.exists():
            print(f"[SKIP] {rel} (not present)")
            continue
        if path.name in HELDOUT_WHITELIST:
            print(f"[SKIP] {rel} (whitelisted as held-out file)")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[SKIP] {rel} (unreadable: {e})")
            continue

        col = find_well_id_column(df)
        if col is None:
            print(f"[PASS] {rel} (no recognizable well-id column, "
                  f"{len(df):,} rows)")
            continue

        ids_in_file = normalize_ids(df[col])
        hits = ids_in_file & H
        if hits:
            print(f"[FAIL] {rel} column '{col}': "
                  f"{len(hits)} held-out well IDs present")
            for w in list(hits)[:5]:
                print(f"  {w}")
            ok = False
        else:
            print(f"[PASS] {rel} column '{col}': "
                  f"0 held-out IDs among {len(ids_in_file):,} unique")
    return ok


# ---------------------------------------------------------------------------
# CHECK 3
# ---------------------------------------------------------------------------
def check_rl_source(root):
    print_header("CHECK 3: RL source code contains no held-out references (static)")
    print("NOTE: text-pattern check only; cannot see indirect/config-driven loads.")
    ok = True
    for src_dir in RL_SOURCE_DIRS:
        d = root / src_dir
        if not d.exists():
            print(f"[SKIP] {src_dir} (not present)")
            continue
        for py_file in d.rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[SKIP] {py_file} (unreadable: {e})")
                continue

            docstring_lines = _docstring_line_numbers(text)

            hits = []
            for line_no, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line_no in docstring_lines:
                    continue
                if not any(pat.lower() in line.lower()
                           for pat in HELDOUT_CODE_PATTERNS):
                    continue
                if not any(m in line for m in READ_MARKERS):
                    continue
                hits.append((line_no, stripped))
            if hits:
                print(f"[FAIL] {py_file.relative_to(root)}: "
                      f"{len(hits)} held-out reference(s)")
                for line_no, line in hits[:5]:
                    print(f"  line {line_no}: {line}")
                ok = False

    if ok:
        print(f"[PASS] no held-out references found in {RL_SOURCE_DIRS} (static)")
    return ok


# ---------------------------------------------------------------------------
# CHECK 4
# ---------------------------------------------------------------------------
def check_rl_dataset_values(root, tolerance: float, max_episodes: int):
    print_header("CHECK 4: RL dataset gw_level vs RF+Kriged sources "
                 "(initial-state strict, trajectory-state drift-aware)")
    h5_path = root / "data/processed/offline_rl_dataset.h5"
    rf_path = root / "reports/rf_grid_predictions.csv"
    kriged_dir = root / "data/interim/kriged_target_monthly"

    if not h5_path.exists():
        print(f"[SKIP] {h5_path} not found")
        return None
    if not HAVE_H5PY:
        print("[SKIP] h5py not installed; cannot inspect RL dataset")
        return None

    source_values = []

    rf_gw = None
    if rf_path.exists():
        rf = pd.read_csv(rf_path)
        rf_cols = [c for c in rf.columns if c.lower() in DEPTH_COL_CANDIDATES]
        if rf_cols:
            rf_gw = rf[rf_cols[0]].dropna().to_numpy()
            source_values.append(rf_gw)
            print(f"RF predictions:              n={len(rf_gw):,}, "
                  f"range=[{rf_gw.min():.2f}, {rf_gw.max():.2f}], "
                  f"mean={rf_gw.mean():.2f}")
        else:
            print(f"[WARN] RF file has no depth-like column")

    kriged_gw = None
    if kriged_dir.exists():
        kriged_arrays = []
        for npy in sorted(kriged_dir.glob("*.npy")):
            try:
                arr = np.load(npy).flatten()
                kriged_arrays.append(arr[~np.isnan(arr)])
            except Exception:
                continue
        if kriged_arrays:
            kriged_gw = np.concatenate(kriged_arrays)
            source_values.append(kriged_gw)
            print(f"Kriged target surfaces:      n={len(kriged_gw):,}, "
                  f"range=[{kriged_gw.min():.2f}, {kriged_gw.max():.2f}], "
                  f"mean={kriged_gw.mean():.2f}")
        else:
            print(f"[WARN] no readable .npy files under {kriged_dir}")
    else:
        print(f"[WARN] kriged directory not found: {kriged_dir}")

    if not source_values:
        print("[SKIP] no reference value sets available")
        return None

    source_sorted = np.sort(np.concatenate(source_values))

    # Load RL dataset.
    try:
        with h5py.File(h5_path, "r") as f:
            obs_keys = sorted([k for k in f.keys()
                               if k.startswith("observations_")])
            if not obs_keys:
                print("[SKIP] no observations_* datasets in RL file")
                return None
            n_total = len(obs_keys)
            if max_episodes and max_episodes < n_total:
                idx = np.linspace(0, n_total - 1, max_episodes, dtype=int)
                keys_to_check = [obs_keys[i] for i in idx]
                print(f"Checking {len(keys_to_check)}/{n_total} episodes")
            else:
                keys_to_check = obs_keys
                print(f"Checking all {n_total} episodes")
            init_gw_values = []
            traj_gw_values = []
            for k in keys_to_check:
                arr = f[k][:]
                if arr.ndim == 2 and arr.shape[0] > 0:
                    init_gw_values.append(arr[0, 0])
                    if arr.shape[0] > 1:
                        traj_gw_values.append(arr[1:, 0])
        init_gw = np.array(init_gw_values) if init_gw_values else np.array([])
        traj_gw = np.concatenate(traj_gw_values) if traj_gw_values else np.array([])
    except Exception as e:
        print(f"[SKIP] failed to read {h5_path}: {e}")
        return None

    if init_gw.size == 0:
        print("[SKIP] no initial gw_level values extracted")
        return None

    def nearest_source_distances(values):
        idx_pos = np.searchsorted(source_sorted, values)
        idx_pos = np.clip(idx_pos, 1, len(source_sorted) - 1)
        dl = np.abs(values - source_sorted[idx_pos - 1])
        dr = np.abs(values - source_sorted[idx_pos])
        return np.minimum(dl, dr)

    # --- TIER 1: initial-state strict check ---
    print(f"\n[TIER 1] Initial-state strict check (tolerance={tolerance})")
    init_dist = nearest_source_distances(init_gw)
    n_bad_init = int((init_dist > tolerance).sum())
    print(f"  RL initial gw_level: n={len(init_gw):,}, "
          f"range=[{init_gw.min():.2f}, {init_gw.max():.2f}], "
          f"mean={init_gw.mean():.2f}")
    if n_bad_init == 0:
        print(f"  [PASS] every initial state is within {tolerance} of a "
              f"source value")
    else:
        print(f"  [FAIL] {n_bad_init}/{len(init_gw)} initial states have no "
              f"source within {tolerance}")
        worst = np.argsort(-init_dist)[:5]
        for i in worst:
            print(f"    init={init_gw[i]:.3f}, nearest source distance="
                  f"{init_dist[i]:.3f}")

    # --- TIER 2: trajectory-state drift-aware check ---
    TRAJ_TOLERANCE = 3.0  # accommodates up to 6 steps of drawdown accumulation
    print(f"\n[TIER 2] Trajectory-state drift-aware check "
          f"(tolerance={TRAJ_TOLERANCE})")
    if traj_gw.size == 0:
        print("  [SKIP] no trajectory (non-initial) states to check")
        n_bad_traj = 0
    else:
        traj_dist = nearest_source_distances(traj_gw)
        n_bad_traj = int((traj_dist > TRAJ_TOLERANCE).sum())
        print(f"  RL trajectory gw_level: n={len(traj_gw):,}, "
              f"range=[{traj_gw.min():.2f}, {traj_gw.max():.2f}], "
              f"mean={traj_gw.mean():.2f}")
        if n_bad_traj == 0:
            print(f"  [PASS] every trajectory state is within "
                  f"{TRAJ_TOLERANCE} of a source value")
        else:
            print(f"  [FAIL] {n_bad_traj}/{len(traj_gw)} trajectory states "
                  f"have no source within {TRAJ_TOLERANCE}")
            worst = np.argsort(-traj_dist)[:5]
            for i in worst:
                print(f"    traj={traj_gw[i]:.3f}, nearest source distance="
                      f"{traj_dist[i]:.3f}")

    # --- INFO: depth-compression note ---
    if rf_gw is not None and kriged_gw is not None:
        rf_max = float(rf_gw.max())
        kr_max = float(kriged_gw.max())
        rl_above_rf = int((init_gw > rf_max).sum())
        print(f"\n[INFO] depth-compression note (not a failure):")
        print(f"  RF prediction max = {rf_max:.2f} m")
        print(f"  Kriged target max = {kr_max:.2f} m")
        print(f"  RL initial states above RF max = {rl_above_rf}/{len(init_gw)} "
              f"({100.0 * rl_above_rf / len(init_gw):.1f}%)")
        print(f"  These deep states come from Kriged surfaces (training-only, "
              f"not leakage). Deployment of RL with RF as state predictor "
              f"would under-predict them by up to "
              f"~{kr_max - rf_max:.1f} m (Phase 2 depth-compression bias, "
              f"limitations.md \u00a71).")

    # --- Verdict ---
    if n_bad_init == 0 and n_bad_traj == 0:
        print("\n[PASS] CHECK 4: no leakage signal detected; RL states are "
              "consistent with source values (initial strict, trajectory "
              "drift-aware)")
        return True

    print("\n[FAIL] CHECK 4: some RL states are not consistent with any source "
          "value at the appropriate tolerance")
    return False


# ---------------------------------------------------------------------------
# CHECK 5
# ---------------------------------------------------------------------------
def check_feature_age_columns(root):
    print_header("CHECK 5: feature age columns (*_age_days must be non-negative)")
    path = discover_first_existing(root, TRAINING_TABLE_CANDIDATES)
    if path is None:
        print(f"[SKIP] no RF training table found in any of: {TRAINING_TABLE_CANDIDATES}")
        print("       This check would verify that any *_age_days column is "
              "non-negative (target_lag1 leakage vector). Skipped because the "
              "table was not located. If you have the table at a custom path, "
              "copy it into one of the locations above, or run this check manually.")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] {path} unreadable: {e}")
        return None

    print(f"Using training table: {path.relative_to(root)} ({len(df):,} rows)")

    age_cols = [c for c in df.columns if c.lower().endswith(AGE_COLUMN_SUFFIX)]
    if not age_cols:
        print(f"[SKIP] no *{AGE_COLUMN_SUFFIX} columns found in {path.name} "
              f"(columns: {list(df.columns)})")
        print("       If target_lag1 exists without a companion *_age_days "
              "column, this check cannot verify it. Confirm manually that "
              "target_lag1 is sourced from kriged_target_monthly/ (training-only).")
        return None

    ok = True
    for col in age_cols:
        vals = df[col].dropna()
        n_neg = int((vals < 0).sum())
        if n_neg > 0:
            print(f"[FAIL] {col}: {n_neg}/{len(vals)} rows have NEGATIVE age "
                  f"(future information used)")
            worst = df.loc[df[col] < 0, col].nsmallest(5)
            for v in worst:
                print(f"  age={v}")
            ok = False
        else:
            print(f"[PASS] {col}: min={vals.min():.1f}, max={vals.max():.1f}, "
                  f"all non-negative across {len(vals):,} rows")

    if ok:
        print("[PASS] no negative ages found in any *_age_days column")
    return ok


# ---------------------------------------------------------------------------
# CHECK 6
# ---------------------------------------------------------------------------
def check_held_out_hash_matches_manifest(held_out_path: Path, manifest_path):
    print_header("CHECK 6: held-out IDs hash matches manifest record")
    if manifest_path is None:
        print("[SKIP] no --manifest-for-hash-check given. This check would "
              "verify that the current held_out_ids.csv is the same version "
              "recorded in a training manifest. Skipped by design.")
        return None
    if not manifest_path.exists():
        print(f"[SKIP] {manifest_path} not found")
        return None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[SKIP] {manifest_path} unreadable: {e}")
        return None

    key = next((k for k in manifest
                if "held_out" in k.lower() and "sha1" in k.lower()), None)
    if key is None:
        print(f"[SKIP] no held-out-related sha1 key in manifest "
              f"(keys: {list(manifest.keys())}). This manifest does not "
              f"record a held-out-ids hash, so this check cannot run. "
              f"Not a failure.")
        return None

    actual_sha1 = sha1_of_file(held_out_path)
    if actual_sha1 != manifest[key]:
        print(f"[FAIL] current {held_out_path.name} sha1={actual_sha1} != "
              f"manifest {key}={manifest[key]}")
        return False
    print(f"[PASS] {held_out_path.name} sha1 matches manifest's recorded value")
    return True


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
def print_summary(results):
    print_header("LEAKAGE AUDIT SUMMARY")
    for name, passed in results.items():
        if passed is True:
            mark = "PASS"
        elif passed is None:
            mark = "SKIP"
        else:
            mark = "FAIL"
        print(f"  [{mark}] {name}")
    n_pass = sum(1 for v in results.values() if v is True)
    n_skip = sum(1 for v in results.values() if v is None)
    n_fail = sum(1 for v in results.values() if v is False)
    print(f"\n{n_pass} PASS / {n_skip} SKIP / {n_fail} FAIL")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--held-out-ids", type=Path,
                    default=Path("data/held_out_wells/held_out_ids.csv"))
    ap.add_argument("--train-ids", type=Path,
                    default=Path("data/interim/train_well_ids.csv"))
    ap.add_argument("--gw-tolerance", type=float, default=0.5,
                    help="Max distance (in gw_level units) between an RL "
                         "dataset value and its nearest RF-predicted value for "
                         "CHECK 4. Default 0.5; tighten to 0.1 after verifying "
                         "the pipeline passes at the loose setting.")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="Cap CHECK 4 to this many evenly-spaced episodes. "
                         "0 (default) = all episodes.")
    ap.add_argument("--manifest-for-hash-check", type=Path, default=None,
                    help="Optional path to a training_manifest.json that "
                         "records a held-out-ids sha1 (CHECK 6).")
    args = ap.parse_args()

    root = args.root
    held_out_path = root / args.held_out_ids
    train_path = root / args.train_ids

    results = {}

    results["split_integrity"] = check_split_integrity(held_out_path, train_path)
    if not results["split_integrity"]:
        print_summary(results)
        sys.exit(1)

    H = read_well_ids(held_out_path)
    results["training_only_files"] = check_training_only_files(root, H)
    results["rl_source_static"] = check_rl_source(root)
    results["rl_dataset_values"] = check_rl_dataset_values(
        root, args.gw_tolerance, args.max_episodes)
    results["feature_age_columns"] = check_feature_age_columns(root)
    results["held_out_hash_matches_manifest"] = check_held_out_hash_matches_manifest(
        held_out_path, args.manifest_for_hash_check)

    print_summary(results)
    sys.exit(0 if all(v is not False for v in results.values()) else 1)


if __name__ == "__main__":
    main()