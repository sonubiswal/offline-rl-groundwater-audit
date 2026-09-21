"""
verify_ci.py — Phase 5 freeze-gate check #8 (CI / statistical consistency)

Verifies reported bootstrap CIs in Phase 5 summary CSVs.

Checks:
  1. Per-seed observed_difference == cql_return - bc_return (exact).
  2. Pooled summary algebraic consistency:
       pooled observed_difference == mean(per-seed observed_differences)
     This is ONLY an algebraic identity check. It does NOT prove that the
     reported pooled bootstrap CI was computed correctly.
  3. CI ordering: ci_lower <= observed_difference <= ci_upper.
  4. ci_contains_zero matches the sign of (ci_lower, ci_upper).
  5. p_positive agrees with the sign of observed_difference.
  6. If raw per-episode return arrays are present recursively under --root,
     recompute the paired bootstrap with seed=2026 and n_bootstrap=5000,
     then compare recomputed values against the reported pooled CI bounds.

If raw arrays are absent or cannot be mapped, checks 1-5 still run and
check 6 is reported as SKIP. Use --require-recompute for final Gate #8:
in that mode, any SKIP/FAIL from check 6 exits non-zero.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("reports/paper_exp")
BOOTSTRAP_SEED = 2026
N_BOOTSTRAP = 5000
CONFIDENCE = 0.95

SUMMARY_CSVS = [
    "scenario_summary.csv",
    "scenario_adaptation_summary.csv",
    "ood_summary.csv",
    "weight_ablation_summary.csv",
    "qr_vs_mean_returns.csv",
]

ATOL_EXACT = 1e-6
ATOL_CI = 1e-6


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Faithful copy of the inline paired_bootstrap_difference used by
# run_paper_experiments.py.
# ---------------------------------------------------------------------------
def paired_bootstrap_difference(returns_a, returns_b, n_bootstrap, seed,
                                confidence_level=0.95):
    returns_a = np.asarray(returns_a, dtype=np.float64)
    returns_b = np.asarray(returns_b, dtype=np.float64)
    if returns_a.shape != returns_b.shape:
        raise ValueError("Paired bootstrap requires equal-length return arrays.")
    if len(returns_a) < 2:
        raise ValueError("At least two episodes are required.")
    if not np.all(np.isfinite(returns_a)) or not np.all(np.isfinite(returns_b)):
        raise ValueError("Inputs contain non-finite values.")

    paired_difference = returns_a - returns_b
    observed_difference = float(np.mean(paired_difference))

    rng = np.random.RandomState(seed)
    sampled = rng.randint(0, len(paired_difference),
                          size=(n_bootstrap, len(paired_difference)))
    boot = np.mean(paired_difference[sampled], axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "observed_difference": observed_difference,
        "ci_lower": lower,
        "ci_upper": upper,
    }


# ---------------------------------------------------------------------------
# Group-column detection
# ---------------------------------------------------------------------------
def find_group_column(df):
    for cand in ("cell", "scenario"):
        if cand in df.columns:
            return cand
    return None


# ---------------------------------------------------------------------------
# Internal consistency checks
# ---------------------------------------------------------------------------
def check_per_seed_identity(df, csv_name, results):
    """Per-seed observed_difference == cql_return - bc_return."""
    if "kind" not in df.columns:
        results.append((csv_name, "per_seed_identity", "SKIP",
                        "no 'kind' column"))
        return

    per_seed = df[df["kind"] == "per_seed"]
    if per_seed.empty:
        results.append((csv_name, "per_seed_identity", "SKIP",
                        "no per_seed rows"))
        return

    if "bc_return" in per_seed.columns and "cql_return" in per_seed.columns:
        a_col, b_col = "cql_return", "bc_return"
    elif "qr_return" in per_seed.columns and "meanq_return" in per_seed.columns:
        a_col, b_col = "qr_return", "meanq_return"
    else:
        results.append((csv_name, "per_seed_identity", "SKIP",
                        f"no recognizable return columns: {list(per_seed.columns)}"))
        return

    bad = []
    for _, row in per_seed.iterrows():
        expected = float(row[a_col]) - float(row[b_col])
        got = float(row["observed_difference"])
        if abs(expected - got) > ATOL_EXACT:
            bad.append((row.get("cell", row.get("scenario", "?")),
                        row.get("seed", "?"), expected, got))

    if bad:
        results.append((csv_name, "per_seed_identity", "FAIL",
                        f"{len(bad)} mismatches, e.g. {bad[:2]}"))
    else:
        results.append((csv_name, "per_seed_identity", "PASS",
                        f"{len(per_seed)} per-seed rows verified"))


def check_pooled_summary_consistency(df, csv_name, results):
    """
    Pooled observed_difference == mean(per-seed observed_difference).

    IMPORTANT: This is an algebraic consistency check only. It does NOT
    independently verify that the reported pooled bootstrap CI is correct.
    """
    if "kind" not in df.columns:
        results.append((csv_name, "pooled_summary_consistency", "SKIP",
                        "no 'kind' column"))
        return

    pooled = df[df["kind"] == "pooled"]
    if pooled.empty:
        results.append((csv_name, "pooled_summary_consistency", "SKIP",
                        "no pooled rows"))
        return

    group_col = find_group_column(df)
    if group_col is None:
        results.append((csv_name, "pooled_summary_consistency", "SKIP",
                        "no group column (cell/scenario)"))
        return

    per_seed = df[df["kind"] == "per_seed"]
    bad = []
    for _, prow in pooled.iterrows():
        key = prow[group_col]
        seed_rows = per_seed[per_seed[group_col] == key]
        if seed_rows.empty:
            continue
        mean_perseed = float(seed_rows["observed_difference"].mean())
        pooled_val = float(prow["observed_difference"])
        if abs(mean_perseed - pooled_val) > ATOL_EXACT:
            bad.append((key, mean_perseed, pooled_val))

    if bad:
        results.append((csv_name, "pooled_summary_consistency", "FAIL",
                        f"{len(bad)} pooled rows mismatched, e.g. {bad[:2]}"))
    else:
        results.append((csv_name, "pooled_summary_consistency", "PASS",
                        f"{len(pooled)} pooled rows algebraically consistent"))


def check_ci_ordering(df, csv_name, results):
    if "ci_lower" not in df.columns or "ci_upper" not in df.columns:
        results.append((csv_name, "ci_ordering", "SKIP",
                        "no ci_lower/ci_upper columns"))
        return
    sub = df.dropna(subset=["ci_lower", "ci_upper", "observed_difference"])
    if sub.empty:
        results.append((csv_name, "ci_ordering", "SKIP", "no CI rows"))
        return
    bad_order = sub[
        (sub["ci_lower"] > sub["observed_difference"] + ATOL_EXACT)
        | (sub["ci_upper"] < sub["observed_difference"] - ATOL_EXACT)
    ]
    if len(bad_order) > 0:
        results.append((csv_name, "ci_ordering", "FAIL",
                        f"{len(bad_order)} rows with ci_lower > obs or obs > ci_upper"))
    else:
        results.append((csv_name, "ci_ordering", "PASS",
                        f"{len(sub)} rows ordered correctly"))


def check_ci_zero_consistency(df, csv_name, results):
    if not {"ci_lower", "ci_upper", "ci_contains_zero"}.issubset(df.columns):
        results.append((csv_name, "ci_contains_zero", "SKIP",
                        "missing ci_lower/ci_upper/ci_contains_zero"))
        return
    sub = df.dropna(subset=["ci_lower", "ci_upper", "ci_contains_zero"])
    if sub.empty:
        results.append((csv_name, "ci_contains_zero", "SKIP", "no rows"))
        return
    expected = (sub["ci_lower"] <= 0.0) & (sub["ci_upper"] >= 0.0)
    got = sub["ci_contains_zero"].astype(bool)
    mismatched = (expected != got).sum()
    if mismatched > 0:
        results.append((csv_name, "ci_contains_zero", "FAIL",
                        f"{mismatched} rows where ci_contains_zero "
                        f"does not match bound sign"))
    else:
        results.append((csv_name, "ci_contains_zero", "PASS",
                        f"{len(sub)} rows verified"))


def check_p_positive_sign(df, csv_name, results):
    if not {"observed_difference", "p_positive"}.issubset(df.columns):
        results.append((csv_name, "p_positive_sign", "SKIP",
                        "missing observed_difference or p_positive"))
        return
    sub = df.dropna(subset=["observed_difference", "p_positive"])
    if sub.empty:
        results.append((csv_name, "p_positive_sign", "SKIP", "no rows"))
        return

    mismatched = 0
    for _, row in sub.iterrows():
        obs = float(row["observed_difference"])
        p = float(row["p_positive"])
        if obs > 0 and p < 0.5:
            mismatched += 1
        elif obs < 0 and p > 0.5:
            mismatched += 1

    if mismatched > 0:
        results.append((csv_name, "p_positive_sign", "FAIL",
                        f"{mismatched} rows where p_positive disagrees with "
                        f"sign of observed_difference"))
    else:
        results.append((csv_name, "p_positive_sign", "PASS",
                        f"{len(sub)} rows verified"))


# ---------------------------------------------------------------------------
# Raw-array lookup and bootstrap recomputation
# ---------------------------------------------------------------------------
def find_raw_return_arrays(root: Path, exp_name: str):
    """
    Recursively search for per-episode return arrays under root.
    Prefers paths mentioning the experiment name; falls back to all candidates.
    """
    patterns = [
        "*returns*.npy", "*returns*.npz",
        "*rollout*returns*.npy", "*rollout*returns*.npz",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(root.rglob(pat))

    candidates = sorted(set(candidates))
    matched = [p for p in candidates if exp_name.lower() in str(p).lower()]
    if matched:
        return matched
    return candidates


def load_arm_arrays(candidates, arm, seeds, group_key=None):
    """
    Best-effort loader. Returns list of arrays, one per seed, or None if
    any seed/arm array cannot be located.

    Naming assumptions tried:
      - .npy file name contains arm and seed, e.g. cql_seed0.npy
      - .npz file contains a key containing arm
      - optionally group_key appears in the file name when multiple groups exist
    """
    arrays = []
    for seed in seeds:
        found = None
        for p in candidates:
            name = p.name.lower()
            if arm.lower() in name and str(seed) in name:
                if group_key is None or str(group_key).lower() in name:
                    found = p
                    break
        if found is None:
            return None

        obj = np.load(found, allow_pickle=False)
        if hasattr(obj, "files"):
            keys = list(obj.keys())
            key_match = [k for k in keys if arm.lower() in k.lower()]
            if not key_match:
                return None
            arr = obj[key_match[0]]
        else:
            arr = obj

        arr = np.asarray(arr, dtype=np.float64).ravel()
        arrays.append(arr)

    return arrays


def try_recompute_ci(root: Path, csv_name: str, df: pd.DataFrame, results):
    """
    Attempt to recompute the pooled paired bootstrap from raw arrays and
    compare against the reported pooled CI.
    """
    exp_name = csv_name.replace("_summary.csv", "").replace("_returns.csv", "")

    candidates = find_raw_return_arrays(root, exp_name)
    if not candidates:
        results.append((csv_name, "recompute_ci", "SKIP",
                        "no *returns*.npy / *.npz found recursively under "
                        f"{root}; raw per-episode arrays were not saved"))
        return

    if "cql_return" in df.columns and "bc_return" in df.columns:
        arm_a, arm_b = "cql", "bc"
    elif "qr_return" in df.columns and "meanq_return" in df.columns:
        arm_a, arm_b = "qr", "meanq"
    else:
        results.append((csv_name, "recompute_ci", "SKIP",
                        "cannot determine arm columns from CSV"))
        return

    if "kind" not in df.columns or "observed_difference" not in df.columns:
        results.append((csv_name, "recompute_ci", "SKIP",
                        "missing kind/observed_difference columns"))
        return

    pooled = df[df["kind"] == "pooled"]
    if pooled.empty:
        results.append((csv_name, "recompute_ci", "SKIP", "no pooled rows"))
        return

    group_col = find_group_column(df)
    if group_col is None:
        results.append((csv_name, "recompute_ci", "SKIP",
                        "no group column (cell/scenario)"))
        return

    per_seed = df[df["kind"] == "per_seed"]
    if per_seed.empty:
        results.append((csv_name, "recompute_ci", "SKIP", "no per_seed rows"))
        return

    multiple_groups = len(pooled) > 1
    verified = 0
    failures = []
    skips = []

    for _, prow in pooled.iterrows():
        group_key = prow[group_col]
        seed_rows = per_seed[per_seed[group_col] == group_key]
        if seed_rows.empty:
            skips.append(f"{group_key}: no per-seed rows")
            continue

        if "seed" in seed_rows.columns:
            seeds = seed_rows["seed"].tolist()
        else:
            seeds = list(range(len(seed_rows)))

        group_filter = group_key if multiple_groups else None

        arrays_a = load_arm_arrays(candidates, arm_a, seeds, group_filter)
        arrays_b = load_arm_arrays(candidates, arm_b, seeds, group_filter)

        if arrays_a is None or arrays_b is None:
            skips.append(f"{group_key}: could not locate arrays for all seeds/arms")
            continue

        returns_a = np.concatenate(arrays_a)
        returns_b = np.concatenate(arrays_b)

        try:
            recomputed = paired_bootstrap_difference(
                returns_a, returns_b,
                n_bootstrap=N_BOOTSTRAP,
                seed=BOOTSTRAP_SEED,
                confidence_level=CONFIDENCE,
            )
        except Exception as e:
            failures.append(f"{group_key}: recompute error {e}")
            continue

        reported_obs = float(prow["observed_difference"])
        reported_lower = float(prow["ci_lower"])
        reported_upper = float(prow["ci_upper"])

        if (
            abs(recomputed["observed_difference"] - reported_obs) > ATOL_CI
            or abs(recomputed["ci_lower"] - reported_lower) > ATOL_CI
            or abs(recomputed["ci_upper"] - reported_upper) > ATOL_CI
        ):
            failures.append(
                f"{group_key}: recomputed "
                f"({recomputed['observed_difference']:.6f}, "
                f"{recomputed['ci_lower']:.6f}, {recomputed['ci_upper']:.6f}) "
                f"!= reported ({reported_obs:.6f}, "
                f"{reported_lower:.6f}, {reported_upper:.6f})"
            )
        else:
            verified += 1

    if failures:
        results.append((csv_name, "recompute_ci", "FAIL",
                        f"{len(failures)} failures, e.g. {failures[:2]}"))
    elif verified > 0:
        results.append((csv_name, "recompute_ci", "PASS",
                        f"{verified} pooled rows recomputed and matched"))
    else:
        results.append((csv_name, "recompute_ci", "SKIP",
                        f"arrays found but no pooled rows verified: {skips[:2]}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Directory containing the Phase 5 summary CSVs.")
    ap.add_argument("--require-recompute", action="store_true",
                    help="Exit 1 if check #6 cannot be completed. Use for final Gate #8.")
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        print(f"FATAL: {root} does not exist")
        sys.exit(1)

    results = []

    for csv_name in SUMMARY_CSVS:
        path = root / csv_name
        if not path.exists():
            results.append((csv_name, "load", "SKIP", "file not found"))
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            results.append((csv_name, "load", "FAIL", f"unreadable: {e}"))
            continue

        print_header(f"Verifying {csv_name}")

        check_per_seed_identity(df, csv_name, results)
        check_pooled_summary_consistency(df, csv_name, results)
        check_ci_ordering(df, csv_name, results)
        check_ci_zero_consistency(df, csv_name, results)
        check_p_positive_sign(df, csv_name, results)
        try_recompute_ci(root, csv_name, df, results)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print_header("CI / STATISTICAL VERIFICATION SUMMARY")
    header = f"{'csv':<32} {'check':<28} {'status':<6} detail"
    print(header)
    print("-" * len(header))
    for csv_name, check, status, detail in results:
        print(f"{csv_name:<32} {check:<28} {status:<6} {detail}")

    n_fail = sum(1 for _, _, status, _ in results if status == "FAIL")
    n_pass = sum(1 for _, _, status, _ in results if status == "PASS")
    n_skip = sum(1 for _, _, status, _ in results if status == "SKIP")
    print(f"\n{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")

    recompute_not_passed = [
        r for r in results
        if r[1] == "recompute_ci" and r[2] != "PASS"
    ]

    if args.require_recompute and recompute_not_passed:
        print("\n--require-recompute set: Gate #8 cannot pass because "
              "recompute_ci did not PASS for all CSVs.")
        for csv_name, check, status, detail in recompute_not_passed:
            print(f"  [{csv_name}] {status}: {detail}")
        sys.exit(1)

    if n_fail > 0:
        print("\n--- FAILURES ---")
        for csv_name, check, status, detail in results:
            if status == "FAIL":
                print(f"[{csv_name}] {check}: {detail}")
        sys.exit(1)

    if args.require_recompute:
        print("\nGate #8 status: PASS (all checks passed and recompute_ci PASSed).")
    else:
        if recompute_not_passed:
            print("\nGate #8 status: PARTIAL — recompute_ci did not PASS for all CSVs. "
                  "Run with --require-recompute for the final gate.")
        else:
            print("\nGate #8 status: PASS (preliminary; recompute_ci PASSed where applicable).")

    sys.exit(0)


if __name__ == "__main__":
    main()