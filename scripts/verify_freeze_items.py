"""
verify_freeze_items.py

Checks the freeze-gate claims in the paper against the actual artifacts.

Reports one PASS/FAIL per item:
  (1) per-episode return arrays (.npy)
  (2) final checkpoints (.d3) and sidecars (.d3.scaler.json)
  (3) freeze manifest (FROZEN.json)
  (4) leakage verification (via check_leakage.py if runnable)

Usage:
    python scripts/verify_freeze_items.py C:\\trishna-opal
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def count(root: Path, pattern: str, recursive: bool = True) -> int:
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    return sum(1 for p in it if p.is_file())


def print_pass(label: str, actual: int, expected: str) -> bool:
    ok = str(actual) == expected or expected == "?"
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}: got {actual}, paper says {expected}")
    return ok


def main(root_str: str):
    root = Path(root_str).resolve()
    print(f"[verify_freeze] root = {root}\n")

    results = []

    # Item 1: per-episode return arrays
    npy_count = count(root, "*.npy")
    results.append(print_pass("per-episode .npy arrays", npy_count, "516"))

    # Where are they?
    for sub in ("reports", "reports_10seed", "_archive"):
        p = root / sub
        if p.exists():
            n = count(p, "*.npy")
            if n:
                print(f"        ({sub}/ contains {n})")

    # Item 2: final checkpoints and sidecars
    d3_total = count(root, "*.d3")
    scaler_total = count(root, "*.d3.scaler.json")
    print(f"[info] total .d3 files in tree: {d3_total}")
    print(f"[info] total .d3.scaler.json files in tree: {scaler_total}")

    # The paper distinguishes final vs intermediate checkpoints. Original programme
    # claimed 114 final. Count what actually exists under models*/.
    final_dirs = [
        root / "models", root / "models_epsilon",
        root / "models_no_oracle", root / "models_no_oracle_alpha1",
        root / "models_no_oracle_alpha10",
        root / "models_epsilon_qr32_verify", root / "models_epsilon_meanq_verify",
    ]
    final_count = sum(count(d, "*.d3", recursive=False) for d in final_dirs if d.exists())
    sidecar_count = sum(
        count(d, "*.d3.scaler.json", recursive=False)
        for d in final_dirs if d.exists()
    )
    results.append(print_pass("final .d3 checkpoints (top-level in models*/)", final_count, "114"))
    results.append(print_pass("sidecars next to final checkpoints", sidecar_count, "36"))

    # Retrained programme (10-seed)
    ten_seed = root / "models_10seed"
    if ten_seed.exists():
        n10 = count(ten_seed, "*.d3")
        s10 = count(ten_seed, "*.d3.scaler.json")
        print(f"[info] models_10seed/: {n10} .d3 files, {s10} sidecars")

    # Item 3: freeze manifest
    frozen = root / "FROZEN.json"
    if frozen.exists():
        try:
            j = json.loads(frozen.read_text())
            if isinstance(j, dict):
                n = len(j.get("files", j))
            elif isinstance(j, list):
                n = len(j)
            else:
                n = 0
            results.append(print_pass("FROZEN.json entry count", n, "280"))
        except Exception as e:
            print(f"[FAIL] FROZEN.json unreadable: {e}")
            results.append(False)
    else:
        print("[FAIL] FROZEN.json not found")
        results.append(False)

    # Item 4: leakage manifest
    manifest = root / "data" / "held_out_wells" / "held_out_ids.manifest.json"
    if manifest.exists():
        print(f"[PASS] leakage manifest present: {manifest.name}")
        results.append(True)
    else:
        print(f"[FAIL] leakage manifest missing: {manifest}")
        results.append(False)

    # Summary
    print()
    print(f"[verify_freeze] {sum(results)}/{len(results)} items PASS")
    if not all(results):
        print("[verify_freeze] The paper must be edited so its audit claims match")
        print("                only the items that PASS. Any item that FAILs is a")
        print("                candidate for a footnote or removal.")


if __name__ == "__main__":
    root_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root_arg)