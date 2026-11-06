"""
Run compare_policies.py for all configs against the 10-seed models_10seed/ tree.
Writes results to reports_10seed/<config>/.

Usage:
    python run_10seed_compare.py                  # all configs
    python run_10seed_compare.py --configs base epsilon
    python run_10seed_compare.py --skip-existing  # resume after a crash
    python run_10seed_compare.py --dry-run        # preflight only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
N_EPISODES = 500

# name -> (dataset, bc_dir, cql_dir)
CONFIGS = {
    "base":          ("data/processed/offline_rl_dataset.h5",
                      "models_10seed/base/bc", "models_10seed/base/cql"),
    "epsilon":       ("data/processed/offline_rl_dataset_epsilon.h5",
                      "models_10seed/epsilon/bc", "models_10seed/epsilon/cql"),
    "no_oracle":     ("data/processed/offline_rl_dataset_no_oracle.h5",
                      "models_10seed/no_oracle/bc", "models_10seed/no_oracle/cql"),
    "no_oracle_a1":  ("data/processed/offline_rl_dataset_no_oracle.h5",
                      "models_10seed/no_oracle_a1/bc", "models_10seed/no_oracle_a1/cql"),
    "no_oracle_a10": ("data/processed/offline_rl_dataset_no_oracle.h5",
                      "models_10seed/no_oracle_a10/bc", "models_10seed/no_oracle_a10/cql"),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def preflight(name: str, dataset: str, bc_dir: str, cql_dir: str) -> list[str]:
    """Return a list of problems (empty list = OK)."""
    problems = []
    if not (REPO / dataset).is_file():
        problems.append(f"dataset missing: {dataset}")
    for kind, d in (("bc", bc_dir), ("cql", cql_dir)):
        for s in SEEDS:
            p = REPO / d / f"{kind}_seed{s}.d3"
            if not p.is_file():
                problems.append(f"missing checkpoint: {d}/{kind}_seed{s}.d3")
    return problems


def run_config(name: str, dataset: str, bc_dir: str, cql_dir: str) -> int:
    out_dir = REPO / "reports_10seed" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    env = dict(os.environ)
    env["TRAINING_SEEDS_OVERRIDE"] = ",".join(map(str, SEEDS))
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "src/rl/compare_policies.py",
        "--dataset", dataset,
        "--bc-dir", bc_dir,
        "--cql-dir", cql_dir,
        "--output-dir", str(out_dir),
        "--n-episodes", str(N_EPISODES),
    ]

    t0 = time.time()
    # Stream output live AND keep the full log on disk.
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()

    meta = {
        "config": name,
        "dataset": dataset,
        "bc_dir": bc_dir,
        "cql_dir": cql_dir,
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "n_episodes": N_EPISODES,
        "command": cmd,
        "git_commit": git_commit(),
        "returncode": rc,
        "started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    default=list(CONFIGS))
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip configs whose run_meta.json has returncode 0")
    ap.add_argument("--dry-run", action="store_true",
                    help="run preflight checks only")
    args = ap.parse_args()

    # Preflight everything first so a missing file fails fast, not mid-batch.
    bad = False
    for name in args.configs:
        problems = preflight(name, *CONFIGS[name])
        if problems:
            bad = True
            print(f"[preflight] {name}: {len(problems)} problem(s)")
            for p in problems[:10]:
                print(f"    - {p}")
    if bad:
        print("[preflight] FAILED; fix the above before running.")
        return 2
    print(f"[preflight] OK for {args.configs} ({len(SEEDS)} seeds each)")
    if args.dry_run:
        return 0

    results: dict[str, str] = {}
    for name in args.configs:
        meta_path = REPO / "reports_10seed" / name / "run_meta.json"
        if args.skip_existing and meta_path.is_file():
            try:
                if json.loads(meta_path.read_text()).get("returncode") == 0:
                    print(f"[10seed-compare] {name}: already done, skipping")
                    results[name] = "skipped"
                    continue
            except Exception:
                pass

        print(f"\n{'=' * 70}\n[10seed-compare] config={name}\n{'=' * 70}")
        rc = run_config(name, *CONFIGS[name])
        results[name] = "ok" if rc == 0 else f"FAILED (rc={rc})"
        # keep going: one failed config should not discard the others

    print("\n[10seed-compare] summary")
    for name, status in results.items():
        print(f"  {name:<15} {status}")
    return 0 if all(s in ("ok", "skipped") for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())