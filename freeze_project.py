"""
freeze_project.py — Phase 5 freeze-gate (final snapshot + verifier)

Records SHA-1, size, and timestamp of every frozen artifact so the exact
state of Phase 1-5 can be restored or verified later. Two modes:

    python freeze_project.py freeze --root . --out FROZEN.md
        Walks the frozen paths, hashes every file, writes FROZEN.md plus a
        machine-readable FROZEN.json next to it.

    python freeze_project.py verify --root . --manifest FROZEN.json
        Re-hashes every recorded path and reports any file that has changed
        (hash mismatch), been deleted (missing), or been added (new file in
        a frozen directory).

Exit codes:
    freeze: 0 on success
    verify: 0 if the tree matches the manifest exactly, 1 otherwise.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Frozen path specification
# ---------------------------------------------------------------------------
# Each entry: (relative_glob, kind). "file" = literal file, "tree" = recursive
# directory walk, "pattern" = recursive glob under the root.
FROZEN_SPECS = [
    # Phase 5 result CSVs and summaries
    ("reports/paper_exp/scenario_summary.csv", "file"),
    ("reports/paper_exp/scenario_adaptation_summary.csv", "file"),
    ("reports/paper_exp/ood_summary.csv", "file"),
    ("reports/paper_exp/weight_ablation_summary.csv", "file"),
    ("reports/paper_exp/rf_sensitivity.csv", "file"),
    ("reports/paper_exp/qr_vs_mean_returns.csv", "file"),

    # Provenance + audit outputs
    ("reports/provenance_audit_final.csv", "file"),

    # RL dataset used by Phase 4/5
    ("data/processed/offline_rl_dataset.h5", "file"),

    # Split files that anchor every downstream experiment
    ("data/held_out_wells/held_out_ids.csv", "file"),
    ("data/interim/train_well_ids.csv", "file"),

    # Trained models and manifests (recursive under each top-level dir)
    ("reports/paper_exp/frozen_models", "tree"),
    ("reports/paper_exp/ood_models", "tree"),
    ("reports/paper_exp/cql_qr", "tree"),
    ("reports/paper_exp/cql_mean", "tree"),
    ("reports/paper_exp/adapt_drought_gw5", "tree"),
    ("reports/paper_exp/adapt_drought_gw15", "tree"),
    ("reports/paper_exp/adapt_drought_gw25", "tree"),
    ("reports/paper_exp/adapt_normal_gw5", "tree"),
    ("reports/paper_exp/adapt_normal_gw15", "tree"),
    ("reports/paper_exp/adapt_normal_gw25", "tree"),
    ("reports/paper_exp/adapt_high_gw5", "tree"),
    ("reports/paper_exp/adapt_high_gw15", "tree"),
    ("reports/paper_exp/adapt_high_gw25", "tree"),
    ("reports/paper_exp/w0.0_2.0_0.5", "tree"),
    ("reports/paper_exp/w1.0_0.0_0.0", "tree"),
    ("reports/paper_exp/w1.0_0.0_0.5", "tree"),
    ("reports/paper_exp/w1.0_1.0_0.5", "tree"),
    ("reports/paper_exp/w1.0_2.0_0.0", "tree"),
    ("reports/paper_exp/w1.0_2.0_0.5", "tree"),
    ("reports/paper_exp/w1.0_4.0_0.5", "tree"),

    # Frozen scripts
    ("run_paper_experiments.py", "file"),
    ("verify_provenance.py", "file"),
    ("check_leakage.py", "file"),
    ("verify_ci.py", "file"),
]

# Directories whose contents should be walked to detect unexpected additions.
# Used by `verify` to flag any file that was not in the manifest.
VERIFY_DIRS = [
    "reports/paper_exp",
    "data/processed",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_frozen_files(root: Path):
    """Yield (relative_path_str, absolute_path) for every frozen artifact."""
    seen = set()
    for rel, kind in FROZEN_SPECS:
        target = root / rel
        if kind == "file":
            if target.exists() and target.is_file():
                key = str(target.relative_to(root)).replace("\\", "/")
                if key not in seen:
                    seen.add(key)
                    yield key, target
        elif kind == "tree":
            if target.exists() and target.is_dir():
                for p in sorted(target.rglob("*")):
                    if p.is_file():
                        key = str(p.relative_to(root)).replace("\\", "/")
                        if key not in seen:
                            seen.add(key)
                            yield key, p


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def cmd_freeze(args):
    root = args.root.resolve()
    entries = []

    print(f"Freezing from root: {root}")
    for rel, path in iter_frozen_files(root):
        try:
            entries.append({
                "path": rel,
                "sha1": sha1_of_file(path),
                "size": path.stat().st_size,
                "mtime": int(path.stat().st_mtime),
            })
            print(f"  hashed {rel}")
        except Exception as e:
            print(f"  [FAIL] {rel}: {e}")
            return 1

    manifest = {
        "root": str(root),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(entries),
        "entries": entries,
    }

    # Machine-readable manifest
    json_out = Path(args.out).with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(manifest, indent=2))

    # Human-readable markdown
    md_lines = [
        "# Trishna-OPAL — Phase 1–5 Freeze Manifest",
        "",
        f"**Created (UTC):** {manifest['created_utc']}",
        f"**Root:** `{root}`",
        f"**Total files:** {manifest['n_files']}",
        "",
        "Any change to the files listed below invalidates the Phase 5 freeze.",
        "To verify, run:",
        "",
        "```powershell",
        f"python freeze_project.py verify --manifest {json_out.name}",
        "```",
        "",
        "| # | Path | SHA-1 | Size | Mtime (UTC) |",
        "|---|------|-------|------|-------------|",
    ]
    for i, e in enumerate(entries, 1):
        mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e["mtime"]))
        md_lines.append(
            f"| {i} | `{e['path']}` | `{e['sha1']}` | "
            f"{human_bytes(e['size'])} | {mtime} |"
        )
    md_lines.append("")

    Path(args.out).write_text("\n".join(md_lines))
    print(f"\nWrote {json_out}")
    print(f"Wrote {args.out}")
    print(f"{manifest['n_files']} files frozen.")
    return 0


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
def cmd_verify(args):
    root = args.root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"FATAL: manifest not found: {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    entries = {e["path"]: e for e in manifest["entries"]}

    mismatched = []
    missing = []

    print(f"Verifying against {manifest_path}")
    print(f"Manifest root: {manifest['root']}")
    print(f"Manifest files: {len(entries)}")
    print()

    for rel, e in sorted(entries.items()):
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        actual = sha1_of_file(p)
        if actual != e["sha1"]:
            mismatched.append((rel, e["sha1"], actual))

    # Detect additions: any file under VERIFY_DIRS not in the manifest.
    additions = []
    for d in VERIFY_DIRS:
        dp = root / d
        if not dp.exists():
            continue
        for p in sorted(dp.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(root)).replace("\\", "/")
                if rel not in entries:
                    additions.append(rel)

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    print("=" * 72)
    print("VERIFY SUMMARY")
    print("=" * 72)
    print(f"  unchanged:  {len(entries) - len(mismatched) - len(missing)}")
    print(f"  mismatched: {len(mismatched)}")
    print(f"  missing:    {len(missing)}")
    print(f"  added:      {len(additions)}")

    if mismatched:
        print("\n--- MISMATCHED ---")
        for rel, expected, actual in mismatched:
            print(f"  {rel}")
            print(f"    expected sha1: {expected}")
            print(f"    actual   sha1: {actual}")

    if missing:
        print("\n--- MISSING ---")
        for rel in missing:
            print(f"  {rel}")

    if additions:
        print("\n--- ADDED (not in manifest) ---")
        for rel in additions:
            print(f"  {rel}")

    clean = not (mismatched or missing or additions)
    if clean:
        print("\n[PASS] every frozen file matches the manifest; no additions.")
        return 0
    print("\n[FAIL] the frozen tree does not match the manifest.")
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_freeze = sub.add_parser("freeze", help="Write FROZEN.md + FROZEN.json")
    p_freeze.add_argument("--root", type=Path, default=Path("."))
    p_freeze.add_argument("--out", type=Path, default=Path("FROZEN.md"))
    p_freeze.set_defaults(func=cmd_freeze)

    p_verify = sub.add_parser("verify", help="Re-hash and compare against manifest")
    p_verify.add_argument("--root", type=Path, default=Path("."))
    p_verify.add_argument("--manifest", type=Path, default=Path("FROZEN.json"))
    p_verify.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()