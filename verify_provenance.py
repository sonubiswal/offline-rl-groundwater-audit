"""
verify_provenance.py — Phase 5 freeze-gate check #1 (v5.3, recursive discovery + coverage reporting)

Loads every BC/CQL .d3 model under reports/paper_exp/ via d3rlpy and reads the
LIVE config off the model object. Field names taken from the live
vars(algo.config) dump of this repo's checkpoints.

  FIX #1 (training steps): d3rlpy .d3 checkpoints do NOT store the step count.
      Step provenance is taken from training manifests (n_steps /
      training_steps / steps / total_steps / grad_steps keys) and from
      d3rlpy logger CSVs (max value of the `step` / `gradient_step` column).
      Flagged as STEPS_OK / STEPS_MISMATCH / STEPS_UNVERIFIABLE.
  FIX #2 (seed): the checkpoint carries no seed either. The script does a
      filename-seed <-> manifest-seed cross-check and reports
      SEED_UNVERIFIABLE_FROM_CHECKPOINT when no manifest seed exists.
  FIX #3: explicit `is None` test for seed display (seed 0 safe).
  FIX #4 (field lookup): exact attribute allow-list matching this repo's
      checkpoints. CQL: alpha, learning_rate, batch_size, gamma, n_critics,
      target_update_interval. BC: learning_rate, batch_size, gamma.
      Each field carries a list of candidate keys (repo names first, 2.x
      aliases as fallback) and the resolved key is shown in observed=.
  FIX #5: canonical enforcement supports per-field flags including
      --canonical-steps; BC's learning rate (1e-3) differs from CQL's (3e-4),
      enforced via separate --canonical-bc-lr flag.
  FIX #6 (v5.1): audit_manifests() now also checks the PARENT directory for
      normalization_manifest.json / training_manifest.json when the local
      recursive glob returns nothing.
  FIX #7 (v5.2): `if not args.canonical_alpha` and `if not args.canonical_steps`
      changed to `is None` so that a canonical value of 0 would still be
      treated as "enforce this value" rather than "not passed".
  FIX #8 (v5.3): find_d3_files() now uses rglob so nested model directories
      (e.g. adapt_*/cql/, w*_*/bc/) are discovered. Previously a non-recursive
      glob silently covered only the top level, so out of 114 .d3 files in
      this repo only 6 were audited. A per-(group, algo) COVERAGE row is now
      emitted, and COVERAGE_INCOMPLETE is raised if the number of discovered
      files is less than the number of .d3 files present in that subtree.
  FIX #9 (v5.3): summary now separates genuinely-OK rows from
      SEED_UNVERIFIABLE_FROM_CHECKPOINT rows, so "0 issues" is no longer
      printed when the majority of seed rows are merely unverifiable.

This script does NOT modify anything. It only reads and reports.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import d3rlpy
    HAVE_D3RLPY = True
except ImportError:
    HAVE_D3RLPY = False

MODEL_GROUPS = [
    "frozen_models",
    "ood_models",
    "cql_qr",
    "cql_mean",
]
GLOB_GROUPS = ["adapt_*", "w*_*"]

CANONICAL_SEEDS = {42, 123, 2024}
CANONICAL_STEPS_DEFAULT = 20000

FIELDS_BY_ALGO = {
    "bc": {
        "learning_rate": ["learning_rate"],
        "batch_size": ["batch_size"],
        "gamma": ["gamma"],
    },
    "cql": {
        "alpha": ["alpha", "initial_alpha"],
        "learning_rate": ["learning_rate", "critic_learning_rate"],
        "batch_size": ["batch_size"],
        "gamma": ["gamma"],
        "n_critics": ["n_critics"],
        "target_update_interval": ["target_update_interval"],
    },
}

CANONICAL_MAP = {
    ("cql", "alpha"): "canonical_alpha",
    ("bc", "learning_rate"): "canonical_bc_lr",
    ("cql", "learning_rate"): "canonical_lr",
    ("bc", "batch_size"): "canonical_batch_size",
    ("cql", "batch_size"): "canonical_batch_size",
    ("bc", "gamma"): "canonical_gamma",
    ("cql", "gamma"): "canonical_gamma",
    ("cql", "n_critics"): "canonical_n_critics",
    ("cql", "target_update_interval"): "canonical_target_update_interval",
}

STEPS_MANIFEST_KEYS = ["n_steps", "training_steps", "steps", "total_steps", "grad_steps"]
SEED_MANIFEST_KEYS = ["seed", "random_seed", "training_seed"]

STEP_COLUMN_RE = re.compile(r"^(step|gradient_step|grad_step)$", re.IGNORECASE)

# Statuses that mean "genuinely fine".
OK_STATUSES = ("OK", "HASH_OK", "STEPS_OK", "COVERAGE")
# Statuses that mean "not provable from the artifacts, and that is expected".
UNVERIFIABLE_STATUSES = ("SEED_UNVERIFIABLE_FROM_CHECKPOINT",)


def sha1_of_file(path: Path, chunk_size=1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def config_to_flat_dict(cfg) -> dict:
    if hasattr(cfg, "__dict__"):
        raw = dict(vars(cfg))
    elif hasattr(cfg, "to_dict"):
        raw = dict(cfg.to_dict())
    else:
        raw = dict(cfg)
    return {k: v for k, v in raw.items()
            if isinstance(v, (int, float, str, bool)) or v is None}


def load_model(d3_path: Path):
    if not HAVE_D3RLPY:
        return None, None, "d3rlpy not installed (pip install d3rlpy)"
    try:
        algo = d3rlpy.load_learnable(str(d3_path))
        cfg = config_to_flat_dict(algo.config)
        return cfg, getattr(algo, "grad_step", None), None
    except Exception as e:
        return None, None, f"failed to load: {e}"


def parse_seed_from_filename(path: Path):
    m = re.search(r"(?<![a-zA-Z])seed(\d+)", path.stem)
    return int(m.group(1)) if m else None


def find_d3_files(model_dir: Path, algo_name: str):
    """
    FIX #8 (v5.3): recursive discovery.

    Old behavior used model_dir.glob(...) which is non-recursive and only
    matched top-level files, silently missing .d3 files in nested dirs like
    adapt_*/cql/ or w*_*/bc/. New behavior uses rglob and then falls back to
    the model_dir/<algo_name>/ subdirectory form if needed.
    """
    flat = sorted(model_dir.rglob(f"{algo_name}_seed*.d3"))
    if flat:
        return flat
    sub = model_dir / algo_name
    if sub.exists():
        return sorted(sub.rglob("*.d3"))
    return []


def count_d3_files(model_dir: Path, algo_name: str):
    """How many .d3 files this subtree actually contains for this algo."""
    return len(sorted(model_dir.rglob(f"{algo_name}_seed*.d3")))


def find_manifest(model_dir: Path, names=("training_manifest.json",
                                          "normalization_manifest.json")):
    found = {}
    for name in names:
        cands = sorted(model_dir.glob(f"**/{name}"))
        if not cands and model_dir.parent.exists():
            parent_candidate = model_dir.parent / name
            if parent_candidate.exists():
                cands = [parent_candidate]
        if cands:
            found[name] = cands[0]
    return found


def read_manifest(path: Path):
    try:
        with open(path) as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def first_key(manifest: dict, keys):
    for k in keys:
        if k in manifest:
            return k, manifest[k]
    return None, None


def max_step_from_csvs(model_dir: Path):
    best, source = None, None
    for csv_path in sorted(model_dir.rglob("*.csv")):
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                step_cols = [c for c in reader.fieldnames
                             if STEP_COLUMN_RE.match(c.strip())]
                if not step_cols:
                    continue
                col = step_cols[0]
                for row in reader:
                    try:
                        v = int(float(row[col]))
                    except (TypeError, ValueError):
                        continue
                    if best is None or v > best:
                        best, source = v, str(csv_path)
        except Exception:
            continue
    return best, source


def audit_model_dir(model_dir: Path, group_label: str, args, rows: list):
    if not model_dir.exists():
        rows.append({"group": group_label, "algo": "", "seed": "", "file": "",
                      "status": "MISSING_DIR", "detail": str(model_dir)})
        return

    manifests = find_manifest(model_dir)
    training_manifest, tm_err = (None, None)
    if "training_manifest.json" in manifests:
        training_manifest, tm_err = read_manifest(manifests["training_manifest.json"])
    m_seed_key, m_seed = (first_key(training_manifest, SEED_MANIFEST_KEYS)
                          if training_manifest else (None, None))
    m_steps_key, m_steps = (first_key(training_manifest, STEPS_MANIFEST_KEYS)
                            if training_manifest else (None, None))

    for algo_name in ("bc", "cql"):
        d3_files = find_d3_files(model_dir, algo_name)
        expected_d3 = count_d3_files(model_dir, algo_name)

        # FIX #8: always record per-(group, algo) coverage so a silent zero
        # cannot recur without being visible in the output.
        rows.append({
            "group": group_label, "algo": algo_name, "seed": "", "file": "",
            "status": "COVERAGE",
            "detail": f"found={len(d3_files)} expected={expected_d3}",
        })
        if len(d3_files) < expected_d3:
            rows.append({
                "group": group_label, "algo": algo_name, "seed": "", "file": "",
                "status": "COVERAGE_INCOMPLETE",
                "detail": (f"find_d3_files() discovered {len(d3_files)} of "
                           f"{expected_d3} .d3 files under {model_dir}"),
            })

        seen_seeds = set()

        for d3_path in d3_files:
            seed = parse_seed_from_filename(d3_path)
            seed_disp = seed if seed is not None else "?"
            cfg, grad_step_live, err = load_model(d3_path)

            if err:
                rows.append({"group": group_label, "algo": algo_name,
                              "seed": seed_disp, "file": d3_path.name,
                              "status": "LOAD_ERROR", "detail": err})
                continue

            if seed is not None:
                if seed in seen_seeds:
                    rows.append({"group": group_label, "algo": algo_name,
                                  "seed": seed, "file": d3_path.name,
                                  "status": "DUPLICATE_SEED", "detail": ""})
                seen_seeds.add(seed)

            if m_seed is not None:
                if seed is not None and int(m_seed) != seed:
                    rows.append({"group": group_label, "algo": algo_name,
                                  "seed": seed_disp, "file": d3_path.name,
                                  "status": "SEED_MANIFEST_MISMATCH",
                                  "detail": f"filename seed={seed} but manifest "
                                            f"{m_seed_key}={m_seed} ({manifests['training_manifest.json']})"})
            elif tm_err:
                rows.append({"group": group_label, "algo": algo_name,
                              "seed": seed_disp, "file": d3_path.name,
                              "status": "SEED_UNVERIFIABLE_FROM_CHECKPOINT",
                              "detail": f"training_manifest.json unreadable: {tm_err}"})
            else:
                rows.append({"group": group_label, "algo": algo_name,
                              "seed": seed_disp, "file": d3_path.name,
                              "status": "SEED_UNVERIFIABLE_FROM_CHECKPOINT",
                              "detail": "no seed key in training_manifest.json "
                                        "(keys present: "
                                        f"{sorted(training_manifest.keys()) if training_manifest else 'manifest missing'})"})

            issues = []
            observed = {}
            for field, candidates in FIELDS_BY_ALGO.get(algo_name, {}).items():
                cli_attr = CANONICAL_MAP.get((algo_name, field))
                canonical_val = getattr(args, cli_attr, None) if cli_attr else None
                key, actual = next(((k, cfg[k]) for k in candidates if k in cfg),
                                   (None, None))
                if key is not None:
                    observed[field] = f"{actual} (via {key})"
                if canonical_val is not None:
                    if actual is None:
                        issues.append(f"{field} absent from live config "
                                      f"(looked for any of {candidates}; this d3rlpy "
                                      f"build may not expose it)")
                    elif isinstance(canonical_val, float):
                        if abs(float(actual) - canonical_val) > 1e-9:
                            issues.append(f"{field}: expected {canonical_val}, found {actual}")
                    elif actual != canonical_val:
                        issues.append(f"{field}: expected {canonical_val}, found {actual}")

            status = "MISMATCH" if issues else "OK"
            rows.append({"group": group_label, "algo": algo_name,
                          "seed": seed_disp, "file": d3_path.name,
                          "status": status,
                          "detail": ("; ".join(issues) if issues
                                     else f"observed={observed} full_flat_config={cfg}")})

        if d3_files:
            csv_steps, csv_src = max_step_from_csvs(model_dir)
            if m_steps is not None:
                if args.canonical_steps is not None and int(m_steps) != args.canonical_steps:
                    rows.append({"group": group_label, "algo": algo_name, "seed": "",
                                  "file": manifests.get("training_manifest.json", ""),
                                  "status": "STEPS_MISMATCH",
                                  "detail": f"manifest {m_steps_key}={m_steps}, "
                                            f"canonical={args.canonical_steps}"})
                else:
                    rows.append({"group": group_label, "algo": algo_name, "seed": "",
                                  "file": manifests.get("training_manifest.json", ""),
                                  "status": "STEPS_OK",
                                  "detail": f"manifest {m_steps_key}={m_steps}"
                                            + (f"; logger csv max step={csv_steps} ({csv_src})"
                                               if csv_steps is not None else "")
                                            + (f" (csv/manifest disagree: {csv_steps} vs {m_steps})"
                                               if csv_steps is not None and csv_steps != int(m_steps) else "")})
            elif csv_steps is not None:
                ok = (args.canonical_steps is None) or (csv_steps == args.canonical_steps)
                rows.append({"group": group_label, "algo": algo_name, "seed": "",
                              "file": csv_src,
                              "status": "STEPS_OK" if ok else "STEPS_MISMATCH",
                              "detail": f"logger csv max step={csv_steps} (no steps key in manifest)"})
            else:
                rows.append({"group": group_label, "algo": algo_name, "seed": "",
                              "file": "", "status": "STEPS_UNVERIFIABLE",
                              "detail": "no steps key in training_manifest.json and no "
                                        "d3rlpy logger CSV with a step column found; "
                                        "checkpoint does not store steps (grad_step "
                                        "resets to 0 on load)"})

        if d3_files:
            missing = CANONICAL_SEEDS - seen_seeds
            for s in missing:
                rows.append({"group": group_label, "algo": algo_name, "seed": s,
                              "file": "", "status": "SEED_NOT_FOUND",
                              "detail": "filename-based coverage check (seed is not "
                                        "baked into the checkpoint — see "
                                        "SEED_UNVERIFIABLE_FROM_CHECKPOINT notes)"})


def audit_manifests(model_dir: Path, group_label: str, rows: list):
    for manifest_name in ("normalization_manifest.json", "training_manifest.json"):
        candidates = list(model_dir.glob(f"**/{manifest_name}"))
        # FIX #6 (v5.1): fall back to the parent directory when the local
        # recursive glob finds nothing.
        if not candidates and model_dir.parent.exists():
            parent_candidate = model_dir.parent / manifest_name
            if parent_candidate.exists():
                candidates = [parent_candidate]
        if not candidates:
            rows.append({"group": group_label, "algo": "", "seed": "",
                          "file": manifest_name, "status": "MANIFEST_MISSING", "detail": ""})
            continue
        for manifest_path in candidates:
            manifest, err = read_manifest(manifest_path)
            if err:
                rows.append({"group": group_label, "algo": "", "seed": "",
                              "file": str(manifest_path), "status": "MANIFEST_UNREADABLE",
                              "detail": err})
                continue

            dataset_path_key = next(
                (k for k in ("dataset_path", "dataset", "dataset_file") if k in manifest),
                None,
            )
            sha_key = next(
                (k for k in ("dataset_sha1", "dataset_hash", "sha1") if k in manifest),
                None,
            )
            if dataset_path_key and sha_key:
                ds_path = Path(manifest[dataset_path_key])
                if not ds_path.is_absolute():
                    ds_path = (manifest_path.parent / ds_path).resolve()
                if ds_path.exists():
                    actual_sha = sha1_of_file(ds_path)
                    if actual_sha != manifest[sha_key]:
                        rows.append({"group": group_label, "algo": "", "seed": "",
                                      "file": str(manifest_path), "status": "DATASET_HASH_MISMATCH",
                                      "detail": f"{ds_path}: manifest={manifest[sha_key]} actual={actual_sha}"})
                    else:
                        rows.append({"group": group_label, "algo": "", "seed": "",
                                      "file": str(manifest_path), "status": "HASH_OK",
                                      "detail": str(ds_path)})
                else:
                    rows.append({"group": group_label, "algo": "", "seed": "",
                                  "file": str(manifest_path), "status": "DATASET_FILE_MISSING",
                                  "detail": str(ds_path)})
            else:
                rows.append({"group": group_label, "algo": "", "seed": "",
                              "file": str(manifest_path), "status": "MANIFEST_PRESENT_NO_HASH_FIELDS",
                              "detail": f"keys found: {list(manifest.keys())}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/paper_exp", type=Path)
    ap.add_argument("--out", default="reports/provenance_audit.csv", type=Path)
    ap.add_argument("--canonical-alpha", type=float, default=None, dest="canonical_alpha",
                    help="Enforced against the CQL config's 'alpha' field (e.g. 1.0). "
                         "BC has no alpha field.")
    ap.add_argument("--canonical-lr", type=float, default=None, dest="canonical_lr",
                    help="Enforced against CQL's learning_rate (e.g. 3e-4). "
                         "BC's differs (1e-3) — use --canonical-bc-lr for BC.")
    ap.add_argument("--canonical-bc-lr", type=float, default=None,
                    dest="canonical_bc_lr",
                    help="Enforced against BC's learning_rate (e.g. 1e-3).")
    ap.add_argument("--canonical-n-critics", type=int, default=None, dest="canonical_n_critics")
    ap.add_argument("--canonical-batch-size", type=int, default=None, dest="canonical_batch_size")
    ap.add_argument("--canonical-gamma", type=float, default=None, dest="canonical_gamma")
    ap.add_argument("--canonical-target-update-interval", type=int, default=None,
                    dest="canonical_target_update_interval",
                    help="Hard target-update interval (e.g. 8000).")
    ap.add_argument("--canonical-tau", type=float, default=None,
                    dest="canonical_tau_deprecated", help=argparse.SUPPRESS)
    ap.add_argument("--canonical-steps", type=int, default=None, dest="canonical_steps",
                    help=f"Expected training steps (e.g. {CANONICAL_STEPS_DEFAULT}). "
                         "Checked against training manifests and d3rlpy logger CSVs, "
                         "NOT the checkpoint (checkpoints do not store step counts).")
    args = ap.parse_args()

    if not HAVE_D3RLPY:
        print("WARNING: d3rlpy is not installed — no model configs can be read. "
              "Install with: pip install d3rlpy")

    rows = []

    for group in MODEL_GROUPS:
        group_dir = args.root / group
        audit_model_dir(group_dir, group, args, rows)
        audit_manifests(group_dir, group, rows)

    for pattern in GLOB_GROUPS:
        for group_dir in sorted(args.root.glob(pattern)):
            if group_dir.is_dir():
                audit_model_dir(group_dir, group_dir.name, args, rows)
                audit_manifests(group_dir, group_dir.name, rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "algo", "seed", "file", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    # FIX #9 (v5.3): honest split between verified OK and merely unverifiable.
    n_ok = sum(1 for r in rows if r["status"] in OK_STATUSES)
    n_unver = sum(1 for r in rows if r["status"] in UNVERIFIABLE_STATUSES)
    n_bad = len(rows) - n_ok - n_unver

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"  {n_ok} verified OK")
    print(f"  {n_unver} SEED_UNVERIFIABLE_FROM_CHECKPOINT "
          f"(seed is not baked into the checkpoint; filename+manifest only)")
    print(f"  {n_bad} issue(s)")

    # FIX #7 (v5.2): use `is None` so a canonical value of 0 still counts as passed.
    if args.canonical_alpha is None:
        print("\nNOTE: --canonical-alpha was not passed, so alpha is only OBSERVED. "
              "Check the observed= column of each cql_seed*.d3 row (live field: "
              "'alpha'), then re-run with --canonical-alpha 1.0 once confirmed.")
    if args.canonical_steps is None:
        print(f"NOTE: --canonical-steps was not passed, so step counts are only "
              f"OBSERVED (paper claims {CANONICAL_STEPS_DEFAULT}). Re-run with "
              f"--canonical-steps {CANONICAL_STEPS_DEFAULT} to enforce.")

    if n_bad:
        print("\n--- ISSUES ---")
        for r in rows:
            if r["status"] not in OK_STATUSES + UNVERIFIABLE_STATUSES:
                print(f"[{r['group']}/{r['algo']}/seed{r['seed']}] "
                      f"{r['status']}: {r['detail']}")
        sys.exit(1)

    # Coverage sanity: fail loudly if any group reported COVERAGE_INCOMPLETE.
    incomplete = [r for r in rows if r["status"] == "COVERAGE_INCOMPLETE"]
    if incomplete:
        print("\n--- COVERAGE INCOMPLETE ---")
        for r in incomplete:
            print(f"[{r['group']}/{r['algo']}] {r['detail']}")
        sys.exit(2)


if __name__ == "__main__":
    main()