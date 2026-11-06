"""
E7: immutable artifact manifest (SHA-256) for the data-availability statement.

Usage (from C:\\trishna-opal):
    python revision_experiments\\e7_make_manifest.py artifacts_manifest.csv ^
        data\\processed models_10seed reports_10seed revision_experiments scripts src
"""
import csv
import hashlib
import pathlib
import sys


EXT = {".h5", ".d3", ".json", ".npy", ".npz", ".csv", ".py", ".yaml", ".yml", ".md"}


def is_included(p: pathlib.Path) -> bool:
    if p.suffix.lower() in EXT:
        return True
    if p.name.endswith(".d3.scaler.json"):
        return True
    if p.name.endswith(".meta.json"):
        return True
    return False


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: e7_make_manifest.py out.csv dir1 [dir2 ...]")
    out = sys.argv[1]
    roots = sys.argv[2:]

    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "sha256"])
        for r in roots:
            root = pathlib.Path(r)
            if not root.exists():
                print(f"[skip] {r} (not found)")
                continue
            for p in sorted(root.rglob("*")):
                if not p.is_file():
                    continue
                if not is_included(p):
                    continue
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                w.writerow([p.as_posix(), p.stat().st_size, h])
                n += 1
    print(f"{n} files hashed -> {out}")


if __name__ == "__main__":
    main()