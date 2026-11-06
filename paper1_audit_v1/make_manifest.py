#!/usr/bin/env python3
"""make_manifest.py — write MANIFEST.csv for the archive.

Walks every file under the archive root (excluding MANIFEST.csv itself),
records path, size and SHA-256, sorted by path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MANIFEST.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "MANIFEST.csv":
            continue
        rel = p.relative_to(ROOT).as_posix()
        rows.append((rel, p.stat().st_size, sha256(p)))

    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("path,bytes,sha256\n")
        for rel, size, h in rows:
            f.write(f"{rel},{size},{h}\n")

    print(f"wrote {OUT}  ({len(rows)} files)")


if __name__ == "__main__":
    main()