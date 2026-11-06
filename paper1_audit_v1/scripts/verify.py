#!/usr/bin/env python3
"""verify.py — reproduce the paper's reported intervals from the per-seed
arrays in this archive, and compare against the summary JSONs.

Runs with numpy and scipy only. Does not need the project tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
ARRAYS = ARCHIVE / "per_seed_arrays"
SUMMARIES = ARCHIVE / "summaries"

# The paper prints contrasts, CIs and LOO values to four decimals and
# medians to two. Values rounded to two decimals can differ from the
# computed value by up to 0.005.
TOL_4DP = 5e-4
TOL_2DP = 5e-3


def t_interval(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def trimmed_mean(x: np.ndarray) -> float:
    return float(np.sort(x)[1:-1].mean())


def loo(x: np.ndarray) -> tuple[int, float, tuple[float, float]]:
    full = x.mean()
    means = np.array([np.delete(x, i).mean() for i in range(len(x))])
    idx = int(np.argmax(np.abs(means - full)))
    m, lo, hi = t_interval(np.delete(x, idx))
    return idx, m, (lo, hi)


class Checker:
    def __init__(self):
        self.failures = 0
        self.total = 0

    def close(self, name: str, got: float, want: float,
              tol: float = TOL_4DP) -> None:
        self.total += 1
        ok = abs(got - want) < tol
        if not ok:
            self.failures += 1
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {name:45s} got {got:+.4f}  want {want:+.4f}")

    def summary(self) -> None:
        print()
        if self.failures == 0:
            print(f"All {self.total} checks passed.")
        else:
            print(f"{self.failures} of {self.total} checks FAILED.")
        sys.exit(0 if self.failures == 0 else 1)


def check_matched_start(c: Checker) -> None:
    print("Matched-start contrast (Table VII / S1)")
    z = np.load(ARRAYS / "matched_start_225.npz")
    seeds = z["seeds"]
    roll = z["rollout"]
    tbl = json.loads((SUMMARIES / "table_vii.json").read_text())

    for tui, key in [(100, "fqe_100"), (200, "fqe_200"), (500, "fqe_500")]:
        fqe = z[key]
        d = fqe - roll
        m, lo, hi = t_interval(d)
        s = tbl[str(tui)]
        c.close(f"tui={tui}  contrast", m, s["contrast"])
        c.close(f"tui={tui}  CI low", lo, s["ci"][0])
        c.close(f"tui={tui}  CI high", hi, s["ci"][1])
        c.close(f"tui={tui}  sd", float(d.std(ddof=1)), s["sd"])
        c.close(f"tui={tui}  median", float(np.median(d)), s["median"],
                tol=TOL_2DP)
        c.close(f"tui={tui}  trimmed mean",
        trimmed_mean(d), s.get("trimmed", trimmed_mean(d)))
        idx, lm, (llo, lhi) = loo(d)
        c.close(f"tui={tui}  LOO mean (drop {seeds[idx]})", lm, s["loo"])
        c.close(f"tui={tui}  LOO CI low", llo, s["loo_ci"][0])
        c.close(f"tui={tui}  LOO CI high", lhi, s["loo_ci"][1])


def check_band_decomposition(c: Checker) -> None:
    print("Reward decomposition endpoint (Table XI)")
    z = np.load(ARRAYS / "band_decomposition.npz")
    d = z["endpoint_w0"] - z["endpoint_w05"]
    m, lo, hi = t_interval(d)
    s = json.loads((SUMMARIES / "table_xi.json").read_text())["endpoint"]
    c.close("endpoint removed - full", m, s["removed_minus_full"])
    c.close("endpoint CI low", lo, s["ci"][0])
    c.close("endpoint CI high", hi, s["ci"][1])


def report_family_grid(c: Checker) -> None:
    print("Family/start grid (Table IX) — summary only")
    z = np.load(ARRAYS / "family_start_grid.npz")
    print(f"  {len(z.files)} arrays present")
    s = json.loads((SUMMARIES / "table_ix.json").read_text())
    n_rows = len(s["rows"]) if "rows" in s else len(s)
    print(f"  summary rows: {n_rows}")
    c.total += 1
    if len(z.files) != n_rows:
        print(f"  [FAIL] array count {len(z.files)} != summary rows {n_rows}")
        c.failures += 1
    else:
        print(f"  [OK  ] array count matches summary rows")


def report_constructions(c: Checker) -> None:
    print("Construction-dependent cells (Table X) — summary only")
    s = json.loads((SUMMARIES / "table_x.json").read_text())
    cells = s if isinstance(s, list) else s.get("cells", [])
    # table_x.json holds all 24 distinct cells. The construction-dependent
    # subset is the one whose three intervals disagree on zero.
    n_all = len(cells)
    n_dep = sum(
        1 for cell in cells
        if not (cell["pooled_excludes"] == cell["seed_t_excludes"]
                == cell["hier_excludes"])
    )
    print(f"  cells in summary: {n_all}   construction-dependent: {n_dep}")
    c.total += 1
    if n_dep != 7:
        print(f"  [FAIL] expected 7 construction-dependent cells, got {n_dep}")
        c.failures += 1
    else:
        print(f"  [OK  ] seven construction-dependent cells present")


def report_bc_sweep(c: Checker) -> None:
    print("BC sweep (Table XII) — summary only")
    s = json.loads((SUMMARIES / "table_xii.json").read_text())
    grid = s.get("grid", [])
    n = len(grid)
    print(f"  cells in summary: {n}")
    c.total += 1
    if n != 18:
        print(f"  [FAIL] expected 18 sweep cells, got {n}")
        c.failures += 1
    else:
        signs_ok = (
            all(g["cql_minus_bc"] > 0 for g in grid if g["family"] == "base") and
            all(g["cql_minus_bc"] < 0 for g in grid if g["family"] == "epsilon") and
            all(g["cql_minus_bc"] > 0 for g in grid if g["family"] == "no_oracle")
        )
        if signs_ok:
            print(f"  [OK  ] sign pattern base>0, epsilon<0, no_oracle>0 holds")
        else:
            print(f"  [FAIL] sign pattern broken")
            c.failures += 1


def main() -> None:
    print(f"Archive: {ARCHIVE}")
    print()
    c = Checker()
    check_matched_start(c)
    print()
    check_band_decomposition(c)
    print()
    report_family_grid(c)
    print()
    report_constructions(c)
    print()
    report_bc_sweep(c)
    c.summary()


if __name__ == "__main__":
    main()