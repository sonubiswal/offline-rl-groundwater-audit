#!/usr/bin/env python3
"""make_figures.py — regenerate all manuscript figures from the
collision-free revision artifacts.

Outputs (figures/):
    fig1_estimator_contrast.pdf   per-seed FQE vs rollout at interval 200
    fig2_interval_sensitivity.pdf FQE-rollout contrast across three intervals
    fig3_family_start_forest.pdf  9 pairings x 3 protocols
    fig4_band_gaps.pdf            reward decomposition, four bands
    fig5_bc_sweep.pdf             18-cell BC tuning grid

Reads from revision_experiments/:
    e3c_test225_constructions_rerun.json
    E2/e2_fqe.json
    E2/seeds/tui{100,200,500}/seed_*.npz
    family_start_rerun.json
    ood_w3_{zero,full}_eval_only.json
    bc_sweep.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(r"C:\trishna-opal")
REV = REPO / "revision_experiments"
FIGDIR = REPO / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
INTERVALS = [100, 200, 500]

FAMS = ["base", "epsilon", "no_oracle"]
PROTOCOLS = ["rf_grid", "common_eps", "own_family"]
PROTOCOL_LABELS = {"rf_grid": "RF-grid (500)",
                   "common_eps": "Common epsilon (225)",
                   "own_family": "Own-family test"}

C_BASE = "#1F5A96"
C_EPS = "#C1272D"
C_NOO = "#2E8B57"
C_FQE = "#1F5A96"
C_ROLL = "#C1272D"
C_RERUN = "#2E8B57"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------- loaders
def _load_fqe_per_seed(tui: int) -> np.ndarray:
    """Ten per-seed FQE CQL-BC diffs at the given interval."""
    if tui == 200:
        p = REV / "E2" / "e2_fqe.json"
        if not p.exists():
            p = REV / "e2_fqe.json"
        d = json.loads(p.read_text())
        return np.array(d["primary_episode_starts"]["per_seed_diffs"])
    out = []
    for s in SEEDS:
        z = np.load(REV / "E2" / "seeds" / f"tui{tui}" / f"seed_{s}.npz")
        out.append(float((z["cql_q_starts"] - z["bc_q_starts"]).mean()))
    return np.array(out)


def _load_rollout_rerun() -> tuple[list[int], np.ndarray]:
    p = REV / "e3c_test225_constructions_rerun.json"
    d = json.loads(p.read_text())
    seeds = [int(r["seed"]) for r in d["per_seed"]]
    gaps = np.array([float(r["mean_gap"]) for r in d["per_seed"]])
    return seeds, gaps


def _t_ci(x: np.ndarray) -> tuple[float, float, float]:
    from scipy import stats
    n = len(x)
    m = float(x.mean())
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


# ---------------------------------------------------------------- Fig 1
def fig1_estimator_contrast():
    seeds, roll = _load_rollout_rerun()
    fqe = _load_fqe_per_seed(200)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    x = np.arange(len(seeds))
    w = 0.38
    ax.bar(x - w / 2, fqe, w, color=C_FQE, label="FQE", alpha=0.9)
    ax.bar(x + w / 2, roll, w, color=C_ROLL, label="Rollout (rerun)",
           alpha=0.9)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("CQL$-$BC (per-seed mean)")
    ax.set_title("Matched-start per-seed gap, interval 200")
    ax.legend(frameon=False, loc="upper right")

    fm, fl, fh = _t_ci(fqe)
    rm, rl, rh = _t_ci(roll)
    print(f"[fig1] FQE  mean {fm:+.4f}  CI [{fl:+.4f}, {fh:+.4f}]")
    print(f"[fig1] Roll mean {rm:+.4f}  CI [{rl:+.4f}, {rh:+.4f}]")

    fig.savefig(FIGDIR / "fig1_estimator_contrast.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2_interval_sensitivity():
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ypos = np.arange(len(INTERVALS))
    for i, tui in enumerate(INTERVALS):
        fqe = _load_fqe_per_seed(tui)
        _, roll = _load_rollout_rerun()
        d = fqe - roll
        m, lo, hi = _t_ci(d)
        color = C_RERUN if tui == 200 else "#888888"
        lw = 2.0 if tui == 200 else 1.2
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o",
                    color=color, lw=lw, capsize=3, markersize=5)
        print(f"[fig2] tui={tui:>3}  contrast {m:+.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]")
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t} (recorded)" if t == 200 else str(t)
                        for t in INTERVALS])
    ax.set_xlabel("Paired contrast (FQE$-$rollout)")
    ax.set_title("Target-update-interval sensitivity")
    ax.invert_yaxis()

    fig.savefig(FIGDIR / "fig2_interval_sensitivity.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig3_family_start_forest():
    d = json.loads((REV / "family_start_rerun.json").read_text())
    rows = d["rows"]

    # recompute_family_start.py writes flat rows
    # {bc, cql, protocol, mean}. Detect that layout; fall back to the
    # wide layout {bc, cql, rf_grid, common_eps, own_family} otherwise.
    if all("protocol" in r for r in rows):
        lookup = {(r["bc"], r["cql"], r["protocol"]): r["mean"]
                  for r in rows}
    else:
        lookup = {(r["bc"], r["cql"], proto): r[proto]
                  for r in rows
                  for proto in PROTOCOLS if proto in r}

    pairings = [(bc, cql) for bc in FAMS for cql in FAMS]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True)

    for ax, proto in zip(axes, PROTOCOLS):
        ys = np.arange(len(pairings))
        for i, (bc, cql) in enumerate(pairings):
            m = lookup.get((bc, cql, proto))
            if m is None:
                continue
            color = {"base": C_BASE, "epsilon": C_EPS,
                     "no_oracle": C_NOO}[bc]
            ax.plot(m, i, "o", color=color, markersize=5)
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{bc} / {cql}" for bc, cql in pairings])
        ax.set_title(PROTOCOL_LABELS[proto])
        ax.invert_yaxis()
        ax.set_xlabel("CQL$-$BC")

    handles = [Line2D([0], [0], marker="o", ls="", color=C_BASE,
                      label="base BC"),
               Line2D([0], [0], marker="o", ls="", color=C_EPS,
                      label="epsilon BC"),
               Line2D([0], [0], marker="o", ls="", color=C_NOO,
                      label="no-oracle BC")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Family-by-start forest", y=1.02)

    fig.savefig(FIGDIR / "fig3_family_start_forest.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def _band_dict(path: Path) -> dict:
    """Return the `bands` dict from an ood_w3_*_eval_only.json file:
    {in_dist|mild|moderate|severe: {obs, lo, hi}}."""
    d = json.loads(path.read_text())
    bands = d.get("bands")
    if not isinstance(bands, dict):
        raise SystemExit(f"{path}: no 'bands' dict")
    return bands


def fig4_band_gaps():
    rem = _band_dict(REV / "ood_w3_zero_eval_only.json")
    full = _band_dict(REV / "ood_w3_full_eval_only.json")

    bands = ["in_dist", "mild", "moderate", "severe"]
    labels = ["in-dist", "mild", "moderate", "severe"]

    x = np.arange(len(bands))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for j, (d, col, lbl) in enumerate(
            [(rem, C_BASE, "$w_3 = 0$ (removed)"),
             (full, C_ROLL, "$w_3 = 0.5$ (full)")]):
        means = np.array([d[b]["obs"] for b in bands])
        los = np.array([d[b]["lo"] for b in bands])
        his = np.array([d[b]["hi"] for b in bands])
        err = np.vstack([means - los, his - means])
        ax.errorbar(x + (j - 0.5) * w, means, yerr=err, fmt="o",
                    color=col, label=lbl, capsize=3, markersize=5)

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Mahalanobis distance band")
    ax.set_ylabel("CQL$-$BC gap (return)")
    ax.set_title("Reward decomposition")
    ax.legend(frameon=False, loc="upper left")

    # Console cross-check against Table XI.
    for b in bands:
        print(f"[fig4] {b:>9}  removed {rem[b]['obs']:+.4f}  "
              f"full {full[b]['obs']:+.4f}")

    fig.savefig(FIGDIR / "fig4_band_gaps.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5_bc_sweep():
    p = REV / "bc_sweep.json"
    d = json.loads(p.read_text())
    cells = d.get("grid")
    if not isinstance(cells, list):
        raise SystemExit(f"{p}: no 'grid' list")

    by_fam: dict[str, list] = {f: [] for f in FAMS}
    for c in cells:
        fam = c.get("family")
        if fam not in FAMS:
            continue
        by_fam[fam].append({
            "lr": float(c["lr"]),
            "hidden": tuple(int(h) for h in c["hidden"]),
            "mean": float(c["cql_minus_bc"]),
            "lo": float(c["lo"]),
            "hi": float(c["hi"]),
        })

    # x-ordering: (lr desc, hidden size asc), matching the manuscript
    # table layout.
    combos = sorted({(c["lr"], c["hidden"])
                     for fam in FAMS for c in by_fam[fam]},
                    key=lambda k: (-k[0], k[1]))
    xpos = {c: i for i, c in enumerate(combos)}

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fam_col = {"base": C_BASE, "epsilon": C_EPS, "no_oracle": C_NOO}
    fam_mk = {"base": "o", "epsilon": "s", "no_oracle": "^"}

    for fam in FAMS:
        xs, ms, los, his = [], [], [], []
        for c in by_fam[fam]:
            key = (c["lr"], c["hidden"])
            if key not in xpos:
                continue
            xs.append(xpos[key])
            ms.append(c["mean"])
            los.append(c["lo"])
            his.append(c["hi"])
        if not xs:
            continue
        ms = np.array(ms)
        err = np.vstack([ms - np.array(los), np.array(his) - ms])
        ax.errorbar(xs, ms, yerr=err, fmt=fam_mk[fam], color=fam_col[fam],
                    capsize=3, markersize=6, linestyle="none",
                    label=fam)

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{lr:g}\n{h[0]}$\\times${h[1]}"
                        for lr, h in combos], fontsize=7)
    ax.set_xlabel("BC learning rate / hidden units")
    ax.set_ylabel("CQL$-$BC (RF-grid)")
    ax.set_title("BC tuning sweep")
    ax.legend(frameon=False, loc="best")

    # Console cross-check against Table XII.
    for fam in FAMS:
        vals = [c["mean"] for c in by_fam[fam]]
        print(f"[fig5] {fam:>9}: n={len(vals)}  "
              f"range [{min(vals):+.4f}, {max(vals):+.4f}]")

    fig.savefig(FIGDIR / "fig5_bc_sweep.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    fig1_estimator_contrast()
    fig2_interval_sensitivity()
    fig3_family_start_forest()
    fig4_band_gaps()
    fig5_bc_sweep()
    print(f"\nwrote five PDFs to {FIGDIR}")


if __name__ == "__main__":
    main()