"""
plot_bc_sweep.py

Reads revision_experiments/bc_sweep.json and renders the Section V-A
hyperparameter-sensitivity figure: CQL-BC vs learning rate, one panel per
hidden-size configuration, one series per family, 95% CI as error bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "revision_experiments" / "bc_sweep.json"
OUT_PNG = REPO / "revision_experiments" / "fig_bc_sweep.png"
OUT_PDF = REPO / "revision_experiments" / "fig_bc_sweep.pdf"

FAMILIES = ["base", "epsilon", "no_oracle"]
COLORS = {"base": "#1f77b4", "epsilon": "#d62728", "no_oracle": "#2ca02c"}
MARKERS = {"base": "o", "epsilon": "s", "no_oracle": "^"}


def load():
    data = json.loads(JSON_PATH.read_text())
    grid = data["grid"]
    # Organize by (hidden, family) -> list of (lr, point, lo, hi)
    out = {}
    for row in grid:
        key = (tuple(row["hidden"]), row["family"])
        out.setdefault(key, []).append({
            "lr": row["lr"],
            "m": row["cql_minus_bc"],
            "lo": row["lo"],
            "hi": row["hi"],
        })
    # Sort each series by lr (descending, so 1e-3 is leftmost)
    for k in out:
        out[k].sort(key=lambda r: -r["lr"])
    return out


def main():
    series = load()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    hidden_configs = [(256, 256), (128, 128)]

    for ax, hidden in zip(axes, hidden_configs):
        # Dashed zero reference
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)

        for fam in FAMILIES:
            key = (hidden, fam)
            if key not in series:
                continue
            rows = series[key]
            lrs = [r["lr"] for r in rows]
            m = np.array([r["m"] for r in rows])
            lo = np.array([r["lo"] for r in rows])
            hi = np.array([r["hi"] for r in rows])
            # Asymmetric error bars
            yerr = np.vstack([m - lo, hi - m])

            ax.errorbar(
                lrs, m, yerr=yerr,
                color=COLORS[fam], marker=MARKERS[fam],
                markersize=7, lw=1.6, capsize=4,
                label=fam, alpha=0.9,
            )

        ax.set_xscale("log")
        # Explicit tick positions/order: 1e-3 -> 3e-4 -> 1e-4 (descending)
        ax.set_xticks([1e-3, 3e-4, 1e-4])
        ax.set_xticklabels([r"$10^{-3}$", r"$3{\cdot}10^{-4}$", r"$10^{-4}$"])
        ax.set_xlabel("BC learning rate")
        ax.set_title(f"hidden = {hidden[0]}\u00d7{hidden[1]}")

    axes[0].set_ylabel("CQL $-$ BC  (95% CI)")
    axes[0].legend(loc="best", frameon=False)

    # Shared y-range with a little headroom
    ymin = min(r["lo"] for v in series.values() for r in v)
    ymax = max(r["hi"] for v in series.values() for r in v)
    pad = 0.08 * (ymax - ymin)
    axes[0].set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(
        "Section V-A: sign of the CQL$-$BC gap is stable across the BC tuning grid",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()