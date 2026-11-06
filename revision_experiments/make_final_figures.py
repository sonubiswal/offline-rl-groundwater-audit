"""
make_professional_figures.py  (v2 - layout fixes)

Regenerates the paper figures in one consistent, publication-ready style.
Writes PNG (300 dpi) + PDF (vector, embedded TrueType fonts) to figures_final/.

  fig1  verdict landscape (forest plot)
  fig2  pipeline schematic (redrawn here -- no more crossing arrows)
  fig3  banded OOD diagnostic
  fig4  reward ablation            (copied from figures/, source script not in this file)
  fig5  A4/A5 heatmap              (redrawn from reports/a4a5_sensitivity.json)
  fig6  E6 within-reward sensitivity

Usage (repo root):  python revision_experiments/make_professional_figures.py
Figure titles are OFF by default (put them in the paper caption); set TITLES = True to show them.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

TITLES = False

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "pdf.fonttype": 42, "ps.fonttype": 42,           # embed TrueType (journal requirement)
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "regular",
    "axes.labelsize": 9.5, "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#1a1a1a", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "lines.linewidth": 1.5, "grid.color": "#E5E7E9", "grid.linewidth": 0.6,
    "mathtext.default": "regular",
})

C_POS, C_NEG, C_TIE = "#C0392B", "#2874A6", "#7F8C8D"
C_NOCI = "#4D5656"
C_QR, C_MEANQ = "#C0392B", "#2874A6"
C_GRID, C_TEXT_DIM = "#E5E7E9", "#566573"
INK = "#1a1a1a"

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "figures"
OUT = REPO / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, pdf=True):
    fig.savefig(OUT / f"{name}.png")
    if pdf:
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"[{name}] wrote {name}.png" + (f", {name}.pdf" if pdf else ""))


def fmt(v):
    return f"{v:+.2f}".replace("-", "\u2212")


# ===========================================================================
# Readers (unchanged logic)
# ===========================================================================
def read_rollout(cfg):
    j = json.loads((REPO / "reports_10seed" / cfg / "policy_comparison.json").read_text())
    d = j.get("cql_minus_bc_paired_bootstrap") or j.get("paired_bootstrap")
    return float(d["observed_difference"]), float(d["ci_lower"]), float(d["ci_upper"])


def read_fqe():
    for p in (REPO / "revision_experiments" / "E2" / "e2_fqe.json",
              REPO / "revision_experiments" / "e2_fqe.json"):
        if p.exists():
            d = json.loads(p.read_text())["primary_episode_starts"]
            return float(d["seed_mean"]), float(d["seed_ci_lower"]), float(d["seed_ci_upper"])
    raise FileNotFoundError("e2_fqe.json not found")


def _t_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else np.nan), np.nan, np.nan
    m, sd = float(x.mean()), float(x.std(ddof=1))
    try:
        from scipy import stats
        crit = float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        crit = 2.262 if n == 10 else 1.96
    h = crit * sd / np.sqrt(n)
    return m, m - h, m + h


def read_ood_bands(path):
    j = json.loads(path.read_text())
    summary = j.get("summary", {})
    per = j.get("per_policy", {}) or j.get("per_policy_per_band", {})
    bands = ("id", "mild", "moderate", "severe")
    s_bc = {b: [] for b in bands}
    s_cql = {b: [] for b in bands}
    for key, d in per.items():
        parts = key.split("|")
        if len(parts) != 2:
            continue
        tgt = s_bc if parts[0].lower() == "bc" else s_cql
        for b in bands:
            if b in d:
                tgt[b].append(float(d[b]))
    out = {}
    for b in bands:
        if b not in summary:
            continue
        obs = float(summary[b].get("cql_minus_bc", summary[b].get("delta_cql_minus_bc", np.nan)))
        if len(s_bc[b]) >= 2 and len(s_cql[b]) == len(s_bc[b]):
            _, lo, hi = _t_ci(np.array(s_cql[b]) - np.array(s_bc[b]))
            out[b] = (obs, lo, hi)
        else:
            out[b] = (obs, None, None)
    return out


def read_a4a5_normal():
    j = json.loads((REPO / "reports" / "a4a5_sensitivity.json").read_text())
    out = {}
    for row in j:
        if row.get("scenario") != "normal":
            continue
        lo, hi = row.get("ci_lower"), row.get("ci_upper")
        out[(float(row["a4"]), float(row["a5"]))] = (
            float(row["cql_minus_bc"]),
            float(lo) if lo is not None else None,
            float(hi) if hi is not None else None)
    return out


def read_e6():
    for c in (REPO / "revision_experiments" / "e6_within_reward.json",
              REPO / "revision_experiments" / "E6" / "e6_within_reward.json"):
        if c.exists():
            return json.loads(c.read_text())
    raise FileNotFoundError("e6_within_reward.json not found")


# ===========================================================================
# Figure 1 -- verdict landscape
# ===========================================================================
def fig1():
    rows = []                       # ("H", title) or ("D", label, (obs, lo, hi))

    def header(t): rows.append(("H", t))
    def data(l, p): rows.append(("D", l, p))

    header("Dataset configuration")
    for cfg, disp in [("base", "base"), ("epsilon", "ε"), ("no_oracle", "no_oracle"),
                      ("no_oracle_a1", "no_oracle, α = 1"), ("no_oracle_a10", "no_oracle, α = 10")]:
        try: data(disp, read_rollout(cfg))
        except FileNotFoundError: pass

    header("Estimator (ε configuration)")
    for disp, fn in [("rollout", lambda: read_rollout("epsilon")), ("FQE", read_fqe)]:
        try: data(disp, fn())
        except FileNotFoundError: pass

    header("State band (QR-32 critic)")
    try:
        bands = read_ood_bands(REPO / "reports" / "ood_return_test.json")
        for b, disp in [("id", "in-distribution"), ("mild", "mild"),
                        ("moderate", "moderate"), ("severe", "severe OOD")]:
            if b in bands: data(disp, bands[b])
    except FileNotFoundError: pass

    header("Reward parameters (normal scenario)")
    try:
        a = read_a4a5_normal()
        for a4 in (10, 15, 25):
            for a5 in (3.0, 2.0, 1.0):
                if (a4, a5) in a:
                    data(f"A4 = {a4}, A5 = {a5:.1f}", a[(a4, a5)])
    except FileNotFoundError: pass

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.30 * n + 1.6))

    y, ticks, labels = 0.0, [], []
    allx = []
    for i, r in enumerate(rows):
        if r[0] == "H":
            if i > 0:
                y -= 0.35
            ax.axhspan(y - 0.5, y + 0.5, color="#F2F4F4", zorder=0, lw=0)
            ax.text(0.008, y, r[1], transform=ax.get_yaxis_transform(), fontsize=8.5,
                    fontweight="bold", color=C_TEXT_DIM, va="center", ha="left")
            y -= 1.0
            continue

        _, name, (obs, lo, hi) = r
        has_ci = lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
        allx += [obs] + ([lo, hi] if has_ci else [])
        if has_ci:
            sig = lo > 0 or hi < 0
            color = (C_POS if obs > 0 else C_NEG) if sig else C_TIE
            face = color if sig else "white"
            ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle="butt", zorder=2)
            for xe in (lo, hi):
                ax.plot([xe, xe], [y - 0.16, y + 0.16], color=color, lw=1.3, zorder=2)
            ax.scatter([obs], [y], s=34, marker="o", facecolors=face, edgecolors=color,
                       linewidths=1.3, zorder=3)
            txt = f"{fmt(obs)}  [{fmt(lo)}, {fmt(hi)}]"
        else:
            ax.scatter([obs], [y], s=30, marker="D", facecolors=C_NOCI, edgecolors=C_NOCI,
                       zorder=3)
            txt = f"{fmt(obs)}  [n/a]"
        ax.text(1.015, y, txt, transform=ax.get_yaxis_transform(), fontsize=7.6,
                va="center", ha="left", color=INK, family="monospace")
        ticks.append(y); labels.append(name)
        y -= 1.0

    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    lo_lim = min(allx + [-1.0]); hi_lim = max(allx)
    ax.set_xlim(lo_lim * 1.8 if lo_lim < -1 else -2.2, hi_lim * 1.6)
    major = [-1, 0, 1, 10, 100]
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_major_formatter(FixedFormatter(["−1", "0", "+1", "+10", "+100"]))
    ax.xaxis.set_minor_locator(NullLocator())

    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(y + 0.4, 0.8)
    ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("CQL − BC (return units; linear within ±1, logarithmic beyond)", labelpad=6)

    # direction labels sit just above the plot, either side of zero
    tr = ax.get_xaxis_transform()
    ax.text(-0.12, 1.012, "◀ BC favoured", transform=tr, ha="right", va="bottom",
            fontsize=8.5, style="italic", color=C_NEG)
    ax.text(0.12, 1.012, "CQL favoured ▶", transform=tr, ha="left", va="bottom",
            fontsize=8.5, style="italic", color=C_POS)
    ax.text(1.015, 1.012, "Δ  [95 % CI]", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=C_TEXT_DIM)

    h = [Line2D([0], [0], marker="o", ls="none", mfc=C_POS, mec=C_POS, ms=6.5, label="CI excludes 0 (CQL better)"),
         Line2D([0], [0], marker="o", ls="none", mfc=C_NEG, mec=C_NEG, ms=6.5, label="CI excludes 0 (BC better)"),
         Line2D([0], [0], marker="o", ls="none", mfc="white", mec=C_TIE, ms=6.5, mew=1.3, label="CI spans 0"),
         Line2D([0], [0], marker="D", ls="none", mfc=C_NOCI, mec=C_NOCI, ms=5.5, label="point estimate only")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.085),
              ncol=2, columnspacing=1.6, handletextpad=0.4)

    if TITLES:
        ax.set_title("Verdict landscape: CQL − BC across configurations", pad=22)
    save(fig, "fig1")


# ===========================================================================
# Figure 2 -- pipeline schematic (clean orthogonal routing, no overlaps)
# ===========================================================================
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(-9, 101); ax.set_ylim(-5, 70); ax.axis("off")

    # lane bands + labels
    lanes = [("Data", 61, "white"), ("RL\nenvironment", 43, "#F6F7F7"),
             ("Training", 25, "white"), ("Evaluation", 7, "#F6F7F7")]
    edges = [70, 52, 34, 16, -4]
    for i, (name, yc, fc) in enumerate(lanes):
        ax.add_patch(plt.Rectangle((-9, edges[i + 1]), 110, edges[i] - edges[i + 1],
                                   fc=fc, ec="none", zorder=0))
        ax.text(-6.5, yc, name, rotation=90, ha="center", va="center", fontsize=8,
                fontweight="bold", color=C_TEXT_DIM, linespacing=1.1)

    H = 9.0
    def box(xc, yc, w, title, sub=None, fc="white", ec=INK, tc=INK, lw=1.0):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - H / 2), w, H,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        if sub:
            ax.text(xc, yc + 1.5, title, ha="center", va="center", fontsize=7.2, color=tc, zorder=3)
            ax.text(xc, yc - 2.0, sub, ha="center", va="center", fontsize=6.4, color=tc, zorder=3,
                    alpha=0.85)
        else:
            ax.text(xc, yc, title, ha="center", va="center", fontsize=7.2, color=tc, zorder=3)

    def path(pts, dashed=False, color=INK):
        ls = (0, (3, 2)) if dashed else "-"
        for p, q in zip(pts[:-2], pts[1:-1]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.0, ls=ls, zorder=1,
                    solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, ls=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=8), zorder=1)

    top, mid, trn, ev = 61, 43, 25, 7
    # Data lane
    xs = [12.5, 37.5, 62.5, 87.5]
    box(xs[0], top, 21, "CGWB quarterly", "well observations")
    box(xs[1], top, 21, "Training-well", "ordinary kriging")
    box(xs[2], top, 21, "RF / HGBR", "downscaling")
    box(xs[3], top, 21, "Downscaled surface", "(companion paper v1)")
    for a, b in zip(xs[:-1], xs[1:]):
        path([(a + 10.5, top), (b - 10.5, top)])
    # RL environment lane
    box(12.5, mid, 25, "Behaviour policies", "random · greedy · ε-greedy", fc="#EEF0F0")
    box(50, mid, 25, "Synthetic simulator", "8×8 blocks · quarterly", fc="#EEF0F0")
    box(87.5, mid, 25, "30,000-transition", "offline dataset", fc="#EEF0F0")
    path([(12.5, top - H / 2), (12.5, mid + H / 2)])
    path([(87.5, top - H / 2), (87.5, mid + H / 2)])
    path([(25, mid), (37.5, mid)])
    path([(62.5, mid), (75, mid)])
    # Training lane
    box(30, trn, 24, "Behavioural cloning", "(BC)", fc=C_NEG, ec=C_NEG, tc="white")
    box(70, trn, 24, "Conservative Q-learning", "(CQL)", fc=C_POS, ec=C_POS, tc="white")
    path([(87.5, mid - H / 2), (87.5, 34), (70, 34), (70, trn + H / 2)])
    ax.plot([87.5, 30], [34, 34], color=INK, lw=1.0, zorder=1)
    path([(30, 34), (30, trn + H / 2)])
    # Evaluation lane
    box(12.5, ev, 21, "Banded OOD", "4 Mahalanobis bands")
    box(37.5, ev, 21, "FQE", "seed bootstrap")
    box(62.5, ev, 21, "Direct rollout", "episode bootstrap")
    box(87.5, ev, 21, "§5.2 finding", "FQE ≠ rollout", fc="#FDEDEC", ec=C_POS)
    bus = 15.5
    ax.plot([30, 30], [trn - H / 2, bus], color=INK, lw=1.0, zorder=1)
    ax.plot([70, 70], [trn - H / 2, bus], color=INK, lw=1.0, zorder=1)
    ax.plot([12.5, 70], [bus, bus], color=INK, lw=1.0, zorder=1)
    for x in (12.5, 37.5, 62.5):
        path([(x, bus), (x, ev + H / 2)])
    # comparison FQE vs rollout -> finding (dashed bracket below the boxes)
    yb = -0.9
    ax.plot([37.5, 37.5], [ev - H / 2, yb], color=C_POS, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.plot([62.5, 62.5], [ev - H / 2, yb], color=C_POS, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.plot([37.5, 87.5], [yb, yb], color=C_POS, lw=1.0, ls=(0, (3, 2)), zorder=1)
    path([(87.5, yb), (87.5, ev - H / 2)], dashed=True, color=C_POS)
    save(fig, "fig2")


# ===========================================================================
# Figure 3 -- banded OOD diagnostic
# ===========================================================================
def fig3():
    qr = read_ood_bands(REPO / "reports" / "ood_return_test.json")
    mq = read_ood_bands(REPO / "reports" / "ood_return_test_meanq.json")
    bands = ["id", "mild", "moderate", "severe"]
    xl = ["in-\ndistribution", "mild", "moderate", "severe\nOOD"]
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    top = 0
    for data, col, mk, lab, dx in [(mq, C_MEANQ, "s", "Mean-Q critic", 0.0),
                                   (qr, C_QR, "o", "QR-32 critic", 0.0)]:
        v = np.array([data[b][0] for b in bands], float)
        top = max(top, v.max())
        ax.plot(x, v, "-", color=col, lw=1.6, zorder=2)
        has = all(data[b][1] is not None and np.isfinite(data[b][1]) for b in bands)
        if has:
            lo = np.array([data[b][1] for b in bands]); hi = np.array([data[b][2] for b in bands])
            ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0, zorder=1)
        ax.plot(x, v, mk, color=col, ms=6.5, mec="white", mew=0.9, label=lab, zorder=3)
        off = 9 if col == C_MEANQ else -13
        for xi, vi in zip(x, v):
            ax.annotate(fmt(vi), (xi, vi), textcoords="offset points", xytext=(0, off),
                        ha="center", fontsize=7.8, color=col)

    ax.axhline(0, color=INK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(xl)
    ax.set_xlabel("Mahalanobis band  (near → far from training distribution)", labelpad=5)
    ax.set_ylabel("CQL − BC (return)")
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-1.5, top * 1.15)
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="upper left", handletextpad=0.5)
    if TITLES:
        ax.set_title("CQL advantage grows with distance from the data")
    save(fig, "fig3")


# ===========================================================================
# Figure 5 -- A4/A5 heatmap (redrawn)
# ===========================================================================
def fig5():
    a = read_a4a5_normal()
    a4s = sorted({k[0] for k in a}); a5s = sorted({k[1] for k in a})
    M = np.array([[a.get((r, c), (np.nan,))[0] for c in a5s] for r in a4s])
    vmin, vmax = np.nanmin(M), np.nanmax(M)

    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    norm = LogNorm(vmin=max(vmin * 0.8, 1e-2), vmax=vmax * 1.1)
    im = ax.imshow(M, cmap="YlOrRd", norm=norm, aspect="auto")

    ax.set_xticks(range(len(a5s))); ax.set_xticklabels([f"{v:.1f}" for v in a5s])
    ax.set_yticks(range(len(a4s))); ax.set_yticklabels([f"{v:.0f}" for v in a4s])
    ax.set_xlabel("A5 (drawdown coefficient)")
    ax.set_ylabel("A4 (sustainability threshold, m)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(a5s)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(a4s)), minor=True)
    ax.grid(which="minor", color="white", lw=2.0); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)

    for i, r in enumerate(a4s):
        for j, c in enumerate(a5s):
            val, lo, hi = a.get((r, c), (np.nan, None, None))
            if not np.isfinite(val):
                continue
            dark = norm(val) > 0.62
            tc = "white" if dark else INK
            ci = lo is not None and hi is not None
            ax.text(j, i - (0.12 if ci else 0), fmt(val), ha="center", va="center",
                    fontsize=9.5, fontweight="bold" if (r == 15 and c == 2.0) else "regular", color=tc)
            if ci:
                ax.text(j, i + 0.22, f"[{lo:.2f}, {hi:.2f}]", ha="center", va="center",
                        fontsize=6.6, color=tc, alpha=0.9)
    if 15 in a4s and 2.0 in a5s:                      # default configuration
        i, j = a4s.index(15), a5s.index(2.0)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=2.2, zorder=5))

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("CQL − BC (return, log colour scale)")
    cb.outline.set_visible(False)
    ticks = [t for t in (0.3, 1, 3, 10, 30, 100) if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ticks); cb.set_ticklabels([f"{t:g}" for t in ticks]); cb.minorticks_off()

    ratio = vmax / vmin if vmin > 0 else float("nan")
    fig.text(0.5, -0.02, f"Range {fmt(vmin)} to {fmt(vmax)} ({ratio:.0f}×).  "
             "Outlined cell = default configuration.", ha="center", va="top",
             fontsize=7.8, color=C_TEXT_DIM)
    if TITLES:
        ax.set_title("Effect size across reward parameters (normal scenario)")
    save(fig, "fig5")


# ===========================================================================
# Figure 6 -- within-reward sensitivity
# ===========================================================================
def fig6():
    j = read_e6(); cells = j["cells"]
    scenarios = ["normal", "drought", "high"]
    a5s = sorted({c["a5"] for c in cells})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cmap = {"normal": "#1F4E79", "drought": "#C0392B", "high": "#27AE60"}
    mk = {"normal": "o", "drought": "s", "high": "^"}
    ls = {"normal": "-", "drought": "--", "high": ":"}
    for s in scenarios:
        ys = []
        for a5 in a5s:
            m = [c for c in cells if c["a5"] == a5 and c["scenario"] == s]
            ys.append(m[0]["cql_minus_bc"] if m else np.nan)
        ax.plot(a5s, ys, ls=ls[s], marker=mk[s], color=cmap[s], lw=1.5, ms=6,
                mec="white", mew=0.9, label=s, zorder=3)
    lo = [min(c["cql_minus_bc"] for c in cells if c["a5"] == a) for a in a5s]
    hi = [max(c["cql_minus_bc"] for c in cells if c["a5"] == a) for a in a5s]
    ax.fill_between(a5s, lo, hi, color=INK, alpha=0.06, lw=0, zorder=1)
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.set_xlabel("A5 (drawdown coefficient)"); ax.set_ylabel("CQL − BC (return at gw = 25 m)")
    ax.set_xticks(a5s)
    ax.grid(color=C_GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ratio = j.get("ratio_summary", {})
    lines = [f"{s:<7} {ratio[s]['min']:+.2f} → {ratio[s]['max']:+.2f}  ({ratio[s]['ratio']:.2f}×)"
             for s in scenarios if s in ratio and {"min", "max", "ratio"} <= set(ratio[s])]
    if lines:
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.4,
                family="monospace", va="top", ha="left", color=C_TEXT_DIM,
                bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA", ec=C_GRID, lw=0.6))
    ax.legend(loc="lower right", title="climate scenario", title_fontsize=8.5, handletextpad=0.6)
    if TITLES:
        ax.set_title("Within-reward sensitivity (A4 = 15, weights = (1, 2, 0.5))")
    save(fig, "fig6")


def copy_through():
    for target, name in [("fig4", "fig4_reward_ablation_two_rates.png")]:
        s = SRC / name
        if s.exists():
            shutil.copy2(s, OUT / f"{target}{s.suffix}"); print(f"[{target}] copied {name}")
        else:
            print(f"[{target}] MISSING: {name}")


def main():
    print(f"output: {OUT}\n")
    for f in (fig1, fig2, fig3, fig5, fig6):
        try:
            f()
        except FileNotFoundError as e:
            print(f"[{f.__name__}] skipped: {e}")
    copy_through()
    print("\n[done]", *sorted(p.name for p in OUT.iterdir()), sep="\n   ")


if __name__ == "__main__":
    main()