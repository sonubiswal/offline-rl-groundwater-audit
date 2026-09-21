#!/usr/bin/env python3
"""
analyze_probe_divergence.py

Quantitative divergence diagnostics for CQL probe runs (A/B/C/D).

This does NOT auto-classify a run as "diverging" or "stable" -- per the
probe decision-tree discussion, that call should stay a human judgment.
It only surfaces the numbers needed to make that call:

  - growth ratio        G = td_loss[last] / td_loss[first]
  - successive increments between checkpoints (Δ2-4, Δ4-6, Δ6-8, ...)
  - second differences of those increments (is the growth ACCELERATING,
    i.e. curving upward, or just climbing at a steady rate?)
  - td_loss / conservative_loss ratio at each checkpoint, and whether
    that ratio is trending up (td component starting to dominate the
    combined objective) or holding roughly flat
  - total loss (when present in the log) and its own growth/increments,
    so a td_loss rise that's just being offset by a falling
    conservative_loss (benign redistribution) can be told apart from
    one where total loss is actually climbing too

Two ways to get data in:
  1. Paste a run's raw console log (the d3rlpy stdout) into a .txt/.log
     file and pass it via --log NAME=path/to/file.log (repeatable).
  2. Hard-code values directly in the RUNS dict below (already filled
     in with probe B's numbers from this session as a worked example).

Usage:
    python analyze_probe_divergence.py
    python analyze_probe_divergence.py --log A=logs/probe_A_raw.txt --log D=logs/probe_D_raw.txt

Either source populates the same RunMetrics structure, so runs parsed
from a log file and runs hard-coded in RUNS can be compared side by
side in one table.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    name: str
    steps: list             # e.g. [2000, 4000, 6000, 8000]
    td_loss: list
    conservative_loss: list
    loss: Optional[list] = None  # total loss, when the log/caller provides it
    classification: str = "unclassified"  # human-assigned label, not computed

    def validate(self) -> None:
        n = len(self.steps)
        if not (len(self.td_loss) == n and len(self.conservative_loss) == n):
            raise ValueError(
                f"[{self.name}] length mismatch: steps={len(self.steps)}, "
                f"td_loss={len(self.td_loss)}, "
                f"conservative_loss={len(self.conservative_loss)}"
            )
        if self.loss is not None and len(self.loss) != n:
            raise ValueError(
                f"[{self.name}] length mismatch: steps={len(self.steps)}, "
                f"loss={len(self.loss)}"
            )
        if n < 2:
            raise ValueError(
                f"[{self.name}] need at least 2 checkpoints to compute deltas, got {n}"
            )


# ---------------------------------------------------------------------------
# Parsing raw d3rlpy console logs
#
# Matches lines like:
#   ... DiscreteCQL_...: epoch=1 step=2000 epoch=1 metrics={'time_sample_batch': 0.0038,
#   'time_algorithm_update': 0.0091, 'loss': 4.220194818615913, 'td_loss': 0.26844760970398784,
#   'conservative_loss': 0.9879368021190167, 'time_step': 0.0131} step=2000
#
# BUGFIX (this revision): the previous pattern required "step=NNN" to be
# immediately followed by "epoch=NNN" then "metrics={...}", with nothing
# else allowed in between. That happens to work for the exact log shape
# above (there's a "step=2000 epoch=1 metrics={...}" substring buried in
# it), but it's brittle -- any d3rlpy log line that puts other tokens
# between step= and metrics=, or that orders epoch= before step=, would
# silently fail to match and the run would parse as "no checkpoints
# found" (or worse, quietly drop checkpoints). The pattern below only
# anchors on step=NNN ... metrics={...} appearing in that order on the
# same line, with anything in between, which covers more real-world
# ordering variants while still finding the same match on the log shape
# in the docstring above.
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"step=(\d+).*?metrics=\{(.*?)\}")
_KV_RE = re.compile(r"'([a-zA-Z_]+)':\s*([-+0-9.eE]+)")


def parse_console_log(log_text: str, name: str) -> RunMetrics:
    """Extract per-checkpoint step/td_loss/conservative_loss (and total
    loss, when present) from a raw d3rlpy console log (multi-seed logs
    are fine -- only the FIRST seed's block is parsed, i.e. the first
    occurrence encountered at each step value; with this repo's current
    training order (TRAINING_SEEDS[0] == 42) that means seed 42's block.
    Pass a log containing a single seed's run if you want a different,
    specific seed instead)."""
    steps, td_losses, cons_losses, losses = [], [], [], []
    seen_steps = set()

    for m in _LINE_RE.finditer(log_text):
        step = int(m.group(1))
        if step in seen_steps:
            continue  # only take the first occurrence per step (first seed)
        metrics_blob = m.group(2)
        kv = dict(_KV_RE.findall(metrics_blob))
        if "td_loss" not in kv or "conservative_loss" not in kv:
            continue
        steps.append(step)
        td_losses.append(float(kv["td_loss"]))
        cons_losses.append(float(kv["conservative_loss"]))
        losses.append(float(kv["loss"]) if "loss" in kv else None)
        seen_steps.add(step)

    if not steps:
        raise ValueError(
            f"[{name}] no epoch metrics lines matched in the provided log text "
            f"-- check the log actually contains d3rlpy's per-epoch 'metrics={{...}}' lines"
        )

    # Only keep total loss if every matched checkpoint had it -- a partial
    # series would silently misalign with steps/td_loss/conservative_loss.
    loss_series = losses if all(v is not None for v in losses) else None

    run = RunMetrics(name=name, steps=steps, td_loss=td_losses,
                      conservative_loss=cons_losses, loss=loss_series)
    run.validate()
    return run


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def growth_ratio(run: RunMetrics) -> float:
    """G = td_loss[last] / td_loss[first]."""
    return run.td_loss[-1] / run.td_loss[0]


def increments(run: RunMetrics) -> list:
    """Successive deltas between checkpoints: Δ(2-4), Δ(4-6), Δ(6-8), ..."""
    return [b - a for a, b in zip(run.td_loss, run.td_loss[1:])]


def second_differences(run: RunMetrics) -> list:
    """Deltas of the deltas. Positive values indicate that the observed
    TD-loss increments are increasing over the measured checkpoints --
    i.e. the growth is accelerating, not decelerating or flat. This is
    a diagnostic signal only, not proof of divergence or proximity to
    a blow-up: a series like 1.0 -> 1.1 -> 1.3 -> 1.6 also has positive
    second differences and could still plateau."""
    incs = increments(run)
    return [b - a for a, b in zip(incs, incs[1:])]


def ratio_series(run: RunMetrics) -> list:
    """td_loss / conservative_loss at each checkpoint. A rising trend
    means the TD term is progressively dominating the combined
    objective, independent of whether td_loss alone looks large."""
    return [td / cons for td, cons in zip(run.td_loss, run.conservative_loss)]


def ratio_trend(run: RunMetrics) -> float:
    """Simple first-vs-last change in the td/conservative ratio."""
    r = ratio_series(run)
    return r[-1] - r[0]


def loss_growth_ratio(run: RunMetrics) -> Optional[float]:
    """G = loss[last] / loss[first], for total loss. None if not available."""
    if not run.loss:
        return None
    return run.loss[-1] / run.loss[0]


def loss_increments(run: RunMetrics) -> list:
    if not run.loss:
        return []
    return [b - a for a, b in zip(run.loss, run.loss[1:])]


def summarize(run: RunMetrics) -> str:
    run.validate()
    incs = increments(run)
    incs_str = ", ".join(f"{x:+.3f}" for x in incs)
    seconds = second_differences(run)
    seconds_str = ", ".join(f"{x:+.3f}" for x in seconds) if seconds else "n/a (need 3+ checkpoints)"
    ratios = ratio_series(run)
    ratios_str = ", ".join(f"{x:.3f}" for x in ratios)

    lines = [
        f"--- {run.name} ({run.classification}) ---",
        f"  steps:                {run.steps}",
        f"  td_loss:              {[round(x, 3) for x in run.td_loss]}",
        f"  conservative_loss:    {[round(x, 3) for x in run.conservative_loss]}",
        f"  growth ratio G:       {growth_ratio(run):.3f}x  (td_loss[last] / td_loss[first])",
        f"  increments:           {incs_str}",
        f"  2nd differences:      {seconds_str}   <- positive & growing => accelerating (not proof of blow-up)",
        f"  td/conservative ratio:{ratios_str}",
        f"  ratio trend (last-first): {ratio_trend(run):+.3f}",
    ]

    if run.loss:
        loss_incs = loss_increments(run)
        loss_incs_str = ", ".join(f"{x:+.3f}" for x in loss_incs)
        lg = loss_growth_ratio(run)
        lines.extend([
            f"  loss (total):         {[round(x, 3) for x in run.loss]}",
            f"  loss growth ratio:    {lg:.3f}x  (loss[last] / loss[first])",
            f"  loss increments:      {loss_incs_str}",
            "  note: rising td_loss with flat/falling total loss suggests a "
            "benign redistribution between td and conservative terms; rising "
            "td_loss WITH rising total loss is the harder case to dismiss.",
        ])
    else:
        lines.append("  loss (total):         not available for this run "
                      "(no 'loss' key parsed at every checkpoint)")

    return "\n".join(lines)


def comparison_table(runs: list) -> str:
    """Side-by-side td_loss table across runs, aligned by checkpoint step.
    Runs with missing checkpoints show '?'."""
    all_steps = sorted(set(s for r in runs for s in r.steps))
    header = ["Run"] + [f"{s//1000}k" for s in all_steps] + ["G", "Classification"]
    rows = [header]
    for r in runs:
        step_to_td = dict(zip(r.steps, r.td_loss))
        row = [r.name]
        for s in all_steps:
            v = step_to_td.get(s)
            row.append(f"{v:.2f}" if v is not None else "?")
        try:
            row.append(f"{growth_ratio(r):.2f}x")
        except (IndexError, ZeroDivisionError):
            row.append("?")
        row.append(r.classification)
        rows.append(row)

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for i, row in enumerate(rows):
        lines.append(" | ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worked example / entry point
# ---------------------------------------------------------------------------

# Hard-coded runs. Fill in A and D as their logs become available; B is
# already populated from this session's probe_B output (seed 42 block).
# B's total loss series is left unset (None) here -- only the four
# individual metrics lines quoted in this session are available, not a
# confirmed step-aligned loss series for B; pull it from the real log
# via --log B=... to get the loss diagnostics populated automatically.
RUNS: dict[str, RunMetrics] = {
    "B": RunMetrics(
        name="B (mean, 3x oversample, tui=500)",
        steps=[2000, 4000, 6000, 8000],
        td_loss=[0.268, 0.361, 0.527, 0.848],
        conservative_loss=[0.988, 0.872, 0.846, 0.829],
        classification="ambiguous (seed-consistent TD-loss growth, no confirmed divergence yet)",
    ),
    # "A": RunMetrics(
    #     name="A (QR, 3x oversample, tui=500)",
    #     steps=[2000, 4000, 6000, 8000],
    #     td_loss=[...],
    #     conservative_loss=[...],
    #     classification="known divergence phenotype",
    # ),
    # "D": RunMetrics(
    #     name="D (mean, 1x oversample, tui=500)",
    #     steps=[2000, 4000, 6000, 8000],
    #     td_loss=[...],
    #     conservative_loss=[...],
    #     classification="pending",
    # ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log", action="append", default=[],
        metavar="NAME=PATH",
        help="Parse a raw d3rlpy console log file and register it as run "
             "NAME (e.g. --log A=logs/probe_A_raw.txt). Repeatable. "
             "Overrides/adds to the hard-coded RUNS dict.",
    )
    args = ap.parse_args()

    runs = dict(RUNS)
    for spec in args.log:
        if "=" not in spec:
            sys.exit(f"--log expects NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        runs[name] = parse_console_log(text, name=name)

    if not runs:
        sys.exit("No runs to analyze -- populate RUNS or pass --log NAME=PATH.")

    print("=" * 70)
    print("PER-RUN DIAGNOSTICS")
    print("=" * 70)
    for run in runs.values():
        print(summarize(run))
        print()

    print("=" * 70)
    print("COMPARISON TABLE (td_loss by checkpoint)")
    print("=" * 70)
    print(comparison_table(list(runs.values())))


if __name__ == "__main__":
    main()