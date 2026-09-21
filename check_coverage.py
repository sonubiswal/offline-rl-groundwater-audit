"""
check_coverage.py

Quick diagnostic: for each of the 38 Kriged-surface dates, report whether
build_feature_stack_for_date() succeeds, and if not, which channel is
likely missing/incomplete. Run this BEFORE deciding whether the ConvLSTM
data-volume problem is fixable (a coverage bug) or fundamental (real data
scarcity).

Usage:
    python check_coverage.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "downscaling"))
from feature_utils import build_feature_stack_for_date

kriged_dir = Path("data/interim/kriged_target_monthly")
interim_dir = "data/interim"

dates = sorted({p.stem for p in kriged_dir.glob("*.npy")})
print(f"Found {len(dates)} Kriged surface dates.\n")

ok, failed = 0, 0
for d in dates:
    stack = build_feature_stack_for_date(interim_dir, d, kriged_dir=None)
    if stack is None:
        print(f"  {d}: FAILED (stack is None -- likely a fully missing source for this date)")
        failed += 1
        continue

    bad_channels = [ch for ch, arr in stack.items() if np.isnan(arr).all()]
    partial_channels = [
        ch for ch, arr in stack.items()
        if 0 < np.isnan(arr).mean() < 1.0 and np.isnan(arr).mean() > 0.1
    ]
    if bad_channels:
        print(f"  {d}: FULLY MISSING channels: {bad_channels}")
        failed += 1
    elif partial_channels:
        print(f"  {d}: OK but >10% NaN in: "
              f"{[(ch, round(float(np.isnan(stack[ch]).mean())*100,1)) for ch in partial_channels]}")
        ok += 1
    else:
        print(f"  {d}: OK, all channels complete")
        ok += 1

print(f"\n{ok}/{len(dates)} dates usable, {failed}/{len(dates)} dates failed or fully missing a channel.")