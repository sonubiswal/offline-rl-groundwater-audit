"""
check_validation_skip_bias.py

Diagnostic for Phase 2: validate_downscale.py skipped 65.7% of held-out
readings due to incomplete covariate coverage for their date. Before
trusting the R²=0.171 result, this checks whether those skips are spread
evenly across the study period (fine -- the evaluated 34% is a fair
sample) or concentrated in specific years/periods (a problem -- the R²
would then reflect performance on an unrepresentative subset, e.g. only
the best-covered years).

Run from repo root:
    python src/downscaling/check_validation_skip_bias.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_utils import build_feature_stack_for_date


def check_skip_distribution(
    held_out_csv: str = "data/held_out_wells/held_out_ids.csv",
    interim_dir: str = "data/interim",
    max_gap_days: int = 45,
) -> None:
    held_df = pd.read_csv(held_out_csv)
    held_df["date"] = held_df["date"].astype(str)
    held_df["year"] = held_df["date"].str[:4]

    unique_dates = held_df["date"].unique()
    print(f"Checking coverage for {len(unique_dates)} unique held-out reading dates...")

    date_status = {}
    for date_str in unique_dates:
        stack = build_feature_stack_for_date(interim_dir, date_str, max_gap_days)
        date_status[date_str] = "ok" if stack is not None else "skipped"

    held_df["coverage_status"] = held_df["date"].map(date_status)

    print()
    print("=" * 60)
    print("SKIP RATE BY YEAR")
    print("=" * 60)
    summary = held_df.groupby("year")["coverage_status"].value_counts().unstack(fill_value=0)
    if "ok" not in summary.columns:
        summary["ok"] = 0
    if "skipped" not in summary.columns:
        summary["skipped"] = 0
    summary["total"] = summary["ok"] + summary["skipped"]
    summary["skip_pct"] = (summary["skipped"] / summary["total"] * 100).round(1)
    print(summary[["ok", "skipped", "total", "skip_pct"]].to_string())

    print()
    overall_skip_pct = (held_df["coverage_status"] == "skipped").mean() * 100
    print(f"Overall skip rate: {overall_skip_pct:.1f}%")
    print()
    print("If skip_pct is roughly SIMILAR across years, the evaluated 34% is a fair")
    print("sample and the R²=0.171 result is trustworthy as-is.")
    print("If skip_pct is heavily concentrated in specific years (e.g. near-100% in")
    print("2013-2015 or 2017-2018), the evaluated sample is biased toward whichever")
    print("years survived, and this should be noted in validation_methodology.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether held-out validation skips are concentrated in specific years.")
    parser.add_argument("--held_out_csv", default="data/held_out_wells/held_out_ids.csv")
    parser.add_argument("--interim_dir", default="data/interim")
    args = parser.parse_args()

    check_skip_distribution(args.held_out_csv, args.interim_dir)