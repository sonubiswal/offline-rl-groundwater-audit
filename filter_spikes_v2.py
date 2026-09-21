"""
filter_spikes_v2.py -- conservative per-reading spike filter.

Flags a reading only if it is BOTH:
  (a) above an absolute depth cap (default 100 m), OR
  (b) more than `ratio` times larger than the mean of its two nearest
      temporal neighbours within `window_days` for the same well.

This preserves real seasonal/interannual variation while catching
transcription errors.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def flag_spikes(df, ratio, abs_max, window_days):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)
    flag = np.zeros(len(df), dtype=bool)

    # (a) absolute cap
    flag |= (df["actual"].values > abs_max)

    # (b) neighbour-based
    for wid, idx in df.groupby("well_id").groups.items():
        idx = list(idx)
        if len(idx) < 3:
            continue
        d = df.loc[idx, "date"].values
        a = df.loc[idx, "actual"].values
        for i in range(1, len(idx) - 1):
            gap_prev = (d[i] - d[i-1]).astype("timedelta64[D]").astype(int)
            gap_next = (d[i+1] - d[i]).astype("timedelta64[D]").astype(int)
            if gap_prev > window_days or gap_next > window_days:
                continue
            neigh_mean = 0.5 * (a[i-1] + a[i+1])
            if neigh_mean > 0 and a[i] / neigh_mean > ratio:
                flag[idx[i]] = True

    return df, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratio", type=float, default=6.0,
                    help="reading must exceed neighbours' mean by this factor")
    ap.add_argument("--abs-max", type=float, default=100.0,
                    help="absolute depth cap in metres")
    ap.add_argument("--window-days", type=int, default=400,
                    help="max gap (days) to previous/next reading for neighbour check")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n0 = len(df)
    df, flag = flag_spikes(df, args.ratio, args.abs_max, args.window_days)

    flagged = df[flag]
    print(f"[filter] flagged {len(flagged)} / {n0} rows:")
    for _, r in flagged.iterrows():
        print(f"  {r['well_id']:<28} {str(r['date'])[:10]}  actual={r['actual']:>8.2f}")

    df = df[~flag]
    df.to_csv(args.out, index=False)
    print(f"\n[filter] wrote {args.out}: {len(df)} rows ({n0 - len(df)} dropped)")


if __name__ == "__main__":
    main()