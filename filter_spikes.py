"""
filter_spikes.py -- remove per-well outliers from a held-out predictions CSV.
Writes a new CSV; never touches the input.
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="reject readings where actual / well_median > this")
    ap.add_argument("--abs-max", type=float, default=100.0,
                    help="reject any reading above this absolute depth (m)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n0 = len(df)

    med = df.groupby("well_id")["actual"].median()
    df = df.merge(med.rename("well_median"), on="well_id")
    df["ratio"] = df["actual"] / df["well_median"].clip(lower=0.5)

    flag_ratio = df["ratio"] > args.ratio
    flag_abs = df["actual"] > args.abs_max
    flagged = df[flag_ratio | flag_abs]
    print(f"[filter] flagged {len(flagged)} / {n0} rows:")
    for _, r in flagged.iterrows():
        print(f"  {r['well_id']:<28} {r['date']}  actual={r['actual']:>7.2f}  "
              f"median={r['well_median']:>7.2f}  ratio={r['ratio']:>6.2f}")

    df = df[~(flag_ratio | flag_abs)].drop(columns=["well_median", "ratio"])
    df.to_csv(args.out, index=False)
    print(f"\n[filter] wrote {args.out}: {len(df)} rows ({n0 - len(df)} dropped)")


if __name__ == "__main__":
    main()