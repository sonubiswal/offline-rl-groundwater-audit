"""
E5 analysis: adjacent-band and monotonicity contrasts, and full-vs-removed
endpoint contrast.

Reads per-seed band gaps from ood_w3_zero_eval_only.json (w3=0) and
ood_w3_full_eval_only.json (w3=0.5). Auto-detects the `band_per_seed`
dict written by run_ood_w3_zero.py.

Run from C:\\trishna-opal:
    python revision_experiments\\e5_band_contrasts.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from stats_utils import t_interval


BANDS = ["in_dist", "mild", "moderate", "severe"]

REM_PATH = "revision_experiments/ood_w3_zero_eval_only.json"
FULL_PATH = "revision_experiments/ood_w3_full_eval_only.json"


def find(o):
    """Recursively locate a dict whose keys are the four bands and whose
    values are lists of per-seed numbers. That is the `band_per_seed`
    block written by run_ood_w3_zero.py."""
    if isinstance(o, dict):
        if all(b in o and isinstance(o[b], list) and len(o[b]) >= 5
               for b in BANDS):
            return o
        for v in o.values():
            r = find(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r:
                return r
    return None


def load(path):
    with open(path) as f:
        payload = json.load(f)
    d = find(payload)
    if d is None:
        sys.exit(f"{path}: could not find per-seed values per band "
                 f"(expected a 'band_per_seed' dict). "
                 f"Top-level keys: {list(payload) if isinstance(payload, dict) else type(payload)}")
    return {
        b: np.array(
            [np.mean(x) if np.ndim(x) else x for x in d[b]], float
        )
        for b in BANDS
    }


def line(name, x):
    m, lo, hi = t_interval(x)
    flag = "excl 0" if (lo > 0 or hi < 0) else "spans 0"
    print(f"{name:42s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]  {flag}")


def main():
    rem = load(REM_PATH)

    print("== w3 = 0.0 (domestic supply removed) ==")
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        line(f"{b} - {a}", rem[b] - rem[a])
    line("severe - in_dist (endpoint)", rem["severe"] - rem["in_dist"])
    line("severe - moderate (saturation test)",
         rem["severe"] - rem["moderate"])

    try:
        full = load(FULL_PATH)
    except FileNotFoundError:
        print(f"\n{FULL_PATH} not found. Run the w3=0.5 configuration first:")
        print("  python revision_experiments\\run_ood_w3_zero.py "
              "--variant eval_only --w3 0.5 --tag w3_full_eval_only")
        return

    print("\n== w3 = 0.5 (full reward) ==")
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        line(f"{b} - {a}", full[b] - full[a])
    line("severe - in_dist (endpoint)", full["severe"] - full["in_dist"])
    line("severe - moderate (saturation test)",
         full["severe"] - full["moderate"])

    print("\n== Paired removed vs full (per seed) ==")
    line("endpoint(removed) - endpoint(full)",
         (rem["severe"] - rem["in_dist"]) - (full["severe"] - full["in_dist"]))
    for b in BANDS:
        line(f"gap({b}) removed - full", rem[b] - full[b])

    print("\n== Interpretation guide ==")
    print("If 'endpoint(removed) - endpoint(full)' spans 0, removing domestic")
    print("supply does NOT significantly change the endpoint contrast — the")
    print("falsification test passes in the strongest sense.")
    print("If it excludes 0, domestic supply partially drives the widening.")


if __name__ == "__main__":
    main()