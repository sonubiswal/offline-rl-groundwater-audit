"""
E6: prove every sweep checkpoint carries the intended lr / hidden units.
Writes a CSV for the supplement. Uses d3rlpy.load_learnable.

Run from C:\\trishna-opal:
    python revision_experiments\\e6_checkpoint_audit.py
"""
import csv
import glob
import re

import d3rlpy


def main():
    rows = []
    bad = 0
    paths = sorted(glob.glob("reports_2/bc_sweep/*.d3"))
    print(f"found {len(paths)} checkpoints")

    for p in paths:
        norm = p.replace("\\", "/")
        m = re.match(r".*/?(\w+?)_lr([0-9.eE+-]+)_h(\d+)_s(\d+)\.d3$", norm)
        if not m:
            print(f"[skip] {p} (name did not parse)")
            continue
        fam = m.group(1)
        lr = float(m.group(2))
        h = int(m.group(3))
        s = int(m.group(4))

        try:
            algo = d3rlpy.load_learnable(p, device="cpu:0")
        except TypeError:
            algo = d3rlpy.load_learnable(p)

        cfg = algo.config
        got_lr = float(cfg.learning_rate)
        hidden_attr = getattr(cfg.encoder_factory, "hidden_units", None)
        got_h = list(hidden_attr) if hidden_attr is not None else []

        ok = (abs(got_lr - lr) < 1e-12
              and len(got_h) >= 1
              and got_h[0] == h)
        bad += (not ok)
        rows.append([fam, lr, h, s, got_lr, got_h, ok])

    with open("revision_experiments/e6_sweep_config_audit.csv",
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "intended_lr", "intended_hidden", "seed",
                    "loaded_lr", "loaded_hidden", "match"])
        w.writerows(rows)

    print(f"{len(rows)} checkpoints audited, {bad} mismatches")
    print("wrote revision_experiments/e6_sweep_config_audit.csv")


if __name__ == "__main__":
    main()