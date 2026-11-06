"""
E3: explain why the epsilon pooled interval moved from [-0.0000, +0.0928]
to [+0.0182, +0.0766]. Tries candidate constructions on the current .npy
files; whichever reproduces the OLD interval identifies the cause.

Run from C:\\trishna-opal:
    python revision_experiments\\e3_pooled_ci_diagnosis.py
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from stats_utils import episode_bootstrap, t_interval


SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]


def main():
    D = {}
    for s in SEEDS:
        bc = np.load(f"reports_10seed/epsilon/returns/returns_bc_seed{s}.npy")
        cq = np.load(f"reports_10seed/epsilon/returns/returns_cql_seed{s}.npy")
        D[s] = cq - bc

    print("episodes per seed:", {s: len(d) for s, d in D.items()})
    print("OLD target: [-0.0000, +0.0928]  (half-width ~0.046)")
    print("NEW target: [+0.0182, +0.0766]  (half-width ~0.029)\n")

    def show(name, m, lo, hi):
        print(f"{name:55s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]")

    print("-- candidate constructions --")
    show("pooled, all 10 seeds, all episodes",
         *episode_bootstrap(np.concatenate(list(D.values()))))

    for k in (3, 5):
        show(f"pooled, first {k} seeds",
             *episode_bootstrap(np.concatenate([D[s] for s in SEEDS[:k]])))

    for n in (100, 200, 225, 250, 500):
        sub = np.concatenate([D[s][:n] for s in SEEDS])
        show(f"pooled, first {n} eps/seed", *episode_bootstrap(sub))

    for nb in (500, 1000, 2000, 5000):
        show(f"pooled, n_boot={nb}",
             *episode_bootstrap(np.concatenate(list(D.values())), n_boot=nb))

    show("seed t-interval", *t_interval([d.mean() for d in D.values()]))

    print("\nIf none matches the OLD interval, the old value came from "
          "different arrays (e.g. pre-fix scaler on the .npy files). "
          "In that case, state exactly that in the manuscript.")


if __name__ == "__main__":
    main()