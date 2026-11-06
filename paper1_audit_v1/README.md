# Same Policies, Different Verdicts — audit archive

Accompanies the manuscript "Same Policies, Different Verdicts: A Matched-Start
Audit of Offline Reinforcement Learning for Synthetic Groundwater Control."

## Contents
per_seed_arrays/ per-seed arrays behind every reported interval
summaries/ the summary JSON for each table in the paper
scripts/ drivers, the patched adapter, and verify.py
environment/ exact package versions used
MANIFEST.csv path, size and SHA-256 of every file

text

## Verifying the reported numbers

`scripts/verify.py` reads only the arrays and summaries in this archive. It
needs numpy and scipy. Run:
cd scripts
python verify.py

text

It recomputes the matched-start contrast, seed-level intervals, leave-one-out
means and the reward-decomposition endpoint, compares each against the
corresponding summary JSON, and exits non-zero on any disagreement. Expected
output: `All 33 checks passed.`

## The rollout RNG repair

The original pipeline constructed rollout generators with
`RandomState(9999 + seed + i)`. That scheme is not injective across
(seed, episode) pairs, so rollouts from different training seeds shared noise
streams. The patched `scripts/adapters.py` uses

- `RandomState(100000 + seed)` for the RF-grid start sampler
- `RandomState(200000 + 1000*seed + i)` for the per-episode rollout generator

For the ten seeds used, the two ranges are disjoint and two rollout streams
differ by at least 1000 integers, exceeding the largest episode index. Every
reported number uses the patched scheme. Table S2 in the paper lists the
original and repaired values side by side.

## Files in `per_seed_arrays/`

| File | Keys | Shape |
|---|---|---|
| `matched_start_225.npz` | `seeds`, `rollout`, `fqe_100`, `fqe_200`, `fqe_500` | each `(10,)` |
| `family_start_grid.npz` | 27 arrays keyed `{bc}__{cql}__{protocol}` | `(10, 500)` / `(10, 225)` / `(10, 150)` |
| `band_decomposition.npz` | per-band arrays at both weightings | each `(10,)` |

## Files in `summaries/`

| File | Table |
|---|---|
| `table_vii.json` | FQE−rollout sensitivity across three intervals |
| `table_viii.json` | FQE and rollout marginals |
| `table_ix.json` | Family/start grid, 27 rows |
| `table_x.json` | All 24 distinct cells and their three interval constructions |
| `table_xi.json` | Reward decomposition |
| `table_xii.json` | BC sweep, 18 cells |

## License

Code: MIT. Data: CC BY 4.0.