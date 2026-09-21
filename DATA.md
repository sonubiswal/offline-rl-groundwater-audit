# Data & Artifact Provenance

This repository tracks code, configs, split definitions, and **frozen result
summaries**. Large or regenerable artifacts (raw data tiles, trained model
checkpoints, RL datasets, rollout outputs, figures) are deliberately excluded
from git and can be rebuilt from the tracked code + configs.

## Tracked in git

- `src/`, `tests/`, `scripts/`, `config/` — source code and configuration
- `data/held_out_wells/`, `data/interim/train_well_ids.csv` — canonical
  train/held-out split definitions (small, must be versioned so results are
  reproducible)
- `reports/**/*.json` — frozen result metadata (model `.meta.json`,
  `training_manifest.json`, `normalization_manifest.json`, `dataset_*.json`)
- Canonical `reports/**/*.csv` summary tables (coverage, heldout comparison,
  RF v2 metrics, policy comparison summaries, scenario/ablation summaries,
  NDVI imputation metrics, provenance audit final)
- `requirements.txt`, `environment.yml` — environment specs
- `FROZEN.md`, `FROZEN.json` — freeze manifest
- `DATA.md` (this file)

## Not tracked (regenerable)

- `data/interim/{chirps,gldas,grace,srtm,lulc_worldcover,sentinel1_vv,sentinel2_ndvi,soil_texture_class,kriged_target*}/`
  — raw and interim downloaded/derived tiles
- `data/processed/` — pipeline outputs (feature tables are tracked
  individually where small and canonical; the directory as a whole is not)
- `models*/`, `*.d3` — trained d3rlpy checkpoints (regenerable from seed +
  config; see `reports/paper_exp/**/*.meta.json` for the exact recipe)
- `reports/**/*.h5` — offline RL MDPDatasets and array archives
- `reports/**/returns/` — per-episode rollout outputs
- `reports/**/*.png`, `reports/**/*.pdf` — figures (regenerated from tracked
  CSVs via the plotting scripts)

## Regenerating excluded artifacts

Canonical entrypoints (see `FROZEN.md` for the frozen commit + command set):

```bash
# Downscaling / NDVI pipeline
python src/downscaling/rf_downscale_v2.py
python src/downscaling/validate_downscale.py

# Offline RL datasets + training
python src/rl/generate_offline_dataset.py
python src/rl/train_cql.py --config config/rl_config.yaml

# Full paper experiment matrix
python run_paper_experiments.py
```

Per-experiment metadata (seeds, hyperparameters, dataset paths, normalization
stats) lives next to each result under `reports/paper_exp/**/*.json`. Those
JSON files are tracked precisely so that any excluded `.d3` / `.h5` / `.png`
can be reproduced bit-for-bit given the frozen code and environment.

## Why these choices

- **Git is for source and small frozen results.** Binary checkpoints and large
  arrays belong in object storage (S3/GCS), DVC, or Git LFS � not in the main
  git history.
- **Every excluded artifact has a tracked recipe.** If you can find the
  `.meta.json` and the code, you can rebuild the artifact. If you cannot, the
  recipe is incomplete and the missing metadata should be added here.
- **Figures are excluded by default.** If a specific figure needs to ship with
  the repo (e.g. for README), `git add -f reports/path/to/figure.png` and note
  it in this file.
