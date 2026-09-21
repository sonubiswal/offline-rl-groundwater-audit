#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDVI imputation experiment - V3 (leakage-safe + temporal transfer + radius sweep)
==================================================================================
Builds on V2. Changes made in response to the V2 review, one section per
review point:

  [Review #1] Spatial fallback interpretation made explicit in docs/config
      (see Experiment C docstring below and `spatial_time_window_days`).

  [Review #2] Added an explicit console/JSON caveat so nobody reads
      "B_spatial ~ A_spatial" as evidence that temporal history improves
      spatial transfer. It doesn't test that - held-out spatial blocks have
      no same-pixel history, so B's temporal features are NaN there BY
      CONSTRUCTION. The caveat is printed right next to the comparison and
      written into ndvi_imputation_metrics.json under "_caveats".

  [Review #3] Added a THIRD validation protocol: temporal_holdout. Train on
      an earlier window, test on a later window (prospective, not shuffled).
      This is the only one of the three CV types that actually tests
      temporal transfer.

          random          -> generic reconstruction
          spatial_block   -> spatial transfer
          temporal_holdout-> temporal transfer   <-- new

  [Review - radius] `spatial_fallback_radius_km` default lowered from 150 to
      20 (not frozen; the 150 km number was a coverage diagnostic, not a
      validated production radius). The radius is also overridable via the
      NDVI_RADIUS_KM env var, and a sensitivity sweep over
      10 / 15 / 20 / 30 km (NDVI_RADIUS_SWEEP=1) reports how much of the
      "still missing after exact match" gap each radius actually closes,
      using this dataset's real donor availability - it does NOT re-derive
      the 42.6%/44.4%/...% figures from the prior diagnostic, since those
      came from a different analysis and this script has no way to
      reproduce them without that diagnostic's own code/data.

  [Review - provenance] Recovery output now also carries
      `sentinel2_ndvi_imputation_model` / `sentinel2_ndvi_imputation_experiment`
      for source-2 (imputed) rows, so the audit trail records not just THAT
      a row was imputed but WHICH frozen model/experiment produced it.

  [Review - missing-covariate handling] Predictor missingness (CHIRPS/GLDAS
      SM/S1 VV/GRACE/etc.) is now measured and reported explicitly in
      ndvi_predictor_missingness.csv and printed to console, rather than
      being left implicit in "RF median-imputes, HGBR uses native NaNs".

Everything else (leakage-safe masking order, GroupKFold spatial blocks,
provenance columns, winner-freeze-on-full-data, hierarchical recovery) is
unchanged from V2 and is NOT re-explained here - see V2's docstring/comments
for the underlying design rationale.

Usage:
    NDVI_INPUT_CSV=feature_table.csv python3 ndvi_imputation_pipeline_v3.py
    NDVI_RADIUS_KM=15 NDVI_RADIUS_SWEEP=1 python3 ndvi_imputation_pipeline_v3.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline

# --------------------------------------------------------------------------
# Configuration - edit to match your feature table
# --------------------------------------------------------------------------
CONFIG = {
    "input_csv": os.environ.get("NDVI_INPUT_CSV", "feature_table.csv"),
    "output_dir": "reports",
    "random_seed": 42,
    "n_splits_random": 5,
    "n_splits_spatial": 5,
    "spatial_block_deg": 1.0,               # lon/lat grid cell size for blocks
    # NOT frozen as the production radius - see Review note above.
    "spatial_fallback_radius_km": float(os.environ.get("NDVI_RADIUS_KM", 20.0)),
    "radius_sweep_km": [10.0, 15.0, 20.0, 30.0],
    "run_radius_sweep": os.environ.get("NDVI_RADIUS_SWEEP", "0") == "1",
    "spatial_time_window_days": 0,          # 0 = same Sentinel-2 composite date
    # Fraction of the OBSERVED date range held out at the end for the
    # temporal_holdout protocol. E.g. 0.2 -> earliest 80% of dates train,
    # latest 20% of dates test. This is a single prospective split, not a
    # k-fold - shuffling it would defeat the point.
    "temporal_holdout_frac": 0.2,
    "models": ["seasonal_median", "random_forest", "hist_gradient_boosting"],
    "cov_predictors": [                     # Experiment A predictors
        "chirps", "gldas_sm", "s1_vv", "grace",
        "elevation", "slope", "month_sin", "month_cos",
    ],
    "id_col": "pixel_id",
    "lon_col": "lon",
    "lat_col": "lat",
    "date_col": "date",
    "target_col": "sentinel2_ndvi",
    "region_col": "region",                 # optional
    "winner_criterion": "spatial_block_r2",  # or "random_r2" / "temporal_holdout_r2"
    # month -> season (edit for your study region)
    "season_map": {
        1: "pre-monsoon", 2: "pre-monsoon", 3: "pre-monsoon",
        4: "pre-monsoon", 5: "pre-monsoon",
        6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "post-monsoon", 11: "post-monsoon", 12: "winter",
    },
}

CV_TYPES = ["random", "spatial_block", "temporal_holdout"]

EARTH_R = 6371.0  # km


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def unit_sphere(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan,
                "bias": np.nan, "corr": np.nan}
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan,
    }


def distribution_compare(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 10:
        return {"ks_stat": np.nan, "ks_p": np.nan, "hist_overlap": np.nan}
    ks = stats.ks_2samp(y_true, y_pred)
    lo = float(np.min(np.concatenate([y_true, y_pred])))
    hi = float(np.max(np.concatenate([y_true, y_pred])))
    bins = np.linspace(lo, hi, 31)
    h1, _ = np.histogram(y_true, bins=bins, density=True)
    h2, _ = np.histogram(y_pred, bins=bins, density=True)
    overlap = float(np.sum(np.minimum(h1, h2)) * (bins[1] - bins[0]))
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "hist_overlap": overlap}


def add_blocks(df, cfg):
    deg = cfg["spatial_block_deg"]
    lon = np.floor(df[cfg["lon_col"]] / deg).astype(int)
    lat = np.floor(df[cfg["lat_col"]] / deg).astype(int)
    return df.assign(block=(lat.astype(str) + "_" + lon.astype(str)))


# --------------------------------------------------------------------------
# Temporal features - computed from a MASKED panel (leakage-safe)
# --------------------------------------------------------------------------
def compute_temporal_features(df, cfg):
    """Causal per-pixel temporal features from OBSERVED NDVI only.

    prev_ndvi / ndvi_age_days come from the nearest observed NDVI strictly
    before t in the same pixel. Rows whose NDVI is NaN (masked validation or
    genuinely missing) are NEVER donors, so a masked target can never appear
    in any feature. No next_ndvi - no future information.

    Returns df (original row order) with columns:
        prev_ndvi, ndvi_age_days, _prev_donor (donor row index, for checks)
    """
    df = df.copy()
    idc, dc, tc = cfg["id_col"], cfg["date_col"], cfg["target_col"]
    df["_orig_idx"] = df.index
    df = df.sort_values([idc, dc]).reset_index(drop=True)
    # original row index of each row's NDVI if observed, else NaN
    df["_obs_row"] = df["_orig_idx"].where(df[tc].notna())
    # nearest observed row strictly before, within pixel
    df["_prev_donor"] = df.groupby(idc, sort=False)["_obs_row"].shift(1)
    df["_prev_donor"] = df.groupby(idc, sort=False)["_prev_donor"].ffill()
    donor_ndvi = df.set_index("_orig_idx")[tc]
    donor_date = df.set_index("_orig_idx")[dc]
    df["prev_ndvi"] = df["_prev_donor"].map(donor_ndvi)
    df["ndvi_age_days"] = (df[dc] - df["_prev_donor"].map(donor_date)).dt.days
    df = df.sort_values("_orig_idx").reset_index(drop=True)
    return df.drop(columns=["_obs_row"])


def feature_matrix(imp, feat, cfg, experiment):
    """Build the feature matrix for an experiment (A or B)."""
    X = imp[cfg["cov_predictors"]].copy()
    if experiment == "B":
        X["prev_ndvi"] = feat["prev_ndvi"].to_numpy()
        X["ndvi_age_days"] = feat["ndvi_age_days"].to_numpy()
    return X


def check_leakage(feat, val_idx, imp, cfg):
    """Assert no validation-row NDVI value appears in any feature column.

    Checks, per validation row:
      (1) the prev_ndvi donor row is NOT a validation row;
      (2) prev_ndvi equals the donor's real observed NDVI (i.e. the feature
          truly came from an unmasked value).
    """
    tc = cfg["target_col"]
    val_set = set(val_idx)
    donor = feat["_prev_donor"].to_numpy()
    is_val = np.asarray(feat.index.isin(val_set))
    donor_is_val = np.isin(donor, list(val_set))
    leaked_donor = int((is_val & donor_is_val & ~np.isnan(donor)).sum())

    # (2) value-integrity check for all rows with a donor
    donor_ok = True
    n_donor = 0
    valid = ~np.isnan(donor)
    if valid.any():
        n_donor = int(valid.sum())
        got = feat["prev_ndvi"].to_numpy()[valid]
        want = imp[tc].to_numpy()[donor[valid].astype(int)]
        donor_ok = bool(np.allclose(got, want, rtol=1e-9, atol=1e-12))

    passed = (leaked_donor == 0) and donor_ok
    return {
        "passed": passed,
        "n_validation_rows": int(len(val_idx)),
        "n_leaked_donors": leaked_donor,
        "n_donor_checks": n_donor,
        "donor_value_integrity_ok": donor_ok,
    }


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def train_model(model_name, X, y, cfg):
    if model_name == "random_forest":
        m = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                  n_jobs=-1, random_state=cfg["random_seed"]),
        )
    elif model_name == "hist_gradient_boosting":
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            random_state=cfg["random_seed"])
    else:
        raise ValueError(f"unknown model: {model_name}")
    m.fit(X, y)
    return m


def seasonal_median_predict(masked, imp, te, cfg):
    """Baseline: per (pixel, season) median from TRAINING rows only.
    masked already has validation rows set to NaN, so they are excluded.
    Fallbacks: pixel median -> global median."""
    idc, tc = cfg["id_col"], cfg["target_col"]
    train = masked[masked[tc].notna()]
    key = train.groupby([idc, "season"])[tc].median().rename("pred")
    pixel_med = train.groupby(idc)[tc].median().rename("pred")
    global_med = float(train[tc].median())
    test = imp.iloc[te].copy()
    out = test[[idc, "season"]].merge(key, on=[idc, "season"], how="left")
    out = out.merge(pixel_med, on=idc, how="left", suffixes=("", "_px"))
    out["pred"] = out["pred"].fillna(out["pred_px"]).fillna(global_med)
    return out["pred"].to_numpy()


# --------------------------------------------------------------------------
# CV splitters - random / spatial_block (from V2) + temporal_holdout (new)
# --------------------------------------------------------------------------
def temporal_holdout_splits(imp, cfg):
    """Single prospective split on UNIQUE acquisition dates: earliest
    (1-frac) of unique dates train, latest `frac` of unique dates test.
    Not shuffled, not k-folded - the whole point is "did the model learn
    something before, that still holds after".

    [Review fix] Uses unique dates rather than a row-weighted quantile of
    `imp[date_col]` - a plain quantile over all rows would let dense
    acquisition dates (more pixels observed that day) pull the cutoff
    toward themselves, so the split would not reliably be "earliest 80% of
    dates" in acquisition-date terms. Sorting and indexing into the unique
    date array makes the 80/20 split unambiguous and paper-defensible.
    """
    dc = cfg["date_col"]
    unique_dates = np.sort(imp[dc].dropna().unique())
    cutoff_idx = int(np.floor((1.0 - cfg["temporal_holdout_frac"]) * len(unique_dates))) - 1
    cutoff_idx = max(0, min(cutoff_idx, len(unique_dates) - 1))
    cutoff = unique_dates[cutoff_idx]
    tr = imp.index[imp[dc] <= cutoff].to_numpy()
    te = imp.index[imp[dc] > cutoff].to_numpy()
    return [(tr, te)], cutoff


def run_cv(imp, cfg, experiment, model_name, cv_type):
    """One (experiment, model, cv_type) combination with per-fold masking."""
    idc, tc = cfg["id_col"], cfg["target_col"]
    cutoff_info = None
    if cv_type == "random":
        splitter = KFold(n_splits=cfg["n_splits_random"], shuffle=True,
                         random_state=cfg["random_seed"])
        splits = splitter.split(imp)
    elif cv_type == "spatial_block":
        splitter = GroupKFold(n_splits=cfg["n_splits_spatial"])
        splits = splitter.split(imp, groups=imp["block"].to_numpy())
    elif cv_type == "temporal_holdout":
        splits, cutoff = temporal_holdout_splits(imp, cfg)
        cutoff_info = str(cutoff)
    else:
        raise ValueError(f"unknown cv_type: {cv_type}")

    folds, checks = [], []
    for fold, (tr, te) in enumerate(splits):
        # 1) mask validation NDVI
        masked = imp.copy()
        masked.loc[te, tc] = np.nan
        # 2) temporal features WITHOUT validation NDVI
        feat = compute_temporal_features(masked, cfg)
        # 3) feature matrix for this experiment
        X = feature_matrix(imp, feat, cfg, experiment)
        y = imp[tc].to_numpy()
        # 4) train on training rows, predict validation rows
        if model_name == "seasonal_median":
            p = seasonal_median_predict(masked, imp, te, cfg)
        else:
            m = train_model(model_name, X.iloc[tr], y[tr], cfg)
            p = m.predict(X.iloc[te])
        # 5) compare against untouched real NDVI
        folds.append(pd.DataFrame({
            "experiment": experiment, "model": model_name, "cv_type": cv_type,
            "fold": fold, "idx": te, "y_true": y[te], "y_pred": p}))
        chk = check_leakage(feat, te, imp, cfg)
        chk.update({"experiment": experiment, "model": model_name,
                    "cv_type": cv_type, "fold": fold,
                    "temporal_cutoff": cutoff_info})
        checks.append(chk)
    return pd.concat(folds, ignore_index=True), pd.DataFrame(checks)


# --------------------------------------------------------------------------
# Recovery (Experiment C) - deployment hierarchy with provenance
# --------------------------------------------------------------------------
def hierarchical_recovery(panel, imp, cfg, model, experiment, feature_cols,
                          radius_km=None):
    """exact pixel -> spatial donor (same Sentinel-2 composite/date, nearest
    valid pixel within radius) -> NDVI imputer.

    [Review #1] Spelling this out explicitly per the review: with
    spatial_time_window_days=0, the spatial donor stage means "find an
    observed NDVI pixel within `radius_km` on EXACTLY the same composite
    date" - it never borrows across dates. That's what matches the
    spatial-fallback diagnostic this radius default was chosen from.

    Adds provenance columns:
      sentinel2_ndvi / sentinel2_ndvi_source (0,1,2) /
      sentinel2_ndvi_age_days / sentinel2_ndvi_spatial_distance_km /
      sentinel2_ndvi_imputation_model / sentinel2_ndvi_imputation_experiment
    """
    df = panel.copy()
    idc, dc, tc = cfg["id_col"], cfg["date_col"], cfg["target_col"]
    missing = df[tc].isna()
    df["sentinel2_ndvi"] = df[tc]
    df["sentinel2_ndvi_source"] = np.where(df[tc].notna(), 0, np.nan)
    df["sentinel2_ndvi_age_days"] = df["ndvi_age_days"]
    df["sentinel2_ndvi_spatial_distance_km"] = np.nan
    # [Review - provenance] which frozen model/experiment produced source-2 rows
    df["sentinel2_ndvi_imputation_model"] = None
    df["sentinel2_ndvi_imputation_experiment"] = None

    if int(missing.sum()) == 0:
        return df

    radius_km = cfg["spatial_fallback_radius_km"] if radius_km is None else radius_km
    win = cfg["spatial_time_window_days"]
    obs = imp.reset_index(drop=True)
    obs_xyz = unit_sphere(obs[cfg["lon_col"]].to_numpy(),
                          obs[cfg["lat_col"]].to_numpy())
    tree = cKDTree(obs_xyz)

    for idx in df.index[missing]:
        row = df.loc[idx]
        q = unit_sphere(np.array([row[cfg["lon_col"]]]),
                        np.array([row[cfg["lat_col"]]]))[0]
        nbrs = tree.query_ball_point(q, radius_km / EARTH_R)
        cand = obs.iloc[nbrs] if nbrs else obs.iloc[[]]
        if len(cand) > 0:
            dt = (cand[dc] - row[dc]).dt.days.abs()
            cand = cand[dt <= win]
        if len(cand) == 0:
            source = 2
        else:
            dist = haversine_km(row[cfg["lon_col"]], row[cfg["lat_col"]],
                                cand[cfg["lon_col"]], cand[cfg["lat_col"]])
            best = cand.loc[dist.idxmin()]
            df.at[idx, "sentinel2_ndvi"] = best[tc]
            df.at[idx, "sentinel2_ndvi_source"] = 1
            df.at[idx, "sentinel2_ndvi_age_days"] = float(
                abs((best[dc] - row[dc]).days))
            df.at[idx, "sentinel2_ndvi_spatial_distance_km"] = float(dist.min())
            continue
        # source 2: imputer
        feat = df.loc[idx, feature_cols].to_frame().T
        df.at[idx, "sentinel2_ndvi"] = float(model.predict(feat)[0])
        df.at[idx, "sentinel2_ndvi_source"] = 2
        df.at[idx, "sentinel2_ndvi_spatial_distance_km"] = np.nan
        df.at[idx, "sentinel2_ndvi_imputation_model"] = experiment  # set below too
    # vectorized fill of the provenance strings for all source==2 rows
    df.loc[df["sentinel2_ndvi_source"] == 2, "sentinel2_ndvi_imputation_model"] = \
        cfg.get("_winner_model_name", experiment)
    df.loc[df["sentinel2_ndvi_source"] == 2, "sentinel2_ndvi_imputation_experiment"] = \
        experiment
    return df


def radius_sweep_report(panel_feat, imp, cfg, model, winner_exp, feature_cols):
    """[Review - radius] Sensitivity of the exact->spatial->imputed split to
    the fallback radius. Reports, for each candidate radius, how many of the
    rows that are missing after exact-match get resolved by the spatial
    donor stage vs. fall through to the imputer, using this dataset's own
    donor geometry (NOT a re-derivation of the earlier 150km-diagnostic
    numbers, which came from separate code/data this script doesn't have).
    """
    rows = []
    for r in cfg["radius_sweep_km"]:
        rec = hierarchical_recovery(panel_feat, imp, cfg, model, winner_exp,
                                    feature_cols, radius_km=r)
        src = rec["sentinel2_ndvi_source"]
        total = int(len(rec))
        rows.append({
            "radius_km": r,
            "n_exact": int((src == 0).sum()),
            "n_spatial_fallback": int((src == 1).sum()),
            "n_imputed": int((src == 2).sum()),
            "pct_spatial_fallback_of_total": 100 * float((src == 1).sum()) / total,
            "pct_imputed_of_total": 100 * float((src == 2).sum()) / total,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    idc, dc, tc = cfg["id_col"], cfg["date_col"], cfg["target_col"]

    # ---- load panel ----
    panel = pd.read_csv(cfg["input_csv"])
    panel[dc] = pd.to_datetime(panel[dc])
    before = len(panel)
    panel = panel.drop_duplicates([idc, dc]).reset_index(drop=True)
    if len(panel) != before:
        print(f"[warn] dropped {before - len(panel)} duplicate (pixel, date) rows")
    panel["month"] = panel[dc].dt.month
    panel["season"] = panel["month"].map(cfg["season_map"])
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
    if cfg["region_col"] not in panel.columns:
        panel["region"] = "all"

    # covariate predictors present in input
    cov = [c for c in cfg["cov_predictors"] if c in panel.columns]
    dropped = [c for c in cfg["cov_predictors"] if c not in panel.columns]
    if dropped:
        print(f"[warn] covariate predictors not in input, dropped: {dropped}")
    cfg["cov_predictors"] = cov

    # [Review - missing-covariate handling] report predictor missingness
    # explicitly, rather than leaving it implicit in RF-median-impute vs
    # HGBR-native-NaN.
    miss_rows = []
    for c in cov:
        n_missing = int(panel[c].isna().sum())
        miss_rows.append({
            "predictor": c,
            "n_missing": n_missing,
            "pct_missing": 100 * n_missing / len(panel),
        })
    miss_df = pd.DataFrame(miss_rows).sort_values("pct_missing", ascending=False)
    miss_df.to_csv(os.path.join(cfg["output_dir"], "ndvi_predictor_missingness.csv"),
                   index=False)
    print("\n=== predictor missingness (panel-wide) ===")
    for _, r in miss_df.iterrows():
        print(f"  {r['predictor']:12s} missing={r['n_missing']:6d} "
              f"({r['pct_missing']:.1f}%)")
    print("  Note: RF median-imputes these before fitting; HGBR consumes "
          "the NaNs natively. Rows where a predictor is missing therefore "
          "still get an NDVI imputation, but from a partially-imputed or "
          "NaN-aware feature row - see this table when auditing individual "
          "imputed values.")

    # ---- imputation table: observed NDVI only ----
    imp = panel[panel[tc].notna()].copy().reset_index(drop=True)
    imp = add_blocks(imp, cfg)
    print(f"\nobserved NDVI rows: {len(imp)} / {len(panel)} "
          f"({100 * len(imp) / len(panel):.1f}% coverage)")

    # ---- validation: A and B x models x {random, spatial_block, temporal_holdout} ----
    overall, cv_rows, by_season_rows, pred_parts, leak_parts = {}, [], [], [], []
    for experiment in ["A", "B"]:
        for model_name in cfg["models"]:
            for cv_type in CV_TYPES:
                preds, checks = run_cv(imp, cfg, experiment, model_name, cv_type)
                m = metrics(preds["y_true"], preds["y_pred"])
                dist = distribution_compare(preds["y_true"], preds["y_pred"])
                overall[f"{experiment}::{model_name}"] = overall.get(
                    f"{experiment}::{model_name}", {})
                overall[f"{experiment}::{model_name}"][cv_type] = {**m, **dist}
                for fold, g in preds.groupby("fold"):
                    fm = metrics(g["y_true"], g["y_pred"])
                    cv_rows.append({"experiment": experiment, "model": model_name,
                                    "cv_type": cv_type, "fold": int(fold), **fm})
                meta = imp[["season", "region", "block"]].copy()
                meta["idx"] = meta.index
                dfp = preds.merge(meta, on="idx", how="left")
                for (season, region), g in dfp.groupby(["season", "region"]):
                    gm = metrics(g["y_true"], g["y_pred"])
                    by_season_rows.append({"experiment": experiment,
                                           "model": model_name, "cv_type": cv_type,
                                           "season": season, "region": region, **gm})
                pred_parts.append(dfp)
                leak_parts.append(checks)

    cv_df = pd.DataFrame(cv_rows)
    by_season_df = pd.DataFrame(by_season_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True)
    leak_df = pd.concat(leak_parts, ignore_index=True)
    leak_df.to_csv(os.path.join(cfg["output_dir"], "ndvi_leakage_check.csv"),
                   index=False)
    n_fail = int((~leak_df["passed"]).sum())
    print(f"\nLEAKAGE CHECK: {len(leak_df)} fold-checks, "
          f"{n_fail} FAILED, {len(leak_df) - n_fail} PASSED")

    # ---- choose winner from validation only ----
    crit = cfg["winner_criterion"]
    crit_cv, crit_metric = crit.rsplit("_", 1)
    if crit_cv not in CV_TYPES and f"{crit_cv}_holdout" in CV_TYPES:
        crit_cv = f"{crit_cv}_holdout"
    best_key, best_r2 = None, -np.inf
    for experiment in ["A", "B"]:
        for model_name in cfg["models"]:
            if model_name == "seasonal_median":
                continue
            key = f"{experiment}::{model_name}"
            r2 = overall[key][crit_cv][crit_metric]
            if r2 > best_r2:
                best_key, best_r2 = key, r2
    winner_exp, winner_model = best_key.split("::")
    cfg["_winner_model_name"] = winner_model
    print(f"\nwinner ({crit}): experiment={winner_exp}, model={winner_model}, "
          f"R2={best_r2:.3f}")

    # [Review #2] explicit caveat, printed right next to the comparison it
    # protects against being misread.
    a_spatial_r2 = overall.get(f"A::{winner_model}", {}).get(
        "spatial_block", {}).get("r2")
    b_spatial_r2 = overall.get(f"B::{winner_model}", {}).get(
        "spatial_block", {}).get("r2")
    caveat = (
        "CAVEAT: Experiment B's spatial_block R2 (%.3f) vs A's (%.3f) does "
        "NOT show that temporal history (prev_ndvi/ndvi_age_days) improves "
        "spatial transfer. Held-out spatial blocks have no same-pixel "
        "history, so B's temporal features are NaN there BY CONSTRUCTION - "
        "B_spatial ~ A_spatial is expected, not a finding about temporal "
        "value. Whether temporal history helps at all is answered by the "
        "temporal_holdout numbers below, comparing A vs B there instead."
        % (b_spatial_r2 if b_spatial_r2 is not None else float("nan"),
           a_spatial_r2 if a_spatial_r2 is not None else float("nan"))
    )
    print("\n" + caveat)
    a_temp_r2 = overall.get(f"A::{winner_model}", {}).get(
        "temporal_holdout", {}).get("r2")
    b_temp_r2 = overall.get(f"B::{winner_model}", {}).get(
        "temporal_holdout", {}).get("r2")
    print(f"  temporal_holdout R2: A={a_temp_r2}, B={b_temp_r2} "
          f"(this comparison IS valid evidence about temporal transfer)")

    # ---- freeze winner on ALL observed data ----
    feat_full = compute_temporal_features(imp, cfg)
    X_full = feature_matrix(imp, feat_full, cfg, winner_exp)
    frozen = train_model(winner_model, X_full, imp[tc].to_numpy(), cfg)

    # ---- temporal features for the full panel (deployment) ----
    panel_feat = compute_temporal_features(panel, cfg)
    feature_cols = list(cfg["cov_predictors"])
    if winner_exp == "B":
        feature_cols += ["prev_ndvi", "ndvi_age_days"]

    recovered = hierarchical_recovery(panel_feat, imp, cfg, frozen, winner_exp,
                                      feature_cols)
    src = recovered["sentinel2_ndvi_source"]
    n0, n1, n2 = int((src == 0).sum()), int((src == 1).sum()), int((src == 2).sum())
    total = int(len(recovered))
    rec_summary = {
        "total_rows": total,
        "observed_exact_source0": n0,
        "spatial_fallback_source1": n1,
        "imputed_source2": n2,
        "still_missing": int(recovered["sentinel2_ndvi"].isna().sum()),
        "coverage_before": float(panel[tc].notna().mean()),
        "coverage_after_exact": n0 / total,
        "coverage_after_spatial": (n0 + n1) / total,
        "coverage_after_imputed": (n0 + n1 + n2) / total,
        "winner_experiment": winner_exp,
        "winner_model": winner_model,
        "winner_criterion": crit,
        "spatial_fallback_radius_km": cfg["spatial_fallback_radius_km"],
        "spatial_time_window_days": cfg["spatial_time_window_days"],
    }
    overall["_winner"] = {"experiment": winner_exp, "model": winner_model,
                          "criterion": crit, "r2": best_r2}
    overall["_recovery"] = rec_summary
    overall["_leakage"] = {"n_checks": int(len(leak_df)), "n_failed": n_fail}
    overall["_caveats"] = {"b_spatial_vs_a_spatial": caveat}
    overall["_config"] = {k: v for k, v in cfg.items() if k != "season_map"}

    # [Review - radius] optional sensitivity sweep
    if cfg["run_radius_sweep"]:
        sweep_df = radius_sweep_report(panel_feat, imp, cfg, frozen, winner_exp,
                                       feature_cols)
        sweep_df.to_csv(os.path.join(cfg["output_dir"], "ndvi_radius_sweep.csv"),
                        index=False)
        print("\n=== radius sensitivity sweep ===")
        print(sweep_df.to_string(index=False))

    # ---- write reports ----
    out = cfg["output_dir"]
    cv_df.to_csv(os.path.join(out, "ndvi_imputation_cv.csv"), index=False)
    by_season_df.to_csv(os.path.join(out, "ndvi_imputation_by_season.csv"),
                        index=False)
    pred_df.to_csv(os.path.join(out, "ndvi_imputation_predictions.csv"),
                   index=False)
    with open(os.path.join(out, "ndvi_imputation_metrics.json"), "w") as f:
        json.dump(overall, f, indent=2, default=float)
    with open(os.path.join(out, "ndvi_recovery_summary.json"), "w") as f:
        json.dump(rec_summary, f, indent=2, default=float)

    # ---- console summary ----
    print("\n=== overall metrics (validation) ===")
    for experiment in ["A", "B"]:
        for model_name in cfg["models"]:
            for cv_type in CV_TYPES:
                m = overall[f"{experiment}::{model_name}"][cv_type]
                print(f"Exp {experiment} {model_name:22s} {cv_type:17s} "
                      f"RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
                      f"R2={m['r2']:.3f} bias={m['bias']:+.4f} "
                      f"corr={m['corr']:.3f} n={m['n']} "
                      f"ks={m['ks_stat']:.3f} overlap={m['hist_overlap']:.3f}")
    print("\n=== recovery ===")
    for k, v in rec_summary.items():
        print(f"  {k}: {v}")
    print(f"\nreports written to {out}/")


if __name__ == "__main__":
    main()