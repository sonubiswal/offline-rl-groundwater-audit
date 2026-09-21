"""
cgwb_behavioral_validation.py
Compare observed CGWB well dynamics against simulated RL-environment dynamics.

Reads:
  data/held_out_wells/held_out_ids.csv
  data/processed/well_level_features_holdout.csv
  data/processed/offline_rl_dataset.h5

Writes:
  reports/cgwb_behavioral_validation.png
  reports/cgwb_behavioral_validation.json

Read-only.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ observed
obs = pd.read_csv("data/held_out_wells/held_out_ids.csv", parse_dates=["date"])
obs["month"] = obs["date"].dt.month

obs_seasonal = obs.groupby("month")["depth_m"].agg(["mean", "std", "count"])

obs = obs.sort_values(["well_id", "date"]).reset_index(drop=True)
obs["d_depth"] = obs.groupby("well_id")["depth_m"].diff()
obs["d_days"]  = obs.groupby("well_id")["date"].diff().dt.days

feat = pd.read_csv("data/processed/well_level_features_holdout.csv",
                   parse_dates=["date"])[["well_id", "date", "chirps"]]

obs_pair = (obs[["well_id", "date", "d_depth", "d_days"]]
            .merge(feat, on=["well_id", "date"], how="left")
            .dropna(subset=["d_depth", "chirps"]))
obs_pair = obs_pair[(obs_pair["d_days"] > 30) & (obs_pair["d_days"] <= 180)]

obs_corr = float(np.corrcoef(obs_pair["d_depth"], obs_pair["chirps"])[0, 1])
obs_slope = float(np.polyfit(obs_pair["chirps"], obs_pair["d_depth"], 1)[0])

# ----------------------------------------------------------------- simulated
with h5py.File("data/processed/offline_rl_dataset.h5", "r") as f:
    ep_ids = sorted(int(k.split("_")[-1]) for k in f.keys()
                    if k.startswith("observations_"))
    gw_list, rain_list, month_list, extract_list = [], [], [], []
    for eid in ep_ids:
        arr = f[f"observations_{eid}"][:]        # (20, 6)
        gw_list.append(arr[:, 0])                # gw_level
        rain_list.append(arr[:, 1])              # recent_rainfall
        extract_list.append(arr[:, 5])           # extraction_rate
        # recover calendar month (0..12, floating)
        m = (np.arctan2(arr[:, 3], arr[:, 4]) % (2 * np.pi)) / (2 * np.pi) * 12
        month_list.append(m)

gw_all      = np.concatenate(gw_list)
rain_all    = np.concatenate(rain_list)
extract_all = np.concatenate(extract_list)
month_all   = np.concatenate(month_list)

# per-episode step differences
d_gw_all, d_rain_all = [], []
for g, r in zip(gw_list, rain_list):
    d_gw_all.append(np.diff(g))
    d_rain_all.append(r[1:])
d_gw   = np.concatenate(d_gw_all)
d_rain = np.concatenate(d_rain_all)

sim_corr  = float(np.corrcoef(d_gw, d_rain)[0, 1])
sim_slope = float(np.polyfit(d_rain, d_gw, 1)[0])

# episode amplitudes
sim_amp = np.array([g.max() - g.min() for g in gw_list])
obs_amp = obs.groupby("well_id")["depth_m"].agg(lambda s: s.max() - s.min())

# simulated seasonal cycle, integer-binned month
sim_df = pd.DataFrame({"month": np.floor(month_all).astype(int) % 12,
                       "gw": gw_all})
sim_seasonal = sim_df.groupby("month")["gw"].mean()

# -------------------------------------------------------------------- report
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax0 = axes[0]
ax0.bar(obs_seasonal.index, obs_seasonal["mean"], yerr=obs_seasonal["std"],
        color="steelblue", alpha=0.7, label="Observed CGWB depth (mean±std)")
ax0.set_xlabel("Month")
ax0.set_ylabel("Observed depth (m)")
ax0.set_title("Seasonal cycle: observed vs simulated")
ax0b = ax0.twinx()
ax0b.plot(sim_seasonal.index, sim_seasonal.values, "o-", color="darkorange",
          label="Simulated gw_level monthly mean")
ax0b.set_ylabel("Simulated gw_level")
ax0.legend(loc="upper left")
ax0b.legend(loc="upper right")

axes[1].scatter(obs_pair["chirps"], obs_pair["d_depth"], s=4, alpha=0.3,
                color="steelblue", label=f"Observed (r={obs_corr:+.2f})")
axes[1].scatter(d_rain, d_gw, s=4, alpha=0.3,
                color="darkorange", label=f"Simulated (r={sim_corr:+.2f})")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_xlabel("Rainfall")
axes[1].set_ylabel("Δ groundwater")
axes[1].set_title("Rainfall response")
axes[1].legend()

fig.tight_layout()
fig.savefig(REPORTS / "cgwb_behavioral_validation.png", dpi=150)

result = {
    "observed": {
        "n_wells": int(obs["well_id"].nunique()),
        "n_readings": int(len(obs)),
        "n_pairs": int(len(obs_pair)),
        "seasonal_amplitude_m": float(obs_seasonal["mean"].max()
                                       - obs_seasonal["mean"].min()),
        "rainfall_response_r": obs_corr,
        "rainfall_response_slope": obs_slope,
        "well_amplitude_mean_m": float(obs_amp.mean()),
        "well_amplitude_std_m": float(obs_amp.std()),
    },
    "simulated": {
        "n_episodes": int(len(ep_ids)),
        "n_steps_total": int(len(gw_all)),
        "amplitude_mean": float(sim_amp.mean()),
        "amplitude_std": float(sim_amp.std()),
        "rainfall_response_r": sim_corr,
        "rainfall_response_slope": sim_slope,
    },
    "comparison": {
        "response_sign_match": bool(np.sign(obs_corr) == np.sign(sim_corr)),
        "observed_response_sign": "positive" if obs_corr > 0 else "negative",
        "simulated_response_sign": "positive" if sim_corr > 0 else "negative",
        "amplitude_ratio_sim_over_obs": float(sim_amp.mean() / obs_amp.mean()),
    },
    "notes": [
        "Observed depth_m = CGWB convention (m below ground level; larger = deeper).",
        "Simulated gw_level sign convention must be confirmed from Phase 4 "
        "environment code. If simulated gw_level also uses depth-below-ground, "
        "the two are directly comparable; if it uses 'height above datum', signs "
        "will be inverted.",
        "Seasonal cycle alignment uses month recovery from month_sin/month_cos "
        "(atan2); binned to integer months for plotting.",
    ],
}
(REPORTS / "cgwb_behavioral_validation.json").write_text(
    json.dumps(result, indent=2))
print(json.dumps(result, indent=2))