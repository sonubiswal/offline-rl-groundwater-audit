"""
load_episodic_h5_dataset.py -- drop-in replacement for load_h5_dataset()
in diagnostics_distribution_shift.py, for d3rlpy's PER-EPISODE HDF5
export format (observations_0, actions_0, rewards_0, terminated_0,
observations_1, actions_1, ... one set of datasets per episode, plus a
root-level 'version' string) rather than one flat set of
observations/actions/rewards/terminals arrays.

HOW TO USE
----------
In diagnostics_distribution_shift.py, replace the load_h5_dataset()
function with load_episodic_h5_dataset() below (same call site:
`behavior = load_episodic_h5_dataset(args.dataset)`), or just import
this function and monkey-patch/replace the name.

WHAT IT DOES
------------
- Finds every episode index N present (from observations_N keys).
- Loads each episode's observations/actions/rewards/terminated.
- Concatenates all episodes into flat arrays matching the schema the
  rest of diagnostics_distribution_shift.py expects:
      observations: (total_timesteps, obs_dim)
      actions:       (total_timesteps,) or (total_timesteps, action_dim)
      rewards:       (total_timesteps,)
      terminals:     (total_timesteps,)  -- per-timestep terminal flag,
                     True only on each episode's LAST timestep, value
                     taken from that episode's terminated_N scalar.
- Does NOT invent data: if an episode's terminated_N is False (i.e. it
  was truncated rather than terminated), the terminals array reflects
  that faithfully rather than forcing True at every episode boundary.

CHECK BEFORE USING
-------------------
- If your file also has truncated_N keys (episode ended by timeout,
  not real termination), inspect_h5_summary.py will show that prefix.
  This loader does not currently combine truncated_N into `terminals`,
  since the original schema's `terminals` field is specifically
  "did the underlying MDP actually terminate" -- if your downstream
  code needs truncation info too, that must be added explicitly, not
  silently merged in.
"""

import re
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def load_episodic_h5_dataset(path: str):
    if h5py is None:
        raise ImportError("h5py is required: pip install h5py")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(p, "r") as f:
        keys = list(f.keys())

        # Find every episode index present, from observations_N keys.
        ep_indices = sorted(
            int(m.group(1))
            for k in keys
            if (m := re.match(r"^observations_(\d+)$", k))
        )

        if not ep_indices:
            raise KeyError(
                "No 'observations_N' keys found in this file. "
                "Run inspect_h5_summary.py on it and check the actual "
                "prefixes -- the schema may differ from what this loader "
                "assumes."
            )

        obs_chunks, act_chunks, rew_chunks, term_chunks = [], [], [], []

        for i in ep_indices:
            obs_key = f"observations_{i}"
            act_key = f"actions_{i}"
            rew_key = f"rewards_{i}"
            term_key = f"terminated_{i}"

            for required in (obs_key, act_key, rew_key, term_key):
                if required not in f:
                    raise KeyError(
                        f"Episode {i} is missing expected key {required!r}. "
                        "Episodes are not uniformly structured -- inspect "
                        "the file manually before proceeding."
                    )

            obs = np.array(f[obs_key])
            act = np.array(f[act_key])
            rew = np.array(f[rew_key])
            terminated_flag = bool(np.array(f[term_key]))

            ep_len = obs.shape[0]

            if act.shape[0] != ep_len or rew.shape[0] != ep_len:
                raise ValueError(
                    f"Episode {i}: observations has {ep_len} timesteps but "
                    f"actions has {act.shape[0]} and rewards has "
                    f"{rew.shape[0]}. Refusing to silently truncate/pad -- "
                    "inspect this episode manually."
                )

            # Per-timestep terminal flag: False for every step except
            # possibly the last, where it's the episode's terminated_N.
            term_arr = np.zeros(ep_len, dtype=bool)
            term_arr[-1] = terminated_flag

            obs_chunks.append(obs)
            act_chunks.append(act)
            rew_chunks.append(rew)
            term_chunks.append(term_arr)

        observations = np.concatenate(obs_chunks, axis=0)
        actions = np.concatenate(act_chunks, axis=0)
        rewards = np.concatenate(rew_chunks, axis=0)
        terminals = np.concatenate(term_chunks, axis=0)

        version = (
            f["version"][()] if "version" in f else None
        )

    print(
        f"Loaded {len(ep_indices)} episodes, "
        f"{observations.shape[0]} total timesteps "
        f"(obs dim={observations.shape[1] if observations.ndim > 1 else 1}) "
        f"from {p}"
        + (f" [dataset version: {version}]" if version is not None else "")
    )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Quick standalone check: load and summarize an "
        "episodic HDF5 offline dataset."
    )
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    data = load_episodic_h5_dataset(args.file)

    for k, v in data.items():
        print(f"{k}: shape={v.shape} dtype={v.dtype}")