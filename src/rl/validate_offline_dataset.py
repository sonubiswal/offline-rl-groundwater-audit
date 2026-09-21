"""
Phase 4 Offline RL Dataset Validation
--------------------------------------

Validates the generated offline RL transition dataset before Phase 5.
Uses h5py to read the dataset (no d3rlpy dependency).
Handles both flat, episode‑group, and per‑episode dataset structures.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import h5py

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "offline_rl_dataset.h5",
    PROJECT_ROOT / "data" / "rl" / "offline_dataset.csv",
    PROJECT_ROOT / "data" / "rl" / "offline_transitions.csv",
    PROJECT_ROOT / "reports" / "offline_dataset.csv",
    PROJECT_ROOT / "reports" / "offline_transitions.csv",
    PROJECT_ROOT / "offline_dataset.csv",
]

EXPECTED_RANGES = {
    "gw_level": (-100.0, 100.0),
    "recent_rainfall": (0.0, 1000.0),
    "crop_water_demand": (0.0, 1000.0),
    "month_sin": (-1.01, 1.01),
    "month_cos": (-1.01, 1.01),
    "extraction_rate": (0.0, 1.0),
}

ACTION_NAMES = {
    0: "0% pumping",
    1: "25% pumping",
    2: "50% pumping",
    3: "75% pumping",
    4: "100% pumping",
}

# ============================================================
# HELPERS
# ============================================================

def locate_dataset():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    print("\nERROR: Could not locate the offline dataset.")
    print("Searched:")
    for path in DATASET_CANDIDATES:
        print("  ", path)
    sys.exit(1)

def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def result(name, passed, message=""):
    symbol = "PASS" if passed else "FAIL"
    print(f"[{symbol}] {name}")
    if message:
        print("       " + message)

# ============================================================
# LOADER: automatic HDF5 structure detection
# ============================================================

def load_dataset_hdf5(path):
    """Read the HDF5 file, detecting flat, episode-group, or per-episode-dataset format."""
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        print(f"Top-level keys: {keys[:10]} ... ({len(keys)} total)")

        # -------- 1. Flat format --------
        flat_candidates = {
            'obs': ['observations', 'obs', 'state', 'states'],
            'action': ['actions', 'action'],
            'reward': ['rewards', 'reward'],
            'next_obs': ['next_observations', 'next_obs', 'next_state'],
            'terminal': ['terminals', 'terminal', 'dones', 'done']
        }
        found_flat = {}
        for role, cands in flat_candidates.items():
            for c in cands:
                if c in f:
                    found_flat[role] = c
                    break
        if len(found_flat) >= 4:
            obs = f[found_flat['obs']][:]
            actions = f[found_flat['action']][:].flatten()
            rewards = f[found_flat['reward']][:].flatten()
            next_obs = f[found_flat['next_obs']][:]
            terminals = f[found_flat['terminal']][:].flatten()
            if obs.shape[0] == actions.shape[0] == rewards.shape[0] == next_obs.shape[0] == terminals.shape[0]:
                rows = [{'obs': obs[i], 'action': actions[i], 'reward': rewards[i],
                         'next_obs': next_obs[i], 'terminal': terminals[i]} for i in range(len(actions))]
                print("Detected flat format.")
                return pd.DataFrame(rows)

        # -------- 2. Episode groups --------
        episode_keys = [k for k in keys if k.startswith('episode_')]
        if episode_keys:
            rows = []
            for ep_key in episode_keys:
                ep = f[ep_key]
                obs = None
                action = None
                reward = None
                terminal = None
                for k in ep.keys():
                    if k in ['observations', 'obs', 'state']:
                        obs = ep[k][:]
                    elif k in ['actions', 'action']:
                        action = ep[k][:].flatten()
                    elif k in ['rewards', 'reward']:
                        reward = ep[k][:].flatten()
                    elif k in ['terminals', 'terminal', 'dones', 'done']:
                        terminal = ep[k][:]
                        # if scalar, convert to array of length action
                        if terminal.shape == ():
                            terminal = np.full(len(action), bool(terminal), dtype=bool)
                        else:
                            terminal = terminal.flatten()
                if obs is not None and action is not None and reward is not None and terminal is not None:
                    for i in range(len(action)):
                        rows.append({
                            'obs': obs[i],
                            'action': action[i],
                            'reward': reward[i],
                            'next_obs': obs[i+1] if i+1 < obs.shape[0] else np.full_like(obs[i], np.nan),
                            'terminal': terminal[i] if i < len(terminal) else False
                        })
            if rows:
                print("Detected episode-group format.")
                return pd.DataFrame(rows)

        # -------- 3. Per-episode datasets (e.g., observations_0, actions_0, ...) --------
        obs_indices = []
        act_indices = []
        for k in keys:
            if k.startswith('observations_') or k.startswith('obs_'):
                try: obs_indices.append(int(k.split('_')[-1]))
                except: pass
            elif k.startswith('actions_') or k.startswith('action_'):
                try: act_indices.append(int(k.split('_')[-1]))
                except: pass
        indices = sorted(set(obs_indices + act_indices))
        if indices:
            print(f"Detected per-episode datasets, indices: {indices[:5]} ... ({len(indices)} total)")
            rows = []
            for idx in indices:
                obs_key = None
                act_key = None
                rew_key = None
                term_key = None
                for prefix in ['observations_', 'obs_']:
                    key = f'{prefix}{idx}'
                    if key in f:
                        obs_key = key
                        break
                for prefix in ['actions_', 'action_']:
                    key = f'{prefix}{idx}'
                    if key in f:
                        act_key = key
                        break
                for prefix in ['rewards_', 'reward_']:
                    key = f'{prefix}{idx}'
                    if key in f:
                        rew_key = key
                        break
                for prefix in ['terminated_', 'terminal_', 'done_']:
                    key = f'{prefix}{idx}'
                    if key in f:
                        term_key = key
                        break
                if obs_key is None or act_key is None or rew_key is None or term_key is None:
                    continue

                obs = f[obs_key][:]
                actions = f[act_key][:]
                if actions.ndim == 2 and actions.shape[1] == 1:
                    actions = actions.flatten()
                elif actions.ndim == 1:
                    pass
                else:
                    actions = actions.flatten()

                rewards = f[rew_key][:]
                if rewards.ndim == 2 and rewards.shape[1] == 1:
                    rewards = rewards.flatten()
                elif rewards.ndim == 1:
                    pass
                else:
                    rewards = rewards.flatten()

                term = f[term_key]
                if term.shape == ():  # scalar
                    terminal = np.full(len(actions), bool(term[()]), dtype=bool)
                else:
                    terminal = term[:].flatten()

                min_len = min(len(actions), len(rewards), len(terminal))
                for i in range(min_len):
                    rows.append({
                        'obs': obs[i],
                        'action': actions[i],
                        'reward': rewards[i],
                        'next_obs': obs[i+1] if i+1 < obs.shape[0] else np.full_like(obs[i], np.nan),
                        'terminal': terminal[i] if i < len(terminal) else False
                    })
            if rows:
                print("Detected per-episode dataset format.")
                return pd.DataFrame(rows)

        # -------- 4. Fallback: try to infer from any 2D dataset with shape (N, 6) --------
        possible_obs = None
        for k in keys:
            if f[k].ndim == 2 and f[k].shape[1] == 6:
                possible_obs = k
                break
        if possible_obs is not None:
            obs = f[possible_obs][:]
            n = obs.shape[0]
            possible_action = None
            possible_reward = None
            possible_next = None
            possible_term = None
            for k in keys:
                if f[k].ndim == 1 and f[k].shape[0] == n:
                    if possible_action is None: possible_action = k
                    elif possible_reward is None: possible_reward = k
                    elif possible_term is None: possible_term = k
                if f[k].ndim == 2 and f[k].shape[0] == n and k != possible_obs:
                    possible_next = k
            if possible_action and possible_reward and possible_next and possible_term:
                actions = f[possible_action][:].flatten()
                rewards = f[possible_reward][:].flatten()
                next_obs = f[possible_next][:]
                terminals = f[possible_term][:].flatten()
                rows = []
                for i in range(n):
                    rows.append({
                        'obs': obs[i],
                        'action': actions[i],
                        'reward': rewards[i],
                        'next_obs': next_obs[i],
                        'terminal': terminals[i]
                    })
                print(f"Inferred structure: obs={possible_obs}, action={possible_action}, reward={possible_reward}, next={possible_next}, term={possible_term}")
                return pd.DataFrame(rows)

        raise ValueError("Could not detect any known HDF5 structure. Please inspect the file keys.")

# ============================================================
# MAIN
# ============================================================

def main():
    print_header("TRISHNA-OPAL PHASE 4 OFFLINE RL DATASET VALIDATION")
    dataset_path = locate_dataset()
    print(f"Dataset: {dataset_path}")

    # Load
    if dataset_path.suffix == '.h5':
        try:
            df = load_dataset_hdf5(dataset_path)
        except Exception as e:
            print(f"ERROR loading HDF5: {e}")
            try:
                with h5py.File(dataset_path, 'r') as f:
                    print("\nTop-level keys (first 20):")
                    for i, key in enumerate(list(f.keys())[:20]):
                        obj = f[key]
                        if isinstance(obj, h5py.Dataset):
                            print(f"  {key}: shape {obj.shape}, dtype {obj.dtype}")
                        else:
                            print(f"  {key}: group")
            except:
                pass
            sys.exit(1)
    else:
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            print(f"ERROR loading CSV: {e}")
            sys.exit(1)

    if df is None or len(df) == 0:
        print("No data loaded. Exiting.")
        sys.exit(1)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    for c in df.columns:
        print(f"  - {c}")

    # Expand 'obs' and 'next_obs' into separate columns if they exist
    if 'obs' in df.columns and 'next_obs' in df.columns:
        state_names = ['gw_level', 'recent_rainfall', 'crop_water_demand', 'month_sin', 'month_cos', 'extraction_rate']
        next_state_names = [f'next_{n}' for n in state_names]
        obs_stack = np.stack(df['obs'].values)
        next_obs_stack = np.stack(df['next_obs'].values)
        for i, name in enumerate(state_names):
            df[name] = obs_stack[:, i]
        for i, name in enumerate(next_state_names):
            df[name] = next_obs_stack[:, i]
        df = df.drop(columns=['obs', 'next_obs'])

    # Detect columns
    state_cols = {key: key for key in EXPECTED_RANGES if key in df.columns}
    next_state_cols = {key: f'next_{key}' for key in EXPECTED_RANGES if f'next_{key}' in df.columns}
    action_col = 'action' if 'action' in df.columns else None
    reward_col = 'reward' if 'reward' in df.columns else None

    if not state_cols:
        for key in EXPECTED_RANGES:
            for candidate in [key, f'state_{key}', f'obs_{key}']:
                if candidate in df.columns:
                    state_cols[key] = candidate
                    break
    if not next_state_cols:
        for key in EXPECTED_RANGES:
            for candidate in [f'next_{key}', f'next_state_{key}', f'next_state.{key}']:
                if candidate in df.columns:
                    next_state_cols[key] = candidate
                    break
    if action_col is None:
        for c in ['action', 'actions']:
            if c in df.columns:
                action_col = c
                break
    if reward_col is None:
        for c in ['reward', 'rewards']:
            if c in df.columns:
                reward_col = c
                break

    print("\nDetected state columns:")
    for k, v in state_cols.items():
        print(f"  {k:20s} -> {v}")
    print("\nDetected next-state columns:")
    for k, v in next_state_cols.items():
        print(f"  {k:20s} -> {v}")
    print(f"\nAction column: {action_col}")
    print(f"Reward column: {reward_col}")

    # ------------------------------------------------------------
    # 1. NaN / INF CHECK
    # ------------------------------------------------------------
    print_header("1. NaN / INF CHECK")
    numeric = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy()).sum())
    result("No Inf values", inf_count == 0, f"Inf count = {inf_count}")

    # Check NaN only in next_obs columns when terminal=True
    nan_count_total = int(numeric.isna().sum().sum())
    if 'terminal' in df.columns:
        terminal_col = 'terminal'
    else:
        terminal_col = None
        # try to find a column that could be terminal
        for c in ['terminal', 'terminals', 'done', 'dones']:
            if c in df.columns:
                terminal_col = c
                break
    if terminal_col is not None:
        # rows with any NaN
        nan_rows = df[numeric.isna().any(axis=1)]
        # among those, check if all are terminal
        if len(nan_rows) > 0:
            all_terminal = nan_rows[terminal_col].all()
            if all_terminal:
                result("No unexpected NaN values (only in terminal rows)", True,
                       f"Total NaN count = {nan_count_total}, all in terminal rows")
            else:
                result("No unexpected NaN values", False,
                       f"NaNs found in non-terminal rows: {nan_rows[~nan_rows[terminal_col]].shape[0]}")
        else:
            result("No NaN values", True, "No NaNs found")
    else:
        result("No NaN values", nan_count_total == 0, f"NaN count = {nan_count_total}")

    # ------------------------------------------------------------
    # 2. STATE RANGE CHECK
    # ------------------------------------------------------------
    print_header("2. STATE VALUE RANGE CHECK")
    range_failures = 0
    for key, col in state_cols.items():
        lo, hi = EXPECTED_RANGES[key]
        values = df[col].dropna().to_numpy()
        if len(values) == 0:
            continue
        actual_min = values.min()
        actual_max = values.max()
        passed = actual_min >= lo and actual_max <= hi
        if not passed:
            range_failures += 1
        result(key, passed, f"range = [{actual_min:.6g}, {actual_max:.6g}], expected [{lo}, {hi}]")
    result("All detected state ranges valid", range_failures == 0, f"range failures = {range_failures}")

    # ------------------------------------------------------------
    # 3. ACTION DISTRIBUTION
    # ------------------------------------------------------------
    print_header("3. ACTION DISTRIBUTION")
    if action_col is None:
        result("Action distribution", False, "Could not detect action column.")
    else:
        counts = df[action_col].value_counts().sort_index()
        print("\nAction counts:")
        for action, count in counts.items():
            pct = 100.0 * count / len(df)
            name = ACTION_NAMES.get(int(action), f"action {action}")
            print(f"  {action}: {name:12s} {count:7,} ({pct:6.2f}%)")
        max_fraction = counts.max() / len(df)
        passed = len(counts) >= 3 and max_fraction < 0.80
        result("Action distribution not badly collapsed", passed,
               f"unique actions = {len(counts)}, maximum action fraction = {max_fraction:.3f}")

    # ------------------------------------------------------------
    # 4. REWARD DISTRIBUTION
    # ------------------------------------------------------------
    print_header("4. REWARD DISTRIBUTION")
    if reward_col is None:
        result("Reward distribution", False, "Could not detect reward column.")
    else:
        rewards = df[reward_col].dropna().to_numpy()
        print(f"\nReward min:    {rewards.min():.6f}")
        print(f"Reward max:    {rewards.max():.6f}")
        print(f"Reward mean:   {rewards.mean():.6f}")
        print(f"Reward median: {np.median(rewards):.6f}")
        print(f"Reward std:    {rewards.std():.6f}")
        unique_rewards = len(np.unique(rewards))
        passed = len(rewards) > 0 and np.isfinite(rewards).all() and rewards.std() > 0 and unique_rewards > 10
        result("Reward distribution is non-degenerate", passed, f"unique reward values = {unique_rewards}")
        reward_upper_bound = 1.500001
        upper_ok = rewards.max() <= reward_upper_bound
        result("Reward respects documented upper bound", upper_ok,
               f"observed max = {rewards.max():.6f}, expected <= 1.5")

    # ------------------------------------------------------------
    # 5. DUPLICATE TRANSITIONS
    # ------------------------------------------------------------
    print_header("5. DUPLICATE TRANSITION CHECK")
    possible_state_cols = list(state_cols.values())
    possible_next_cols = list(next_state_cols.values())
    transition_cols = (possible_state_cols + ([action_col] if action_col else []) +
                       possible_next_cols + ([reward_col] if reward_col else []))
    transition_cols = [c for c in transition_cols if c in df.columns]
    if len(transition_cols) >= 2:
        duplicate_count = int(df.duplicated(subset=transition_cols).sum())
        duplicate_fraction = duplicate_count / len(df)
        print(f"Exact duplicate transitions: {duplicate_count:,}")
        print(f"Duplicate fraction: {duplicate_fraction:.4%}")
        passed = duplicate_fraction < 0.10
        result("Transitions are not excessively duplicated", passed, f"duplicate fraction = {duplicate_fraction:.4%}")
    else:
        result("Duplicate transition check", False, "Insufficient detectable transition columns.")

    # ------------------------------------------------------------
    # 6. NEXT STATE RESPONSE TO ACTION
    # ------------------------------------------------------------
    print_header("6. NEXT-STATE RESPONSE TO ACTION")
    if action_col is None or not next_state_cols:
        result("Action -> next_state response", False, "Could not detect required columns.")
    else:
        print("\nMean groundwater next-state by action:")
        gw_next_col = next_state_cols.get("gw_level")
        if gw_next_col:
            grouped = df.groupby(action_col)[gw_next_col].agg(["count", "mean", "std"]).sort_index()
            print(grouped.to_string())
            result("Different actions produce different next states",
                   grouped["mean"].nunique() > 1,
                   "next groundwater means vary across actions")
        else:
            result("Groundwater next-state response", False, "next groundwater column not detected.")

    # ------------------------------------------------------------
    # 7. HIGHER PUMPING -> GREATER DRAWDOWN
    # ------------------------------------------------------------
    print_header("7. PUMPING -> DRAWDOWN CHECK")
    if (action_col is not None and "gw_level" in state_cols and "gw_level" in next_state_cols):
        state_gw = state_cols["gw_level"]
        next_gw = next_state_cols["gw_level"]
        temp = df[[action_col, state_gw, next_gw]].dropna().copy()
        # Correct drawdown: pumping increases depth, so drawdown = next - current
        temp["drawdown"] = temp[next_gw] - temp[state_gw]
        grouped = temp.groupby(action_col)["drawdown"].agg(["count", "mean", "median"]).sort_index()
        print("\nDrawdown by action (positive = deeper):")
        print(grouped.to_string())
        means = grouped["mean"].to_numpy()
        monotonic_pairs = 0
        total_pairs = max(0, len(means) - 1)
        for i in range(len(means) - 1):
            if means[i + 1] >= means[i]:
                monotonic_pairs += 1
        if total_pairs > 0:
            monotonic_fraction = monotonic_pairs / total_pairs
        else:
            monotonic_fraction = 0
        passed = monotonic_fraction >= 0.75
        result("Higher pumping generally causes greater drawdown", passed,
               f"monotonic adjacent pairs = {monotonic_pairs}/{total_pairs}")
    else:
        result("Pumping -> drawdown", False, "Required state/action columns not detected.")

    # ------------------------------------------------------------
    # 8. RAINFALL -> RECHARGE
    # ------------------------------------------------------------
    print_header("8. RAINFALL -> RECHARGE CHECK")
    if ("recent_rainfall" in state_cols and "gw_level" in state_cols and "gw_level" in next_state_cols):
        rain_col = state_cols["recent_rainfall"]
        gw_col = state_cols["gw_level"]
        next_gw_col = next_state_cols["gw_level"]
        temp = df[[rain_col, gw_col, next_gw_col]].dropna().copy()
        temp["gw_change"] = temp[next_gw_col] - temp[gw_col]   # positive = deeper, negative = shallower
        q25 = temp[rain_col].quantile(0.25)
        q75 = temp[rain_col].quantile(0.75)
        low_rain = temp[temp[rain_col] <= q25]["gw_change"]
        high_rain = temp[temp[rain_col] >= q75]["gw_change"]
        low_mean = low_rain.mean()
        high_mean = high_rain.mean()
        print(f"Low rainfall threshold:  {q25:.6f}")
        print(f"High rainfall threshold: {q75:.6f}")
        print(f"Low-rainfall mean GW change:  {low_mean:.6f} (positive = deeper)")
        print(f"High-rainfall mean GW change: {high_mean:.6f} (positive = deeper)")
        # Recharge should reduce depth (more negative change), so high_mean < low_mean
        passed = high_mean < low_mean
        result("Rainfall generally produces recharge", passed,
               "Higher-rainfall states show greater groundwater rise (more negative change).")
    else:
        result("Rainfall -> recharge", False, "Required columns not detected.")

    # ------------------------------------------------------------
    # 9. REWARD VS ACTION
    # ------------------------------------------------------------
    print_header("9. REWARD BEHAVIOR BY ACTION")
    if action_col is not None and reward_col is not None:
        grouped = df.groupby(action_col)[reward_col].agg(["count", "mean", "median", "std"]).sort_index()
        print("\nReward statistics by action:")
        print(grouped.to_string())
        result("Reward varies across actions", grouped["mean"].nunique() > 1,
               "Mean reward is not identical for every action.")
    else:
        result("Reward vs action", False, "Action/reward columns not detected.")

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    print_header("VALIDATION COMPLETE")
    print("""
Interpretation:

PASS means the transition generator behaves consistently with
the current documented synthetic assumptions.

This validation does NOT establish that the synthetic environment
is a physically calibrated representation of the real aquifer.

If the checks pass, the dataset is suitable to proceed to:

    Phase 5
      1. Behavior Cloning (BC)
      2. CQL
      3. FQE
      4. Bootstrap confidence intervals
      5. CQL vs BC comparison
      6. Unsafe-action validation

If any important check FAILS, fix the transition generator before
training CQL.
""")

if __name__ == "__main__":
    main()
    