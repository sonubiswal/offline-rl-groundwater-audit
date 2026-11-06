"""Merge the two BCQ control-run result JSONs into a single file with all
ten seeds. The first file has 3 seeds, the second has 7. The union is 10.
"""
import json
from pathlib import Path

REP = Path(r"C:\trishna-opal\reports")
a = json.loads((REP / "epsilon_bcq_control_results.json").read_text())
b = json.loads((REP / "epsilon_bcq_control_results_part2.json").read_text())

merged = dict(a)
merged["per_seed"] = sorted(a["per_seed"] + b["per_seed"],
                            key=lambda r: r["seed"])
merged["training_seeds"] = [r["seed"] for r in merged["per_seed"]]

import numpy as np
vals = [r["val_action_match"]["overall"] for r in merged["per_seed"]]
tests = [r["test_action_match"]["overall"] for r in merged["per_seed"]]
merged["val_action_match_mean"] = float(np.mean(vals))
merged["val_action_match_std"]  = float(np.std(vals))
merged["test_action_match_mean"] = float(np.mean(tests))
merged["test_action_match_std"]  = float(np.std(tests))
merged["n_seeds_total"] = len(merged["per_seed"])
merged["merged_from"] = [
    "epsilon_bcq_control_results.json",
    "epsilon_bcq_control_results_part2.json",
]

(REP / "epsilon_bcq_control_results.json").write_text(
    json.dumps(merged, indent=2))

print(f"merged: {len(merged['per_seed'])} seeds")
print(f"seeds:  {merged['training_seeds']}")
print(f"test mean: {merged['test_action_match_mean']:.4f}")
print(f"val  mean: {merged['val_action_match_mean']:.4f}")