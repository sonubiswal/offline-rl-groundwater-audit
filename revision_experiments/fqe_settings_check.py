"""
Inspect the FQE json and source for gamma, target-update interval,
critic count, and terminal handling. Determines whether FQE is
configured to estimate the 20-step undiscounted return, matching
the rollout.

Run from C:\\trishna-opal:
    python revision_experiments\\fqe_settings_check.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
fqe_path = REPO / "revision_experiments" / "E2" / "e2_fqe.json"
if not fqe_path.exists():
    fqe_path = REPO / "revision_experiments" / "e2_fqe.json"

fqe = json.loads(fqe_path.read_text())

# Recursively find any key matching gamma/terminal/target/discount/critic
INTEREST = ("gamma", "discount", "terminal", "timeout", "target",
            "critic", "n_critics", "reward_scaler", "learning_rate",
            "n_steps", "batch_size")

def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if any(x in k.lower() for x in INTEREST):
                print(f"  {p} = {v}")
            walk(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node[:3]):
            walk(v, f"{path}[{i}]")

print("=== keys containing FQE settings ===")
walk(fqe)

print("\n=== top-level protocol block ===")
print(json.dumps(fqe.get("protocol", {}), indent=2))

print("\n=== convergence block ===")
print(json.dumps(fqe.get("convergence", {}), indent=2)[:800])