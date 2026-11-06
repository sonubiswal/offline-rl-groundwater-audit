import json
from pathlib import Path

d = json.loads(Path(r"C:\trishna-opal\revision_experiments"
                    r"\family_start_rerun.json").read_text())

for r in d["rows"]:
    print(f"{r['bc']:>10} & {r['cql']:>10} & "
          f"{r['rf_grid']:+.4f} & {r['common_eps']:+.4f} \\\\")