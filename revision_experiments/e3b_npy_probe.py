import numpy as np
SEEDS = [42, 123, 2024, 7, 99, 777, 1234, 5678, 12345, 99999]
print("seed      n     bc_mean    cq_mean    cq-bc")
for s in SEEDS:
    bc = np.load(f"reports_10seed/epsilon/returns/returns_bc_seed{s}.npy")
    cq = np.load(f"reports_10seed/epsilon/returns/returns_cql_seed{s}.npy")
    print(f"{s:7d}  {len(bc):4d}  {bc.mean():+.4f}  {cq.mean():+.4f}  {(cq-bc).mean():+.4f}")
print()
all_bc = np.concatenate([np.load(f"reports_10seed/epsilon/returns/returns_bc_seed{s}.npy") for s in SEEDS])
all_cq = np.concatenate([np.load(f"reports_10seed/epsilon/returns/returns_cql_seed{s}.npy") for s in SEEDS])
print(f"pooled  bc_mean={all_bc.mean():+.4f}  cq_mean={all_cq.mean():+.4f}  gap={(all_cq-all_bc).mean():+.4f}")
