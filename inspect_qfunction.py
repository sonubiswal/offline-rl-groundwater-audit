"""
inspect_qfunction.py -- one-off diagnostic to print the real structure
of a loaded QR CQL checkpoint's q-function, so we can stop guessing
attribute names for the per-quantile extraction.

USAGE
-----
python inspect_qfunction.py models_epsilon_qr32_verify/cql_seed42.d3
"""
import sys
import torch
import d3rlpy

if len(sys.argv) != 2:
    print("Usage: python inspect_qfunction.py <path_to_checkpoint.d3>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"Loading {ckpt_path} ...")
cql = d3rlpy.load_learnable(ckpt_path)

# Try both the public and private accessor -- if only one exists this
# tells us which, if both exist we confirm they're the same object.
impl_public = getattr(cql, "impl", None)
impl_private = getattr(cql, "_impl", None)
print(f"\ncql.impl  -> {type(impl_public)}")
print(f"cql._impl -> {type(impl_private)}")
if impl_public is not None and impl_private is not None:
    print(f"same object? {impl_public is impl_private}")

impl = impl_public if impl_public is not None else impl_private
if impl is None:
    print("\nNeither cql.impl nor cql._impl exists. Top-level cql attributes:")
    print([a for a in dir(cql) if not a.startswith("__")])
    sys.exit(1)

for attr_name in ("q_function", "_q_func", "q_func"):
    qf = getattr(impl, attr_name, None)
    if qf is None:
        print(f"\nimpl.{attr_name} -> not found")
        continue
    print(f"\nimpl.{attr_name} -> {type(qf)}")
    if hasattr(qf, "named_children"):
        children = list(qf.named_children())
        print(f"  named_children: {[n for n, _ in children]}")
        for name, child in children:
            print(f"  -- child '{name}': {type(child)}")
            print(f"     public attrs/methods: {[m for m in dir(child) if not m.startswith('_')]}")
            if hasattr(child, "named_children"):
                grandchildren = list(child.named_children())
                if grandchildren:
                    print(f"     its own children: {[n for n, _ in grandchildren]}")

    # Try an actual forward pass with a dummy input matching this
    # checkpoint's observation shape, and print what comes back.
    try:
        obs_shape = impl.observation_shape
    except Exception:
        obs_shape = (6,)  # fallback: known from this project's config
    dummy_x = torch.zeros((2,) + tuple(obs_shape), dtype=torch.float32)

    if hasattr(qf, "__getitem__"):
        try:
            sub = qf[0]
            print(f"\n  Trying qf[0](dummy_x) where qf[0] is {type(sub)} ...")
            with torch.no_grad():
                out = sub(dummy_x)
            print(f"  -> output type: {type(out)}")
            if isinstance(out, torch.Tensor):
                print(f"  -> output shape: {tuple(out.shape)}")
            else:
                print(f"  -> output public attrs: {[a for a in dir(out) if not a.startswith('_')]}")
                for cand in ("quantiles", "q_value", "values", "logits", "taus"):
                    val = getattr(out, cand, None)
                    if val is not None:
                        shape = tuple(val.shape) if isinstance(val, torch.Tensor) else type(val)
                        print(f"     .{cand} -> {shape}")
        except Exception as e:
            print(f"  qf[0](dummy_x) raised: {type(e).__name__}: {e}")

print("\nDone. Paste this whole output back for the next fix.")