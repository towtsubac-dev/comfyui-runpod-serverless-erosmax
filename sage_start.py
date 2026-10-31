"""Select and CUDA-test SageAttention; fall back safely to PyTorch attention."""
import os
import subprocess
import sys

import torch

architecture = torch.cuda.get_device_capability(0)
sage_dirs = {(8, 9): "/opt/sage/sm89", (12, 0): "/opt/sage/sm120"}
sage_dir = sage_dirs.get(architecture)
print(f"[ErosMax] GPU: {torch.cuda.get_device_name(0)}, capability: {architecture}", flush=True)

if sage_dir is None:
    print("[ErosMax] Unsupported SageAttention GPU; using PyTorch attention.", flush=True)
    os.environ["EROSMAX_DISABLE_SAGE"] = "1"
else:
    os.environ["PYTHONPATH"] = sage_dir + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    # Isolate CUDA failures from the long-lived worker process.
    smoke_test = """
import torch
from sageattention import sageattn
q = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
torch.cuda.synchronize()
assert out.shape == q.shape
assert torch.isfinite(out).all().item()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", smoke_test],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout)[-1600:])
        print(f"[ErosMax] SageAttention CUDA test passed: {sage_dir}", flush=True)
    except Exception as exc:
        print(f"[ErosMax] SageAttention CUDA test failed: {exc}", flush=True)
        print("[ErosMax] Falling back to PyTorch attention.", flush=True)
        os.environ["EROSMAX_DISABLE_SAGE"] = "1"

os.execvpe(sys.executable, [sys.executable, "/workspace/handler.py"], os.environ)
