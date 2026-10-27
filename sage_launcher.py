import os
import sys
import torch

# Select before ComfyUI starts; unsupported GPUs use native PyTorch attention.
try:
    capability = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
except Exception as exc:
    raise RuntimeError("Cannot identify the allocated CUDA device") from exc

paths = {(8, 9): "/opt/sage/sm89", (12, 0): "/opt/sage/sm120"}
path = paths.get(capability)
env = os.environ.copy()
env["EROS_USE_SAGE"] = "1" if path else "0"
if path:
    env["PYTHONPATH"] = path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
print(f"[ErosMax] GPU: {name}, capability: {capability}, attention: {'sage ' + path if path else 'pytorch'}", flush=True)
os.execve(sys.executable, [sys.executable, "/workspace/handler.py"], env)
