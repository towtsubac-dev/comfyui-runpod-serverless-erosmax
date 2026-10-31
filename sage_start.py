"""Select the pre-bundled SageAttention wheel matching the assigned GPU."""
import os
import sys

import torch

major, minor = torch.cuda.get_device_capability(0)
architecture = (major, minor)
if architecture == (8, 9):
    sage_dir = "/opt/sage/sm89"
elif architecture == (12, 0):
    sage_dir = "/opt/sage/sm120"
else:
    raise RuntimeError(
        f"Unsupported GPU capability {major}.{minor}; expected SM89 or SM120"
    )

os.environ["PYTHONPATH"] = sage_dir + (
    os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
)
print(
    f"[ErosMax] GPU: {torch.cuda.get_device_name(0)}, "
    f"capability: {architecture}, SageAttention: {sage_dir}",
    flush=True,
)
os.execvpe(sys.executable, [sys.executable, "/workspace/handler.py"], os.environ)
