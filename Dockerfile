FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth 1 \
    https://github.com/comfyanonymous/ComfyUI.git \
    /workspace/ComfyUI

# H3 video-context extension nodes (test branch only)
RUN git clone --depth 1 https://github.com/kat3ri/ComfyUI-MiniMax-H3-Extend.git /workspace/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Extend
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

WORKDIR /workspace/ComfyUI

RUN pip install --break-system-packages -r requirements.txt
RUN pip install --break-system-packages runpod requests huggingface_hub hf_transfer
RUN pip install --break-system-packages -r /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# SageAttention 2.2.0 wheel matched to torch 2.12.0 / CUDA 13.0 / Python 3.12 / RTX 4090 (SM89).
RUN pip install --break-system-packages --no-deps \
    https://github.com/tcpassos/sage-wheels-linux/releases/download/sage-2.2.0-torch-2.12.0-cu130-py312/sageattention-2.2.0-89-cp312-cp312-linux_x86_64.whl
RUN python -c "import sageattention; print('SageAttention import OK')"

WORKDIR /workspace

# Bundle the optional FL2VA LoRA at build time.
# The build fails if Civitai does not provide the actual safetensors file.
RUN --mount=type=secret,id=civitai_api_token python - <<'PY'
import os
import requests

with open("/run/secrets/civitai_api_token", "r") as secret_file:
    token = secret_file.read().strip()
if not token:
    raise RuntimeError("CIVITAI_API_TOKEN secret is empty")
url = "https://civitai.com/api/download/models/3258590"
path = "/workspace/ComfyUI/models/loras/bounceV07_fl2va-000230_Intense.safetensors"
os.makedirs(os.path.dirname(path), exist_ok=True)
with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=(30, 180)) as response:
    response.raise_for_status()
    with open(path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
with open(path, "rb") as downloaded:
    header = downloaded.read(8)
    if len(header) != 8 or not 0 < int.from_bytes(header, "little") < 100_000_000:
        raise RuntimeError("Civitai response is not a valid safetensors file")
print("[LoRA] bundled in Docker image:", os.path.getsize(path), "bytes", flush=True)
PY

# Bundle Mystic XXX V4 alongside the existing Bounce LoRA.
RUN --mount=type=secret,id=civitai_api_token python - <<'PY'
import hashlib
import os
import requests

with open("/run/secrets/civitai_api_token", "r") as secret_file:
    token = secret_file.read().strip()
if not token:
    raise RuntimeError("CIVITAI_API_TOKEN secret is empty")
url = "https://civitai.com/api/download/models/3266628"
path = "/workspace/ComfyUI/models/loras/MysticXXX_MMH3-V4.safetensors"
expected_sha256 = "fc3e856d14c6c19557c888f48662d591e4794e281233ec0d987be5003068afba"
os.makedirs(os.path.dirname(path), exist_ok=True)
hasher = hashlib.sha256()
with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=(30, 180)) as response:
    response.raise_for_status()
    with open(path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                hasher.update(chunk)
                output.write(chunk)
if hasher.hexdigest().lower() != expected_sha256:
    os.remove(path)
    raise RuntimeError("Mystic XXX V4 download SHA256 mismatch")
print("[LoRA] Mystic XXX V4 bundled:", os.path.getsize(path), "bytes", flush=True)
PY

COPY download_models.py /workspace/download_models.py
COPY handler.py /workspace/handler.py
COPY registered_lora.py /workspace/registered_lora.py

CMD ["python", "/workspace/handler.py"]
