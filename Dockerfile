FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

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

COPY download_models.py /workspace/download_models.py
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
