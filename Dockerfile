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

# Test-only: SageAttention 2.2.0 matched to torch 2.12.0 / CUDA 13.0 / Python 3.12 / Blackwell (SM120).
# The production RTX 4090 (SM89) image remains untouched.
RUN pip install --break-system-packages --no-deps \
    https://github.com/tcpassos/sage-wheels-linux/releases/download/sage-2.2.0-torch-2.12.0-cu130-py312/sageattention-2.2.0-120-cp312-cp312-linux_x86_64.whl
RUN python -c "import sageattention; print('SageAttention import OK')"

WORKDIR /workspace

# Bake only shared MiniMax H3 support weights into the image.
# The Eros Max diffusion model remains on RunPod's host-side model cache.
# Files are placed where download_models.py already checks for them.
RUN python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in (
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
):
    hf_hub_download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=filename,
        local_dir="/workspace/ComfyUI/models",
    )
    print(f"[BUILD] bundled {filename}", flush=True)
PY

COPY download_models.py /workspace/download_models.py
COPY handler.py /workspace/handler.py
COPY registered_lora.py /workspace/registered_lora.py

CMD ["python", "/workspace/handler.py"]
