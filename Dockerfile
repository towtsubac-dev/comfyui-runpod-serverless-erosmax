# Build SageAttention against the exact PyTorch/CUDA runtime used by ErosMax.
# The build stage contains nvcc; the final image stays on the smaller runtime base.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS cuda_toolkit
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime AS sage_builder
COPY --from=cuda_toolkit /usr/local/cuda-13.0 /usr/local/cuda-13.0
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=/usr/local/cuda-13.0/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ ninja-build libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages --no-cache-dir packaging ninja setuptools wheel
RUN git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
RUN cd /tmp/SageAttention && pip wheel --no-build-isolation --no-deps -w /tmp/sage-wheels .

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

# Locally built SageAttention supports both RTX 4090 (SM89) and Blackwell (SM120).
COPY --from=sage_builder /tmp/sage-wheels/ /tmp/sage-wheels/
RUN pip install --break-system-packages --no-deps /tmp/sage-wheels/sageattention*.whl && rm -rf /tmp/sage-wheels
RUN python -c "import sageattention, torch; print('SageAttention import OK; torch=', torch.__version__)"

WORKDIR /workspace

COPY download_models.py /workspace/download_models.py

# Download all required model weights during image build, not after a
# Serverless GPU worker starts. Existing handler checks these files and
# skips downloads when they are already present.
RUN python /workspace/download_models.py

COPY handler.py /workspace/handler.py
COPY registered_lora.py /workspace/registered_lora.py

CMD ["python", "/workspace/handler.py"]
