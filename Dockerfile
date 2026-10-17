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

WORKDIR /workspace/ComfyUI

RUN pip install --break-system-packages -r requirements.txt
RUN pip install --break-system-packages runpod requests huggingface_hub hf_transfer

WORKDIR /workspace

COPY download_models.py /workspace/download_models.py
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
