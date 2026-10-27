FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface

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

# Keep architecture-specific wheels isolated; choose at container startup.
RUN pip install --break-system-packages --no-deps --target /opt/sage/sm89 \
    https://github.com/tcpassos/sage-wheels-linux/releases/download/sage-2.2.0-torch-2.12.0-cu130-py312/sageattention-2.2.0-89-cp312-cp312-linux_x86_64.whl
RUN pip install --break-system-packages --no-deps --target /opt/sage/sm120 \
    https://github.com/tcpassos/sage-wheels-linux/releases/download/sage-2.2.0-torch-2.12.0-cu130-py312/sageattention-2.2.0-120-cp312-cp312-linux_x86_64.whl

WORKDIR /workspace

COPY download_models.py /workspace/download_models.py
COPY handler.py /workspace/handler.py
COPY registered_lora.py /workspace/registered_lora.py

COPY sage_launcher.py /workspace/sage_launcher.py
CMD ["python", "/workspace/sage_launcher.py"]
