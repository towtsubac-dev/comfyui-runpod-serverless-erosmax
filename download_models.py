import os
from huggingface_hub import hf_hub_download

COMFYUI_MODELS = "/workspace/ComfyUI/models"


def download(repo_id: str, filename: str) -> None:
    target = os.path.join(COMFYUI_MODELS, filename)
    os.makedirs(os.path.dirname(target), exist_ok=True)

    if os.path.isfile(target) and os.path.getsize(target) > 0:
        print(f"[MODEL] already exists: {target}", flush=True)
        return

    print(f"[MODEL] downloading {repo_id}/{filename}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=COMFYUI_MODELS,
    )
    print(f"[MODEL] ready: {target}", flush=True)


# Eros Max beta6: 8-step TURBO, W4A8.
download(
    "berryber09/10Eros-Max-h3-turbo-beta6-pruned-w4a8",
    "10Eros_Max_h3_8stepTURBO_beta6_pruned_w4a8.safetensors",
)

# Official MiniMax H3 NVFP4/AWQ text encoder.
download(
    "Comfy-Org/MiniMax-H3",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
)

# Official INT8 ConvRot video VAE.
download(
    "Comfy-Org/MiniMax-H3",
    "vae/minimax_h3_video_vae_int8_convrot.safetensors",
)

# Official H3 audio VAE.
download(
    "Comfy-Org/MiniMax-H3",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
)

print("[MODEL] All Eros Max / MiniMax H3 models are ready.", flush=True)
