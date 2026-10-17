import os
from huggingface_hub import hf_hub_download

COMFYUI_DIR = "/workspace/ComfyUI"
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")


def download(repo_id: str, filename: str, subfolder: str) -> None:
    target_dir = os.path.join(MODELS_DIR, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    print(f"[MODEL] {repo_id}/{filename}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
    )


# Eros Max beta6, 8-step TURBO, W4A8.
download(
    "berryber09/10Eros-Max-h3-turbo-beta6-pruned-w4a8",
    "10Eros_Max_h3_8stepTURBO_beta6_pruned_w4a8.safetensors",
    "diffusion_models",
)

# Official MiniMax H3 text encoder.
download(
    "Comfy-Org/MiniMax-H3",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "text_encoders",
)

# Official lightweight INT8 ConvRot video VAE.
download(
    "Comfy-Org/MiniMax-H3",
    "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "vae",
)

# Official H3 audio VAE.
download(
    "Comfy-Org/MiniMax-H3",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
    "vae",
)

print("[MODEL] All Eros Max / MiniMax H3 models downloaded.", flush=True)
