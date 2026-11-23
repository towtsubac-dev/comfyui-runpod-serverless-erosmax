"""Pinned ErosMax v1.0 Compact assets, isolated from beta6 Production.

The v1.0 transformer is verified against the publisher's SHA-256 sidecar.
Cold starts reuse image-baked weights and do not redownload or rehash them.
"""
import hashlib
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path("/workspace/ComfyUI/models")
MARKER = Path("/workspace/.eros_v1_compact_verified")
V1_REPO = "LokkenJP/TenEros_Max_optimized_learned"
V1_REVISION = "3415250db9e2cb89b886f5375c1306fc933e215f"
V1_FILENAME = "10Eros_Max_8stepTurbo_v1.0_hybrid_activation_learned_compact_w4w6int8_14.50gb.safetensors"
V1_SHA256 = "569f4337f6101b909c5fcf96efb207464a862a540394cff2723534caea4c73ed"
OTHER_REPO = "Comfy-Org/MiniMax-H3"

ASSETS = [
    (V1_REPO, V1_FILENAME, "diffusion_models/" + V1_FILENAME, V1_REVISION, 14_000_000_000),
    (OTHER_REPO, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", None, 1_000_000_000),
    (OTHER_REPO, "vae/minimax_h3_video_vae_int8_convrot.safetensors",
     "vae/minimax_h3_video_vae_int8_convrot.safetensors", None, 100_000_000),
    (OTHER_REPO, "vae/minimax_h3_audio_vae_fp32.safetensors",
     "vae/minimax_h3_audio_vae_fp32.safetensors", None, 1_000_000),
]

def fast_valid():
    return MARKER.exists() and all(
        (ROOT / dst).is_file() and (ROOT / dst).stat().st_size >= minimum
        for _, _, dst, _, minimum in ASSETS
    )

def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def main():
    verify = "--verify" in sys.argv
    if fast_valid() and not verify:
        print("[EROS V1] image-baked checked models available", flush=True)
        return
    for repo, remote_file, destination, revision, minimum in ASSETS:
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.stat().st_size < minimum:
            print(f"[EROS V1] downloading {repo}/{remote_file}", flush=True)
            local_dir = ROOT / "diffusion_models" if repo == V1_REPO else ROOT
            src = Path(hf_hub_download(
                repo_id=repo,
                filename=remote_file,
                revision=revision,
                local_dir=str(local_dir),
                token=os.getenv("HF_TOKEN") or None,
            ))
            if src.resolve() != target.resolve():
                src.replace(target)
        if target.stat().st_size < minimum:
            raise RuntimeError(f"Model weight truncated: {target.name}")
        print(f"[EROS V1] ready {target.name}: {target.stat().st_size} bytes", flush=True)
    if not MARKER.exists() or verify:
        actual = sha256(ROOT / "diffusion_models" / V1_FILENAME)
        if actual.lower() != V1_SHA256:
            raise RuntimeError(f"ErosMax v1 Compact SHA256 mismatch: {actual}")
        MARKER.write_text("v1.0-compact\n" + actual + "\n")
        print("[EROS V1] SHA256 matches official sidecar", flush=True)

if __name__ == "__main__":
    main()
