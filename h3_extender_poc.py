"""Isolated RunPod PoC for tritant/ComfyUI_MiniMax_H3_Extender v3.0.14.

No changes to existing ErosMax extension workflow. This deliberately builds a
single-clip, 5-second REF2VA/Motion Context workflow from a native VIDEO input.
A GPU test is required to establish checkpoint and ComfyUI compatibility.
"""
from __future__ import annotations

import json
import uuid


UPSTREAM_SHA = "b3e23bded3bcfb2398ec2470943b35330c7faaff"
REQUIRED_NODES = (
    "LoadVideo",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "MiniMaxH3SigmaShift",
    "MiniMaxH3Extender",
    "MiniMaxH3MotionContextDiskFinalDecode",
)


def _default_clip(prompt: str, seconds: int, seed: int) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty continuation prompt is required.")
    if seconds != 5:
        raise ValueError("PoC supports only a single 5-second continuation.")
    if not 0 <= int(seed) < 10_000_000_000:
        raise ValueError("Seed must be between 0 and 9,999,999,999.")
    return {
        "id": "clip_1",
        "name": "RunPod PoC continuation",
        "prompt": prompt.strip(),
        "seed": int(seed),
        "seed_mode": "fixed",
        "duration": 5.0,
        "validated": False,
        "color_adjustment": {
            "saturation": 100.0, "contrast": 100.0, "brightness": 100.0,
        },
        "loras": [],
        "local_refs": {"version": 1, "images": [], "videos": [], "audios": []},
        "first_frame": None,
        "last_frame": None,
        "guides": [],
        "first_source": "manual",
    }


def build_workflow(
    *,
    video_filename: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    model_filename: str,
    clip_filename: str,
    video_vae_filename: str,
    audio_vae_filename: str,
    seconds: int = 5,
    context_length: int = 5,
    steps: int = 8,
    filename_prefix: str | None = None,
) -> dict:
    """Build native LoadVideo → Extender → Final Decode ComfyUI API JSON.

    Refine, additional LoRAs, and source-image reference management are
    intentionally disabled for the first single-inference comparison.
    """
    if not video_filename or "/" in video_filename or "\\" in video_filename:
        raise ValueError("video_filename must be a single basename in ComfyUI/input.")
    if not width or not height or width % 32 or height % 32:
        raise ValueError("PoC source width/height must be multiples of 32.")
    if int(context_length) not in (5, 22):
        raise ValueError("PoC context_length must be 5 or 22.")
    if int(steps) != 8:
        raise ValueError("PoC is fixed at 8 sampling steps.")

    clips_json = json.dumps(
        {"version": 2, "generation_mode": "ref2va", "motion_context": True,
         "clips": [_default_clip(prompt, seconds, seed)],
         "continue_video_resize": {"mode": "original"}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    refs_json = json.dumps(
        {"version": 2, "refs": [None] * 9},
        separators=(",", ":"),
    )
    prefix = filename_prefix or ("ErosH3Extender_PoC_" + uuid.uuid4().hex)
    if "/" in prefix or "\\" in prefix or not prefix.startswith("ErosH3Extender_PoC_"):
        raise ValueError("PoC output prefix must be namespaced under ErosH3Extender_PoC_.")

    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_filename}},
        "2": {"class_type": "UNETLoader",
              "inputs": {"unet_name": model_filename, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip_filename, "type": "minimax",
                         "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_filename}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_filename}},
        "6": {"class_type": "MiniMaxH3SigmaShift",
              "inputs": {"model": ["2", 0], "shift_video": 12.0,
                         "shift_audio": 3.0}},
        "7": {
            "class_type": "MiniMaxH3Extender",
            "inputs": {
                "model": ["6", 0],
                "clip": ["3", 0],
                "vae": ["4", 0],
                "audio_vae": ["5", 0],
                "continue_existing_video": ["1", 0],
                "run_mode": "full_batch",
                "width": int(width),
                "height": int(height),
                "ref_image_size": "match",
                "steps": int(steps),
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "context_length": str(context_length),
                "audio_context_length": 0,
                "clips_json": clips_json,
                "resolution_mode": "manual",
                "megapixels": round(width * height / 1_000_000, 3),
                "refs_json": refs_json,
                "generation_mode": "ref2va",
                "motion_context": True,
                "refine_enabled": False,
                "refine_scale": 1.5,
                "refine_steps": 4,
                "refine_denoise": 0.30,
            },
        },
        "8": {
            "class_type": "MiniMaxH3MotionContextDiskFinalDecode",
            "inputs": {
                "cache": ["7", 0],
                "vae": ["4", 0],
                "audio_vae": ["5", 0],
                "fps": 24.0,
                "filename_prefix": prefix,
                "output_directory": "",
                "codec": "H.264 CPU (libx264)",
                "crf": 17,
                "preset": "fast",
                "audio_bitrate": "192k",
                "autoplay": False,
                "auto_save_project": False,
                "save_individual_clips": False,
            },
        },
    }


def summarize_comfy_contract(object_info: dict) -> dict:
    """Bounded diagnostic, without serializing all model names/media metadata."""
    found = {}
    for node in REQUIRED_NODES:
        metadata = object_info.get(node)
        if not isinstance(metadata, dict):
            found[node] = {"available": False}
            continue
        inputs = metadata.get("input") or {}
        found[node] = {
            "available": True,
            "required": sorted((inputs.get("required") or {}).keys()),
            "optional": sorted((inputs.get("optional") or {}).keys()),
            "return_types": metadata.get("output") or [],
        }
    return {
        "status": "ok" if all(v["available"] for v in found.values()) else "missing_nodes",
        "upstream_sha": UPSTREAM_SHA,
        "nodes": found,
        "missing": [k for k, v in found.items() if not v["available"]],
        "next_step": "GPU-generation comparison is not yet validated",
    }
