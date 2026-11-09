"""Isolated RunPod PoC for tritant/ComfyUI_MiniMax_H3_Extender v3.0.14.

No changes to existing ErosMax extension workflow. This deliberately builds a
single-clip, 5-second REF2VA/Motion Context workflow from a native VIDEO input.
A GPU test is required to establish checkpoint and ComfyUI compatibility.
"""
from __future__ import annotations

import json
import math
import re
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


def continuation_frame_plan(seconds: int, context_length: int) -> dict:
    """Match the *visible* continuation length across H3 Motion Context sizes.

    Upstream Extender uses 17k+5 native frames and trims the first N context
    frames from the generated segment. These are the only two tested context
    lengths. Both can yield 119 new frames (4.9583 s at 24 fps).

    Passing 5.0 seconds for context=5 preserves the already-verified output:
    upstream rounds 120 to 124 native frames. Context=22 must request 141
    native frames instead (141/24=5.875 seconds), otherwise it yields only
    102 visible frames. This plan is a computation, not a GPU validation.
    """
    if int(seconds) != 5:
        raise ValueError("PoC supports only a single 5-second continuation.")
    context = int(context_length)
    if context not in (5, 22):
        raise ValueError("PoC context_length must be 5 or 22.")
    fps = 24
    wanted_visible_frames = 119
    native_frames = wanted_visible_frames + context
    if native_frames % 17 != 5:
        raise RuntimeError("Invalid H3 17k+5 temporal grid.")
    return {
        "fps": fps,
        "requested_seconds": int(seconds),
        "context_frames": context,
        "native_frames": native_frames,
        "visible_frames": wanted_visible_frames,
        "clip_duration_seconds": 5.0 if context == 5 else native_frames / fps,
    }

def normalize_extender_loras(loras=None) -> list[dict]:
    """Validate a bounded, ordered MODEL-only stack for Extender clip 1.

    Names are filenames returned by the existing verified R2 download helper,
    never remote URLs or client-supplied paths. We do not chain model loaders
    globally: upstream Extender supports per-clip LoRAs and patches its own
    incoming MODEL only while generating that clip.
    """
    if loras is None:
        return []
    if not isinstance(loras, list) or len(loras) > 5:
        raise ValueError("Extender supports at most 5 ordered LoRAs.")
    results = []
    seen = set()
    for item in loras:
        if not isinstance(item, dict):
            raise ValueError("Extender LoRAs must be resolved filename/strength objects.")
        name = item.get("filename")
        if not isinstance(name, str) or not re.fullmatch(
            r"library_[a-z0-9][a-z0-9_-]{0,63}_[a-f0-9]{16}\\.safetensors", name
        ):
            raise ValueError("Extender LoRA must be a verified library safetensors filename.")
        if name in seen:
            raise ValueError("Extender LoRA filenames must be unique.")
        seen.add(name)
        value = item.get("strength", 1.0)
        if isinstance(value, bool):
            raise ValueError("Invalid Extender LoRA strength.")
        try:
            strength = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Extender LoRA strength.") from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
            raise ValueError("Extender LoRA strength must be between 0 and 2.")
        results.append({"name": name, "strength": strength})
    return results


def _default_clip(prompt: str, seconds: int, seed: int, context_length: int = 5, loras=None) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty continuation prompt is required.")
    frame_plan = continuation_frame_plan(seconds, context_length)
    if not 0 <= int(seed) < 10_000_000_000:
        raise ValueError("Seed must be between 0 and 9,999,999,999.")
    return {
        "id": "clip_1",
        "name": "RunPod PoC continuation",
        "prompt": prompt.strip(),
        "seed": int(seed),
        "seed_mode": "fixed",
        "duration": frame_plan["clip_duration_seconds"],
        "validated": False,
        "color_adjustment": {
            "saturation": 100.0, "contrast": 100.0, "brightness": 100.0,
        },
        "loras": normalize_extender_loras(loras),
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
    loras: list[dict] | None = None,
) -> dict:
    """Build native LoadVideo → Extender → Final Decode ComfyUI API JSON.

    Upstream Extender owns the per-clip model-only LoRA loader. Refine and
    source-image reference management remain disabled in this isolated PoC.
    """
    if not video_filename or "/" in video_filename or "\\" in video_filename:
        raise ValueError("video_filename must be a single basename in ComfyUI/input.")
    if not width or not height or width % 32 or height % 32:
        raise ValueError("PoC source width/height must be multiples of 32.")
    continuation_frame_plan(seconds, context_length)
    ordered_loras = normalize_extender_loras(loras)
    if int(steps) != 8:
        raise ValueError("PoC is fixed at 8 sampling steps.")

    clips_json = json.dumps(
        {"version": 2, "generation_mode": "ref2va", "motion_context": True,
         "clips": [_default_clip(prompt, seconds, seed, context_length, ordered_loras)],
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
