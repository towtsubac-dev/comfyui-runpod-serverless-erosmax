import base64
import json
import os
import subprocess
import time
import uuid

import requests
import runpod

from registered_lora import download_registered_lora

COMFYUI_DIR = "/workspace/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"

DIFFUSION_MODEL = "10Eros_Max_h3_8stepTURBO_beta6_pruned_w4a8.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_LENGTH = 124
DEFAULT_STEPS = 8
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0
DEFAULT_TIMEOUT = 900

_comfy_process = None
_models_ready = False


def ensure_models():
    global _models_ready
    if _models_ready:
        return
    print("[ErosMax] Preparing models...", flush=True)
    subprocess.run(
        ["python", "/workspace/download_models.py"],
        check=True,
        timeout=3600,
    )
    _models_ready = True
    print("[ErosMax] Models ready.", flush=True)


def start_comfyui():
    global _comfy_process
    if _comfy_process is not None:
        return

    print("[ErosMax] Starting ComfyUI...", flush=True)
    _comfy_process = subprocess.Popen(
        [
            "python", "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--use-sage-attention",
        ],
        cwd=COMFYUI_DIR,
    )


def wait_for_comfyui(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        if _comfy_process is not None and _comfy_process.poll() is not None:
            raise RuntimeError(
                f"ComfyUI exited during startup: {_comfy_process.returncode}"
            )
        try:
            response = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if response.status_code == 200:
                print("[ErosMax] ComfyUI is ready.", flush=True)
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI did not become ready in time.")


def ensure_comfyui():
    ensure_models()
    if _comfy_process is None:
        start_comfyui()
        wait_for_comfyui()
        return
    try:
        if requests.get(f"{COMFYUI_URL}/system_stats", timeout=5).status_code == 200:
            return
    except Exception:
        pass
    wait_for_comfyui()


def upload_image(image_bytes, filename):
    response = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, image_bytes, "application/octet-stream")},
        data={"overwrite": "true"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["name"]


def queue_prompt(workflow):
    response = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("node_errors"):
        raise RuntimeError(f"ComfyUI node validation failed: {data['node_errors']}")
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id returned: {data}")
    print(f"[ErosMax] queued prompt: {prompt_id}", flush=True)
    return prompt_id


def wait_for_result(prompt_id, timeout=DEFAULT_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            record = history[prompt_id]
            status = record.get("status", {}).get("status_str", "")
            if status == "success":
                return record
            if status in {"error", "failed", "canceled"}:
                raise RuntimeError(f"ComfyUI job failed: {record}")
        time.sleep(2)
    raise TimeoutError(f"ComfyUI job timed out after {timeout} seconds.")


def extract_outputs(record):
    results = []
    for node_id, node_output in record.get("outputs", {}).items():
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []) or []:
                if item.get("filename"):
                    results.append({
                        "type": key,
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "file_type": item.get("type", "output"),
                        "node_id": node_id,
                    })
    return results


def download_output(item):
    response = requests.get(
        f"{COMFYUI_URL}/view",
        params={
            "filename": item["filename"],
            "subfolder": item["subfolder"],
            "type": item["file_type"],
        },
        timeout=120,
    )
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")



def optional_lora(job_input):
    """Return selected LoRAs in order; preserve legacy input.lora support."""
    selections = job_input.get("loras")
    if selections is None:
        legacy = job_input.get("lora", "none")
        selections = [] if legacy in (None, "none", False) else [
            {"name": legacy, "strength": job_input.get("lora_strength")}
        ]
    if not isinstance(selections, list):
        raise ValueError("input.loras must be a list.")
    if len(selections) > 5:
        raise ValueError("Too many LoRAs selected (maximum 5).")

    selected = []
    seen = set()
    for entry in selections:
        if isinstance(entry, str):
            name, raw_strength = entry, None
        elif isinstance(entry, dict):
            name, raw_strength = entry.get("name"), entry.get("strength")
        else:
            raise ValueError("Each LoRA must be a name or {name, strength}.")
        registered = entry.get("registered") if isinstance(entry, dict) else None
        if registered is not None:
            if not isinstance(registered, dict) or name != "library:" + str(registered.get("id")):
                raise ValueError("Registered LoRA ID mismatch.")
            if name in seen:
                raise ValueError(f"Duplicate LoRA: {name}.")
            seen.add(name)
            strength = float(1.0 if raw_strength is None else raw_strength)
            if not 0 <= strength <= 2:
                raise ValueError("Registered LoRA strength must be between 0 and 2.")
            filename = download_registered_lora(registered)
            selected.append({"filename": filename, "strength": strength})
            continue
        raise ValueError("Only registered library LoRAs are supported.")
    return selected


def apply_optional_lora(workflow, loras):
    """Chain model-only LoRAs before MiniMaxH3SigmaShift."""
    model_ref = ["2", 0]
    for index, lora in enumerate(loras or []):
        node_id = str(30 + index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora["filename"],
                "strength_model": lora["strength"],
            },
        }
        model_ref = [node_id, 0]
    workflow["8"]["inputs"]["model"] = model_ref
    return workflow


def build_i2v_workflow(image_filename, prompt, width, height, length, steps, seed):
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": DIFFUSION_MODEL,
                "weight_dtype": "default",
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "6": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "first_frame": ["1", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "7": {
            "class_type": "EmptyMiniMaxH3LatentAV",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "8": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["2", 0],
                "shift_video": DEFAULT_SHIFT_VIDEO,
                "shift_audio": DEFAULT_SHIFT_AUDIO,
            },
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "10": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["8", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "11": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["8", 0],
                "conditioning": ["6", 0],
            },
        },
        "12": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["12", 0],
                "guider": ["11", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["7", 0],
            },
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["4", 0],
            },
        },
        "15": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["5", 0],
            },
        },
        "16": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["14", 0],
                "audio": ["15", 0],
                "fps": 24,
                "bit_depth": 8,
            },
        },
        "17": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["16", 0],
                "filename_prefix": "video/ErosMax_H3_I2V",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }



def build_t2v_workflow(prompt, width, height, length, steps, seed):
    # Reuse the proven Eros Max sampler, model, VAE and MP4 pipeline.
    # MiniMaxH3ImageToVideo accepts a text-only request when first_frame
    # is omitted, as in the original H3 serverless implementation.
    workflow = build_i2v_workflow(
        "", prompt, width, height, length, steps, seed
    )
    del workflow["1"]
    del workflow["6"]["inputs"]["first_frame"]
    workflow["17"]["inputs"]["filename_prefix"] = "video/ErosMax_H3_T2V"
    return workflow


def h3_t2v_generate(job_input):
    ensure_comfyui()
    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing input.prompt.")

    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    length = int(job_input.get("length", DEFAULT_LENGTH))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    seed = int(job_input.get(
        "seed", int.from_bytes(os.urandom(8), "big") % 10_000_000_000
    ))

    workflow = build_t2v_workflow(
        prompt, width, height, length, steps, seed
    )
    workflow = apply_optional_lora(workflow, optional_lora(job_input))
    prompt_id = queue_prompt(workflow)
    history = wait_for_result(prompt_id)

    outputs = []
    for item in extract_outputs(history):
        outputs.append({
            "type": item["type"],
            "filename": item["filename"],
            "data": download_output(item),
        })
    return {"prompt_id": prompt_id, "outputs": outputs, "mode": "t2v", "seed": seed}



def cache_check():
    """Inspect cache locations without downloading or modifying models."""
    import glob

    roots = {
        "runpod_model_cache": "/runpod-volume/huggingface-cache/hub",
        "huggingface_cache": os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        ),
        "comfyui_models": f"{COMFYUI_DIR}/models",
    }
    filenames = {
        "diffusion_model": DIFFUSION_MODEL,
        "text_encoder": TEXT_ENCODER,
        "video_vae": VIDEO_VAE,
        "audio_vae": AUDIO_VAE,
    }
    locations = {}
    for key, root in roots.items():
        locations[key] = {
            "path": root,
            "exists": os.path.isdir(root),
        }

    models = {}
    for key, filename in filenames.items():
        subdir = {
            "diffusion_model": "diffusion_models",
            "text_encoder": "text_encoders",
            "video_vae": "vae",
            "audio_vae": "vae",
        }[key]
        local_path = os.path.join(COMFYUI_DIR, "models", subdir, filename)
        local_exists = os.path.isfile(local_path)
        matches = []
        for cache_root in (
            "/runpod-volume/huggingface-cache/hub",
            os.path.join(roots["huggingface_cache"], "hub"),
        ):
            if not os.path.isdir(cache_root):
                continue
            # HF hub snapshots may store symlinks to cached blobs.
            pattern = os.path.join(cache_root, "models--*", "snapshots", "*", "**", filename)
            for path in glob.glob(pattern, recursive=True)[:5]:
                matches.append({
                    "path": path,
                    "size_bytes": os.path.getsize(path),
                })
        models[key] = {
            "local_path": local_path,
            "local_exists": local_exists,
            "local_size_bytes": os.path.getsize(local_path) if local_exists else 0,
            "cache_matches": matches,
        }
    return {"status": "ok", "locations": locations, "models": models}


def h3_check():
    ensure_comfyui()
    files = {
        "diffusion_model": f"{COMFYUI_DIR}/models/diffusion_models/{DIFFUSION_MODEL}",
        "text_encoder": f"{COMFYUI_DIR}/models/text_encoders/{TEXT_ENCODER}",
        "video_vae": f"{COMFYUI_DIR}/models/vae/{VIDEO_VAE}",
        "audio_vae": f"{COMFYUI_DIR}/models/vae/{AUDIO_VAE}",
    }
    models = {}
    for name, path in files.items():
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        models[name] = {"exists": exists, "size_bytes": size}

    try:
        object_info = requests.get(f"{COMFYUI_URL}/object_info", timeout=30).json()
        h3_nodes = {
            name: data.get("display_name")
            for name, data in object_info.items()
            if "minimaxh3" in name.lower() or "minimax" in name.lower()
        }
    except Exception as exc:
        h3_nodes = {"error": str(exc)}

    ready = all(item["exists"] for item in models.values()) and "MiniMaxH3ImageToVideo" in h3_nodes
    return {"status": "ok" if ready else "error", "h3_ready": ready, "models": models, "h3_nodes": h3_nodes}


def h3_generate(job_input):
    ensure_comfyui()

    image_b64 = job_input.get("image")
    image_url = job_input.get("image_url")
    if not image_b64 and not image_url:
        raise ValueError("Missing input.image or input.image_url.")

    prompt = job_input.get(
        "prompt",
        "Create a natural cinematic video from the input image. Preserve the subject identity and overall composition. Use smooth realistic motion and subtle camera movement.",
    )
    width = int(job_input.get("width", DEFAULT_WIDTH))
    height = int(job_input.get("height", DEFAULT_HEIGHT))
    length = int(job_input.get("length", DEFAULT_LENGTH))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    seed = int(job_input.get("seed", int.from_bytes(os.urandom(8), "big") % 10_000_000_000))

    if image_url:
        if not image_url.startswith(("http://", "https://")):
            raise ValueError("input.image_url must start with http:// or https://")
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        image_bytes = response.content
    else:
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            raise ValueError(f"Invalid base64 image: {exc}")

    uploaded = upload_image(image_bytes, f"runpod_erosmax_{uuid.uuid4().hex}.png")
    workflow = build_i2v_workflow(uploaded, prompt, width, height, length, steps, seed)
    workflow = apply_optional_lora(workflow, optional_lora(job_input))
    prompt_id = queue_prompt(workflow)
    history = wait_for_result(prompt_id)

    outputs = []
    for item in extract_outputs(history):
        outputs.append({
            "type": item["type"],
            "filename": item["filename"],
            "data": download_output(item),
        })

    return {"prompt_id": prompt_id, "outputs": outputs, "mode": "i2v", "seed": seed}


# Test-only video-context extension. Requires custom nodes installed by test Dockerfile.
def _probe_video(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,codec_type",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError("The uploaded file does not contain a video stream.")
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    frames = int(video_stream.get("nb_read_frames") or round(duration * 24))
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "frames": frames,
        "duration": duration,
        "has_audio": audio_stream is not None,
    }


def _normalize_video_for_concat(source, destination, width, height):
    info = _probe_video(source)
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "fps=24,setsar=1,format=yuv420p"
    )
    cmd = ["ffmpeg", "-y", "-i", source]
    if info["has_audio"]:
        cmd += [
            "-map", "0:v:0", "-map", "0:a:0",
            "-vf", video_filter,
            "-af", "aresample=32000:async=1:first_pts=0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", destination,
        ]
    else:
        cmd += [
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=32000",
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", video_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-shortest", "-movflags", "+faststart", destination,
        ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)


def _concat_extended_video(original_path, segment_path, output_path):
    info = _probe_video(original_path)
    work_dir = os.path.dirname(output_path)
    normalized_original = os.path.join(work_dir, "normalized_original.mp4")
    normalized_segment = os.path.join(work_dir, "normalized_segment.mp4")
    _normalize_video_for_concat(
        original_path, normalized_original, info["width"], info["height"]
    )
    _normalize_video_for_concat(
        segment_path, normalized_segment, info["width"], info["height"]
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", normalized_original,
            "-i", normalized_segment,
            "-filter_complex",
            "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[outv][outa]",
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "32000", "-ac", "2",
            "-movflags", "+faststart", output_path,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    for path in (normalized_original, normalized_segment):
        try:
            os.remove(path)
        except OSError:
            pass



# ErosMax Preview-only extension flash repair, ported from MiniMax H3 v11.
# Apply after one successful inference, before concatenating with the input video.
# No impact on T2V/I2V, LoRA selection, or extension sampling.
def _log_extension_start_luma(path, label, count=8):
    # Diagnostic only: inspect every decoded frame in the first ~0.33 seconds.
    # No changes to output video, audio, or existing RGB matching.
    import numpy as np
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-map", "0:v:0", "-frames:v", str(count),
        "-vf", "scale=160:90,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True, timeout=60).stdout
    frame_size = 160 * 90 * 3
    frames = len(raw) // frame_size
    if frames == 0 or len(raw) % frame_size:
        raise ValueError("Unable to decode extension-start frames")
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(frames, 90, 160, 3).astype(np.float32) / 255.0
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = pixels @ weights
    means = np.round(luma.mean(axis=(1, 2)), 4).tolist()
    print(f"[EROS EXTEND FRAME] {label} first_{frames}_luma={means}", flush=True)


def _probe_extension_start_rgb(path, count=12):
    """Decode the opening frames once so calibration never samples a flash by timestamp."""
    import numpy as np
    cmd = [
        "ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
        "-frames:v", str(count), "-vf", "scale=160:90,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True, timeout=60).stdout
    size = 160 * 90 * 3
    if len(raw) < 6 * size or len(raw) % size:
        raise ValueError("Insufficient extension frames for seam analysis")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 90, 160, 3).astype(np.float32) / 255.0


def _read_extension_source_last_rgb(path):
    """Read the final decoded source frame, not a frame at an approximate timestamp."""
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
         "-vf", "scale=160:90,format=rgb24", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "pipe:1"],
        check=True, capture_output=True, timeout=180,
    ).stdout
    size = 160 * 90 * 3
    if len(raw) < size or len(raw) % size:
        raise ValueError("Cannot inspect final source frame")
    return np.frombuffer(raw[-size:], dtype=np.uint8).reshape(90, 160, 3).astype(np.float32) / 255.0


def _v10_flash_plan(frames, source, label):
    """Locate an exposure-only opening glitch using the actual source end frame."""
    import numpy as np
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    means = (frames @ weights).mean(axis=(1, 2))
    reference_y = source @ weights
    source_mean = float(reference_y.mean())
    reference_std = float(reference_y.std())
    corrs = []
    for frame in frames:
        current_y = frame @ weights
        if min(reference_std, float(current_y.std())) < 0.01:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(reference_y.ravel(), current_y.ravel())[0, 1]))
    corrs = np.asarray(corrs, dtype=np.float32)
    peak = int(np.argmax(means[:min(6, len(means))]))
    first_excess = float(means[0] - source_mean)
    peak_excess = float(means[peak] - source_mean)
    print(
        f"[EROS EXTEND FLASH] {label}: v10 scan source={source_mean:.4f} "
        f"first_12={means[:12].round(4).tolist()} "
        f"corr={corrs[:12].round(3).tolist()}", flush=True,
    )
    if float(corrs[0]) < 0.92:
        return None
    residual = False
    if first_excess >= 0.032:
        start = 0
    elif (abs(first_excess) < 0.020 and peak in (1, 2, 3, 4)
          and peak_excess > 0.032
          and bool(np.all(corrs[:peak + 1] > 0.93))):
        start, residual = 1, True
    else:
        return None

    anchor = None
    # A bright three-frame plateau may still be part of a fading flash.
    # Only accept a recovered window at most 0.014 brighter than the source.
    # A slightly darker window is allowed for natural movement / darkening.
    for i in range(max(start + 1, peak + 1 if residual else 1),
                   min(11, len(means) - 2)):
        window = means[i:i + 3]
        window_delta = float(window.mean() - source_mean)
        if (float(np.max(np.abs(np.diff(window)))) >= 0.014
                or window_delta > 0.014 or window_delta < -0.035
                or float(corrs[i]) < 0.72):
            continue
        if max(float(means[k] - window.mean())
               for k in range(start, min(i, 7))) < 0.030:
            continue
        anchor = i
        break
    # v7-style isolated 1-frame flash followed immediately by a normal frame.
    if anchor is None and start == 0 and len(means) > 2:
        if (abs(float(means[1] - source_mean)) < 0.025
                and abs(float(means[2] - source_mean)) < 0.025
                and abs(float(means[1] - means[2])) < 0.018
                and float(corrs[1]) > 0.92
                and float(means[0] - means[1]) > 0.030):
            anchor = 1
    if anchor is None:
        return None

    offsets = []
    # TV-range YUV420p uses 219 luma codes per unit; do not touch U/V chroma.
    # Interpolate *exposure* only; never duplicate frames or change motion.
    for frame_i in range(start, anchor):
        fraction = (frame_i - start) / (anchor - start)
        target = source_mean * (1.0 - fraction) + float(means[anchor]) * fraction
        y_offset = int(round((target - float(means[frame_i])) * 219))
        if abs(y_offset) > 48:
            return None
        offsets.append(y_offset)
    return {
        "start": start, "anchor": anchor, "offsets": offsets,
        "source_mean": source_mean, "recovery_mean": float(means[anchor]),
    }


def _v11_fade_flash_plan(frames, source):
    """Detect a strongly source-correlated, 4-10 frame fading exposure flash.

    This is deliberately a narrow fallback to v10: normal motion or a new
    lighting condition must not be flattened simply to match mean luminance.
    """
    import numpy as np
    if len(frames) < 11:
        return None
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    means = (frames @ weights).mean(axis=(1, 2))
    reference = source @ weights
    source_mean = float(reference.mean())
    source_std = float(reference.std())
    if source_std < 0.01:
        return None

    corrs = []
    for frame in frames[:3]:
        current = frame @ weights
        if float(current.std()) < 0.01:
            return None
        corrs.append(float(np.corrcoef(reference.ravel(), current.ravel())[0, 1]))
    if not (
        float(means[0] - source_mean) >= 0.07
        and corrs[0] >= 0.955
        and corrs[1] >= 0.90
        and corrs[2] >= 0.74
    ):
        return None

    early_drops = np.diff(means[:5])
    if not (
        bool(np.all(early_drops < 0.012))
        and float(means[0] - means[2]) > 0.035
        and float(means[0] - means[4]) > 0.065
        and int(np.count_nonzero(early_drops < -0.006)) >= 3
        and float(np.max(early_drops)) < 0.009
    ):
        return None

    anchor = None
    for i in range(4, min(11, len(means) - 2)):
        window = means[i:i + 3]
        if (
            float(np.ptp(window)) > 0.018
            or abs(float(np.median(window) - source_mean)) > 0.035
            or float(means[i] - source_mean) > 0.021
            or float(means[i] - means[0]) > -0.065
            or bool(np.any(means[1:i] < means[i] - 0.028))
            or bool(np.any(np.diff(means[:i + 1]) > 0.014))
        ):
            continue
        anchor = i
        break
    if anchor is None:
        return None

    offsets = []
    for i in range(anchor):
        target = source_mean + (float(means[anchor]) - source_mean) * i / anchor
        offset = int(round((target - float(means[i])) * 219))
        if abs(offset) > 40:
            return None
        offsets.append(offset)
    print(
        f"[EROS EXTEND FLASH] v11 gradual fade confirmed: "
        f"source={source_mean:.4f} first_12={means[:12].round(4).tolist()} "
        f"first_corr={[round(c, 3) for c in corrs]} "
        f"anchor={anchor} offsets={offsets}",
        flush=True,
    )
    return {
        "start": 0, "anchor": anchor, "offsets": offsets,
        "source_mean": source_mean, "recovery_mean": float(means[anchor]),
    }



def _v12_slow_fade_flash_plan(frames, source):
    """ErosMax-specific 8-14 frame fade fallback for a source-matching exposure flash.

    Keep v10/v11 decisions unchanged. A scene transition or persistent
    lighting change must not be corrected: require high source correlation,
    sustained decay and a stabilized recovery near the source brightness.
    """
    import numpy as np
    if len(frames) < 15:
        return None
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    means = (frames @ weights).mean(axis=(1, 2))
    reference = source @ weights
    source_mean = float(reference.mean())
    if (
        float(reference.std()) < 0.025
        or float(means[0] - source_mean) < 0.085
        or float(np.min(means[:3]) - source_mean) < 0.065
    ):
        return None

    corrs = []
    for frame in frames[:16]:
        luminance = frame @ weights
        if float(luminance.std()) < 0.025:
            return None
        corrs.append(float(np.corrcoef(reference.ravel(), luminance.ravel())[0, 1]))
    # ErosMax can preserve the scene with a lower initial correlation (~0.90)
    # than MiniMax v11, while reaching ~0.95 by frames 4-8.
    if (
        min(corrs[:4]) < 0.88
        or max(corrs[2:8]) < 0.935
        or float(np.median(corrs[2:8])) < 0.91
    ):
        return None

    anchor = None
    for i in range(8, min(15, len(means) - 2)):
        window = means[i:i + 3]
        drops = np.diff(means[:i + 1])
        if not (
            float(np.ptp(window)) <= 0.021
            and abs(float(np.median(window) - source_mean)) <= 0.035
            and abs(float(means[i] - source_mean)) <= 0.030
            and float(means[0] - means[i]) >= 0.085
            and float(np.max(drops)) < 0.012
            and int(np.count_nonzero(drops < -0.006)) >= 5
            and float(np.min(corrs[:i + 1])) >= 0.86
            and float(np.mean(corrs[max(i - 2, 0):i + 1])) >= 0.89
        ):
            continue
        anchor = i
        break
    if anchor is None:
        return None

    offsets = []
    for i in range(anchor):
        target = source_mean + (float(means[anchor]) - source_mean) * i / anchor
        offset = int(round((target - float(means[i])) * 219))
        if abs(offset) > 40:
            return None
        offsets.append(offset)
    print(
        "[EROS EXTEND FLASH] v12 slow fade confirmed: "
        f"source={source_mean:.4f} first_18={means[:18].round(4).tolist()} "
        f"first_corr={[round(c, 3) for c in corrs]} "
        f"anchor={anchor} offsets={offsets}",
        flush=True,
    )
    return {
        "start": 0, "anchor": anchor, "offsets": offsets,
        "source_mean": source_mean, "recovery_mean": float(means[anchor]),
    }



def _v13_isolated_flash_plan(frames, source):
    """Conservative single-frame fallback for ErosMax continuity flashes.

    Requires the *same source scene* to continue: the first generated frame
    is overexposed, frames 1-4 are already close to the source brightness,
    and frame 0 remains structurally similar to both source and frame 1.
    Do not correct scene cuts, persistent relighting or normal camera motion.
    """
    import numpy as np
    if len(frames) < 6:
        return None

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    reference = source @ weights
    initial = frames[0] @ weights
    recovered = frames[1] @ weights
    reference_mean = float(reference.mean())
    means = (frames[:6] @ weights).mean(axis=(1, 2))
    excess = float(means[0] - reference_mean)

    if (
        float(reference.std()) < 0.025
        or float(initial.std()) < 0.025
        or float(recovered.std()) < 0.025
        or not 0.085 <= excess <= 0.18
        or float(means[0] - means[1]) < 0.085
        or float(np.max(np.abs(means[1:5] - reference_mean))) > 0.026
        or float(np.ptp(means[1:5])) > 0.018
    ):
        return None

    # A real single-frame exposure glitch should affect all RGB channels,
    # not introduce a different-colored frame or new scene.
    channel_change = frames[0].mean(axis=(0, 1)) - frames[1].mean(axis=(0, 1))
    if (
        float(np.min(channel_change)) < 0.07
        or float(np.ptp(channel_change)) > 0.045
    ):
        return None

    source_y = reference.ravel()
    first_y = initial.ravel()
    next_y = recovered.ravel()
    source_corr_first = float(np.corrcoef(source_y, first_y)[0, 1])
    source_corr_next = float(np.corrcoef(source_y, next_y)[0, 1])
    first_next_corr = float(np.corrcoef(first_y, next_y)[0, 1])
    if (
        source_corr_first < 0.85
        or source_corr_next < 0.85
        or first_next_corr < 0.94
        or first_next_corr - max(source_corr_first, source_corr_next) < 0.025
    ):
        return None

    offset = int(round((reference_mean - float(means[0])) * 219))
    if not -40 <= offset <= -12:
        return None
    print(
        "[EROS EXTEND FLASH] v13 isolated frame confirmed: "
        f"source={reference_mean:.4f} first_6={means[:6].round(4).tolist()} "
        f"corr_source_first={source_corr_first:.3f} "
        f"corr_source_next={source_corr_next:.3f} "
        f"corr_first_next={first_next_corr:.3f} "
        f"anchor=1 offsets={[offset]}",
        flush=True,
    )
    return {
        "start": 0, "anchor": 1, "offsets": [offset],
        "source_mean": reference_mean, "recovery_mean": float(means[1]),
    }


def _repair_extension_start_flash(segment_path, temp_dir, label, original_path=None):
    """Adjust only the Y plane before exposure recovery, preserving v5 motion."""
    if original_path is None:
        return segment_path
    frames = _probe_extension_start_rgb(segment_path, count=18)
    source = _read_extension_source_last_rgb(original_path)
    plan = _v10_flash_plan(frames, source, label)
    version = "v10"
    if plan is None:
        plan = _v11_fade_flash_plan(frames, source)
        version = "v11"
    if plan is None:
        plan = _v12_slow_fade_flash_plan(frames, source)
        version = "v12"
    if plan is None:
        plan = _v13_isolated_flash_plan(frames, source)
        version = "v13"
    if plan is None:
        print(
            f"[EROS EXTEND FLASH] {label}: v10/v11/v12/v13 no confirmed exposure flash; unchanged",
            flush=True,
        )
        return segment_path
    start, anchor, offsets = plan["start"], plan["anchor"], plan["offsets"]
    filters = [
        f"lutyuv=y='clip(val+{offset},16,235)':enable='eq(n\\,{i})'"
        for i, offset in enumerate(offsets, start=start)
    ]
    result = os.path.join(temp_dir, f"flash_repaired_{label}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", segment_path, "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "medium",
         "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", result],
        check=True, capture_output=True, text=True, timeout=300,
    )
    print(
        f"[EROS EXTEND FLASH] {label}: {version} Y-only repair applied "
        f"frames={start}..{anchor-1} offsets={offsets} "
        f"source={plan['source_mean']:.4f} recovery={plan['recovery_mean']:.4f}; "
        f"remaining motion and chroma unchanged",
        flush=True,
    )
    _log_extension_start_luma(result, f"{label} post_repair")
    return result


def build_h3_extend_workflow(video_filename, prompt, length, skip_first_frames,
                             frame_load_cap, steps=DEFAULT_STEPS, seed=None,
                             optional_loras=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") % 10_000_000_000

    workflow = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_filename,
                "force_rate": 24,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_load_cap,
                "skip_first_frames": skip_first_frames,
                "select_every_nth": 1,
                "format": "AnimateDiff",
            },
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
        },
        "5": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE},
        },
        "6": {
            "class_type": "MiniMaxH3EncodeAVPatched",
            "inputs": {
                "vae": ["4", 0],
                "images": ["1", 0],
                "audio_vae": ["5", 0],
                "audio": ["1", 2],
            },
        },
        "7": {
            "class_type": "MiniMaxH3VideoExtendPatched",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "context_latent": ["6", 0],
                "audio_vae": ["5", 0],
                "prompt": prompt,
                "length": length,
                "context_frames": 4,
                "pin_last_frame": True,
            },
        },
        "8": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["2", 0],
                "shift_video": DEFAULT_SHIFT_VIDEO,
                "shift_audio": DEFAULT_SHIFT_AUDIO,
            },
        },
        "10": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["8", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "12": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["8", 0], "conditioning": ["7", 0]},
        },
        "13": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": ["12", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["7", 1],
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["14", 0], "vae": ["4", 0]},
        },
        "16": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["14", 0], "vae": ["5", 0]},
        },
        "17": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": 24, "bit_depth": 8},
        },
        "18": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["17", 0],
                "filename_prefix": f"video/H3_Extend_{uuid.uuid4().hex}",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }

    apply_optional_lora(workflow, optional_loras)

    return workflow



def h3_extend_generate(job_input):
    import json
    import tempfile
    ensure_comfyui()
    prompt = str(job_input.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Missing input.prompt.")
    seconds = int(job_input.get("extend_seconds", 5))
    if seconds not in (3, 5, 8, 10):
        raise ValueError("ErosMax extension supports 3, 5, 8 or 10 seconds.")
    video_url = job_input.get("video_url")
    video_b64 = job_input.get("video")
    if not video_url and not video_b64:
        raise ValueError("Missing input.video or input.video_url.")
    with tempfile.TemporaryDirectory(prefix="eros_extend_") as temp_dir:
        if video_url:
            from urllib.parse import urlparse
            parsed = urlparse(video_url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("input.video_url must be an HTTPS URL.")
            response = requests.get(video_url, timeout=90)
            response.raise_for_status()
            video_bytes = response.content
        else:
            video_bytes = base64.b64decode(str(video_b64).split(",")[-1], validate=True)
        if not video_bytes or len(video_bytes) > 100 * 1024 * 1024:
            raise ValueError("Input video must be 1-100 MiB.")
        job_id = uuid.uuid4().hex
        filename = f"eros_extend_{job_id}.mp4"
        input_path = os.path.join(COMFYUI_DIR, "input", filename)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        try:
            with open(input_path, "wb") as file:
                file.write(video_bytes)
            info = _probe_video(input_path)
            if info["duration"] <= 0 or info["duration"] > 120:
                raise ValueError("Input video duration must be 0-120 seconds.")
            if info["width"] % 16 or info["height"] % 16:
                raise ValueError("Input video dimensions must be divisible by 16.")
            frame_cap = min(81, max(2, info["frames"]))
            skip = max(0, info["frames"] - frame_cap)
            seed = int(job_input.get("seed", int.from_bytes(os.urandom(8), "big") % 10_000_000_000))
            workflow = build_h3_extend_workflow(filename, prompt, {3: 76, 5: 124, 8: 196, 10: 244}[seconds], skip, frame_cap, steps=8, seed=seed, optional_loras=optional_lora(job_input))
            prompt_id = queue_prompt(workflow)
            record = wait_for_result(prompt_id)
            outputs = [item for item in extract_outputs(record) if item["filename"].lower().endswith(".mp4")]
            if not outputs:
                # Test-only diagnostics: capture the actual ComfyUI output structure.
                # Do not include input video bytes or prompt text in the error.
                node_outputs = record.get("outputs", {})
                diagnostics = {
                    "prompt_id": prompt_id,
                    "comfy_status": record.get("status", {}).get("status_str"),
                    "messages": [
                        {"event": event, "details": (
                            {k: v for k, v in details.items()
                             if k in ("node_id", "node_type", "exception_type", "exception_message")}
                            if isinstance(details, dict) else str(details)[:300]
                        )}
                        for event, details in record.get("status", {}).get("messages", [])[-12:]
                    ],
                    "output_nodes": {
                        str(node_id): {
                            "keys": list(value.keys()) if isinstance(value, dict) else [],
                            "files": [
                                {"key": key, "filename": item.get("filename"), "type": item.get("type")}
                                for key in ("videos", "gifs", "images")
                                for item in ((value.get(key) or []) if isinstance(value, dict) else [])
                                if isinstance(item, dict)
                            ][:10],
                        }
                        for node_id, value in node_outputs.items()
                    },
                    "extracted_files": extract_outputs(record),
                }
                print("[ErosMax] Extension missing MP4: " + json.dumps(diagnostics, ensure_ascii=False)[:12000], flush=True)
                raise RuntimeError("Extension completed without MP4 output. Diagnostics: " + json.dumps(diagnostics, ensure_ascii=False)[:9000])
            segment = outputs[-1]
            segment_b64 = download_output(segment)
            segment_path = os.path.join(temp_dir, "segment.mp4")
            with open(segment_path, "wb") as file:
                file.write(base64.b64decode(segment_b64))
            raw_segment_path = segment_path
            flash_requested = job_input.get("extension_flash_repair", False) is True
            if flash_requested:
                try:
                    _log_extension_start_luma(raw_segment_path, "raw")
                    segment_path = _repair_extension_start_flash(
                        raw_segment_path, temp_dir, "corrected", input_path
                    )
                except Exception as exc:
                    # A detection/FFmpeg failure must never prevent a usable
                    # extension from being returned; keep the original segment.
                    print(f"[EROS EXTEND FLASH] repair failed; using original segment: {exc}", flush=True)
                    segment_path = raw_segment_path
            else:
                print("[EROS EXTEND FLASH] off (Preview opt-in)", flush=True)
            combined = os.path.join(temp_dir, "extended.mp4")
            _concat_extended_video(input_path, segment_path, combined)
            with open(combined, "rb") as file:
                result_b64 = base64.b64encode(file.read()).decode("ascii")
            returned_outputs = [{
                "type": "videos",
                "filename": f"ErosMax_Extended_{job_id}.mp4",
                "data": result_b64,
            }]
            # Optional A/B comparison of the same inferred segment: no extra
            # GPU inference. Keep the primary video if the second file is too
            # large for RunPod's inline job result.
            MAX_RUNPOD_INLINE_RESULT_BYTES = 19_500_000
            if (job_input.get("extension_compare_flash", False) is True
                    and flash_requested and segment_path != raw_segment_path):
                try:
                    raw_combined = os.path.join(temp_dir, "extended_no_flash.mp4")
                    _concat_extended_video(input_path, raw_segment_path, raw_combined)
                    with open(raw_combined, "rb") as file:
                        raw_b64 = base64.b64encode(file.read()).decode("ascii")
                    if len(result_b64) + len(raw_b64) + 4096 <= MAX_RUNPOD_INLINE_RESULT_BYTES:
                        returned_outputs.append({
                            "type": "videos",
                            "filename": f"ErosMax_Extended_NO_FLASH_{job_id}.mp4",
                            "data": raw_b64,
                        })
                    else:
                        print("[EROS EXTEND FLASH] A/B skipped: inline result safety limit 19.5MB", flush=True)
                except Exception as exc:
                    print(f"[EROS EXTEND FLASH] A/B unavailable: {exc}", flush=True)
            if len(result_b64) + 4096 > MAX_RUNPOD_INLINE_RESULT_BYTES:
                print("[EROS EXTEND RETURN] primary video exceeds inline 19.5MB safety margin", flush=True)
            return {
                "status": "ok", "mode": "extend", "prompt_id": prompt_id,
                "flash_repair_requested": flash_requested,
                "flash_repair_applied": segment_path != raw_segment_path,
                "outputs": returned_outputs,
            }
        finally:
            if os.path.isfile(input_path):
                os.remove(input_path)



def h3_extender_probe():
    """Report actual installed ComfyUI node sockets; never queue a GPU job."""
    from h3_extender_poc import summarize_comfy_contract
    ensure_comfyui()
    response = requests.get(f"{COMFYUI_URL}/object_info", timeout=90)
    response.raise_for_status()
    return summarize_comfy_contract(response.json())


def h3_extender_generate(job_input):
    """Isolated one-clip v3.0.14 Extender PoC, not used by Studio Production."""
    import glob
    import tempfile
    from h3_extender_poc import build_workflow, continuation_frame_plan
    from extender_r2_io import download_r2_source, upload_r2_output

    prompt = str(job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing input.prompt.")
    seconds = int(job_input.get("extend_seconds", 5))
    if seconds != 5:
        raise ValueError("Extender PoC is limited to a single 5-second continuation.")

    video_url = job_input.get("video_url")
    video_b64 = job_input.get("video")
    output_upload = job_input.get("output_upload")
    if not video_url and not video_b64:
        raise ValueError("Missing input.video or input.video_url.")

    contract = h3_extender_probe()
    if contract["missing"]:
        raise RuntimeError("Missing Extender PoC nodes: " + ", ".join(contract["missing"]))

    job_id = uuid.uuid4().hex
    filename = f"eros_h3_extender_poc_{job_id}.mp4"
    output_prefix = f"ErosH3Extender_PoC_{job_id}"
    input_path = os.path.join(COMFYUI_DIR, "input", filename)
    try:
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        if video_url:
            # Studio passes a signed R2 GET. Stream with a hard size cap and
            # reject arbitrary hosts/redirects to avoid worker-side SSRF.
            download_r2_source(video_url, input_path)
        else:
            video_bytes = base64.b64decode(str(video_b64).split(",")[-1], validate=True)
            if not 16 <= len(video_bytes) <= 100 * 1024 * 1024:
                raise ValueError("Input video must be 16 bytes to 100 MiB.")
            with open(input_path, "wb") as handle:
                handle.write(video_bytes)

        meta = _probe_video(input_path)
        if not (0 < float(meta["duration"]) <= 120):
            raise ValueError("Source video must be 0-120 seconds.")
        if int(meta["width"]) % 32 or int(meta["height"]) % 32:
            raise ValueError("PoC requires source dimensions divisible by 32.")
        seed = int(job_input.get("seed", 123456))
        context_length = int(job_input.get("context_length", 5))
        frame_plan = continuation_frame_plan(seconds, context_length)
        # Existing ErosMax library downloader validates the R2 host, SHA-256,
        # safetensors structure and per-item strength before returning filenames.
        # Preserve user-specified LoRA order. The Extender node itself applies
        # these model-only patches per clip; do not chain global LoraLoader nodes.
        resolved_loras = optional_lora(job_input)
        workflow = build_workflow(
            video_filename=filename,
            prompt=prompt,
            width=int(meta["width"]),
            height=int(meta["height"]),
            seed=seed,
            model_filename=DIFFUSION_MODEL,
            clip_filename=TEXT_ENCODER,
            video_vae_filename=VIDEO_VAE,
            audio_vae_filename=AUDIO_VAE,
            seconds=seconds,
            context_length=context_length,
            steps=8,
            filename_prefix=output_prefix,
            loras=resolved_loras,
        )
        print(
            f"[EROS EXTENDER POC] native VIDEO; ref2va; Motion Context ON; "
            f"context={context_length}; native_frames={frame_plan['native_frames']}; "
            f"expected_visible_frames={frame_plan['visible_frames']}; "
            f"source={meta['width']}x{meta['height']}; "
            f"seed={seed}; loras={len(resolved_loras)}",
            flush=True,
        )
        prompt_id = queue_prompt(workflow)
        record = wait_for_result(prompt_id, timeout=DEFAULT_TIMEOUT)

        # Extender Final Decode reports custom h3_video UI metadata rather
        # than the standard SaveVideo videos[] array. Its documented full_batch
        # path writes the final file into the normal ComfyUI output directory.
        candidates = glob.glob(os.path.join(
            COMFYUI_DIR, "output", output_prefix + "*.mp4"
        ))
        if not candidates:
            node_keys = {
                str(node_id): list(value.keys()) if isinstance(value, dict) else []
                for node_id, value in (record.get("outputs") or {}).items()
            }
            raise RuntimeError(
                "Extender PoC completed without an output MP4. "
                f"ComfyUI output keys: {node_keys}"
            )
        final_path = max(candidates, key=os.path.getmtime)
        result = {
            "status": "ok",
            "mode": "h3_extender_poc",
            "prompt_id": prompt_id,
            "node_revision": contract["upstream_sha"],
            "context_length": context_length,
            "frame_plan": frame_plan,
            "lora_count": len(resolved_loras),
        }
        if output_upload is not None:
            # Output is sent directly to one presigned R2 staging key.
            # The RunPod result contains only a small signed receipt;
            # no R2 secrets or signed URLs are returned.
            result["r2_upload"] = upload_r2_output(final_path, output_upload)
            result["outputs"] = []
            print("[EROS EXTENDER R2] uploaded video bytes="
                  + str(result["r2_upload"]["bytes"]), flush=True)
        else:
            # Legacy direct-RunPod tests remain compatible if the new
            # Preview feature flag is OFF; keep the old 14MB safety cap.
            if os.path.getsize(final_path) > 14_000_000:
                raise RuntimeError("Extender inline result exceeds 14 MB; "
                                   "enable the Studio R2 direct-output Preview")
            with open(final_path, "rb") as handle:
                data = handle.read()
            result["outputs"] = [{
                "type": "videos",
                "filename": output_prefix + ".mp4",
                "data": base64.b64encode(data).decode("ascii"),
            }]
        return result
    finally:
        if os.path.isfile(input_path):
            os.remove(input_path)


def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "h3_generate")

    try:
        if action == "health":
            ensure_comfyui()
            return {"status": "ok", "message": "ComfyUI is running", "eros_extend_revision": "v13-isolated-flash-preview", "system_stats": requests.get(f"{COMFYUI_URL}/system_stats", timeout=30).json()}
        if action == "eros_ref2va_check":
            from eros_ref2va import check
            ensure_comfyui()
            return check(
                comfyui_dir=COMFYUI_DIR,
                comfyui_url=COMFYUI_URL,
                model=DIFFUSION_MODEL,
                clip=TEXT_ENCODER,
                video_vae=VIDEO_VAE,
                audio_vae=AUDIO_VAE,
            )
        if action == "eros_ref2va_generate":
            from eros_ref2va import generate
            return generate(
                job_input,
                comfyui_dir=COMFYUI_DIR,
                model=DIFFUSION_MODEL,
                clip=TEXT_ENCODER,
                video_vae=VIDEO_VAE,
                audio_vae=AUDIO_VAE,
                ensure_comfyui=ensure_comfyui,
                upload_image=upload_image,
                queue_prompt=queue_prompt,
                wait_for_result=wait_for_result,
                extract_outputs=extract_outputs,
                download_output=download_output,
            )
        if action == "h3_check":
            return h3_check()
        if action == "h3_extend_generate":
            return h3_extend_generate(job_input)
        if action == "h3_extender_probe":
            return h3_extender_probe()
        if action == "h3_extender_generate":
            return h3_extender_generate(job_input)
        if action == "cache_check":
            return cache_check()
        if action in {"h3_generate", "h3_i2v_generate", "generate"}:
            return h3_generate(job_input)
        if action in {"h3_t2v_generate", "t2v_generate"}:
            return h3_t2v_generate(job_input)
        raise ValueError(f"Unsupported action: {action}")
    except Exception as exc:
        print(f"[ErosMax] ERROR: {exc}", flush=True)
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    ensure_models()
    start_comfyui()
    wait_for_comfyui()
    print("[ErosMax] revision=v13-isolated-flash-preview; flash=v10+v11 Y-only; Preview opt-in", flush=True)
    print("[ErosMax] Starting RunPod Serverless handler...", flush=True)
    runpod.serverless.start({"handler": handler})
