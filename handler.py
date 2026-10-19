import base64
import json
import os
import subprocess
import time
import uuid

import requests
import runpod

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



LORA_SPECS = {
    "bounce_fl2va": {
        "filename": "bounceV07_fl2va-000230_Intense.safetensors",
        "version_id": 3258590,
        "default_strength": 0.8,
    },
    "mystic_xxx_v4": {
        "filename": "MysticXXX_MMH3-V4.safetensors",
        "version_id": 3266628,
        "default_strength": 1.0,
    },
}


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
    if len(selections) > len(LORA_SPECS):
        raise ValueError("Too many LoRAs selected.")

    selected = []
    seen = set()
    for entry in selections:
        if isinstance(entry, str):
            name, raw_strength = entry, None
        elif isinstance(entry, dict):
            name, raw_strength = entry.get("name"), entry.get("strength")
        else:
            raise ValueError("Each LoRA must be a name or {name, strength}.")
        if name not in LORA_SPECS:
            raise ValueError(f"Unknown LoRA: {name!r}.")
        if name in seen:
            raise ValueError(f"Duplicate LoRA: {name}.")
        seen.add(name)
        spec = LORA_SPECS[name]
        strength = float(spec["default_strength"] if raw_strength is None else raw_strength)
        if not 0 <= strength <= 2:
            raise ValueError(f"LoRA strength for {name} must be between 0 and 2.")

        filename = spec["filename"]
        target = os.path.join(COMFYUI_DIR, "models", "loras", filename)
        if not os.path.isfile(target) or os.path.getsize(target) == 0:
            # Keep compatibility with previously built images. New images bundle both.
            url = f"https://civitai.com/api/download/models/{spec['version_id']}"
            token = os.environ.get("CIVITAI_API_TOKEN", "").strip()
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temp = target + ".part"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                    response.raise_for_status()
                    with open(temp, "wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                with open(temp, "rb") as downloaded:
                    header = downloaded.read(8)
                    if len(header) != 8 or not 0 < int.from_bytes(header, "little") < 100_000_000:
                        raise RuntimeError(f"Invalid safetensors download: {filename}")
                os.replace(temp, target)
                print(f"[LoRA] downloaded: {filename}", flush=True)
            finally:
                if os.path.exists(temp):
                    os.remove(temp)
        selected.append({"filename": filename, "strength": strength})
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
                "context_frames": 2,
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
            combined = os.path.join(temp_dir, "extended.mp4")
            _concat_extended_video(input_path, segment_path, combined)
            with open(combined, "rb") as file:
                result_b64 = base64.b64encode(file.read()).decode("ascii")
            return {"status": "ok", "mode": "extend", "prompt_id": prompt_id, "outputs": [{"type": "videos", "filename": f"ErosMax_Extended_{job_id}.mp4", "data": result_b64}]}
        finally:
            if os.path.isfile(input_path):
                os.remove(input_path)


def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "h3_generate")

    try:
        if action == "health":
            ensure_comfyui()
            return {"status": "ok", "message": "ComfyUI is running", "system_stats": requests.get(f"{COMFYUI_URL}/system_stats", timeout=30).json()}
        if action == "h3_check":
            return h3_check()
        if action == "h3_extend_generate":
            return h3_extend_generate(job_input)
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
    print("[ErosMax] Starting RunPod Serverless handler...", flush=True)
    runpod.serverless.start({"handler": handler})
