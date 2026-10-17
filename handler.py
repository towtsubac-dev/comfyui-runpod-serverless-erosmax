import base64
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


def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "h3_generate")

    try:
        if action == "health":
            ensure_comfyui()
            return {"status": "ok", "message": "ComfyUI is running", "system_stats": requests.get(f"{COMFYUI_URL}/system_stats", timeout=30).json()}
        if action == "h3_check":
            return h3_check()
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
