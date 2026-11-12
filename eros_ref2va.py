"""Preview-only ErosMax beta6 Ref2VA using the *existing* diffusion weights.

This module does not download or swap model weights. It creates a native
MiniMaxH3ReferenceToVideo ComfyUI API workflow, with flat dotted autogrow keys.
Compatibility and identity fidelity must be established with an actual GPU run.
"""
import base64
import io
import os
import re
import uuid
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGES = 3
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 12_000_000
R2_HOST = re.compile(r"^[a-f0-9]{32}\.r2\.cloudflarestorage\.com$", re.I)
SUPPORTED_SIZES = {(864, 480), (480, 864), (768, 768)}


def _image_from_url(url):
    """Read only signed Cloudflare R2 HTTPS objects (no redirects or SSRF)."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or not R2_HOST.fullmatch(host)
            or parsed.port is not None or parsed.username or parsed.password
            or not parsed.query or not parsed.path or not parsed.query.startswith("X-Amz-")):
        raise ValueError("reference image_url must be a signed Cloudflare R2 GET URL")
    with requests.get(url, timeout=(10, 60), stream=True, allow_redirects=False) as response:
        if response.status_code != 200:
            raise ValueError("Failed to download a signed R2 reference image")
        chunks, total = [], 0
        for chunk in response.iter_content(65536):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("Reference image exceeds 12 MiB")
            chunks.append(chunk)
    return b"".join(chunks)


def _read_image(item):
    if isinstance(item, dict):
        raw_url = item.get("image_url")
        raw_b64 = item.get("image")
        if bool(raw_url) == bool(raw_b64):
            raise ValueError("Each reference must contain exactly one of image_url or image")
    elif isinstance(item, str):
        raw_url = item if item.startswith("https://") else None
        raw_b64 = item if raw_url is None else None
    else:
        raise ValueError("Each reference must be a signed R2 URL or base64 image")

    if raw_url:
        if not isinstance(raw_url, str):
            raise ValueError("Invalid image URL")
        raw = _image_from_url(raw_url)
    else:
        if not isinstance(raw_b64, str):
            raise ValueError("Invalid base64 image")
        if raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[-1]
        if len(raw_b64) > MAX_IMAGE_BYTES * 4 // 3 + 16:
            raise ValueError("Reference image exceeds 12 MiB")
        try:
            raw = base64.b64decode(raw_b64, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 image") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Reference image is empty or exceeds 12 MiB")

    # Clean orientation and re-encode to actual PNG bytes. Preserve the original
    # files in R2; temporary worker input files are deleted after generation.
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("Reference images must be PNG, JPEG or WebP")
            if img.width * img.height > MAX_PIXELS or min(img.size) < 128:
                raise ValueError("Reference image dimensions must be at least 128px and at most 12MP")
            cleaned = ImageOps.exif_transpose(img).convert("RGB")
            buf = io.BytesIO()
            cleaned.save(buf, format="PNG")
            return buf.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid reference image data") from exc


def build_ref2va_workflow(filenames, *, prompt, model, clip, video_vae, audio_vae,
                          width=864, height=480, length=124, steps=8,
                          seed=123456, ref_image_size="match", optional_loras=None):
    if not isinstance(filenames, list) or not 2 <= len(filenames) <= MAX_IMAGES:
        raise ValueError("Ref2VA preview accepts 2–3 reference images")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
        raise ValueError("Missing or oversized prompt")
    if (width, height) not in SUPPORTED_SIZES or length != 124:
        raise ValueError("Unsupported preview resolution or length")
    if steps not in (8, 20):
        raise ValueError("Eros beta6 preview supports 8 or 20 steps")
    if ref_image_size not in ("match", "max"):
        raise ValueError("ref_image_size must be match or max")
    if not isinstance(seed, int) or seed < 0 or seed > 999999999:
        raise ValueError("seed must be an integer between 0 and 999999999")

    refs = {
        "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
        "prompt": prompt, "width": width, "height": height,
        "length": length, "ref_image_size": ref_image_size
    }
    workflow = {
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": refs},
        "8": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["2", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {"model": ["8", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["8", 0], "conditioning": ["6", 0]}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["13", 0], "guider": ["12", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["6", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["4", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "audio": ["16", 0], "fps": 24, "bit_depth": 8}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "video/Eros_beta6_Ref2VA_" + uuid.uuid4().hex,
            "format": "mp4", "codec": "h264"}},
    }
    for idx, filename in enumerate(filenames):
        node_id = str(20 + idx)
        workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
        refs["ref_images.ref_image_" + str(idx)] = [node_id, 0]
    # Model-only LoRAs use the existing ErosMax library. Keep their order
    # exactly as selected in Studio; the native Ref2VA base model stays beta6.
    selected = optional_loras or []
    if len(selected) > 5:
        raise ValueError("At most 5 optional LoRAs are supported")
    model_ref = ["2", 0]
    for index, item in enumerate(selected):
        strength = float(item["strength"])
        if not 0.0 <= strength <= 2.0:
            raise ValueError("ErosMax Ref2VA LoRA strength must be 0–2")
        node_id = str(30 + index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": item["filename"],
                "strength_model": strength,
            },
        }
        model_ref = [node_id, 0]
    workflow["8"]["inputs"]["model"] = model_ref
    return workflow


def check(*, comfyui_dir, comfyui_url, model, clip, video_vae, audio_vae):
    paths = {
        "diffusion_model": os.path.join(comfyui_dir, "models/diffusion_models", model),
        "text_encoder": os.path.join(comfyui_dir, "models/text_encoders", clip),
        "video_vae": os.path.join(comfyui_dir, "models/vae", video_vae),
        "audio_vae": os.path.join(comfyui_dir, "models/vae", audio_vae),
    }
    models = {key: {"exists": os.path.isfile(path),
                    "size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0}
              for key, path in paths.items()}
    response = requests.get(comfyui_url + "/object_info/MiniMaxH3ReferenceToVideo", timeout=30)
    response.raise_for_status()
    node_info = response.json().get("MiniMaxH3ReferenceToVideo")
    ready = all(x["exists"] for x in models.values()) and node_info is not None
    return {
        "status": "ok" if ready else "error",
        "ref2va_ready": ready,
        "native_node_present": node_info is not None,
        "existing_model": model,
        "model_changed": False,
        "compatibility_proven": False,
        "models": models,
        "supported_ref_count": [2, MAX_IMAGES],
    }


def generate(job_input, *, comfyui_dir, model, clip, video_vae, audio_vae,
             ensure_comfyui, upload_image, queue_prompt, wait_for_result,
             extract_outputs, download_output, resolve_loras=None):
    refs = job_input.get("reference_images")
    if not isinstance(refs, list) or not 2 <= len(refs) <= MAX_IMAGES:
        raise ValueError("reference_images must contain 2–3 images")
    prompt = job_input.get("prompt")
    width = int(job_input.get("width", 864))
    height = int(job_input.get("height", 480))
    steps = int(job_input.get("steps", 8))
    length = int(job_input.get("length", 124))
    seed_raw = job_input.get("seed")
    seed = int.from_bytes(os.urandom(8), "big") % 1000000000 if seed_raw is None else int(seed_raw)
    ref_size = job_input.get("ref_image_size", "match")
    optional_loras = []
    if job_input.get("loras") or job_input.get("lora"):
        if resolve_loras is None:
            raise ValueError("LoRA resolver is not available")
        optional_loras = resolve_loras(job_input)
        if len(optional_loras) > 5:
            raise ValueError("At most 5 optional LoRAs are supported")
    # Validate parameters before paying to load the model or uploading inputs.
    build_ref2va_workflow(["placeholder-a.png", "placeholder-b.png"],
                          prompt=prompt, model=model, clip=clip,
                          video_vae=video_vae, audio_vae=audio_vae,
                          width=width, height=height, length=length,
                          steps=steps, seed=seed, ref_image_size=ref_size)
    ensure_comfyui()
    uploaded = []
    try:
        for idx, entry in enumerate(refs):
            data = _read_image(entry)
            filename = "eros_ref2va_" + uuid.uuid4().hex + "_" + str(idx) + ".png"
            uploaded.append(upload_image(data, filename))
        workflow = build_ref2va_workflow(uploaded, prompt=prompt, model=model, clip=clip,
                                         video_vae=video_vae, audio_vae=audio_vae,
                                         width=width, height=height, length=length,
                                         steps=steps, seed=seed, ref_image_size=ref_size,
                                         optional_loras=optional_loras)
        print("[Eros Ref2VA] model=" + model + " refs=" + str(len(uploaded))
              + " steps=" + str(steps) + " size=" + ref_size + " seed=" + str(seed), flush=True)
        prompt_id = queue_prompt(workflow)
        history = wait_for_result(prompt_id, timeout=1800)
        outputs = []
        for item in extract_outputs(history):
            if item["filename"].lower().endswith(".mp4"):
                outputs.append({"type": item["type"], "filename": item["filename"],
                                "data": download_output(item)})
        if not outputs:
            raise RuntimeError("Eros Ref2VA completed without an MP4")
        return {"status": "ok", "mode": "ref2va", "prompt_id": prompt_id,
                "model": model, "reference_image_count": len(uploaded),
                "steps": steps, "seed": seed, "lora_count": len(optional_loras), "outputs": outputs}
    finally:
        root = os.path.join(comfyui_dir, "input")
        for filename in uploaded:
            if (filename.startswith("eros_ref2va_") and filename.endswith(".png")
                    and filename == os.path.basename(filename)):
                try:
                    os.remove(os.path.join(root, filename))
                except FileNotFoundError:
                    pass
