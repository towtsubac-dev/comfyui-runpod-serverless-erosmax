"""Download a registered R2 LoRA using a short-lived signed URL.

This module does not change the existing bundled Civitai LoRAs.
Only call it with a trusted descriptor created by the Vercel backend.
"""
import hashlib
import json
import os
import re
import tempfile
from urllib.parse import urlparse

import requests

MAX_BYTES = 2 * 1024 ** 3
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def download_registered_lora(descriptor, directory="/workspace/ComfyUI/models/loras"):
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid LoRA descriptor")
    identifier = descriptor.get("id")
    sha256 = descriptor.get("sha256")
    url = descriptor.get("url")
    size = descriptor.get("size")
    if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
        raise ValueError("Invalid registered LoRA ID")
    if not isinstance(sha256, str) or not HASH_PATTERN.fullmatch(sha256):
        raise ValueError("Invalid registered LoRA SHA256")
    if not isinstance(size, int) or not 16 <= size <= MAX_BYTES:
        raise ValueError("Invalid registered LoRA size")
    if not isinstance(url, str) or len(url) > 4096:
        raise ValueError("Invalid LoRA download URL")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".r2.cloudflarestorage.com"):
        raise ValueError("Unexpected LoRA download host")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Invalid LoRA URL")
    os.makedirs(directory, exist_ok=True)
    filename = f"library_{identifier}_{sha256[:16]}.safetensors"
    target = os.path.join(directory, filename)
    if os.path.isfile(target) and os.path.getsize(target) == size:
        with open(target, "rb") as existing:
            digest = hashlib.file_digest(existing, "sha256").hexdigest()
        if digest == sha256:
            return filename
    fd, temporary = tempfile.mkstemp(prefix=".lora-", suffix=".part", dir=directory)
    try:
        hasher = hashlib.sha256()
        received = 0
        with os.fdopen(fd, "wb") as output:
            with requests.get(url, stream=True, timeout=(30, 120), allow_redirects=False) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > size:
                        raise ValueError("LoRA exceeds expected size")
                    hasher.update(chunk)
                    output.write(chunk)
        if received != size or hasher.hexdigest() != sha256:
            raise ValueError("LoRA checksum or size mismatch")
        with open(temporary, "rb") as f:
            header = f.read(8)
            if len(header) != 8 or not 2 <= int.from_bytes(header, "little") <= 16 * 1024 * 1024:
                raise ValueError("Invalid safetensors header")
            header_length = int.from_bytes(header, "little")
            metadata = json.loads(f.read(header_length).decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Invalid safetensors metadata")
            tensors = [value for key, value in metadata.items() if key != "__metadata__"]
            if not tensors:
                raise ValueError("Safetensors contains no tensors")
            intervals = []
            for tensor in tensors:
                if not isinstance(tensor, dict) or not isinstance(tensor.get("data_offsets"), list):
                    raise ValueError("Invalid safetensors tensor descriptor")
                offsets = tensor["data_offsets"]
                if len(offsets) != 2 or any(type(x) is not int for x in offsets):
                    raise ValueError("Invalid safetensors offsets")
                intervals.append(offsets)
            intervals.sort()
            end = 0
            for start, stop in intervals:
                if start != end or stop < start:
                    raise ValueError("Invalid safetensors tensor layout")
                end = stop
            if end != size - 8 - header_length:
                raise ValueError("Safetensors tensor data size mismatch")
        os.replace(temporary, target)
        return filename
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
