"""Strict, memory-bounded R2 media transport for the opt-in H3 Extender.

Only backend-provided Cloudflare R2 presigned URLs are supported. No R2
credentials are stored in the Worker, and no unrestricted HTTP fetching.
"""
import os
import re
import time
from urllib.parse import parse_qs, unquote, urlsplit

STAGING_KEY_RE = re.compile(
    r"^video/staging/extender/[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\.mp4$"
)
SOURCE_PATH_RE = re.compile(
    r"^/[a-z0-9][a-z0-9-]{2,62}/uploads/[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\.mp4$"
)
MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024 * 1024


def validate_r2_signed_url(url, *, source=False, staging_key=None):
    """Reject redirects, arbitrary hosts, URL credentials and unexpected R2 keys."""
    if not isinstance(url, str) or not (100 <= len(url) <= 4096):
        raise ValueError("Invalid signed R2 URL")
    p = urlsplit(url)
    if p.scheme != "https" or p.username or p.password or p.port or p.fragment:
        raise ValueError("Only plain HTTPS R2 presigned URLs are allowed")
    if not re.fullmatch(r"[0-9a-f]{32}\.r2\.cloudflarestorage\.com", p.hostname or ""):
        raise ValueError("R2 URL host not allowed")
    path = unquote(p.path)
    if source:
        if not SOURCE_PATH_RE.fullmatch(path):
            raise ValueError("R2 input URL must point to a Studio upload")
    else:
        if not isinstance(staging_key, str) or not STAGING_KEY_RE.fullmatch(staging_key):
            raise ValueError("Invalid Extender staging key")
        if not re.fullmatch(r"/[a-z0-9][a-z0-9-]{2,62}/" + re.escape(staging_key), path):
            raise ValueError("R2 output URL does not match staging key")
    params = parse_qs(p.query)
    if params.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]:
        raise ValueError("R2 signature algorithm missing")
    signature = params.get("X-Amz-Signature", [""])[0]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", signature):
        raise ValueError("R2 presigned signature missing")
    if not params.get("X-Amz-Expires") or not params.get("X-Amz-Date"):
        raise ValueError("R2 signature expiry missing")
    return True


def download_r2_source(url, path):
    """Write the MP4 to disk without buffering it in RAM, max 100 MiB."""
    import requests
    validate_r2_signed_url(url, source=True)
    received = 0
    with requests.get(url, stream=True, allow_redirects=False,
                      timeout=(25, 120)) as response:
        if response.status_code != 200:
            raise RuntimeError(f"R2 source download HTTP {response.status_code}")
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_INPUT_BYTES:
            raise ValueError("R2 input exceeds 100 MiB")
        with open(path, "wb") as dest:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                received += len(chunk)
                if received > MAX_INPUT_BYTES:
                    raise ValueError("R2 input exceeds 100 MiB")
                dest.write(chunk)
    if received < 16:
        raise ValueError("R2 input MP4 is empty")
    return received


def upload_r2_output(path, descriptor):
    """Stream output to a single-key presigned PUT, without inline base64."""
    if not isinstance(descriptor, dict):
        raise ValueError("Invalid R2 upload descriptor")
    key, url = descriptor.get("key"), descriptor.get("url")
    proof, expires = descriptor.get("proof"), descriptor.get("proof_expires")
    validate_r2_signed_url(url, staging_key=key)
    if not isinstance(proof, str) or not re.fullmatch(r"[0-9a-f]{64}", proof):
        raise ValueError("Invalid output receipt proof")
    if type(expires) is not int or not int(time.time()) < expires <= int(time.time()) + 86400:
        raise ValueError("Invalid output receipt expiry")
    size = os.path.getsize(path)
    if not 16 <= size <= MAX_OUTPUT_BYTES:
        raise ValueError("Extender output must be 16 bytes to 512 MiB")
    import requests
    for attempt in range(2):
        try:
            with open(path, "rb") as handle:
                response = requests.put(url, data=handle, allow_redirects=False,
                    headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
                    timeout=(30, 300))
            if response.status_code not in (200, 201, 204):
                raise RuntimeError(f"R2 output PUT HTTP {response.status_code}")
            return {
                "storage": "r2-staged",
                "key": key,
                "bytes": size,
                "proof": proof,
                "proof_expires": expires,
            }
        except (requests.RequestException, RuntimeError):
            if attempt:
                raise
            time.sleep(2)
    raise RuntimeError("R2 upload failed")
