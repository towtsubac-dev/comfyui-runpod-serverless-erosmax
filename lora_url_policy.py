"""Strict allowlist for presigned LoRA download sources and HF CDN redirects.

The Studio signs only the registered LoRA object. Never follow an unchecked
redirect: a signed URL alone is not a safe network destination.
"""
from urllib.parse import parse_qs, urljoin, urlparse

HF_S3_HOST = "twsub.s3.hf.co"
HF_BUCKET = "ai-studio-loras"
HF_DOWNLOAD_HOSTS = frozenset({
    "cas-bridge.xethub.hf.co",
    "cas-bridge.xethub-eu.hf.co",
    "cas-server.xethub.hf.co",
    "cas-server.xethub-eu.hf.co",
    "transfer.xethub.hf.co",
    "transfer.xethub-eu.hf.co",
    "us.aws.cdn.hf.co",
    "us.gcp.cdn.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
})


def _safe_https(parsed):
    try:
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except ValueError:
        return False


def lora_source(url, lora_id):
    """Return 'r2' or 'huggingface'; reject all other download sources."""
    parsed = urlparse(url if isinstance(url, str) else "")
    if not _safe_https(parsed):
        raise ValueError("Invalid signed LoRA URL")
    if parsed.hostname.endswith(".r2.cloudflarestorage.com"):
        # Existing R2 object contract; do not change its URL handling.
        return "r2"
    if (
        parsed.hostname == HF_S3_HOST
        and parsed.path == f"/{HF_BUCKET}/loras/library/{lora_id}.safetensors"
    ):
        params = parse_qs(parsed.query)
        if (
            params.get("X-Amz-Algorithm") == ["AWS4-HMAC-SHA256"]
            and params.get("X-Amz-Signature")
            and params.get("X-Amz-Expires")
        ):
            return "huggingface"
    raise ValueError("Invalid signed LoRA URL")


def hf_redirect_url(current_url, location):
    """Validate a single HF CDN redirect without exposing signed URL fields."""
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Missing Hugging Face LoRA redirect")
    next_url = urljoin(current_url, location)
    parsed = urlparse(next_url)
    if not _safe_https(parsed) or parsed.hostname not in HF_DOWNLOAD_HOSTS:
        raise ValueError("Unapproved Hugging Face LoRA redirect")
    return next_url

# Redirects are checked individually; never let the HTTP client follow
# untrusted Location headers automatically.
from contextlib import contextmanager


@contextmanager
def open_signed_lora(url, provider, timeout=(30, 120)):
    """Requests stream with a strict redirect allowlist and bounded hops."""
    import requests

    current = url
    for hop in range(4):
        with requests.get(current, stream=True, timeout=timeout, allow_redirects=False) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                if provider != "huggingface" or hop >= 3:
                    raise ValueError("Unapproved LoRA download redirect")
                current = hf_redirect_url(current, response.headers.get("Location"))
                continue
            if response.status_code != 200:
                raise ValueError(f"LoRA download failed (HTTP {response.status_code})")
            yield response
            return
    raise ValueError("Too many Hugging Face LoRA redirects")


@contextmanager
def open_signed_lora_urllib(url, provider, opener, timeout=120):
    """urllib stream preserving no-redirect policy, for Illustrious/NoobAI."""
    from urllib.error import HTTPError
    from urllib.request import Request

    current = url
    for hop in range(4):
        req = Request(current, headers={"User-Agent": "AI-Studio-Registered-LoRA/1"}, method="GET")
        try:
            response = opener.open(req, timeout=timeout)
        except HTTPError as error:
            if error.code in (301, 302, 303, 307, 308):
                if provider != "huggingface" or hop >= 3:
                    error.close()
                    raise ValueError("Unapproved LoRA download redirect") from None
                location = error.headers.get("Location")
                error.close()
                current = hf_redirect_url(current, location)
                continue
            error.close()
            raise ValueError(f"LoRA download failed (HTTP {error.code})") from None
        with response:
            if response.status != 200:
                raise ValueError(f"LoRA download failed (HTTP {response.status})")
            yield response
        return
    raise ValueError("Too many Hugging Face LoRA redirects")
