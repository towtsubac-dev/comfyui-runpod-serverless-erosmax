# H3 Extender LoRA: fast RunPod direct GitHub build

Status: **prepared / not yet built in RunPod / Production unchanged**.

## Why

RunPod GitHub integration limits the Docker build step to 30 minutes.
The original `Dockerfile.extender-poc` recompiles SageAttention with
CUDA SM89/SM120, installs ComfyUI and nodes, and downloads four large
ErosMax / H3 weights. This is unnecessary for a code-only LoRA change.

A successful **previous Context 22** GitHub Actions image build exists:
https://github.com/towtsubac-dev/comfyui-runpod-serverless-erosmax/actions/runs/42596086457

Its published immutable OCI manifest digest is:

```text
ghcr.io/towtsubac-dev/comfyui-runpod-serverless-erosmax@sha256:9ba7121bd64b86f798b97deaec7b1d1aca4bcfe9e64cc361046d50ee790111ff
```

`Dockerfile.extender-lora-fast` inherits exactly this image; it only copies
`handler.py`, `h3_extender_poc.py` and `registered_lora.py`.
BuildKit `COPY --link` allows reusing its existing OCI base layers.

**An anonymous GHCR manifest-read check passed** on GitHub Actions:
https://github.com/towtsubac-dev/comfyui-runpod-serverless-erosmax/actions/runs/42633207471
Twelve CPU workflow tests and Python syntax checks passed in the same run.
RunPod itself has NOT built this new image yet; actual transfer and push
times must still be measured.

## RunPod direct-build settings (test endpoint ONLY)

- GitHub repository: `towtsubac-dev/comfyui-runpod-serverless-erosmax`
- Branch: `fix/erosmax-extender-lora-fast-build`
- Dockerfile: `Dockerfile.extender-lora-fast`
- Build context: `.`
- Endpoint: **existing H3 Extender test**, not ErosMax Production
- GPU: existing 24GB profile, Active Workers 0, Network Volume none
- Keep Dockerfile.extender-poc and earlier successful image URL unchanged.
- Record the current test endpoint's image reference before updating it.
- Do not enable Studio's `EROSMAX_EXTENDER_LORA_ENABLED` until the new
  test worker is Ready and isolated LoRA inference is verified.

Expected: no CUDA compilation and no Hugging Face downloads in this new
Dockerfile. Initial transfer of the large prebuilt image can still take
time and incur image-pull delay; **the build is not yet timed**.

If RunPod's direct GitHub builder still times out while fetching or exporting
large image layers, use a separate GitHub Actions build/push of this tiny
Dockerfile to an isolated GHCR tag, then configure only the existing
test endpoint to use that image URL. Do NOT run the shared `build.yml`,
which can overwrite Production tags.

Roll back by reselecting the previous Extender PoC test image; Studio
Standard remains unchanged. No Production cutover has been authorized.
