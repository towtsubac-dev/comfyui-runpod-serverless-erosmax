# ErosMax beta6 W4A8 Ref2VA — isolated Preview experiment

## Objective

Prove or disprove whether **the exact current ErosMax beta6 diffusion model** can
use the native ComfyUI `MiniMaxH3ReferenceToVideo` reference-image conditioning.
A successful job alone does not prove that the extra images improve identity
consistency; compare output quality with MiniMax H3 Ref2VA Preview before release.

Production I2V / T2V / Extend / LoRA remain untouched.

### Unchanged weights

- `10Eros_Max_h3_8stepTURBO_beta6_pruned_w4a8.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_int8_convrot.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`

No Ref2VA beta2 model, no MiniMax Ref2VA W6A8, and no extra Turbo LoRA.

## Build directly on RunPod (preferred)

Use **RunPod Serverless → New Endpoint → Import Git Repository**. The endpoint
must be created using the **GitHub repository source**, not a pre-built GHCR
container image, to use RunPod's own builder.

- Repo: `towtsubac-dev/comfyui-runpod-serverless-erosmax`
- Branch: `feature/erosmax-beta6-ref2va-preview`
- Dockerfile Path: `Dockerfile.ref2va-beta6`
- Build context: repository root
- Endpoint name: `erosmax-beta6-ref2va-runpod-test`
- Serverless mode: Queue
- GPU: 24 GB compatible
- Active/min workers: 0; max workers: 1; idle timeout: 5 seconds
- Queue delay: 4 seconds
- Container disk: 80 GB; Network Volume: None
- Timeout: 900–1800 seconds where available

**Do not select the GHCR image option or production branch.** RunPod
builds the specified Dockerfile and deploys its own managed container
image into this new test endpoint. The immutable GHCR digest in the
Dockerfile is a **public model-bundled base image**, not an output
target. As a result, no additional diffusion model download or model
replacement occurs during this preview build.

The optional `.github/workflows/build-eros-beta6-ref2va-preview.yml`
GHCR job is manually triggered only. It should not be run for this
workflow. An initial unintended GitHub Actions build may remain in
GitHub's history; it is separate from RunPod Builds.

Check the **Builds** tab of the new RunPod endpoint until it shows
Completed and the worker passes its initialization health check.
No GPU is needed merely to build the image.


## First diagnostic request

```json
{"input":{"action":"eros_ref2va_check"}}
```

Read `native_node_present`, `ref2va_ready`, and `existing_model`. The
`compatibility_proven` field intentionally remains false until full GPU
generation has been evaluated. A successful node check does not imply the
beta6 model has been trained to follow the references.

## Generate request

```json
{
  "input": {
    "action": "eros_ref2va_generate",
    "reference_images": [
      {"image_url": "https://YOUR_ACCOUNT.r2.cloudflarestorage.com/BUCKET/IMAGE1?X-Amz-Algorithm=..."},
      {"image_url": "https://YOUR_ACCOUNT.r2.cloudflarestorage.com/BUCKET/IMAGE2?X-Amz-Algorithm=..."}
    ],
    "prompt": "Use <Picture 1> as the identity reference for the adult person on the left, and <Picture 2> as the reference for a different adult person on the right. Keep two distinct faces and outfits throughout. Both smile, gently turn toward one another, then wave. Fixed camera.",
    "width": 864,
    "height": 480,
    "length": 124,
    "steps": 8,
    "ref_image_size": "match",
    "seed": 123456
  }
}
```

Both URLs must be fresh signed **Cloudflare R2 HTTPS GET URLs**. Alternately
each image can be `{"image":"<base64>"}`. 2–3 images supported; default 8
steps. Experimental 20 steps is available, but the existing 8-step beta6
Turbo model is not guaranteed to benefit from 20. Optional LoRAs disabled.

Output follows existing ErosMax inline format:
`output.outputs[0].data` is base64-encoded MP4.

## Test progression

1. `eros_ref2va_check` — verify **same** beta6 weights and native node present
2. 2 images of the same person, 8 steps, `match`
3. Two different adults, 8 steps, `match` — check face blending, left/right
   swaps, outfit mixing, person count
4. Only if useful: 20 steps / `max` performance and quality comparisons
5. If no identity effect, do not label as supported; keep the model unchanged
   and record the limitation instead of silently replacing it

RunPod billing begins when workers execute jobs. Keep Active Workers = 0.
