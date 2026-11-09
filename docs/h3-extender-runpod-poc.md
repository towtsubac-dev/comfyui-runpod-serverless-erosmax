# ErosMax MiniMax H3 Extender PoC — isolated RunPod trial

## What this branch does

Compare the existing ErosMax `MiniMaxH3VideoExtendPatched + v13` extension
with tritant's **MiniMax H3 Extender v3.0.14** using the same video and prompt.

- Upstream (Apache-2.0): `tritant/ComfyUI_MiniMax_H3_Extender`
- Locked commit: `b3e23bded3bcfb2398ec2470943b35330c7faaff`
- Branch: `feature/erosmax-h3-extender-poc`
- Build file: **`Dockerfile.extender-poc`** (not `Dockerfile`)
- Existing normal `h3_extend_generate`, old test Endpoint, Studio and Production are unchanged.
- Do NOT merge this branch into `fix/erosmax-isolated-flash-v13-preview`
  or `main` without the user explicitly deciding to promote it.

## Startup/configuration

Create a **separate test Endpoint** that points to this GitHub branch and
`Dockerfile.extender-poc`. Do not retarget the current v13 Endpoint.

RunPod GPU: RTX 4090 or PRO 6000 MIG **24GB**; Active Workers **0**;
Network Volume **none**, cached models in image, Pause idle enabled, Idle timeout
5 seconds. Keep `Container Disk` large enough for the image and transient
Extender clip caches (e.g. 80GB as in current ErosMax).

The large model-inclusive Docker image may take the full 30-minute build limit
to produce/export. Never trigger GitHub Actions `build.yml` manually for this
branch: that workflow has shared GHCR tags.

## Stage 1: probe only (no inference)

From the **new** test Endpoint's RunPod JSON tester:

```json
{
  "input": {
    "action": "h3_extender_probe"
  }
}
```

Expected `output.status == "ok"` and all required nodes available:
`LoadVideo`, `UNETLoader`, `CLIPLoader`, `VAELoader`,
`MiniMaxH3SigmaShift`, `MiniMaxH3Extender`, and
`MiniMaxH3MotionContextDiskFinalDecode`. If
`output.status == "missing_nodes"`, **stop before generating** and inspect
Worker startup logs for custom-node import errors.

This probe only checks imports and API socket contracts. It is not a GPU
compatibility test and the cold start can still incur GPU billing.

## Stage 2: one controlled five-second test

Only if the probe succeeds, provide a short source MP4 stored at a
time-limited HTTPS Cloudflare R2 URL (or supply base64 in `input.video`).
Use the same source, prompt and seed as the existing-v13 comparison.

```json
{
  "input": {
    "action": "h3_extender_generate",
    "video_url": "https://REPLACE-WITH-YOUR-OWN-TEMPORARY-SIGNED-R2-URL",
    "prompt": "Continue the existing shot seamlessly. The same woman keeps smiling at the camera and gently swaying while holding the same double peace signs. Keep the same person, face, outfit, camera framing, lighting and background. No scene change or cut.",
    "extend_seconds": 5,
    "context_length": 5,
    "seed": 123456
  }
}
```

For the first run: **no LoRAs**, `context_length=5`,
`refine_enabled=false`, **8 steps**, source width/height divisible by 32
(e.g. 896×1152), and only one generated clip. The exact generated length
may differ slightly due to H3's temporal framing.

If successful, RunPod output includes `mode=h3_extender_poc` and an inline
base64 MP4. The safeguard rejects outputs larger than 14MB; in that case
the inference may still have completed (check Worker logs). **Do not resubmit
blindly and pay for a second sample.**

## What to compare

- First/last frame source and generated continuity (face, clothes, background)
- Frame 0 brightness at extension seam and whether a one-frame flash occurs
- Repeated-shots/scene cuts and natural movements
- Audio continuity and any duplicated frames
- VRAM, queue delay and execution time, RunPod costs
- Node cache size and availability after Worker recycle

## Known Serverless constraints

The Extender keeps clip caches under its custom-node `cache/` directory. With
Active 0 and no Network Volume these caches are ephemeral and cannot be
assumed to survive a Worker recycle; do not promise persisted project saves
or multi-job resume until R2 cache import/export is implemented.

This PoC uses `run_mode=full_batch` for a single segment so there is no
repeated clip-by-clip request. That deliberately avoids the browser-only
interactive card/validation workflow.

The current ErosMax checkpoint's compatibility with Extender's REF2VA
conditioning **is unverified**. A contract probe can pass even when model
loading or sampling later fails. Keep existing native v13 extension as fallback.

## Rollback

No changes to the active Endpoint, its registry image, Studio Preview env
variables or Production are needed. Delete/disable the new PoC Endpoint if
the model does not work, without changing existing service endpoints.
