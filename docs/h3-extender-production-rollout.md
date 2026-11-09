# ErosMax H3 Extender: Production-ready rollout plan (NOT deployed)

## Source evidence (2026-11-09)

Successful isolated RunPod/ComfyUI jobs using RTX PRO 6000 MIG 1g.24gb:

| Input | Source length | Context | Native H3 clip frames | Actual new frames | Actual final frames | ComfyUI wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 107631.mp4 | 124 | 5 | 124 | 119 | 243 | 238.49 s |
| output-outputs-0-data (1).mp4 | 124 | 22 | 124 | 102 | 226 | 273.42 s |

Both completed with audio. An earlier side-by-side review found a small
brightness gap at the seam for both; **not a guarantee** for arbitrary clips.

Upstream MiniMax H3 Extender (`tritant/ComfyUI_MiniMax_H3_Extender`, commit
`b3e23bded3bcfb2398ec2470943b35330c7faaff`) aligns native clip
frames to `17n+5` and trims the context window from the start of each new
clip. `duration=5.0` gives 124 native frames; with 22 context frames, only
102 visible new frames remain.

## Context-independent visible duration correction

This branch uses a small, isolated `continuation_frame_plan()` in
`h3_extender_poc.py`:

- Context 5: 124 native frames - 5 context = 119 new frames (~4.96s).
  Keep `clips_json.clips[0].duration=5.0` to preserve the verified baseline.
- Context 22: 141 native frames - 22 context = 119 new frames (~4.96s).
  Set `clips_json.clips[0].duration=5.875` (141/24).
- No changes to `h3_extend_generate`, Model, LoRAs, or Production;
  the new `h3_extender_generate` is still explicit/opt-in.

**Inference with 141 native frames is NOT tested yet**. The expected 119
frames are a deterministic plan based on upstream's source code, not an
actual GPU result. The next step is a single controlled 22-context generation
using the same source file and seed; log effective frames, VRAM and time.
Stop if OOM or the output has less than 119 visible frames.

## Safe Production topology

1. Keep the current ErosMax Production endpoint and existing
   `h3_extend_generate` action untouched as a safe default.
2. Use a separate, pinned Extender image and test endpoint during validation.
   Do not merge the experimental branch or reuse shared GHCR tags.
3. For a future Studio *Preview-only* integration, add a server-side
   `EROSMAX_EXTENDER_ENDPOINT_ID` and an explicit flag
   `EROSMAX_EXTENDER_ENABLED` defaulting to false. Only when enabled,
   a new extension-engine selector can offer `Extender` alongside
   `Standard`; default to Standard.
4. Server-side route must validate engine, mode, duration=5,
   context_length in {5,22}, upload/password authorization, MP4 R2 key and
   optional LoRAs. Unknown or missing test endpoint should reject clearly,
   **not** silently route to production.
5. Preserve Studio's R2 GET-signed video URL with <=3600s expiry;
   do not send R2 API credentials to the client. Worker must restrict URL
   hosts/redirects and download size to avoid SSRF. The present PoC's
   permissive HTTPS fetch is **not production-ready**.
6. Reuse the existing Studio `/api/video/status` saveVideo workflow only
   after confirming it archives Extender's outputs. Ensure >14 MB MP4s
   are handled without exposing base64 errors or silently reporting
   a completed job as a saved result.
7. Explicitly mark Extender as *no LoRA support yet* at the UI level.
   Do not enable production until existing LoRA combinations (0,1,4)
   are supported/tested or until product requirements explicitly approve
   a separate restricted engine.
8. Test audio alignment, multiple seeds, other resolutions/ratios, scene
   switches, and repeated extension; compare against the original engine.
9. With Active Workers 0 / no Network Volume, cache cannot survive a
   recycled worker. Single-job full_batch is fine; multi-job continuity
   requires an R2-backed project/latent cache or a new design.
10. Promotion requires explicit user authorization. The rollback is to turn
    the feature flag off; no data migration or change of the legacy endpoint.

## Acceptance criteria before enabling in Production

- One full output at context 22 contains 119 new frames at 24fps.
- Context 5 remains the verified 119-new-frame baseline.
- No OOM / unhealthy workers during 24GB trials.
- Detectable seam brightness not worse than legacy after varied clips.
- No source audio loss, duplication or source truncation.
- Identical seed/prompt/source comparisons with measured costs.
- Studio Preview can archive the resulting MP4 in Cloudflare R2 reliably.
- Server-side security checks, timeouts and large-file response path complete.
- LoRA support and rollback controls resolved.
- Production and legacy endpoint remain unchanged until a separate go/no-go.
