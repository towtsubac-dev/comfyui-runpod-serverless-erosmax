# ErosMax standard extension: 32-pixel alignment / quality Preview

Status: **CPU checks done; GPU generation NOT YET verified; Production unchanged.**

## Isolated test build (RunPod GitHub integration)

- Repository: `towtsubac-dev/comfyui-runpod-serverless-erosmax`
- GitHub branch: `fix/erosmax-extension-32px-align-20261120`
- Dockerfile Path: `Dockerfile.eros-extend-alignment-preview`
- Container is a code-only rebase on the pinned production-proven GHCR digest.
- Dedicated test Endpoint only. **Do not point `xbunhcsz4i2lf3` (ErosMax-Production) at this branch.**
- Reusable retired endpoint candidate: `554rzgrdr80g1o` (SeedVR2 preview). Reassign the GitHub source/build settings and image only once the user has accepted reuse.
- GPU: existing 24GB configuration; container disk: 80GB; min workers: 0; max workers: 1; idle timeout: 5 sec; queue delay: 4 sec; timeout >= 900 sec; no network volume.
- Test model input action: `h3_extend_generate`; durations 3, 5, 8 or 10 seconds; keep the existing 8-step workflow.
- First job must use no LoRAs. Next confirm 2 LoRAs, and extension flash repair opt-in separately.

## Safety / RunPod API

The connected RunPod management key currently lacks `serverless:write` (HTTP 403), so it cannot repurpose the retired endpoint or create a new one. This is **not** a code/build failure. Grant narrowly scoped Serverless write access only to the selected test endpoint, or change the endpoint's GitHub build configuration in the RunPod console. No Production permission should be needed for testing.

## Pre-GPU checks

- Local FFmpeg transformations were executed on synthetic audio-bearing 0.5s MP4s:
  - 720x1280 => 736x1280 => 720x1280, merged ~1s, audio kept
  - 1920x1080 => 1344x768 => 1920x1080, merged ~1s, audio kept
- GitHub Actions CPU contract: 6/8 passed; two FFmpeg-specific tests skipped on hosted runner without FFmpeg. On the Docker image, the new build Dockerfile explicitly runs those tests with FFmpeg present.
- No real RunPod generation, visual continuity or GPU peak-VRAM measurement yet.

## GPU acceptance criteria

1. 720x1280 source at 24fps: 5-second standard extension completes without `patchify_video` reshape error.
2. Output video remains **720x1280**, source segment preserved, output frame count/duration and audio consistent.
3. No black border in generated frames, no unexpected stretching or seams.
4. 1920x1080 only if 720x1280 passes, with expected inference size 1344x768 and original resolution restored.
5. Worker remains healthy at 24GB and result MP4 is stored/displayed correctly by Studio Preview.
6. Existing aligned 864x480 extension path still works.
7. No Production rollout until the test endpoint has succeeded in actual GPU inference and the user accepts the result.

## Rollback

Do not modify Production. A test-endpoint rollback is simply restoring its previous pinned image or GitHub build configuration.
