# ErosMax H3 Extender — R2 direct MP4 upload (Preview only)

Status: **CPU tests passed; no GPU build or R2 PUT test yet; no Production changes**.

## Problem

The existing RunPod handler reads the generated MP4 into memory and returns
base64 in the job output. It rejects the resulting file above 14,000,000
bytes to stay below RunPod's inline response size. Thus a normal 10-second
896x1152 MP4 might fail even though ComfyUI finished successfully.

## New opt-in flow (without RunPod or browser R2 API credentials)

1. The authenticated Studio server uses existing R2 API credentials to
   presign a single PUT to a random
   `video/staging/extender/<uuid>.mp4` key, 1-hour expiry.
   The Studio server also produces an HMAC receipt proof with the existing
   `UPLOAD_SECRET` valid for up to 24 hours. The signed PUT URL is sent only
   to the RunPod job, not to the user's browser.
2. The worker streams Studio's signed R2 source GET directly to its
   ComfyUI input file. Only HTTPS account R2 API URLs, a Studio
   `uploads/<uuid>.mp4` key, and AWS V4 signatures are accepted.
   No redirects, arbitrary URLs or local network access.
3. The existing H3 Extender workflow, Context 5/22, LoRAs and audio are
   unchanged.
4. The worker streams its finished MP4 with `requests.put` to the
   designated staging R2 key. Signed Content-Type is `video/mp4`.
   It returns ONLY `r2_upload` receipt fields (key, byte count, HMAC
   proof, expiry), not the media bytes or presigned URL.
5. Studio status verifies the HMAC, key format and 16-byte to 512-MiB
   byte count, then HEAD-checks the staging object, copies it server-side
   to `video/history/<RunPod-job-id>.mp4`, verifies copied object size,
   writes the usual history JSON, and deletes the staging object.
   Download/history remain the existing authenticated routes.
6. If archive processing temporarily fails, the status endpoint exposes
   `ARCHIVE_PENDING` (no receipt proof or signing URL), and the UI
   keeps polling to retry.

When Studio's `EROSMAX_EXTENDER_R2_OUTPUT_ENABLED` flag is absent/false,
the worker uses the previous inline return path (14MB cap) so older
Preview deployments and existing direct RunPod clients remain compatible.

## Test and constraints

- A new `extender_r2_io.py` module contains validation and streaming code.
- CPU tests cover trusted and untrusted URL shapes, a >14MB streaming PUT,
  malformed receipts, LoRA & Context contracts, and overlay Dockerfile.
- Single HTTP PUT is appropriate for small/medium MP4 outputs;
  R2 single PUT limit is 5 GiB. This PoC bounds output to **512 MiB** and
  input to **100 MiB** (Studio source upload UI still caps at **20 MiB**).
  For routinely large (>100 MiB) outputs, use multipart upload and a
  resumable transfer in a future revision.
- After disabling a job or a failed RunPod worker, its staging key might
  remain. Configure an R2 lifecycle rule to delete objects under
  `video/staging/extender/` after 1–2 days, or periodically clean up.
- The 1-hour signed PUT URL can expire if the RunPod queue is delayed
  excessively; this is a retryable job error, not an automatic fallback
  to embedding MP4 in RunPod output.
- The worker requires the existing `requests` library; NO AWS
  credentials, new Python dependencies, model downloads or CUDA build
  stages. Use `Dockerfile.extender-lora-fast` on this branch.

## First GPU test

1. Preserve currently successful Extender test image URL for rollback.
2. RunPod direct GitHub build: branch
   `feature/erosmax-extender-r2-direct-preview`,
   Dockerfile `Dockerfile.extender-lora-fast`, build context `.`.
   Update only the existing Extender **test** endpoint, not Production.
3. When the worker is Ready, enable
   `EROSMAX_EXTENDER_R2_OUTPUT_ENABLED=true` only for the **separate**
   Studio R2 Preview branch
   `feature/erosmax-extender-r2-direct-preview-ui` and redeploy it.
4. Test Context 5, LoRA 0, then Context 22 + LoRA 4. On success inspect
   response `output.r2_upload.storage = r2-staged`, R2 history entry
   `storage = r2-direct`, video playback, audio, and an output >14MB.
5. If direct upload fails, turn the Preview flag off and redeploy;
   revert only the **test endpoint** image if necessary.
6. Production remains untouched until explicitly approved.

Security: treat signed PUT URL as a bearer token. The one-hour URL is never
logged or included in the handler response; R2 access and the HMAC secret
never leave Vercel. Do not post full RunPod input JSON publicly.
