# ErosMax H3 Extender / R2 LoRA integration (isolated candidate)

Status: **CPU contract checked; not GPU-tested; NOT deployed** (2026-11-09).

## Mechanism

The pinned `tritant/ComfyUI_MiniMax_H3_Extender` includes
`_normalize_clip_loras()` and `_apply_per_clip_loras()`. In REF2VA Motion
Context the node reads `clips_json.clips[i].loras`, resolves each via
`folder_paths.get_full_path_or_raise("loras", name)`, and calls
`comfy.sd.load_lora_for_models(model, None, weights, strength, 0.0)`.
Each card receives an independent patched MODEL; TEXT encoder is not modified.
LoRAs are applied in serialized order.

No extra ComfyUI `LoraLoaderModelOnly` chain is required before Sigma Shift
for this Extender path: such a chain could duplicate the stack or leak
per-card settings into other clips.

The new opt-in `h3_extender_generate` path reuses the existing ErosMax
`optional_lora(job_input)` and `registered_lora.download_registered_lora()`.
The helper accepts R2 metadata + a short-lived signed URL provided by the
authenticated Studio backend. It validates host, SHA-256, byte count, and
safetensors layout, stores weights under `/workspace/ComfyUI/models/loras`,
and returns a local basename.

`h3_extender_poc.build_workflow(loras=[...])` converts the already-resolved
list into the per-card `clips_json` stack, rejecting unverified/unexpected
filenames, path traversals, duplicates, NaN/infinite or out-of-range
strengths. Max 5 items, 0 <= strength <= 2.0. LoRA selection order is kept.
No change to `h3_extend_generate`, the default ErosMax model, or other modes.

## GPU acceptance sequence

1. Complete existing Context 22 Studio Preview test **before** modifying
   any RunPod test endpoint or Docker image.
2. Create a separate, pinned image from this branch with
   `Dockerfile.extender-poc`. Only use the existing Extender test endpoint
   **after** the Context 22 run finishes and its results are recorded.
3. Enable LoRA controls on a separate Studio Preview branch by setting
   `EROSMAX_EXTENDER_LORA_ENABLED=true` **only after** the new image is
   Ready. Do not set this flag in Production.
4. Run A/B same source, prompt, seed, Context 5: 0 LoRA versus 1 LoRA
   at strength 1.0; check output and logs, identity, seam, audio, VRAM,
   execution time, download size/time. Do not assume effect from success
   alone—compare the video appearance and request metadata.
5. Run 4 LoRAs at 1.0 each; check order, memory, clip length, and identical
   `context=22` case. Optional 5 only if 4 are stable.
6. Verify R2 generation history, replay and download. Retry worker cold
   start: no Network Volume; download will recur on a new worker.
7. Review 14 MB inline response cap and larger output behavior; harden
   worker source URL checks and protect secrets before Production rollout.

## Rollback

- Keep `EROSMAX_EXTENDER_LORA_ENABLED` unset or false until GPU-tested.
- Existing Extender PoC branch/image and Production images stay untouched.
- If testing fails, change the test endpoint back to its previous image.
- Always leave Standard extension as the default engine.
