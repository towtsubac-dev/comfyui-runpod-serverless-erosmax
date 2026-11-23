# ErosMax v1.0 Compact RTX 4090 Preview — 2026-11-23

This is an **isolated test**, not a Production replacement.

- Branch: `test/erosmax-v1-compact-rtx4090`, forked from Production code commit `43144efdf104640b2da2c870b0eee27f5a6f33c3`
- RunPod Preview Endpoint: `tesdb3yl2pdtle`
- Production Endpoint (do not edit): `xbunhcsz4i2lf3`
- GPU: `ADA_24` (NVIDIA GeForce RTX 4090, 24 GB), one GPU; minimum host CUDA `13.2`
- Allowed locations: `CA-MTL-3` (Canada), `EUR-NO-1` (Norway)
- Workers initially `min=0, max=0` until the image is published and verified; idle timeout 5 s; 80 GB container disk
- Model: `LokkenJP/TenEros_Max_optimized_learned` at HF revision `3415250db9e2cb89b886f5375c1306fc933e215f`
- Checkpoint: `10Eros_Max_8stepTurbo_v1.0_hybrid_activation_learned_compact_w4w6int8_14.50gb.safetensors`
- Publisher SHA-256: `569f4337f6101b909c5fcf96efb207464a862a540394cff2723534caea4c73ed`
- ComfyUI v0.46.0 and existing MiniMax H3 text encoder and VAEs; SageAttention 2.2.0
- Dockerfile: `Dockerfile.eros-v1-compact`; build workflow: `.github/workflows/build-eros-v1-compact.yml`
- Preview tag: `ghcr.io/towtsubac-dev/comfyui-runpod-serverless-erosmax:eros-v1-compact-48ed2bd12a20294ef1b10eefa059e8975301917f`
- Checkpoint, 720p, I2V, T2V, Ref2VA, LoRA and extension to be tested before Production decisions.

The official MiniMax H3 Community License excludes the USA, EU, UK and South Korea. The Preview Endpoint restricts placements to Canada and Norway. Only run in a location and use case permitted by the license.

**Preflight status:** Python syntax and existing 720p test suite passed in GitHub Actions workflow run `45178520242`. Image build publication and actual GPU generation remain subject to verification.

**Never start test jobs before v1.0 image publication and inspection.** A queued job with `maxWorkers=0` will not execute.
