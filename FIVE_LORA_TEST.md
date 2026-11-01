# ErosMax five-LoRA isolated verification

This branch is for testing up to five registered LoRAs without changing Production.

- Handler: accepts at most five registered LoRAs; preserves order and individual strengths.
- Build workflow: pushes `ghcr.io/towtsubac-dev/comfyui-runpod-serverless-erosmax:five-loras-test`.
- GPU: test with the same 24 GB RunPod GPU class before promoting.
- Verification: submit four registered LoRAs, then inspect worker logs for successful download, prompt validation, and GPU OOM.
- Keep the existing Production endpoint and digest unchanged until the test completes.

Note: accepting four LoRAs does not guarantee they fit in 24 GB VRAM.
