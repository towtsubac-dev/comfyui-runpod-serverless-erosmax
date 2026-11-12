"""Offline regression checks for ErosMax beta6 Ref2VA integration; no GPU required."""
import base64
import io
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from eros_ref2va import (
    _read_image, _image_from_url, build_ref2va_workflow, generate
)

WEIGHTS = {
    "model": "10Eros_Max_h3_8stepTURBO_beta6_pruned_w4a8.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


def workflow(images, **kw):
    return build_ref2va_workflow(images, prompt="Two adult people. Keep <Picture 1> and <Picture 2> distinct.",
                                 seed=123456, **WEIGHTS, **kw)


class Ref2vaGraphTests(unittest.TestCase):
    def test_same_beta6_diffusion_model(self):
        wf = workflow(["one.png", "two.png"])
        self.assertEqual(wf["2"]["inputs"]["unet_name"], WEIGHTS["model"])
        self.assertEqual(wf["3"]["inputs"]["clip_name"], WEIGHTS["clip"])
        self.assertEqual(wf["4"]["inputs"]["vae_name"], WEIGHTS["video_vae"])
        self.assertEqual(wf["5"]["inputs"]["vae_name"], WEIGHTS["audio_vae"])
        self.assertFalse(any(n["class_type"] == "LoraLoaderModelOnly" for n in wf.values()))

    def test_flat_native_reference_sockets_and_latent(self):
        wf = workflow(["one.png", "two.png", "three.png"])
        inputs = wf["6"]["inputs"]
        self.assertEqual(wf["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(inputs["ref_images.ref_image_0"], ["20", 0])
        self.assertEqual(inputs["ref_images.ref_image_1"], ["21", 0])
        self.assertEqual(inputs["ref_images.ref_image_2"], ["22", 0])
        self.assertNotIn("ref_images", inputs)
        self.assertEqual(wf["12"]["inputs"]["conditioning"], ["6", 0])
        self.assertEqual(wf["14"]["inputs"]["latent_image"], ["6", 1])
        self.assertEqual(wf["11"]["inputs"]["steps"], 8)
        self.assertEqual(wf["17"]["inputs"]["audio"], ["16", 0])

    def test_experimental_steps_and_match_max(self):
        wf = workflow(["one.png", "two.png"], steps=20, ref_image_size="max")
        self.assertEqual(wf["11"]["inputs"]["steps"], 20)
        self.assertEqual(wf["6"]["inputs"]["ref_image_size"], "max")

    def test_invalid_inputs_rejected(self):
        for images in ([], ["one.png"], ["a", "b", "c", "d"]):
            with self.assertRaises(ValueError):
                workflow(images)
        with self.assertRaises(ValueError):
            workflow(["a","b"], width=1024)
        with self.assertRaises(ValueError):
            workflow(["a","b"], steps=24)
        with self.assertRaises(ValueError):
            workflow(["a","b"], ref_image_size="big")

    def test_base64_images_sanitized_as_png(self):
        im = Image.new("RGB", (256, 256), "blue")
        buf = io.BytesIO()
        im.save(buf, "JPEG")
        data = {"image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}
        decoded = _read_image(data)
        with Image.open(io.BytesIO(decoded)) as output:
            self.assertEqual(output.format, "PNG")
            self.assertEqual(output.size, (256, 256))

    def test_signed_url_ignores_local_urls_without_request(self):
        for url in (
            "http://example.com/image.png",
            "https://localhost/image.png",
            "https://127.0.0.1/image.png",
            "https://metadata.google.internal/file",
            "https://abcd.r2.cloudflarestorage.com/image.png",
        ):
            with self.assertRaises(ValueError):
                _image_from_url(url)

    def test_runpod_action_inline_response_without_real_gpu(self):
        im = Image.new("RGB", (256, 256), "green")
        buf = io.BytesIO()
        im.save(buf, "PNG")
        src = base64.b64encode(buf.getvalue()).decode("ascii")
        queued = []
        with tempfile.TemporaryDirectory() as tmp:
            result = generate(
                {"reference_images": [{"image": src}, {"image": src}],
                 "prompt": "Adult A from <Picture 1> on the left; Adult B from <Picture 2> on the right.",
                 "seed": 777},
                comfyui_dir=tmp, **WEIGHTS,
                ensure_comfyui=lambda: None,
                upload_image=lambda raw, name: name,
                queue_prompt=lambda graph: queued.append(graph) or "p1",
                wait_for_result=lambda id, timeout: {"outputs": {}},
                extract_outputs=lambda history: [{"filename": "ok.mp4", "type": "videos"}],
                download_output=lambda item: "dGVzdC1tcDQ=",
            )
        self.assertEqual(result["model"], WEIGHTS["model"])
        self.assertEqual(result["reference_image_count"], 2)
        self.assertEqual(result["seed"], 777)
        self.assertEqual(queued[0]["13"]["inputs"]["noise_seed"], 777)
        self.assertEqual(result["outputs"][0]["data"], "dGVzdC1tcDQ=")

    def test_ordered_library_loras_on_original_beta6_model(self):
        refs = [{"filename": f"library_{i}.safetensors", "strength": strength}
                for i, strength in enumerate([0.8, 0.3, 1.2, 0.5, 0.75])]
        wf = workflow(["one.png", "two.png"], optional_loras=refs)
        self.assertEqual(wf["2"]["inputs"]["unet_name"], WEIGHTS["model"])
        self.assertEqual(wf["30"]["inputs"]["model"], ["2", 0])
        self.assertEqual(wf["34"]["inputs"]["model"], ["33", 0])
        self.assertEqual(wf["8"]["inputs"]["model"], ["34", 0])
        self.assertEqual(wf["31"]["inputs"]["strength_model"], 0.3)
        self.assertEqual(wf["34"]["inputs"]["lora_name"], "library_4.safetensors")
        self.assertEqual(len([v for v in wf.values() if v["class_type"] == "LoraLoaderModelOnly"]), 5)

    def test_lora_strength_and_count_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "5"):
            workflow(["one.png", "two.png"], optional_loras=[
                {"filename": str(i), "strength": 1} for i in range(6)])
        with self.assertRaisesRegex(ValueError, "strength"):
            workflow(["one.png", "two.png"], optional_loras=[
                {"filename": "bad", "strength": 2.5}])

    def test_generate_resolves_loras_and_applies_to_beta6(self):
        img = Image.new("RGB", (256, 256), "red")
        buffer = io.BytesIO()
        img.save(buffer, "PNG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        queued = []
        selected = []
        with tempfile.TemporaryDirectory() as tmp:
            output = generate({
                "reference_images": [{"image": data}, {"image": data}],
                "prompt": "Test <Picture 1> and <Picture 2>",
                "loras": [{"name": "library:good", "strength": 0.8}],
                "seed": 999,
            }, comfyui_dir=tmp, **WEIGHTS,
                ensure_comfyui=lambda: None,
                upload_image=lambda raw, name: name,
                queue_prompt=lambda wf: queued.append(wf) or "p1",
                wait_for_result=lambda id, timeout: {"outputs": {}},
                extract_outputs=lambda history: [{"filename": "ok.mp4", "type": "videos"}],
                download_output=lambda item: "dGVzdA==",
                resolve_loras=lambda inp: selected.append(inp["loras"]) or [
                    {"filename": "library_good.safetensors", "strength": 0.8}
                ],
            )
        self.assertEqual(output["lora_count"], 1)
        self.assertEqual(queued[0]["30"]["inputs"]["model"], ["2", 0])
        self.assertEqual(queued[0]["8"]["inputs"]["model"], ["30", 0])
        self.assertEqual(len(selected), 1)

    def test_loras_requires_a_trusted_resolver(self):
        with self.assertRaisesRegex(ValueError, "resolver"):
            generate({"reference_images": [{}, {}], "prompt": "test",
                      "loras": [{"name": "foo"}]},
                     comfyui_dir="/tmp", **WEIGHTS, ensure_comfyui=None, upload_image=None,
                     queue_prompt=None, wait_for_result=None, extract_outputs=None, download_output=None)


if __name__ == "__main__":
    unittest.main()
