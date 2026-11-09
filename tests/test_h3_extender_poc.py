"""No-GPU contract checks for the isolated H3 Extender v3.0.14 PoC."""
import json
import math
from pathlib import Path
import unittest

from h3_extender_poc import build_workflow, summarize_comfy_contract, continuation_frame_plan, normalize_extender_loras, UPSTREAM_SHA

ROOT = Path(__file__).resolve().parents[1]


def workflow(**changes):
    defaults = dict(
        video_filename="clip0.mp4",
        prompt="Continue the same person and outfit in the same room.",
        width=896,
        height=1152,
        seed=123456,
        model_filename="eros.safetensors",
        clip_filename="qwen.safetensors",
        video_vae_filename="video_vae.safetensors",
        audio_vae_filename="audio_vae.safetensors",
    )
    defaults.update(changes)
    return build_workflow(**defaults)


class ExtenderPocTests(unittest.TestCase):
    def test_native_video_and_ref2va_motion_chain(self):
        nodes = workflow()
        self.assertEqual(nodes["1"]["class_type"], "LoadVideo")
        self.assertEqual(nodes["1"]["inputs"]["file"], "clip0.mp4")
        self.assertEqual(nodes["7"]["class_type"], "MiniMaxH3Extender")
        inputs = nodes["7"]["inputs"]
        self.assertEqual(inputs["continue_existing_video"], ["1", 0])
        self.assertEqual(inputs["model"], ["6", 0])
        self.assertEqual(inputs["generation_mode"], "ref2va")
        self.assertTrue(inputs["motion_context"])
        self.assertEqual(inputs["run_mode"], "full_batch")
        self.assertEqual(inputs["context_length"], "5")
        self.assertEqual(inputs["steps"], 8)
        self.assertFalse(inputs["refine_enabled"])
        self.assertEqual(nodes["8"]["class_type"], "MiniMaxH3MotionContextDiskFinalDecode")
        self.assertEqual(nodes["8"]["inputs"]["cache"], ["7", 0])
        self.assertEqual(nodes["8"]["inputs"]["audio_vae"], ["5", 0])

    def test_one_fixed_seed_five_second_clip(self):
        nodes = workflow(prompt="the same woman continues smiling", seed=789)
        payload = json.loads(nodes["7"]["inputs"]["clips_json"])
        self.assertEqual(len(payload["clips"]), 1)
        cfg = payload["clips"][0]
        self.assertEqual(cfg["duration"], 5.0)
        self.assertEqual(cfg["seed_mode"], "fixed")
        self.assertEqual(cfg["seed"], 789)
        self.assertEqual(cfg["prompt"], "the same woman continues smiling")
        self.assertEqual(cfg["loras"], [])
        self.assertEqual(len(json.loads(nodes["7"]["inputs"]["refs_json"])["refs"]), 9)

    def test_context22_pads_native_frames_for_equal_visible_duration(self):
        context5 = continuation_frame_plan(5, 5)
        context22 = continuation_frame_plan(5, 22)
        self.assertEqual(context5["native_frames"], 124)
        self.assertEqual(context22["native_frames"], 141)
        self.assertEqual(context5["visible_frames"], 119)
        self.assertEqual(context22["visible_frames"], 119)
        self.assertEqual(context5["native_frames"] - 5, 119)
        self.assertEqual(context22["native_frames"] - 22, 119)
        self.assertEqual(context5["clip_duration_seconds"], 5.0)
        self.assertEqual(context22["clip_duration_seconds"], 5.875)
        self.assertEqual(context5["native_frames"] % 17, 5)
        self.assertEqual(context22["native_frames"] % 17, 5)

        # Upstream _duration_to_frames rounds duration*24 and then aligns UP
        # to the next H3 17k+5 frame count.
        def upstream_frame_count(duration):
            n = max(5, round(duration * 24))
            while n % 17 != 5:
                n += 1
            return n
        self.assertEqual(upstream_frame_count(context5["clip_duration_seconds"]), 124)
        self.assertEqual(upstream_frame_count(context22["clip_duration_seconds"]), 141)

    def test_context22_video_workflow_preserves_context_and_fixed_seed(self):
        context5 = workflow(context_length=5)
        context22 = workflow(context_length=22, seed=123456)
        inputs = context22["7"]["inputs"]
        self.assertEqual(inputs["context_length"], "22")
        cfg = json.loads(inputs["clips_json"])["clips"][0]
        self.assertEqual(cfg["duration"], 5.875)
        self.assertEqual(cfg["seed_mode"], "fixed")
        self.assertEqual(cfg["seed"], 123456)
        self.assertEqual(context5["7"]["inputs"]["context_length"], "5")
        self.assertEqual(json.loads(context5["7"]["inputs"]["clips_json"])["clips"][0]["duration"], 5.0)
        self.assertEqual(context22["8"]["inputs"]["cache"], ["7", 0])

    @staticmethod
    def lora(name="portrait", prefix="a", strength=1.0):
        return {"filename": f"library_{name}_{prefix * 16}.safetensors",
                "strength": strength}

    def test_clip_lora_stack_keeps_order_and_strength_context5_and_22(self):
        chosen = [
            self.lora("first", "a", 0.25),
            self.lora("second", "b", 1.0),
            self.lora("third", "c", 1.7),
            self.lora("fourth", "d", 2.0),
        ]
        expected = [{"name": x["filename"], "strength": x["strength"]} for x in chosen]
        for context in (5, 22):
            with self.subTest(context=context):
                nodes = workflow(context_length=context, loras=chosen)
                cfg = json.loads(nodes["7"]["inputs"]["clips_json"])["clips"][0]
                self.assertEqual(cfg["loras"], expected)
                self.assertEqual(cfg["duration"], 5.0 if context == 5 else 5.875)
                # Per-clip upstream support; don't add global model patch nodes.
                self.assertEqual(nodes["6"]["inputs"]["model"], ["2", 0])
                self.assertEqual(sorted(nodes.keys()), [str(i) for i in range(1, 9)])

    def test_no_lora_and_max_five_are_valid(self):
        self.assertEqual(normalize_extender_loras(None), [])
        self.assertEqual(normalize_extender_loras([]), [])
        five = [self.lora("pick" + str(i), str(i), i / 2) for i in range(5)]
        self.assertEqual(len(normalize_extender_loras(five)), 5)
        self.assertEqual(len(json.loads(workflow(loras=five)["7"]["inputs"]["clips_json"])["clips"][0]["loras"]), 5)

    def test_lora_stack_rejects_invalid_and_duplicate_entries(self):
        x = self.lora()
        invalid = [
            "not-a-list", [x] * 2, [self.lora(str(i), str(i)) for i in range(6)],
            [{"name": "user-input.safetensors", "strength": 1.0}],
            [{"filename": "../escape.safetensors", "strength": 1.0}],
            [{"filename": "mismatch.safetensors", "strength": 1.0}],
            [self.lora(strength=-0.1)], [self.lora(strength=2.1)],
            [self.lora(strength=float("nan"))], [self.lora(strength=float("inf"))],
            [self.lora(strength=True)], [self.lora(strength="abc")],
            [self.lora(strength=None)], [None],
        ]
        for case in invalid:
            with self.subTest(case=str(case)[:70]):
                with self.assertRaises(ValueError):
                    workflow(loras=case)

    def test_backend_uses_existing_registered_download_without_global_loader(self):
        src = (ROOT / "handler.py").read_text(encoding="utf-8")
        section = src.split("def h3_extender_generate(job_input):", 1)[1].split("def handler(job):", 1)[0]
        self.assertIn("resolved_loras = optional_lora(job_input)", section)
        self.assertIn("loras=resolved_loras", section)
        self.assertIn('"lora_count": len(resolved_loras)', section)
        self.assertNotIn("apply_optional_lora(workflow", section)
        lib = (ROOT / "registered_lora.py").read_text(encoding="utf-8")
        self.assertIn("sha256", lib)
        self.assertIn("safetensors", lib)
        self.assertIn("allow_redirects=False", lib)

    def test_invalid_inputs_fail_early(self):
        for data in (
            {"video_filename": "../escape.mp4"},
            {"video_filename": "a\\b.mp4"},
            {"width": 890},
            {"height": 1150},
            {"prompt": " "},
            {"seconds": 10},
            {"context_length": 56},
            {"steps": 4},
            {"seed": -1},
            {"seed": 10_000_000_000},
            {"filename_prefix": "../other"},
        ):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    workflow(**data)

    def test_missing_node_probe(self):
        result = summarize_comfy_contract({
            "LoadVideo": {
                "input": {"required": {"file": []}, "optional": {}},
                "output": ["VIDEO"],
            },
        })
        self.assertEqual(result["status"], "missing_nodes")
        self.assertNotIn("LoadVideo", result["missing"])
        self.assertIn("MiniMaxH3Extender", result["missing"])
        self.assertIn("MiniMaxH3MotionContextDiskFinalDecode", result["missing"])

    def test_isolated_dockerfile_and_pinned_upstream(self):
        poc = (ROOT / "Dockerfile.extender-poc").read_text(encoding="utf-8")
        prod = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertEqual(poc.count(UPSTREAM_SHA), 2)
        self.assertIn("ComfyUI_MiniMax_H3_Extender", poc)
        self.assertNotIn("ComfyUI_MiniMax_H3_Extender", prod)
        self.assertIn("RUN python /workspace/download_models.py", poc)
        self.assertIn("COPY handler.py", poc)
        self.assertEqual((ROOT / "h3_extender_poc.py").read_text().count("class_type"), 8)

    def test_opt_in_handler_does_not_replace_existing_extend(self):
        source = (ROOT / "handler.py").read_text(encoding="utf-8")
        self.assertIn('if action == "h3_extend_generate"', source)
        self.assertIn('if action == "h3_extender_probe"', source)
        self.assertIn('if action == "h3_extender_generate"', source)
        self.assertIn("def h3_extend_generate(job_input):", source)
        self.assertIn("def h3_extender_generate(job_input):", source)
        self.assertNotIn("h3_extender_generate(job_input)", source.split('if action == "h3_extend_generate"', 1)[1].split('if action == "h3_extender_probe"',1)[0])


if __name__ == "__main__":
    unittest.main()
