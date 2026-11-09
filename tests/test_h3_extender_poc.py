"""No-GPU contract checks for the isolated H3 Extender v3.0.14 PoC."""
import json
from pathlib import Path
import unittest

from h3_extender_poc import build_workflow, summarize_comfy_contract, UPSTREAM_SHA

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
