"""CPU-only regression tests for ErosMax Preview's H3 v10/v11 seam detector."""
import ast
from pathlib import Path
import unittest

import numpy as np

HANDLER = Path(__file__).resolve().parents[1] / "handler.py"
NAMES = {"_v10_flash_plan", "_v11_fade_flash_plan", "_v12_slow_fade_flash_plan"}


def load_detectors():
    tree = ast.parse(HANDLER.read_text(encoding="utf-8"), filename=str(HANDLER))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    assert {node.name for node in functions} == NAMES, "Missing detector functions"
    standalone = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    scope = {}
    exec(compile(standalone, str(HANDLER), "exec"), scope)
    return scope["_v10_flash_plan"], scope["_v11_fade_flash_plan"], scope["_v12_slow_fade_flash_plan"]


v10, v11, v12 = load_detectors()


def fixture(kind):
    rng = np.random.default_rng(12345)
    source = rng.uniform(0.25, 0.70, size=(90, 160, 3)).astype(np.float32)
    frames = np.repeat(source[None, ...], 18, axis=0)
    if kind == "single":
        frames[0] = source + np.float32(0.10)
    elif kind == "gradual":
        for idx, offset in enumerate([0.115, 0.085, 0.068, 0.045]):
            frames[idx] = source + np.float32(offset)
        decorrelated = rng.uniform(0.25, 0.70, size=(90, 160, 3)).astype(np.float32)
        for idx in range(4, 18):
            frames[idx] = decorrelated + np.float32({4: 0.015, 5: 0.005}.get(idx, 0.0))
    elif kind == "unrelated":
        frames = rng.uniform(0.25, 0.70, size=(18, 90, 160, 3)).astype(np.float32)
        frames[0] += np.float32(0.10)
    return source, frames


class FlashDetectorTests(unittest.TestCase):
    def test_short_flash_v10(self):
        source, frames = fixture("single")
        plan = v10(frames, source, "synthetic")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["start"], 0)
        self.assertEqual(plan["anchor"], 1)
        self.assertLess(plan["offsets"][0], 0)

    def test_gradual_flash_requires_v11(self):
        source, frames = fixture("gradual")
        self.assertIsNone(v10(frames, source, "synthetic"))
        plan = v11(frames, source)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["start"], 0)
        self.assertEqual(plan["anchor"], 4)
        self.assertEqual(len(plan["offsets"]), 4)
        self.assertTrue(all(x < 0 for x in plan["offsets"]))

    def test_unrelated_scene_is_not_overcorrected(self):
        source, frames = fixture("unrelated")
        self.assertIsNone(v10(frames, source, "synthetic"))
        self.assertIsNone(v11(frames, source))
        self.assertIsNone(v12(frames, source))

    def test_long_slow_fade_matching_real_erosmax_log(self):
        # Reproduce the shape seen in the 2026-11-09 ErosMax test log:
        # a strong +0.108 exposure that decays over ~10 frames, then settles.
        rng = np.random.default_rng(20261109)
        source = rng.uniform(0.25, 0.70, size=(90, 160, 3)).astype(np.float32)
        frames = np.repeat(source[None, ...], 18, axis=0)
        for i, offset in enumerate([
            0.108, 0.104, 0.098, 0.088, 0.086,
            0.070, 0.050, 0.048, 0.035, 0.012,
            -0.008, -0.015, -0.019,
        ]):
            frames[i] = np.clip(source + np.float32(offset), 0, 1)
        self.assertIsNone(v10(frames, source, "synthetic"))
        self.assertIsNone(v11(frames, source))
        plan = v12(frames, source)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["anchor"], 10)
        self.assertEqual(len(plan["offsets"]), 10)
        self.assertLess(plan["offsets"][0], -15)
        self.assertLess(abs(plan["offsets"][-1]), 10)

    def test_persistent_brightness_change_is_not_a_flash(self):
        rng = np.random.default_rng(20261110)
        source = rng.uniform(0.25, 0.70, size=(90, 160, 3)).astype(np.float32)
        frames = np.repeat(np.clip(source + np.float32(0.12), 0, 1)[None, ...], 18, axis=0)
        self.assertIsNone(v12(frames, source))

    def test_single_frame_flash_uses_v10_not_v12(self):
        source, frames = fixture("single")
        self.assertIsNotNone(v10(frames, source, "synthetic"))
        self.assertIsNone(v12(frames, source))

    def test_preview_opt_in_and_y_only(self):
        content = HANDLER.read_text(encoding="utf-8")
        self.assertIn('job_input.get("extension_flash_repair", False) is True', content)
        self.assertIn('job_input.get("extension_compare_flash", False) is True', content)
        self.assertIn("lutyuv=y=", content)
        self.assertIn("MAX_RUNPOD_INLINE_RESULT_BYTES = 19_500_000", content)
        body = content.split("def h3_extend_generate(job_input):", 1)[1].split("def handler(job):", 1)[0]
        self.assertEqual(body.count("prompt_id = queue_prompt(workflow)"), 1)


if __name__ == "__main__":
    unittest.main()
