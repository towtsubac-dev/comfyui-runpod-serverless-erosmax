"""CPU-only regression tests for ErosMax Preview's H3 v10/v11 seam detector."""
import ast
from pathlib import Path
import unittest

import numpy as np

HANDLER = Path(__file__).resolve().parents[1] / "handler.py"
NAMES = {"_v10_flash_plan", "_v11_fade_flash_plan"}


def load_detectors():
    tree = ast.parse(HANDLER.read_text(encoding="utf-8"), filename=str(HANDLER))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    assert {node.name for node in functions} == NAMES, "Missing detector functions"
    standalone = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    scope = {}
    exec(compile(standalone, str(HANDLER), "exec"), scope)
    return scope["_v10_flash_plan"], scope["_v11_fade_flash_plan"]


v10, v11 = load_detectors()


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
