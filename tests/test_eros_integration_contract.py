"""Regression guard for the combined ErosMax beta6 production handler."""
import ast
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ErosProductionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler_src = (ROOT / "handler.py").read_text()
        cls.handler_ast = ast.parse(cls.handler_src)
        cls.workflow_src = (ROOT / "eros_ref2va.py").read_text()
        cls.dockerfile = (ROOT / "Dockerfile.extender-lora-fast").read_text()

    def test_legacy_actions_remain_supported(self):
        expected = ["health", "h3_check", "h3_generate", "h3_i2v_generate",
                    "generate", "h3_t2v_generate", "t2v_generate",
                    "h3_extend_generate", "h3_extender_probe",
                    "h3_extender_generate", "cache_check"]
        source = ast.get_source_segment(
            self.handler_src,
            next(n for n in self.handler_ast.body
                 if isinstance(n, ast.FunctionDef) and n.name == "handler"),
        )
        for action in expected:
            with self.subTest(action=action):
                self.assertIn(repr(action), source)

    def test_ref2va_routes_are_separate_and_model_is_unchanged(self):
        for action in ("eros_ref2va_check", "eros_ref2va_generate"):
            self.assertIn(repr(action), self.handler_src)
        self.assertIn("resolve_loras=optional_lora", self.handler_src)
        self.assertIn("MiniMaxH3ReferenceToVideo", self.workflow_src)
        self.assertIn("10Eros_Max_h3_8stepTURBO_beta6_pruned_w4a8.safetensors",
                      self.handler_src)

    def test_production_docker_base_is_unchanged_and_ref2va_included(self):
        self.assertIn(
            "sha256:9ba7121bd64b86f798b97deaec7b1d1aca4bcfe9e64cc361046d50ee790111ff",
            self.dockerfile,
        )
        copied = next(s for s in self.dockerfile.splitlines()
                      if s.startswith("COPY --link "))
        for module in ["handler.py", "h3_extender_poc.py",
                       "registered_lora.py", "extender_r2_io.py", "eros_ref2va.py"]:
            self.assertIn(module, copied)

    def test_no_unintended_downloads_or_python_install(self):
        self.assertNotRegex(self.dockerfile, r"(?m)^RUN\\s+.*(?:wget|curl|pip install|apt-get)")

if __name__ == "__main__":
    unittest.main()
