"""CPU-only contract tests for secure presigned R2 Extender input/output."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from extender_r2_io import (
    MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, STAGING_KEY_RE, SOURCE_PATH_RE,
    validate_r2_signed_url, download_r2_source, upload_r2_output
)

UUID = "81a2bd49-4d26-41c8-87c9-572c31cfbade"
KEY = "video/staging/extender/" + UUID + ".mp4"
SIG = "a" * 64
BASE = "https://" + "a"*32 + ".r2.cloudflarestorage.com/studio-bucket/"
SUFFIX = "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=" + SIG + "&X-Amz-Expires=3600&X-Amz-Date=20261110T010000Z"


class IoContracts(unittest.TestCase):
    def test_studio_upload_and_staging_put_urls(self):
        self.assertTrue(validate_r2_signed_url(BASE + "uploads/" + UUID + ".mp4" + SUFFIX, source=True))
        self.assertTrue(validate_r2_signed_url(BASE + KEY + SUFFIX, staging_key=KEY))

    def test_unsafe_urls_cannot_reach_network(self):
        good = BASE + KEY + SUFFIX
        invalid = [
            good.replace("https:", "http:"),
            good.replace("a"*32, "127.0.0.1"),
            good.replace(".r2.cloudflarestorage.com", ".r2.cloudflarestorage.com.evil.test"),
            good.replace("studio-bucket", "user:password@studio-bucket"),
            good.replace(KEY, "video/staging/extender/../../private.mp4"),
            good.replace(KEY, "video/staging/extender/" + UUID + ".txt"),
            good.replace(SIG, "invalid"),
            good.replace("X-Amz-Signature", "x-Signature"),
            BASE + KEY + SUFFIX + "#fragment",
        ]
        for candidate in invalid:
            with self.subTest(candidate=candidate[:80]), self.assertRaises(ValueError):
                validate_r2_signed_url(candidate, staging_key=KEY)
        with self.assertRaises(ValueError):
            validate_r2_signed_url(BASE+"video/history/"+UUID+".mp4"+SUFFIX,source=True)

    def test_large_mp4_is_streamed_and_receipt_is_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "test.mp4"
            with file.open("wb") as out:
                out.truncate(15 * 1024 * 1024 + 1)  # > old 14 MB response cap
            expire = int(time.time()) + 3600
            desc = {"key": KEY, "url": BASE + KEY + SUFFIX,
                    "proof": "f"*64, "proof_expires": expire}
            sent = []
            class Response:
                status_code = 200
            def fake_put(url, data, **kwargs):
                sent.append((url, data.read(32), kwargs["headers"]["Content-Type"]))
                return Response()
            class FakeRequests:
                RequestException = IOError
                put = staticmethod(fake_put)
            with patch.dict("sys.modules", {"requests": FakeRequests}):
                receipt = upload_r2_output(file,desc)
            self.assertEqual(receipt["storage"], "r2-staged")
            self.assertEqual(receipt["bytes"], 15 * 1024 * 1024 + 1)
            self.assertNotIn("url",receipt)
            self.assertEqual(sent[0][2],"video/mp4")

    def test_output_rejects_expired_or_unbounded_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "test.mp4"
            file.write_bytes(b"x"*32)
            desc = {"key":KEY, "url":BASE+KEY+SUFFIX, "proof":"e"*64,
                    "proof_expires":int(time.time())-1}
            with self.assertRaises(ValueError):
                upload_r2_output(file,desc)
            desc["proof_expires"]=int(time.time())+3600
            desc["key"]="video/history/"+UUID+".mp4"
            with self.assertRaises(ValueError):
                upload_r2_output(file,desc)

    def test_backend_has_legacy_inline_fallback_and_new_streaming_path(self):
        src = (Path(__file__).resolve().parents[1]/"handler.py").read_text()
        part = src.split("def h3_extender_generate(job_input):",1)[1].split("def handler(job):",1)[0]
        self.assertIn("download_r2_source(video_url, input_path)",part)
        self.assertIn("upload_r2_output(final_path, output_upload)",part)
        self.assertIn('result["r2_upload"]',part)
        self.assertIn("14_000_000",part)
        docker = (Path(__file__).resolve().parents[1]/"Dockerfile.extender-lora-fast").read_text()
        self.assertIn("extender_r2_io.py /workspace/",docker)
        self.assertNotIn("RUN python /workspace/download_models.py",docker)


if __name__=="__main__":
    unittest.main()
