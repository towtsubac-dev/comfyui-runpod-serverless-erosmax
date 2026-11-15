import unittest

from lora_url_policy import lora_source, hf_redirect_url

ID = "1234abcd-5678-9abc-0123-ffffffffffff"
HF_URL = (
    f"https://twsub.s3.hf.co/ai-studio-loras/loras/library/{ID}.safetensors"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Signature=abc123"
)


class SignedLoraUrlTests(unittest.TestCase):
    def test_existing_r2_signed_url(self):
        self.assertEqual(lora_source("https://account.r2.cloudflarestorage.com/bucket/loras/library/a.safetensors?X-Amz-Signature=abc", ID), "r2")

    def test_expected_hf_signed_url(self):
        self.assertEqual(lora_source(HF_URL, ID), "huggingface")

    def test_hf_wrong_bucket_or_id(self):
        for url in (HF_URL.replace("ai-studio-loras", "other-bucket"),
                    HF_URL.replace(ID, "other"),
                    HF_URL.replace("twsub.s3.hf.co", "other.s3.hf.co"),
                    HF_URL.replace("X-Amz-Signature=abc123", "")):
            with self.subTest(url=url.split("?")[0]):
                with self.assertRaisesRegex(ValueError, "Invalid signed LoRA URL"):
                    lora_source(url, ID)

    def test_reject_http_fake_suffix_userinfo_and_ports(self):
        for url in (HF_URL.replace("https:", "http:"),
                    HF_URL.replace("twsub.s3.hf.co", "twsub.s3.hf.co.evil.test"),
                    HF_URL.replace("twsub.s3.hf.co", "username@twsub.s3.hf.co"),
                    HF_URL.replace("twsub.s3.hf.co", "twsub.s3.hf.co:8080"),
                    HF_URL + "#trailing"):
            with self.subTest(case=url.split("?")[0]):
                with self.assertRaisesRegex(ValueError, "Invalid signed LoRA URL"):
                    lora_source(url, ID)

    def test_allow_explicit_hf_cdn_redirects(self):
        for hostname in ("cas-bridge.xethub.hf.co", "cas-bridge.xethub-eu.hf.co",
                         "cas-server.xethub.hf.co", "cas-server.xethub-eu.hf.co",
                         "transfer.xethub.hf.co", "transfer.xethub-eu.hf.co",
                         "us.aws.cdn.hf.co", "us.gcp.cdn.hf.co",
                         "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co"):
            target = f"https://{hostname}/download?signature=test"
            self.assertEqual(hf_redirect_url(HF_URL, target), target)

    def test_reject_cross_origin_and_unsafe_hf_redirects(self):
        for loc in ("http://cas-bridge.xethub.hf.co/path",
                    "https://cas-bridge.xethub.hf.co.evil.test/path",
                    "https://localhost:8443/file",
                    "https://127.0.0.1/file",
                    "https://evil.test/file",
                    "https://cas-bridge.xethub.hf.co:8443/path",
                    "https://user@cas-bridge.xethub.hf.co/file",
                    "/internal/file", "//evil.test/file", ""):
            with self.subTest(location=loc):
                with self.assertRaises(ValueError):
                    hf_redirect_url(HF_URL, loc)



class DownloadRedirectTests(unittest.TestCase):
    def test_requests_hf_redirect_checked_per_hop(self):
        from unittest.mock import patch
        from lora_url_policy import open_signed_lora

        class Response:
            def __init__(self, code, location=None):
                self.status_code = code
                self.headers = {"Location": location} if location else {}
            def __enter__(self): return self
            def __exit__(self, *args): return False

        response = Response(200)
        redirect = Response(302, "https://cas-bridge.xethub.hf.co/signed")
        with patch("requests.get", side_effect=[redirect, response]) as getter:
            with open_signed_lora(HF_URL, "huggingface") as actual:
                self.assertIs(actual, response)
            self.assertEqual(getter.call_count, 2)
            self.assertFalse(getter.call_args_list[0].kwargs["allow_redirects"])
            self.assertFalse(getter.call_args_list[1].kwargs["allow_redirects"])

    def test_requests_reject_unapproved_hf_redirect(self):
        from unittest.mock import patch
        from lora_url_policy import open_signed_lora

        class Redirect:
            status_code = 302
            headers = {"Location": "https://evil.example/private"}
            def __enter__(self): return self
            def __exit__(self, *args): return False

        with patch("requests.get", return_value=Redirect()) as getter:
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                with open_signed_lora(HF_URL, "huggingface"):
                    pass
            self.assertEqual(getter.call_count, 1)

if __name__ == "__main__":
    unittest.main()
