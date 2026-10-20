import unittest
from registered_lora import download_registered_lora

class RegisteredLoraTests(unittest.TestCase):
    def test_rejects_missing_descriptor(self):
        with self.assertRaises(ValueError):
            download_registered_lora(None)

    def test_rejects_untrusted_host(self):
        item = {
            "id": "sample",
            "sha256": "a" * 64,
            "size": 32,
            "url": "https://example.com/file",
        }
        with self.assertRaisesRegex(ValueError, "host"):
            download_registered_lora(item)

if __name__ == "__main__":
    unittest.main()
