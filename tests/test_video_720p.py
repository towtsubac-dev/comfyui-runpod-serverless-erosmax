import copy
import unittest

from video_720p import apply_720p_crop


class Output720pTests(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "15": {"class_type": "VAEDecode", "inputs": {}},
            "16": {"class_type": "CreateVideo", "inputs": {
                "images": ["15", 0], "audio": ["14", 0], "fps": 24}},
            "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0]}},
        }

    def test_native_output_preserved(self):
        original = copy.deepcopy(self.graph)
        self.assertIs(apply_720p_crop(self.graph, 864, 480), self.graph)
        self.assertEqual(original, self.graph)

    def test_landscape(self):
        apply_720p_crop(self.graph, 1280, 736, "720p")
        self.assertEqual(self.graph["18"]["inputs"],
                         {"image": ["15", 0], "width": 1280, "height": 720,
                          "x": 0, "y": 8})
        self.assertEqual(self.graph["16"]["inputs"]["images"], ["18", 0])
        self.assertEqual(self.graph["16"]["inputs"]["audio"], ["14", 0])

    def test_portrait(self):
        apply_720p_crop(self.graph, 736, 1280, "720p")
        self.assertEqual(self.graph["18"]["inputs"],
                         {"image": ["15", 0], "width": 720, "height": 1280,
                          "x": 8, "y": 0})

    def test_rejects_other_dimensions(self):
        with self.assertRaises(ValueError):
            apply_720p_crop(self.graph, 864, 480, "720p")

    def test_fails_closed_if_video_node_missing(self):
        with self.assertRaises(ValueError):
            apply_720p_crop({"1": {"class_type": "SaveVideo", "inputs": {}}},
                            1280, 736, "720p")


if __name__ == "__main__":
    unittest.main()
