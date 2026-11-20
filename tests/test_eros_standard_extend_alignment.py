"""Offline regression for ErosMax standard H3 extension's 32-pixel patch alignment."""
import ast
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "handler.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def load_alignment_helpers():
    names = {"_h3_extend_aligned_size", "_pad_h3_extend_source", "_crop_h3_extend_segment"}
    selected = [item for item in TREE.body
                if isinstance(item, ast.FunctionDef) and item.name in names]
    assert len(selected) == len(names)
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "handler.py", "exec"), namespace)
    return namespace


class ErosExtensionAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_alignment_helpers()

    def test_1280x720_requires_padding_not_downscaling(self):
        align = self.helpers["_h3_extend_aligned_size"]
        self.assertEqual(align(1280, 720), (1280, 736))
        self.assertEqual(align(720, 1280), (736, 1280))
        self.assertEqual(align(864, 480), (864, 480))
        self.assertEqual(align(896, 1152), (896, 1152))

    def test_alignment_corrects_original_patchify_failure(self):
        # 1280x720 -> VAE latent grid 80x45. Patches of 2x2 cannot
        # reshape a grid with an odd width/height.
        self.assertEqual((1280 // 16, 720 // 16), (80, 45))
        w, h = self.helpers["_h3_extend_aligned_size"](1280, 720)
        self.assertEqual((w // 16) % 2, 0)
        self.assertEqual((h // 16) % 2, 0)

    def test_ffmpeg_operations_preserve_original_geometry(self):
        commands = []
        fake = Mock()
        fake.run.side_effect = lambda command, **kw: commands.append(command)
        env = dict(self.helpers)
        env["subprocess"] = fake
        env["_probe_video"] = lambda path: {"width": 1280, "height": 720}
        exec(compile(ast.Module(
            body=[n for n in TREE.body if isinstance(n, ast.FunctionDef)
                  and n.name in {"_pad_h3_extend_source", "_crop_h3_extend_segment"}],
            type_ignores=[],
        ), "handler.py", "exec"), env)
        self.assertEqual(env["_pad_h3_extend_source"]("original.mp4","padded.mp4",1280,720), (0,8))
        env["_crop_h3_extend_segment"]("segment.mp4","restored.mp4",1280,720,0,8)
        self.assertIn("pad=1280:736:0:8:black", commands[0])
        self.assertIn("crop=1280:720:0:8,setsar=1,format=yuv420p", commands[1])
        self.assertEqual(commands[0][commands[0].index("-map") + 1], "0:v:0")
        self.assertIn("0:a:0?",commands[0])
        self.assertIn("0:a:0?",commands[1])

    def test_unaligned_source_uses_temp_file_and_cleans_up(self):
        self.assertIn('if info["width"] % 32 or info["height"] % 32:', SOURCE)
        self.assertIn("build_h3_extend_workflow(model_filename,", SOURCE)
        self.assertIn("_crop_h3_extend_segment(", SOURCE)
        self.assertIn('os.remove(aligned_path)', SOURCE)
        self.assertIn('_concat_extended_video(input_path, segment_path, combined)', SOURCE)

    def test_failure_logs_never_include_full_comfy_record(self):
        self.assertNotIn('f"ComfyUI job failed: {record}"', SOURCE)
        self.assertIn("ComfyUI job failed at {node}: {reason}", SOURCE)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg unavailable")
    def test_real_ffmpeg_pad_crop_roundtrip_with_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = pathlib.Path(temporary)
            original, padded, restored = (folder / name for name in
                                         ("original.mp4","padded.mp4","restored.mp4"))
            subprocess.run([
                "ffmpeg","-y","-v","error","-f","lavfi",
                "-i","testsrc2=size=320x176:rate=24:duration=0.5",
                "-f","lavfi","-i","anullsrc=channel_layout=stereo:sample_rate=32000",
                "-t","0.5","-c:v","libx264","-threads","1","-pix_fmt","yuv420p",
                "-c:a","aac",str(original),
            ],check=True,timeout=90)
            self.helpers["_pad_h3_extend_source"](str(original),str(padded),320,176)
            self.helpers["_probe_video"] = lambda path: {"width":320,"height":176}
            self.helpers["_crop_h3_extend_segment"](str(padded),str(restored),320,176,0,8)
            for path, expected in [(padded,(320,192)),(restored,(320,176))]:
                probe=subprocess.run([
                    "ffprobe","-v","error","-select_streams","v:0",
                    "-show_entries","stream=width,height","-of","csv=p=0",str(path)
                ],check=True,capture_output=True,text=True,timeout=20)
                self.assertEqual(tuple(map(int,probe.stdout.strip().split(","))),expected)
                streams=subprocess.run([
                    "ffprobe","-v","error","-select_streams","a:0",
                    "-show_entries","stream=codec_type","-of","csv=p=0",str(path)
                ],check=True,capture_output=True,text=True,timeout=20)
                self.assertEqual(streams.stdout.strip(),"audio")


if __name__ == "__main__":
    unittest.main()
