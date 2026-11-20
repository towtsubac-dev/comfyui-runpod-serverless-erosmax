"""CPU/FFmpeg regression for ErosMax extension on ordinary portrait/1080p MP4."""
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
NAMES = {
    "_h3_extend_aligned_size", "_h3_extend_geometry",
    "_pad_h3_extend_source", "_crop_h3_extend_segment",
    "_concat_extended_video_onepass",
}


def load_helpers():
    selected = [
        n for n in TREE.body
        if isinstance(n, ast.FunctionDef) and n.name in NAMES
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in {
            "H3_EXTEND_MAX_PIXELS", "H3_EXTEND_MAX_AXIS",
        } for t in n.targets)
    ]
    assert len([x for x in selected if isinstance(x, ast.FunctionDef)]) == len(NAMES)
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 "handler.py", "exec"), namespace)
    return namespace


class ErosExtensionAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_helpers()

    def test_common_sizes_and_inference_limit(self):
        align = self.helpers["_h3_extend_aligned_size"]
        expect = {
            (720, 1280): (736, 1280),
            (1280, 720): (1280, 736),
            (1920, 1080): (1344, 768),
            (1080, 1920): (768, 1344),
            (864, 480): (864, 480),
            (896, 1152): (896, 1152),
            (1024, 1024): (1024, 1024),
        }
        for source, target in expect.items():
            with self.subTest(source=source):
                self.assertEqual(align(*source), target)
                self.assertEqual(target[0] % 32, 0)
                self.assertEqual(target[1] % 32, 0)
                self.assertLessEqual(target[0]*target[1], 1024*1024)
        for source in [(3840, 2160), (2160, 3840), (7680, 4320)]:
            with self.subTest(source=source):
                w, h = align(*source)
                self.assertEqual(w % 32, 0)
                self.assertEqual(h % 32, 0)
                self.assertLessEqual(w*h, 1024*1024)
                self.assertLessEqual(max(w,h), 1536)

    def test_edge_padding_and_aspect_are_consistent(self):
        geom = self.helpers["_h3_extend_geometry"]
        for width, height in [(720,1280),(1280,720),(1920,1080),(1080,1920),(864,480)]:
            with self.subTest(source=(width,height)):
                plan=geom(width,height)
                self.assertEqual(plan["left"] + plan["inner_width"] + plan["right"], plan["width"])
                self.assertEqual(plan["top"] + plan["inner_height"] + plan["bottom"], plan["height"])
                self.assertLessEqual(abs(plan["left"]-plan["right"]),1)
                self.assertLessEqual(abs(plan["top"]-plan["bottom"]),1)
                self.assertEqual(plan["inner_width"] % 2, 0)
                self.assertEqual(plan["inner_height"] % 2, 0)
        self.assertFalse(geom(720,1280)["downscaled"])
        self.assertTrue(geom(1920,1080)["downscaled"])
        with self.assertRaises(ValueError):
            self.helpers["_h3_extend_aligned_size"](721,1280)

    def test_patchification_regression_720x1280(self):
        self.assertEqual((720//16,1280//16),(45,80))
        w,h=self.helpers["_h3_extend_aligned_size"](720,1280)
        self.assertEqual((w//16)%2,0)
        self.assertEqual((h//16)%2,0)

    def test_command_uses_smear_lanczos_crf14_and_copies_original_audio(self):
        env=dict(self.helpers)
        commands=[]
        mock=Mock()
        mock.run.side_effect=lambda cmd, **kw: commands.append(cmd)
        env["subprocess"]=mock
        plan=env["_pad_h3_extend_source"]("original.mp4","aligned.mp4",720,1280)
        self.assertEqual((plan["width"],plan["height"]),(736,1280))
        cmd=commands[0]
        vf=cmd[cmd.index("-vf")+1]
        self.assertIn("fillborders=left=8:right=8:top=0:bottom=0:mode=smear",vf)
        self.assertIn("scale=720:1280:flags=lanczos",vf)
        self.assertIn("pad=736:1280:8:0:black",vf)
        self.assertEqual(cmd[cmd.index("-crf")+1],"14")
        self.assertEqual(cmd[cmd.index("-c:a")+1],"copy")
        self.assertNotIn("scale=736:1280",vf)

    def test_concat_crops_inline_without_second_video_encode(self):
        env=dict(self.helpers)
        mock=Mock()
        commands=[]
        mock.run.side_effect=lambda cmd, **kw: commands.append(cmd)
        env["subprocess"]=mock
        original={"width":720,"height":1280,"duration":0.5,"has_audio":True}
        segment={"width":736,"height":1280,"duration":0.5,"has_audio":True}
        env["_probe_video"]=lambda path: (
            original if path=="src.mp4" or path=="out.mp4" else segment
        )
        plan=env["_h3_extend_geometry"](720,1280)
        env["_concat_extended_video_onepass"]("src.mp4","seg.mp4","out.mp4",plan)
        cmd=commands[0]
        fc=cmd[cmd.index("-filter_complex")+1]
        self.assertIn("crop=720:1280:8:0",fc)
        self.assertIn("concat=n=2:v=1:a=1",fc)
        self.assertEqual(cmd[cmd.index("-crf")+1],"16")
        self.assertEqual(sum(x=="libx264" for x in cmd),1)

    def test_only_non_aligned_sources_are_reencoded_before_inference(self):
        self.assertIn('if expected != (info["width"], info["height"]):',SOURCE)
        self.assertIn("build_h3_extend_workflow(model_filename,",SOURCE)
        self.assertIn("_concat_extended_video_onepass(",SOURCE)
        self.assertIn('os.remove(aligned_path)',SOURCE)
        self.assertIn("_concat_extended_video(input_path, segment_path, combined)",SOURCE)
        self.assertNotIn('f"ComfyUI job failed: {record}"',SOURCE)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg unavailable")
    def test_real_ffmpeg_edge_colors_audio_and_onepass_output(self):
        env=self.helpers
        with tempfile.TemporaryDirectory() as directory:
            folder=pathlib.Path(directory)
            original=folder/"original.mp4"
            padded=folder/"padded.mp4"
            output=folder/"joined.mp4"
            subprocess.run([
                "ffmpeg","-y","-v","error","-f","lavfi",
                "-i","color=c=red:s=320x176:r=24:d=0.5",
                "-f","lavfi","-i","sine=frequency=1000:sample_rate=32000:duration=0.5",
                "-t","0.5","-c:v","libx264","-threads","1","-pix_fmt","yuv420p",
                "-c:a","aac",str(original),
            ],check=True,timeout=60)
            env["_pad_h3_extend_source"](str(original),str(padded),320,176)
            # This is the actual decoded pixel at the upper padded border.
            raw=subprocess.run([
                "ffmpeg","-v","error","-i",str(padded),
                "-frames:v","1","-f","rawvideo","-pix_fmt","rgb24","pipe:1",
            ],capture_output=True,check=True,timeout=20).stdout
            self.assertGreater(raw[0],180)
            self.assertLess(raw[1],70)
            self.assertLess(raw[2],70)
            self.assertGreater(raw[(8*320+16)*3],180)
            probe_video=lambda path: _probe(path)
            def _probe(path):
                result=subprocess.run([
                    "ffprobe","-v","error","-show_entries",
                    "stream=codec_type,width,height:format=duration",
                    "-of","json",str(path),
                ],capture_output=True,text=True,check=True,timeout=20)
                import json
                obj=json.loads(result.stdout)
                video=next(x for x in obj["streams"] if x["codec_type"]=="video")
                return {
                    "width":video["width"],"height":video["height"],
                    "duration":float(obj["format"]["duration"]),
                    "has_audio":any(x["codec_type"]=="audio" for x in obj["streams"]),
                }
            self.assertEqual((_probe(padded)["width"],_probe(padded)["height"]),(320,192))
            local=dict(env)
            local["_probe_video"]=_probe
            import types
            # The function's globals, rather than a separate source import, are explicit.
            func=types.FunctionType(
                local["_concat_extended_video_onepass"].__code__,local)
            func(str(original),str(padded),str(output),local["_h3_extend_geometry"](320,176))
            result=_probe(output)
            self.assertEqual((result["width"],result["height"]),(320,176))
            self.assertTrue(result["has_audio"])
            self.assertGreater(result["duration"],0.8)
            self.assertLess(result["duration"],1.3)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg unavailable")
    def test_real_ffmpeg_audio_absent_gets_silent_output(self):
        import json
        import types
        env=dict(self.helpers)
        with tempfile.TemporaryDirectory() as directory:
            folder=pathlib.Path(directory)
            original=folder/"silent.mp4"
            joined=folder/"joined.mp4"
            subprocess.run([
                "ffmpeg","-y","-v","error","-f","lavfi",
                "-i","testsrc2=size=320x192:rate=24:duration=0.3",
                "-c:v","libx264","-threads","1","-an",str(original),
            ],check=True,timeout=60)
            def _probe(path):
                x=subprocess.run([
                    "ffprobe","-v","error","-show_entries",
                    "stream=codec_type,width,height:format=duration",
                    "-of","json",str(path),
                ],capture_output=True,text=True,check=True,timeout=20)
                obj=json.loads(x.stdout)
                video=next(s for s in obj["streams"] if s["codec_type"]=="video")
                return {
                    "width":video["width"],"height":video["height"],
                    "duration":float(obj["format"]["duration"]),
                    "has_audio":any(s["codec_type"]=="audio" for s in obj["streams"]),
                }
            env["_probe_video"]=_probe
            func=types.FunctionType(
                env["_concat_extended_video_onepass"].__code__,env)
            func(str(original),str(original),str(joined))
            self.assertTrue(_probe(joined)["has_audio"])


if __name__=="__main__":
    unittest.main()
