import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from backend.video import convert_to_h264_mp4, probe


def _write_avi(path: Path, codec: str):
    with av.open(str(path), mode="w", format="avi") as out:
        stream = out.add_stream(codec, rate=10)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        if codec == "libx264":
            stream.options = {"preset": "ultrafast", "crf": "22", "bf": "0"}
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


class ConvertTests(unittest.TestCase):
    def test_h264_avi_becomes_probeable_mp4(self):
        with tempfile.TemporaryDirectory() as folder:
            avi = Path(folder) / "clip.avi"
            _write_avi(avi, "libx264")
            dest = Path(folder) / "clip.mp4"
            convert_to_h264_mp4(avi, dest)
            self.assertEqual(probe(dest)["codec"], "h264")

    def test_other_codec_is_transcoded(self):
        with tempfile.TemporaryDirectory() as folder:
            avi = Path(folder) / "clip.avi"
            _write_avi(avi, "mpeg4")
            dest = Path(folder) / "clip.mp4"
            convert_to_h264_mp4(avi, dest)
            self.assertEqual(probe(dest)["codec"], "h264")
