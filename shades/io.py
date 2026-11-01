"""Probe, 10-bit decode and H.264 encode through ffmpeg pipes.

Frames travel as rgb48le (16 bit per channel) in both directions, so the
10-bit source is never truncated to 8 bit before grading; the only 8-bit
step is the final yuv420p encode, which is dithered beforehand.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import Encode


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    n_frames: int
    duration: float
    has_audio: bool
    pix_fmt: str


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,nb_read_packets,pix_fmt"
         ":format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    return VideoInfo(
        width=int(v["width"]), height=int(v["height"]),
        fps=Fraction(v["r_frame_rate"]), n_frames=int(v["nb_read_packets"]),
        duration=float(d["format"]["duration"]),
        has_audio=any(s["codec_type"] == "audio" for s in d["streams"]),
        pix_fmt=v.get("pix_fmt", ""))


def iter_frames(path: Path, info: VideoInfo, scale_w: int | None = None):
    """Yield float32 RGB frames in 0..1 (BT.709, full-range RGB)."""
    w, h = info.width, info.height
    vf = "scale=in_range=tv:in_color_matrix=bt709"
    if scale_w:
        h = int(round(info.height * scale_w / info.width / 2)) * 2
        w = scale_w
        vf += f":w={w}:h={h}:flags=area"
    vf += ",format=rgb48le"
    proc = subprocess.Popen(
        # passthrough: emit exactly the coded frames. Phone footage is often
        # slightly VFR and the rawvideo muxer would otherwise duplicate/drop
        # frames to force CFR, breaking frame <-> mask index alignment.
        ["ffmpeg", "-v", "fatal", "-i", str(path), "-fps_mode", "passthrough",
         "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
        stdout=subprocess.PIPE, bufsize=w * h * 6 * 2)
    size = w * h * 3 * 2
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint16).reshape(h, w, 3).astype(
                np.float32) / 65535.0
    finally:
        if proc.poll() is None:  # caller stopped early: stop ffmpeg quietly
            proc.kill()
        proc.stdout.close()
        proc.wait()


def read_frames_at(path: Path, info: VideoInfo, indices: list[int],
                   scale_w: int | None = None) -> dict[int, np.ndarray]:
    want, out = set(indices), {}
    for i, f in enumerate(iter_frames(path, info, scale_w)):
        if i in want:
            out[i] = f
            if len(out) == len(want):
                break
    return out


class Encoder:
    """Pipe float frames into ffmpeg -> H.264 High yuv420p BT.709, copying
    the source audio stream untouched."""

    def __init__(self, src: Path, dst: Path, info: VideoInfo, enc: Encode):
        self.info, self.enc = info, enc
        self.rng = np.random.default_rng(1234)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", f"{info.width}x{info.height}",
               "-r", str(info.fps), "-i", "-",
               "-i", str(src),
               "-map", "0:v:0"]
        if info.has_audio:
            cmd += ["-map", "1:a:0", "-c:a", "copy"]
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:"
                       "flags=accurate_rnd+full_chroma_int,format=yuv420p",
                "-c:v", "libx264", "-profile:v", enc.profile,
                "-preset", enc.preset, "-crf", str(enc.crf),
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-color_range", "tv",
                "-movflags", "+faststart", str(dst)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        x = rgb
        if self.enc.dither:
            # TPDF dither of +-1 LSB at 8 bit: breaks up banding in smooth
            # water gradients when the encoder quantises to 8 bit.
            n = (self.rng.random(x.shape[:2] + (1,), np.float32)
                 - self.rng.random(x.shape[:2] + (1,), np.float32)) / 255.0
            x = x + n
        u16 = (np.clip(x, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
        self.proc.stdin.write(u16.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")
