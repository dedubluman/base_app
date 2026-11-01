"""FFmpeg-based reading/writing.

Frames are decoded as 16-bit RGB so 10-bit HEVC/ProRes sources keep their
precision through the grade, then converted to float32 in [0, 1]. Every
reader applies the same 9:16 fit (scale to cover + center crop), so frame N
here is frame N in the SAM 3.1 mask cache.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator, Optional

import numpy as np


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH (install FFmpeg 6+)")


def probe(path: str) -> dict:
    _require_ffmpeg()
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
            "-of", "json", path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "duration": float(info["format"]["duration"]),
    }


def fit_filter(width: int, height: int) -> str:
    """Scale to cover WxH, then center-crop: any input becomes a 9:16 frame."""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase"
        f":flags=lanczos:in_color_matrix=auto,"
        f"crop={width}:{height}"
    )


def read_frames(path: str, width: int, height: int) -> Iterator[np.ndarray]:
    """Yield float32 RGB frames (H, W, 3) in [0, 1], in decode order."""
    _require_ffmpeg()
    cmd = [
        "ffmpeg", "-v", "error", "-i", path, "-an", "-fps_mode", "passthrough",
        "-vf", fit_filter(width, height),
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "-",
    ]
    frame_bytes = width * height * 3 * 2
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype="<u2").reshape(height, width, 3)
            yield arr.astype(np.float32) * (1.0 / 65535.0)
    finally:
        if proc.poll() is None:  # consumer stopped early: stop ffmpeg quietly
            proc.kill()
        proc.stdout.close()
        proc.wait()


class SequentialFrameSource:
    """Random-ish access for monotonically increasing frame indices.

    The time-remapped render only ever moves forward, so a small window of
    decoded frames is enough; nothing is held in RAM beyond that.
    """

    def __init__(self, path: str, width: int, height: int, window: int = 3):
        self._it = read_frames(path, width, height)
        self._cache: dict[int, np.ndarray] = {}
        self._next = 0
        self._window = window
        self.exhausted = False

    def get(self, idx: int) -> Optional[np.ndarray]:
        while idx >= self._next and not self.exhausted:
            try:
                self._cache[self._next] = next(self._it)
                self._next += 1
            except StopIteration:
                self.exhausted = True
        for k in [k for k in self._cache if k < idx - self._window]:
            del self._cache[k]
        return self._cache.get(idx)


def count_frames(path: str) -> int:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-an", "-fps_mode", "passthrough",
         "-f", "null", "-", "-progress", "pipe:1"],
        check=True, capture_output=True, text=True,
    ).stdout
    frames = [line for line in out.splitlines() if line.startswith("frame=")]
    return int(frames[-1].split("=")[1])


class VideoWriter:
    """Pipe float RGB frames to FFmpeg; BT.709 tagged, web/Reels friendly."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        crf: int = 16,
        preset: str = "slow",
        dither: bool = True,
        music: Optional[str] = None,
        music_fade: float = 1.0,
        duration: Optional[float] = None,
    ):
        _require_ffmpeg()
        self.width, self.height, self.dither = width, height, dither
        self._rng = np.random.default_rng(0)
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", f"{fps}", "-i", "-",
        ]
        if music:
            cmd += ["-i", music]
        cmd += [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart",
        ]
        if codec == "libx265":
            cmd += ["-tag:v", "hvc1"]
        if music:
            cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
            if duration:
                start = max(0.0, duration - music_fade)
                cmd += ["-af", f"afade=t=out:st={start:.3f}:d={music_fade:.3f}"]
        cmd.append(path)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        x = np.clip(frame, 0.0, 1.0) * 255.0
        if self.dither:
            x += self._rng.uniform(-0.5, 0.5, size=x.shape[:2])[..., None].astype(np.float32)
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.clip(x + 0.5, 0, 255).astype(np.uint8).tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
