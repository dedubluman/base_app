"""Speed-ramp retiming + optical-flow frame synthesis (RIFE flow, Lanczos re-render)."""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np


# ------------------------------------------------------------------ speed map --

def _ease(u):  # smootherstep: C2-continuous, zero slope at both ends
    return u * u * u * (u * (6 * u - 15) + 10)


def _ramp(n: int, speed_from: float, speed_to: float) -> np.ndarray:
    """Per-output-frame source step for an n-frame ramp whose total source advance is
    an exact integer, so the retimed clip lands back on an original frame."""
    u = (np.arange(n) + 0.5) / n
    base = speed_from + (speed_to - speed_from) * _ease(u)
    w = np.sin(np.pi * u)                       # bump that vanishes at both ends
    target = round(base.sum())
    c = (target - base.sum()) / w.sum()
    return base + c * w


@dataclass
class RampSpec:
    ramp_in_start: int = 20      # source frame where the deceleration begins (normal speed before)
    ramp_in_frames: int = 14     # output frames spent decelerating
    speed: float = 0.5           # hold speed
    ramp_out_end: int = 104      # source frame by which normal speed is regained (approx.)
    ramp_out_frames: int = 16
    n_src: int = 141


def build_speed_map(spec: RampSpec) -> np.ndarray:
    """Source position (float, in source frames) of every output frame."""
    s_in = _ramp(spec.ramp_in_frames, 1.0, spec.speed)
    s_out = _ramp(spec.ramp_out_frames, spec.speed, 1.0)
    m_in, m_out = round(s_in.sum()), round(s_out.sum())
    hold_src = spec.ramp_out_end - spec.ramp_in_start - m_in - m_out
    n_hold = int(round(hold_src / spec.speed / 2)) * 2        # even -> integer source advance
    steps = np.concatenate([np.ones(spec.ramp_in_start), s_in, np.full(n_hold, spec.speed), s_out])
    pos = np.concatenate([[0.0], np.cumsum(steps)])
    last = spec.n_src - 1
    tail_start = pos[-1]
    assert abs(tail_start - round(tail_start)) < 1e-6, tail_start
    tail = np.arange(tail_start + 1, last + 1)
    pos = np.concatenate([pos, tail])
    pos = pos[pos <= last + 1e-9]
    return pos


# ------------------------------------------------------------------ source IO --

def probe_video(path: Path) -> dict:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)])
    return json.loads(out)["streams"][0]


def decode_frames(path: Path, w: int, h: int) -> np.ndarray:
    """All frames as uint16 RGB (N,H,W,3); BT.709 limited-range YUV -> full-range RGB."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-fps_mode", "passthrough",
           "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, np.uint16).reshape(-1, h, w, 3)


class Encoder:
    """Streams rgb48 frames into an H.264 / BT.709 encode (video only)."""

    def __init__(self, out: Path, w: int, h: int, fps: str, crf: int = 12, preset: str = "slow"):
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
             "-r", fps, "-i", "-",
             "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def write(self, f16: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(f16).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def to_f32(u16):
    return u16.astype(np.float32) * (1.0 / 65535.0)


def to_u16(f):
    return np.clip(f * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
