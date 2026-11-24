"""Shared helpers for the deep-abyss pipeline (abyss_masks / abyss_propagate /
abyss_grade / abyss_qc): ffprobe, frame IO through ffmpeg, small image ops."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)   # BT.709


# --------------------------------------------------------------------------- io
def probe(path: Path) -> dict:
    """Width/height/frame count/fps/duration of the first video stream (+ audio flag)."""
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) in (90, 270):                         # ffmpeg autorotates; report display size
        w, h = h, w
    avg, nominal = Fraction(v.get("avg_frame_rate", "0/1")), Fraction(v.get("r_frame_rate", "0/1"))
    # prefer the clean nominal rate (30000/1001) over the odd averaged one (96000/3203) when they agree
    fps = nominal if nominal and (not avg or abs(float(nominal) - float(avg)) < 0.05) else (avg or nominal)
    n = int(v["nb_frames"]) if v.get("nb_frames", "N/A").isdigit() else int(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)]).strip())
    return {"w": w, "h": h, "frames": n, "fps": fps, "duration": float(j["format"]["duration"]),
            "audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def read_frames(path: Path, w: int, h: int, scale_to: tuple[int, int] | None = None):
    """Yield float32 RGB frames (0..1, BT.709 full-range display values) decoded with
    ffmpeg's TV-range -> full-range BT.709 conversion at 16 bit.
    ``scale_to=(w,h)`` decodes straight to a smaller size (fast analysis passes)."""
    ow, oh = scale_to or (w, h)
    vf = "scale=in_color_matrix=bt709:in_range=tv"
    if scale_to:
        vf += f":w={ow}:h={oh}:flags=area"
    vf += ",format=rgb48le"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n = ow * oh * 3 * 2
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint16).reshape(oh, ow, 3).astype(np.float32) / 65535
    finally:                                   # also runs when the consumer stops early
        p.stdout.close()
        if p.poll() is None:
            p.kill()
        p.wait()


def open_encoder(out: Path, w: int, h: int, fps: Fraction, audio_src: Path | None, crf: int = 16,
                 preset: str = "slow") -> subprocess.Popen:
    """H.264 (libx264) / yuv420p / BT.709 / faststart encoder fed with rgb48le frames on stdin.
    Audio (if the source has it) is stream-copied. aq-mode=3 biases bits toward dark flat areas,
    which is where a deep-abyss grade would otherwise show banding."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
           "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-"]
    if audio_src is not None:
        cmd += ["-i", str(audio_src)]
    cmd += ["-map", "0:v"]
    if audio_src is not None:
        cmd += ["-map", "1:a?", "-c:a", "copy"]
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
            "-x264-params", "aq-mode=3:aq-strength=1.0:deblock=-1,-1",
            "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ------------------------------------------------------------------ image maths
def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def gblur(x: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), s)


def lowpass(x: np.ndarray, s: float, f: int = 4) -> np.ndarray:
    """Gaussian blur of sigma ``s`` px computed at 1/f resolution (use for s >= ~10 px): ~f^2 faster
    and indistinguishable for smooth fields (tone base, chroma, light beams)."""
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), s / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter with a single-channel guide I (edge-aware smoothing of p)."""
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.blur(I, k), cv2.blur(p, k)
    a = (cv2.blur(I * p, k) - mI * mp) / (cv2.blur(I * I, k) - mI * mI + eps)
    b = mp - a * mI
    return cv2.blur(a, k) * I + cv2.blur(b, k)


def hexrgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def gradient_map(t: np.ndarray, stops: list[tuple[float, str]]) -> np.ndarray:
    """HxW in 0..1 -> HxWx3 by piecewise-linear interpolation of (position, hex) stops."""
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.stack([hexrgb(s[1]) for s in stops])
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, xs, cs[:, c]) for c in range(3)], -1).astype(np.float32)


def srgb_to_lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_to_lab(c: np.ndarray) -> np.ndarray:
    l = srgb_to_lin(c)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]], np.float32)
    xyz = (l @ m.T) / np.array([.95047, 1, 1.08883], np.float32)
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
