"""Measurements used both while tuning and in the final verification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from .config import Swatch
from .grade import LUMA, _rgb_to_hsv, hex_to_rgb, rgb_to_hex


def _lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(np.clip(rgb, 0, 1).reshape(1, -1, 3)).reshape(-1, 3)


def measure_palette(graded: list[np.ndarray], palette: list[Swatch]) -> dict:
    """Find each anchor's region in the graded frames and measure ΔE2000.

    shadow    : pixels between the 2nd and 12th luminance percentile
    mid       : warm-hued (10-60 deg), mid-luminance, reasonably saturated
                pixels (sand, rock, skin), the most saturated 40% of them
    highlight : the brightest 0.7% of pixels
    """
    px = np.concatenate([f[::2, ::2].reshape(-1, 3) for f in graded])
    lum = px @ LUMA
    h, s, v = _rgb_to_hsv(px)
    regions = {}
    lo, hi = np.percentile(lum, [2, 12])
    regions["shadow"] = px[(lum >= lo) & (lum <= hi)]
    warm = ((h * 360 > 10) & (h * 360 < 60) & (lum > 0.25) & (lum < 0.75)
            & (s > 0.12))
    cand = px[warm]
    if len(cand) > 200:
        cs = s[warm]
        regions["mid"] = cand[cs >= np.percentile(cs, 60)]
    else:
        regions["mid"] = cand
    regions["highlight"] = px[lum >= np.percentile(lum, 99.3)]

    out = {}
    for sw in palette:
        reg = regions.get(sw.role, np.empty((0, 3)))
        if len(reg) < 50:
            out[sw.role] = {"name": sw.name, "hex": sw.hex, "measured": None,
                            "delta_e": None, "pixels": int(len(reg))}
            continue
        mean = reg.mean(0)
        de = float(deltaE_ciede2000(_lab(hex_to_rgb(sw.hex)[None]),
                                    _lab(mean[None]))[0])
        out[sw.role] = {"name": sw.name, "hex": sw.hex,
                        "measured": rgb_to_hex(mean), "delta_e": round(de, 2),
                        "pixels": int(len(reg))}
    return out


def palette_strip(palette_result: dict, path: Path, w: int = 900) -> None:
    """Sidecar swatch image: top half = anchor, bottom half = measured."""
    cells = []
    for r in palette_result.values():
        a = hex_to_rgb(r["hex"])
        m = hex_to_rgb(r["measured"]) if r["measured"] else a * 0
        cell = np.zeros((240, w // 3, 3), np.float32)
        cell[:150], cell[150:] = a, m
        cells.append(cell)
    img = (np.hstack(cells) * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ffprobe_summary(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_packets,"
         "pix_fmt,color_primaries,color_transfer,color_space:format=duration",
         "-of", "json", str(path)], check=True, capture_output=True,
        text=True).stdout
    return json.loads(out)


def contact_sheet(video: Path, times: list[float], path: Path,
                  thumb_w: int = 270) -> None:
    dur = float(ffprobe_summary(video)["format"]["duration"])
    thumbs = []
    for t in [t for t in times if t < dur - 0.05]:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"scale={thumb_w}:-2", "-f", "image2pipe",
             "-vcodec", "png", "-"], check=True, capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        cv2.putText(img, f"{t:.1f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)  # sheet label, not video
        thumbs.append(img)
    cv2.imwrite(str(path), np.hstack(thumbs), [cv2.IMWRITE_JPEG_QUALITY, 88])


def raw_frame_diff(src: Path, out: Path, t: float) -> float:
    """Mean absolute difference (0..255 scale) of one frame, source vs output.
    Used to prove the raw section is untouched apart from encode noise."""
    def grab(p):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(p),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).astype(np.float32)
    a, b = grab(src), grab(out)
    return float(np.abs(a - b).mean())
