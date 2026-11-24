#!/usr/bin/env python3
"""Side-by-side comparison frames: original | grade A | grade B ...

    python deepocean_compare.py --original clip.mov \
        --graded output/prev.mp4 output/new.mp4 --labels "previous grade" "skin-fixed" \
        --times 2 6 12 20 --out output/compare

Writes one full-frame sheet per timestamp (``<out>_<t>s.jpg``) and, with
``--zoom-masks DIR`` (SAM keyframe masks), a second sheet cropped tightly
around the diver so skin detail can be judged (``<out>_<t>s_zoom.jpg``).
Every clip is decoded at the same size, so the panels line up exactly.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def grab(path: Path, t: float, w: int, h: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", f"scale={w}:{h}:flags=lanczos", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    if len(raw) != w * h * 3:
        raise SystemExit(f"could not decode a frame at {t}s from {path}")
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def diver_box(mask_dir: Path, t: float, fps: float, step: int, w: int, h: int, pad=0.18):
    k = min(int(round(t * fps / step)), len(list((mask_dir / "subject").glob("*.png"))) - 1)
    m = cv2.imread(str(mask_dir / "subject" / f"{k:05d}.png"), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(cv2.resize(m, (w, h)) > 127)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad)) + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    side = max(min(side, min(w, h)), 240)
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return x0, y0, side, side


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--graded", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--times", type=float, nargs="+", default=[2, 6, 12, 20])
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--panel-width", type=int, default=560)
    ap.add_argument("--zoom-masks", type=Path, default=None)
    ap.add_argument("--sam-step", type=int, default=8)
    a = ap.parse_args()
    labels = ["original"] + (a.labels or [p.stem for p in a.graded])
    if len(labels) != len(a.graded) + 1:
        ap.error("--labels must have one entry per --graded file")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,avg_frame_rate", "-of", "csv=p=0", str(a.original)],
                           check=True, capture_output=True, text=True).stdout.strip().split(",")
    ow, oh = int(probe[0]), int(probe[1])
    n, d = probe[2].split("/")
    fps = float(n) / float(d)
    W = 1234 if abs(ow / oh - 1234 / 1080) < 0.01 else ow     # grades are 1080-short-edge
    H = 1080 if W == 1234 else oh
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for t in a.times:
        frames = [grab(a.original, t, W, H)] + [grab(g, t, W, H) for g in a.graded]
        pw = a.panel_width
        ph = int(round(pw * H / W))
        sheet = np.hstack([label(cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA), l)
                           for f, l in zip(frames, labels)])
        cv2.imwrite(f"{a.out}_{t:g}s.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.zoom_masks:
            x0, y0, sw, sh = diver_box(a.zoom_masks, t, fps, a.sam_step, W, H)
            zp = 520
            crops = [cv2.resize(f[y0:y0 + sh, x0:x0 + sw], (zp, zp), interpolation=cv2.INTER_CUBIC)
                     for f in frames]
            cv2.imwrite(f"{a.out}_{t:g}s_zoom.jpg",
                        np.hstack([label(c, l) for c, l in zip(crops, labels)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {a.out}_{t:g}s.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
