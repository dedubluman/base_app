#!/usr/bin/env python
"""Full-frame 1920x1080 H.264 export at the clip's natural speed (no bars, no retime).

    python uw_export.py --graded graded.mkv --masks masks/merged --out out.mp4

The source is nearly square (1232x1078), so filling 16:9 means cropping the
height.  Instead of a static centre crop, the 16:9 window follows the swimmer:
its top edge tracks the top of the SAM 3.1 subject mask (the head), smoothed
over time.  Frames without a subject (sun-only start, water-only end) drift
back to the centred window.  The crop is scaled to 1920x1080 with Lanczos.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe(path):
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)]))["streams"][0]
    num, den = map(int, j["r_frame_rate"].split("/"))
    return j["width"], j["height"], num, den


def crop_track(mdir: Path, n: int, h: int, w: int, ch: int, head_margin: float, sigma: float):
    """Top edge of the crop window per frame."""
    centre = (h - ch) / 2.0
    top = np.full(n, np.nan)
    for i in range(n):
        m = cv2.imread(str(mdir / "subject" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None or (m > 127).sum() < 20000:
            continue
        ys = np.nonzero((m > 127).any(1))[0]
        top[i] = ys[int(len(ys) * 0.02)]          # ~top of the head
    want = np.where(np.isnan(top), centre, np.clip(top - head_margin * ch, 0, h - ch))
    # smooth with edge padding so start/end do not snap
    k = int(sigma * 4) | 1
    ker = cv2.getGaussianKernel(k, sigma).ravel()
    pad = np.pad(want, k, mode="edge")
    sm = np.convolve(pad, ker, mode="same")[k:-k]
    return np.clip(sm, 0, h - ch)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graded", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--head-margin", type=float, default=0.14,
                    help="space kept above the head, fraction of the crop height")
    ap.add_argument("--sigma", type=float, default=7.0, help="temporal smoothing (frames)")
    ap.add_argument("--crf", type=int, default=17)
    a = ap.parse_args()

    w, h, num, den = probe(a.graded)
    ch = int(round(w * a.height / a.width))            # crop height at full source width
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.graded), "-f", "rawvideo",
                           "-pix_fmt", "rgb48le", "-"], stdout=subprocess.PIPE)
    sz = w * h * 6
    frames = []
    while True:
        b = pr.stdout.read(sz)
        if len(b) < sz:
            break
        frames.append(np.frombuffer(b, "<u2").reshape(h, w, 3))
    n = len(frames)
    top = crop_track(a.masks, n, h, w, ch, a.head_margin, a.sigma)
    print(f"[export] {n} frames {w}x{h} -> crop {w}x{ch} -> {a.width}x{a.height}; "
          f"crop top {top.min():.0f}..{top.max():.0f}px")

    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
         "-s", f"{a.width}x{a.height}", "-framerate", f"{num}/{den}", "-i", "-",
         "-vf", "scale=in_range=full:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-profile:v", "high",
         "-level", "4.2", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart",
         str(a.out)], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        # sub-pixel vertical shift via affine warp keeps the motion of the crop smooth
        y0 = float(top[i])
        M = np.float32([[a.width / w, 0, 0], [0, a.height / ch, -y0 * a.height / ch]])
        out = cv2.warpAffine(f, M, (a.width, a.height), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REPLICATE)
        enc.stdin.write(np.ascontiguousarray(out).astype("<u2").tobytes())
    enc.stdin.close()
    enc.wait()
    np.savetxt(str(a.out.with_suffix(".croptrack.txt")), top, fmt="%.1f")
    print("[export] done ->", a.out)


if __name__ == "__main__":
    main()
