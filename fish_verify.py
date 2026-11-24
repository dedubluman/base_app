#!/usr/bin/env python3
"""Verification strip: original | matte (old = green, new = red contour + alpha tint) | graded (from the ENCODED file).

    python fish_verify.py --input inputs/fish.mov --video output/video_1_final.mp4 --canvas-crop 608:1080:656:0 \
        --old-matte DIR --new-matte DIR --times 0.5 2 4 6 6.5 9 12.5 14 --out strip.jpg
Also: --stability prints temporal-stability numbers of the two mattes per time window."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from fish_refine import decode_source

FPS = 30000 / 1001


def graded_frame(video, t, crop):
    vf = f"select=eq(n\\,{t}),scale=in_range=tv:out_range=pc,format=rgb24"
    cw, ch, cx, cy = [int(x) for x in crop.split(":")] if crop else (None,) * 4
    pv = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                  "-of", "csv=p=0", str(video)]).decode().strip().split(",")
    W, H = int(pv[0]), int(pv[1])
    b = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf, "-frames:v", "1", "-f", "rawvideo", "-"])
    im = np.frombuffer(b, np.uint8).reshape(H, W, 3)[..., ::-1]
    if crop:
        im = im[cy:cy + ch, cx:cx + cw]
        im = cv2.resize(im, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(im)


def col(src, gr, t, old, new, h=420, pad=70):
    mo = cv2.imread(str(Path(old) / f"{t:05d}.png"), 0)
    mn = cv2.imread(str(Path(new) / f"{t:05d}.png"), 0)
    u = np.maximum(mo, mn)
    ys, xs = np.nonzero(u > 60)
    H, W = u.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    s = h / (y1 - y0)
    w = int((x1 - x0) * s)
    rs = lambda a: cv2.resize(a[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_CUBIC)
    a = np.clip(np.asarray(src[t])[..., ::-1].astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
    orig = rs(a)
    ov = orig.copy()
    mnr = cv2.resize(mn[y0:y1, x0:x1], (w, h))
    mor = cv2.resize(mo[y0:y1, x0:x1], (w, h))
    al = (mnr.astype(np.float32) / 255)[..., None]
    ov = (ov * (1 - 0.5 * al) + 0.5 * al * np.array([40, 150, 255], np.float32)).astype(np.uint8)
    for m, c in ((mor, (0, 255, 0)), (mnr, (0, 0, 255))):
        cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(ov, cs, True, c, 1)
    g = rs(gr)
    for im, tag in ((orig, f"{t / FPS:.1f}s f{t}"), (ov, "mask old(g) new(r)"), (g, "graded")):
        cv2.putText(im, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    out = np.vstack([orig, ov, g])
    return cv2.copyMakeBorder(out, 0, 0, 0, max(0, 300 - out.shape[1]), cv2.BORDER_CONSTANT)[:, :300]


def stability(old, new, n=429):
    wins = [(0, 1.0), (1.5, 4.5), (5.0, 7.5), (5.5, 7.0), (8.0, 11.0), (11.5, 14.0)]
    rd = lambda d, t: cv2.imread(str(Path(d) / f"{t:05d}.png"), 0).astype(np.float32) / 255
    res = {}
    for name, d in (("old", old), ("new", new)):
        A = np.array([rd(d, t).sum() for t in range(n)])
        B = [rd(d, t) > 0.5 for t in range(n)]
        iou = np.array([(B[t] & B[t + 1]).sum() / max((B[t] | B[t + 1]).sum(), 1) for t in range(n - 1)])
        d2 = np.abs(A[2:] - 2 * A[1:-1] + A[:-2]) / np.maximum(A[1:-1], 1)
        res[name] = (iou, d2)
    print("window        | consecutive-frame IoU (mean / p5)     | area 2nd-diff % (mean / p95)")
    for a, b in wins:
        s = slice(int(a * FPS), min(int(b * FPS), n - 3))
        line = f"{a:4.1f}-{b:4.1f}s    "
        for name in ("old", "new"):
            iou, d2 = res[name]
            line += f"| {name}: {iou[s].mean():.3f} / {np.percentile(iou[s], 5):.3f}  {d2[s].mean() * 100:.2f} / {np.percentile(d2[s], 95) * 100:.2f}  "
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--canvas-crop", default=None)
    ap.add_argument("--old-matte", type=Path, required=True)
    ap.add_argument("--new-matte", type=Path, required=True)
    ap.add_argument("--times", nargs="*", type=float, default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stability", action="store_true")
    a = ap.parse_args()
    if a.stability:
        stability(a.old_matte, a.new_matte)
    if a.times:
        src = decode_source(a.input)
        cols = []
        for tt in a.times:
            t = min(int(round(tt * FPS)), len(src) - 1)
            cols.append(col(src, graded_frame(a.video, t, a.canvas_crop), t, a.old_matte, a.new_matte))
        cv2.imwrite(str(a.out), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", a.out)


if __name__ == "__main__":
    main()
