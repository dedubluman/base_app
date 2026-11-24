#!/usr/bin/env python3
"""Diagnostic sheets for the fish matte: source (brightened) with matte contour + tinted alpha, at ~100 %.

    python fish_diag.py --input inputs/fish.mov --times 0.5 2 4 6 --matte-dir masks/fish__mattes/fish --out sheet.jpg
Each tile: top = source with contour(s), bottom = matte alpha as a heat tint on the source.
Optional --compare DIR2 draws a second matte contour (green) next to the first (red)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from fish_refine import decode_source

FPS = 30000 / 1001


def tile(src, t, dirs, h=380, pad=60, bright=2.0):
    im = np.clip(np.asarray(src[t])[..., ::-1].astype(np.float32) * bright, 0, 255).astype(np.uint8)
    ms = [cv2.imread(str(Path(d) / f"{t:05d}.png"), 0) for d in dirs]
    u = np.zeros_like(ms[0])
    for m in ms:
        u = np.maximum(u, m)
    ys, xs = np.nonzero(u > 60)
    H, W = u.shape
    if not len(ys):
        y0, y1, x0, x1 = 0, 400, 0, 300
    else:
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    c = im[y0:y1, x0:x1].copy()
    s = h / c.shape[0]
    c = cv2.resize(c, (max(int(c.shape[1] * s), 50), h), interpolation=cv2.INTER_CUBIC)
    row2 = c.copy()
    cols = [(0, 0, 255), (0, 255, 0)]
    for k, m in enumerate(ms):
        mm = cv2.resize(m[y0:y1, x0:x1], (c.shape[1], h), interpolation=cv2.INTER_LINEAR)
        cs, _ = cv2.findContours((mm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(c, cs, True, cols[k], 1)
        if k == 0:
            a = (mm.astype(np.float32) / 255)[..., None]
            row2 = (row2 * (1 - 0.55 * a) + 0.55 * a * np.array([40, 160, 255], np.float32)).astype(np.uint8)
    cv2.putText(c, f"f{t} {t / FPS:.2f}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    w = 330
    out = np.vstack([c, row2])
    return cv2.copyMakeBorder(out, 0, 0, 0, max(0, w - out.shape[1]), cv2.BORDER_CONSTANT)[:, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--times", nargs="+", type=float, required=True)
    ap.add_argument("--matte-dir", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-row", type=int, default=6)
    a = ap.parse_args()
    src = decode_source(a.input)
    dirs = [a.matte_dir] + ([a.compare] if a.compare else [])
    tiles = [tile(src, min(int(round(t * FPS)), len(src) - 1), dirs) for t in a.times]
    rows = [np.hstack(tiles[i:i + a.per_row] + [np.zeros_like(tiles[0])] * (a.per_row - len(tiles[i:i + a.per_row])))
            for i in range(0, len(tiles), a.per_row)]
    cv2.imwrite(str(a.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
