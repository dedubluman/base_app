#!/usr/bin/env python3
"""Before/after contact sheets cropped to the diver: [source | previous grade | refined grade] per frame.
    python compare_sheet.py --frames 5,30,60,90 --refined DIR_OR_VIDEO --out sheet.jpg [--tile 560]
--refined: a directory of r_{frame:05d}.png stills or a video file. The previous render has its pillarbox bars cropped
(columns 343-1577 of the 1920x1080 file) and resized back to the source frame size for the side-by-side."""
import argparse
from pathlib import Path
import cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True); ap.add_argument("--refined", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--prev", default="output/UNDERWATER_CINEMATIC_FINAL_1080P.mp4"); ap.add_argument("--src", default="inputs/player_export.mov")
ap.add_argument("--tile", type=int, default=560); ap.add_argument("--pad", type=int, default=40)
ap.add_argument("--stills-prefix", default="r"); ap.add_argument("--wide", action="store_true", help="full frame instead of diver crop")
a = ap.parse_args()


def vid(path, i, prev=False):
    c = cv2.VideoCapture(path); c.set(1, i); ok, f = c.read(); c.release()
    if prev: f = cv2.resize(f[:, 343:1577], (1232, 1078), interpolation=cv2.INTER_AREA)
    return f


rows = []
for i in [int(x) for x in a.frames.split(",")]:
    src = vid(a.src, i); prev = vid(a.prev, i, True)
    rp = Path(a.refined)
    new = cv2.imread(str(rp / f"{a.stills_prefix}_{i:05d}.png")) if rp.is_dir() else vid(a.refined, i)
    dv = cv2.imread(f"masks_refined/player_export/{i:05d}.png", 0) > 127; ys, xs = np.nonzero(dv)
    if a.wide or not len(ys): x0, y0, x1, y1 = 0, 0, 1232, 1078
    else: x0, x1, y0, y1 = max(0, xs.min() - a.pad), min(1232, xs.max() + a.pad), max(0, ys.min() - a.pad), min(1078, ys.max() + a.pad)
    cw, ch = x1 - x0, y1 - y0; s = max(cw / 4, ch / 3); cw, ch = int(4 * s), int(3 * s)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2; x0 = int(np.clip(cx - cw // 2, 0, max(0, 1232 - cw))); y0 = int(np.clip(cy - ch // 2, 0, max(0, 1078 - ch)))
    tiles = []
    for name, im in (("source", src), ("previous", prev), ("refined", new)):
        t = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (a.tile, a.tile * 3 // 4), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{name} f{i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2); tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
