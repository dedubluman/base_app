#!/usr/bin/env python3
"""Repair isolated bad frames of a SAM 3.1 part mask by flow-chained warping of the nearest good masks.

    python repair_masks.py --part shorts --spans 168:171 --spans 436:436

For every bad frame t in a span [a,b] the mask of the last good frame before (a-1) is carried forward frame by frame with
dense optical flow (Farneback on the source frames), the mask of the first good frame after (b+1) is carried backward the
same way, and the two soft results are blended by temporal distance (a span at the clip end uses the one-sided carry only).
Originals are kept in masks/<stem>__<part>/subject_before_repair/."""
import argparse, shutil
from pathlib import Path
import cv2, numpy as np
from shades.io import probe, iter_frames

SRC = Path("inputs/player_export.mov")


def warp(img, fl):
    h, w = img.shape[:2]; gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True); ap.add_argument("--spans", action="append", required=True)
    ap.add_argument("--mode", choices=["both", "fwd", "bwd"], default="both", help="fwd: carry only from the frame before the span; bwd: only from the frame after")
    a = ap.parse_args()
    d = Path(f"masks/player_export__{a.part}/subject"); keep = d.parent / "subject_before_repair"; keep.mkdir(exist_ok=True)
    spans = [tuple(int(x) for x in s.split(":")) for s in a.spans]
    need = set()
    for s, e in spans: need |= set(range(max(0, s - 1), min(436, e + 1) + 1))
    g = {}
    for i, f in enumerate(iter_frames(SRC, probe(SRC))):
        if i in need: g[i] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if i > max(need): break
    rd = lambda i: (cv2.imread(str(d / f"{i:05d}.png"), 0) > 127).astype(np.float32)
    for s, e in spans:
        for t in range(s, e + 1):
            shutil.copy(d / f"{t:05d}.png", keep / f"{t:05d}.png") if (d / f"{t:05d}.png").exists() else None
        fw = rd(s - 1); fwd = {}
        for t in (range(s, e + 1) if a.mode != "bwd" else []):                      # carry forward: flow from t to t-1 samples the previous mask
            fl = cv2.calcOpticalFlowFarneback(g[t], g[t - 1], None, 0.5, 4, 25, 4, 7, 1.5, 0); fw = warp(fw, fl); fwd[t] = fw
        bwd = {}
        if a.mode != "fwd" and e + 1 <= 436 and (d / f"{e + 1:05d}.png").exists():
            bw = rd(e + 1)
            for t in range(e, s - 1, -1):
                fl = cv2.calcOpticalFlowFarneback(g[t], g[t + 1], None, 0.5, 4, 25, 4, 7, 1.5, 0); bw = warp(bw, fl); bwd[t] = bw
        for t in range(s, e + 1):
            if bwd and fwd:
                w = (t - s + 1) / (e - s + 2); m = (1 - w) * fwd[t] + w * bwd[t]      # nearer the earlier good frame -> more forward carry
            else:
                m = bwd[t] if bwd else fwd[t]
            cv2.imwrite(str(d / f"{t:05d}.png"), ((m > 0.5).astype(np.uint8)) * 255)
            print(a.part, "repaired frame", t, "area", int((m > .5).sum()))


if __name__ == "__main__":
    main()
