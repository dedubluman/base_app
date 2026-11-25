#!/usr/bin/env python3
"""Refine raw SAM 3.1 diver masks into temporally stable soft alpha mattes.

    python refine_masks.py --raw masks/player_export__fwd/subject --out masks_refined/player_export

Steps: (1) drop tiny disconnected specks, (2) fill pin-holes / close hairline gaps,
(3) repair frames where SAM lost or truncated the diver (area collapses vs. temporal neighbours)
by flow-warping the nearest good mask, (4) temporal smoothing (flow-compensated 3-tap), (5) a
small physical feather. Output: 8-bit PNG alpha per frame, source resolution."""
import argparse, json
from pathlib import Path
import cv2, numpy as np
from shades.io import probe, iter_frames


def clean(m, min_frac=0.0015):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = np.zeros_like(m)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_frac * m.size:
            out |= lab == k
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    # fill enclosed holes
    inv = (~out).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    for kk in range(1, n):
        x, y, w, h, a = st[kk]
        if a < 0.004 * m.size and x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]:
            out |= lab == kk
    return out


def flow(a_gray, b_gray):
    return cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, 0.5, 4, 25, 4, 7, 1.5, 0)


def warp(img, fl):
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--input", type=Path, default=Path("inputs/player_export.mov"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--feather", type=float, default=1.6)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    info = probe(a.input)
    n = info.n_frames
    raw = []
    for i in range(n):
        p = a.raw / f"{i:05d}.png"
        raw.append(cv2.imread(str(p), 0) > 127 if p.exists() else None)
    h, w = next(m for m in raw if m is not None).shape
    raw = [clean(m) if m is not None else np.zeros((h, w), bool) for m in raw]
    area = np.array([m.sum() for m in raw], float)
    med = np.array([np.median(area[max(0, i - 7):i + 8]) for i in range(n)])
    gray = [cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for f in iter_frames(a.input, info)]
    bad = (area < 0.55 * med) | (area == 0)
    print("suspect frames (lost/truncated):", np.flatnonzero(bad).tolist())
    fixed = [m.astype(np.float32) for m in raw]
    for i in np.flatnonzero(bad):
        cands = [j for j in range(max(0, i - 12), min(n, i + 13)) if not bad[j]]
        if not cands:
            continue
        j = min(cands, key=lambda j: abs(j - i))
        fl = flow(gray[i], gray[j])  # sample frame j at positions displaced from i
        fixed[i] = np.maximum(fixed[i], warp(fixed[j], fl))
    # flow-compensated temporal smoothing
    out = []
    for i in range(n):
        acc, wsum = fixed[i] * 0.5, 0.5
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                acc = acc + warp(fixed[j], flow(gray[i], gray[j])) * 0.25; wsum += 0.25
        out.append(acc / wsum)
    for i, m in enumerate(out):
        m = cv2.GaussianBlur(m, (0, 0), a.feather)
        m = np.clip((m - 0.5) * 1.6 + 0.5, 0, 1)        # keep soft edge but avoid grey halo
        cv2.imwrite(str(a.out / f"{i:05d}.png"), (m * 255 + .5).astype(np.uint8))
    json.dump({"bad": np.flatnonzero(bad).tolist(), "area": area.tolist()}, open(a.out / "meta.json", "w"))
    print("wrote", n, "alphas")


if __name__ == "__main__":
    main()
