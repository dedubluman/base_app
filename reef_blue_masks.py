#!/usr/bin/env python3
"""Soft per-frame masks for the small blue fish (SAM 3.1 multi-instance tracking).

SAM 3.1 key-frame masks (every `stride`-th frame, all instances united) are flow-warped to every
frame, grown by a few pixels (fish swim several body-lengths between key frames), then gated by the
frame's own blue chroma so the mask snaps to the real fish pixels on every frame. A final Gaussian
makes the edge soft. Output: uint8 (N, H/2, W/2).
"""
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from reef_grade import srgb2lin, to_lab, sstep
from reef_masks_interp import decode_small

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, nargs="+", required=True, help="one or more SAM 3.1 run dirs (united)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--grow", type=int, default=9)
    a = ap.parse_args()
    W, H = 540, 960
    nkey = max(json.loads((d / "meta.json").read_text())["num_frames"] for d in a.masks)
    fr = decode_small(a.input, W, H); N = len(fr)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr])
    key = np.zeros((nkey, H, W), np.float32)
    for d in [o for r in a.masks for o in sorted(r.glob("obj_*"))]:
        for p in d.glob("*.png"):
            i = int(p.stem)
            if i < nkey:
                key[i] = np.maximum(key[i], cv2.resize(cv2.imread(str(p), 0), (W, H), interpolation=cv2.INTER_AREA) / 255.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def warp(m, s, d):
        fl = cv2.calcOpticalFlowFarneback(d, s, None, 0.5, 4, 25, 5, 7, 1.5, 0)
        return cv2.remap(m, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR)
    region = np.zeros((N, H, W), np.float32)
    for n in range(N):
        i0 = n // a.stride; i1 = min(i0 + 1, nkey - 1); t = (n - i0 * a.stride) / a.stride
        k0, k1 = i0 * a.stride, min(i1 * a.stride, N - 1)
        m0 = key[i0] if n == k0 else warp(key[i0], gray[k0], gray[n])
        r = m0 if (t == 0 or i1 == i0) else np.maximum((1 - t) * m0, t * warp(key[i1], gray[k1], gray[n]))
        region[n] = r
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1, 2 * a.grow + 1))
    out = np.zeros((N, H, W), np.uint8)
    for n in range(N):
        reg = cv2.dilate((region[n] > 0.15).astype(np.uint8), k).astype(np.float32)
        lab = to_lab(srgb2lin(fr[n].astype(np.float32) / 255))
        C = np.hypot(lab[..., 1], lab[..., 2]); h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        blue = sstep(0.035, 0.085, C) * np.exp(-0.5 * (((h - 255 + 180) % 360 - 180) / 38) ** 2)
        m = np.maximum(blue * reg, 0.6 * region[n].clip(0, 1) * (blue > 0.15))
        out[n] = (np.clip(cv2.GaussianBlur(m, (0, 0), 1.2), 0, 1) * 255).astype(np.uint8)
    out = gaussian_filter1d(out.astype(np.float32), 0.7, axis=0).astype(np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, masks=out)
    cov = (out > 60).reshape(N, -1).sum(1)
    print(f"[blue] {N} frames, mask px (@{W}x{H}) median {int(np.median(cov[cov>0])) if (cov>0).any() else 0}, frames with fish {(cov>0).sum()}")

if __name__ == "__main__":
    main()
