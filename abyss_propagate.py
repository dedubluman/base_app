#!/usr/bin/env python3
"""Carry the SAM 3.1 keyframe masks (abyss_masks.py) to every frame of the clip.

For a frame t between keyframes a <= t <= b the mask is the blend of
    mask_a warped to t   and   mask_b warped to t
with weights (b-t)/(b-a) and (t-a)/(b-a); the warps follow DIS optical flow estimated
between the actual frames (t -> a, t -> b), so masks track camera motion and swimming fish
instead of cross-fading. The coarse masks are then smoothed over time (5-tap triangular)
and snapped to the real edges with a luma-guided filter (so thin fins / diver straps keep
partial coverage and no halo of the old background colour is carried into the grade).

Output: masks/<stem>__abyss/<job>_full/<frame:05d>.png   uint8 soft alpha, full source res

    python abyss_propagate.py --input inputs/dive.mov [--jobs diver,fish]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from abyss_common import gblur, guided, luma, probe, read_frames

REPO = Path(__file__).resolve().parent
FLOW_W, FLOW_H = 540, 960          # flow is estimated at half resolution (px / 2 on the full frame)


def load_keyframes(d: Path) -> dict[int, np.ndarray]:
    return {int(p.stem[3:]): cv2.imread(str(p), 0) for p in sorted(d.glob("kf_*.png"))}


def warp(mask: np.ndarray, flow_full: np.ndarray, grid: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """mask_t(x) = mask_src(x + flow(x)) with flow = flow(t -> src)."""
    return cv2.remap(mask, grid[0] + flow_full[..., 0], grid[1] + flow_full[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--jobs", default="diver,fish")
    ap.add_argument("--masks-dir", type=Path)
    ap.add_argument("--smooth", type=int, default=2, help="temporal half-window (frames)")
    ap.add_argument("--guide-radius", type=int, default=4)
    ap.add_argument("--guide-eps", type=float, default=2e-4)
    a = ap.parse_args()

    info = probe(a.input)
    w, h, n = info["w"], info["h"], info["frames"]
    root = a.masks_dir or REPO / "masks" / f"{a.input.stem}__abyss"
    jobs = [j for j in a.jobs.split(",") if (root / j).exists()]
    kfs = {j: load_keyframes(root / j) for j in jobs}
    for j in jobs:
        ks = sorted(kfs[j])
        print(f"[prop] {j}: {len(ks)} keyframes ({ks[0]}..{ks[-1]}), "
              f"{sum(1 for k in ks if kfs[j][k].any())} with a subject", flush=True)

    t0 = time.time()
    gray = np.stack([cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                     for f in read_frames(a.input, w, h, scale_to=(FLOW_W, FLOW_H))])
    assert len(gray) == n, (len(gray), n)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = w / FLOW_W, h / FLOW_H

    def flow_to(t: int, k: int) -> np.ndarray:
        if t == k:
            return np.zeros((h, w, 2), np.float32)
        f = dis.calc(gray[t], gray[k], None)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        f[..., 0] *= sx
        f[..., 1] *= sy
        return f

    # ---- stage 1: coarse masks (flow-warped keyframe blend) ------------------------------
    for j in jobs:
        shutil.rmtree(root / f"{j}_coarse", ignore_errors=True)
        (root / f"{j}_coarse").mkdir()
    ks = {j: sorted(kfs[j]) for j in jobs}
    for t in range(n):
        need = {}
        for j in jobs:
            lo = max((k for k in ks[j] if k <= t), default=None)
            hi = min((k for k in ks[j] if k >= t), default=None)
            need[j] = (lo if lo is not None else hi, hi if hi is not None else lo)
        flows = {k: flow_to(t, k) for k in {k for pair in need.values() for k in pair}}
        for j in jobs:
            lo, hi = need[j]
            ma = warp(kfs[j][lo].astype(np.float32) / 255, flows[lo], grid)
            if hi == lo:
                m = ma
            else:
                mb = warp(kfs[j][hi].astype(np.float32) / 255, flows[hi], grid)
                wb = (t - lo) / (hi - lo)
                m = (1 - wb) * ma + wb * mb
            cv2.imwrite(str(root / f"{j}_coarse" / f"{t:05d}.png"), (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
        if t % 60 == 0:
            print(f"[prop] coarse {t}/{n}  {time.time() - t0:.0f}s", flush=True)

    # ---- stage 2: temporal smoothing + guided-filter edge refinement ---------------------------
    tw = np.array([a.smooth + 1 - abs(k) for k in range(-a.smooth, a.smooth + 1)], np.float32)
    tw /= tw.sum()
    for j in jobs:
        shutil.rmtree(root / f"{j}_full", ignore_errors=True)
        (root / f"{j}_full").mkdir()
    for t, f in enumerate(read_frames(a.input, w, h)):
        L = luma(f)
        for j in jobs:
            acc = np.zeros((h, w), np.float32)
            for k, wk in zip(range(-a.smooth, a.smooth + 1), tw):
                tt = min(max(t + k, 0), n - 1)
                acc += wk * cv2.imread(str(root / f"{j}_coarse" / f"{tt:05d}.png"), 0).astype(np.float32) / 255
            out = np.zeros((h, w), np.float32)
            ys, xs = np.nonzero(acc > 0.02)
            if len(ys):
                y0, y1 = max(ys.min() - 24, 0), min(ys.max() + 25, h)
                x0, x1 = max(xs.min() - 24, 0), min(xs.max() + 25, w)
                sub = acc[y0:y1, x0:x1]
                g = np.clip(guided(L[y0:y1, x0:x1], sub, a.guide_radius, a.guide_eps), 0, 1)
                support = cv2.dilate((sub > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
                g = gblur(g * support, 0.7)
                # pull partial coverage a little toward 0/1 so edges are crisp, not milky
                out[y0:y1, x0:x1] = np.clip((g - 0.08) / 0.84, 0, 1)
            cv2.imwrite(str(root / f"{j}_full" / f"{t:05d}.png"), (out * 255 + 0.5).astype(np.uint8))
        if t % 60 == 0:
            print(f"[prop] refined {t}/{n}  {time.time() - t0:.0f}s", flush=True)
    for j in jobs:
        shutil.rmtree(root / f"{j}_coarse", ignore_errors=True)
    print(f"[prop] done in {time.time() - t0:.0f}s -> {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
