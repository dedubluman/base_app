#!/usr/bin/env python3
"""Turn SAM 3.1 key-frame masks (every `stride`-th frame) into one soft mask per source frame.

Stages
  1. union of every SAM object for each key frame
  2. optical-flow (Farneback) warp of both neighbouring key masks onto frame n, blended by time
  3. temporal Gaussian smoothing (no flicker / no popping)
  4. guided-filter edge refinement against the frame itself (follows thin fins / tail edges)
Output: <out>.npz with `masks` uint8 (N, H/2, W/2), 0..255
"""
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np

def decode_small(path, w, h):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-vf",
                        f"scale={w}:{h}:flags=area:in_range=tv:out_range=pc:in_color_matrix=bt709",
                        "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)

def guided(I, p, r, eps):
    """He et al. guided filter, gray guide I (float32 0..1), input p (float32)."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return box(a) * I + box(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True, help="masks/<stem>__cuttle dir from cuttle_masks.py")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-prob", type=float, default=0.0)
    a = ap.parse_args()
    W, H = 540, 960
    meta = json.loads((a.masks / "meta.json").read_text())
    objs = sorted(a.masks.glob("obj_*"))
    nkey = meta["num_frames"]
    frames = decode_small(a.input, W, H)
    N = len(frames)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    key = np.zeros((nkey, H, W), np.float32)
    for d in objs:
        for p in d.glob("*.png"):
            i = int(p.stem)
            if i < nkey:
                key[i] = np.maximum(key[i], cv2.resize(cv2.imread(str(p), 0), (W, H), interpolation=cv2.INTER_AREA) / 255.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def warp(mask, src_gray, dst_gray):
        # flow from dst(n) -> src(key): where each pixel of n sits in the key frame
        fl = cv2.calcOpticalFlowFarneback(dst_gray, src_gray, None, 0.5, 4, 25, 5, 7, 1.5, 0)
        return cv2.remap(mask, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    raw = np.zeros((N, H, W), np.float32)
    for n in range(N):
        i0 = n // a.stride; i1 = min(i0 + 1, nkey - 1)
        t = (n - i0 * a.stride) / a.stride
        k0, k1 = i0 * a.stride, min(i1 * a.stride, N - 1)
        m0 = key[i0] if n == k0 else warp(key[i0], gray[k0], gray[n])
        if t == 0 or i1 == i0:
            raw[n] = m0
        else:
            m1 = warp(key[i1], gray[k1], gray[n])
            # a fish that exists in only one key frame fades with its distance in time
            raw[n] = (1 - t) * m0 + t * m1
    # temporal smoothing (sigma 1.4 frames)
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(raw, 1.4, axis=0, mode="nearest")
    out = np.zeros((N, H, W), np.uint8)
    for n in range(N):
        g = gray[n].astype(np.float32) / 255.0
        m = guided(g, sm[n], r=3, eps=2e-3)
        m = np.clip((m - 0.25) / 0.55, 0, 1)          # tighten: kill the soft skirt, keep fin edges
        m = m * (cv2.GaussianBlur(sm[n], (0, 0), 6) > 0.02)  # never let refinement invent mask far from SAM
        out[n] = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, masks=out)
    cov = (out > 127).reshape(N, -1).sum(1)
    print(f"[interp] {N} frames; fish present (>50 px) in {(cov > 50).sum()} frames; "
          f"median area {int(np.median(cov[cov > 50])) if (cov > 50).any() else 0} px @ {W}x{H}")

if __name__ == "__main__":
    main()
