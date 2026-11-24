"""Flow-guided temporal interpolation of the sub-sampled SAM 3.1 masks.

SAM 3.1 ran on every Nth frame (uw_masks.py). For every source frame t between
two key frames a < t < b we warp both neighbouring key masks onto frame t with
dense optical flow (Farneback, t -> a and t -> b) and blend them by temporal
distance. That keeps mask edges glued to moving content (a plain cross-fade
of two masks would ghost on fast motion), and the result is temporally smooth.

Output: <masks_dir>/<concept>_full.npy  uint8[T, h, w] (0..255, soft), one mask
per source frame, at the SAM working resolution.

    python uw_interp.py --video clip.mov --masks masks_uw/clip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe


def _flow(gray_t: np.ndarray, gray_k: np.ndarray) -> np.ndarray:
    """Dense flow at half resolution, upsampled: gray_t(x) ~ gray_k(x + flow)."""
    h, w = gray_t.shape
    a = cv2.resize(gray_t, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray_k, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0


def _warp(mask: np.ndarray, flow: np.ndarray, grid) -> np.ndarray:
    gx, gy = grid
    return cv2.remap(mask, gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def interpolate_concept(stack: np.ndarray, src_idx: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """stack uint8[K,h,w] at key frames src_idx; gray float32[T,h,w] of every source frame."""
    t_total, h, w = gray.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = np.zeros((t_total, h, w), np.uint8)
    key_of = {int(s): k for k, s in enumerate(src_idx)}
    keys = np.asarray(src_idx)
    for t in range(t_total):
        if t in key_of:
            out[t] = stack[key_of[t]]
            continue
        kb = int(np.searchsorted(keys, t))          # first key after t
        ka = kb - 1                                 # last key before t
        if kb >= len(keys):                         # tail: only a previous key
            ia, ib = ka, None
        elif ka < 0:
            ia, ib = kb, None
        else:
            ia, ib = ka, kb
        ma = stack[ia].astype(np.float32)
        wa = _warp(ma, _flow(gray[t], gray[keys[ia]]), (gx, gy))
        if ib is None:
            res = wa
        else:
            mb = stack[ib].astype(np.float32)
            wb = _warp(mb, _flow(gray[t], gray[keys[ib]]), (gx, gy))
            alpha = (t - keys[ia]) / float(keys[ib] - keys[ia])
            res = (1 - alpha) * wa + alpha * wb
        out[t] = np.clip(res, 0, 255).astype(np.uint8)
    return out


def merge_override(base_npz: Path, override_npz: Path) -> dict:
    """Replace the key-frame masks of `base` that fall inside the frame range
    covered by `override` (a dedicated, re-prompted SAM run) with its masks."""
    b, o = np.load(base_npz), np.load(override_npz)
    masks = b["masks"].copy()
    lo, hi = int(o["src_idx"].min()), int(o["src_idx"].max())
    pos = {int(s): i for i, s in enumerate(o["src_idx"])}
    n = 0
    for k, s in enumerate(b["src_idx"]):
        if int(s) in pos:
            masks[k] = o["masks"][pos[int(s)]]
            n += 1
    print(f"[interp] override {override_npz.name}: replaced {n} key masks in frames {lo}-{hi}", flush=True)
    return {"masks": masks, "src_idx": b["src_idx"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--override", action="append", default=[],
                    help="concept=path.npz : replace that concept's key masks inside the override's frame range")
    ap.add_argument("--extra", action="append", default=[],
                    help="concept=path.npz : add a concept computed in a separate SAM run")
    a = ap.parse_args()
    ov = dict(x.split("=", 1) for x in a.override)
    extra = {k: Path(v) for k, v in (x.split("=", 1) for x in a.extra)}
    info = probe(a.video)
    first = next(a.masks.glob("*.npz"))
    h, w = np.load(first)["masks"].shape[1:]
    gray = np.stack([cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                     for f in iter_frames(a.video, info, scale_w=w)]).astype(np.float32)
    gray = np.ascontiguousarray(gray[:, :h])
    print(f"[interp] {gray.shape[0]} frames at {w}x{h}", flush=True)
    sources = {p.stem: p for p in sorted(a.masks.glob("*.npz"))}
    sources.update(extra)
    for name, npz in sources.items():
        d = merge_override(npz, Path(ov[name])) if name in ov else np.load(npz)
        full = interpolate_concept(d["masks"], d["src_idx"], gray)
        np.save(a.masks / f"{name}_full.npy", full)
        print(f"[interp] {name}: {full.shape} mean coverage {full.mean() / 2.55:.1f}%", flush=True)


if __name__ == "__main__":
    main()
