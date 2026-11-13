"""Diver alpha for every frame from the stride-2 SAM 3.1 tracking.

* even source frames: SAM 3.1 mask used directly
* odd source frames: both SAM neighbours are warped onto the frame with dense
  optical flow (DIS), averaged, thresholded
* every frame: edge snap with a guided filter on the image luminance, kept in
  a narrow band around the SAM / warped boundary (no feather, no halo)
* frames where SAM lost the diver (empty / sudden area jump) are rebuilt from
  the nearest good neighbours the same way
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from matte import guided_filter  # noqa: E402  (project helper, He et al. guided filter)

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _ell(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def gray_u8(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)


_DIS = None


def flow(g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    """flow such that g0(x) ~ g1(x + flow(x)); computed at half resolution."""
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    h, w = g0.shape
    a = cv2.resize(g0, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    b = cv2.resize(g1, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    f = _DIS.calc(a, b, None)
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    return f


def warp(img: np.ndarray, fl: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def snap(raw: np.ndarray, rgb01: np.ndarray, r_in: int = 4, r_out: int = 1) -> np.ndarray:
    """raw: bool mask. Returns float alpha following the image edge inside the
    band [boundary - r_in, boundary + r_out]."""
    m = raw.astype(np.uint8)
    if int(m.sum()) == 0:
        return np.zeros(raw.shape, np.float32)
    small = int(m.sum()) < 4000
    inner = cv2.erode(m, _ell(min(r_in, 2) if small else r_in)).astype(np.float32)
    outer = cv2.dilate(m, _ell(r_out)).astype(np.float32)
    I = (rgb01 @ LUMA).astype(np.float32)
    p = cv2.erode(m, _ell(1)).astype(np.float32)
    g = np.clip(guided_filter(I, p, 2 if small else 4, 1e-3), 0, 1)
    a = np.minimum(np.maximum(g, inner), outer)
    return np.minimum(cv2.GaussianBlur(a, (0, 0), 0.7), outer).astype(np.float32)


def load_sam(sdir: Path, n_proxy: int, shape: tuple[int, int]) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for i in range(n_proxy):
        f = sdir / "subject" / f"{i:05d}.png"
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
        out.append(None if m is None else (m > 127))
    return out


def good_flags(masks: list[np.ndarray | None], jump: float = 2.2, min_frac: float = 0.0008) -> np.ndarray:
    area = np.array([0 if m is None else int(m.sum()) for m in masks], np.float64)
    h, w = next(m for m in masks if m is not None).shape
    ok = area > min_frac * h * w
    med = np.array([np.median(area[max(0, i - 3):i + 4][ok[max(0, i - 3):i + 4]]) if ok[max(0, i - 3):i + 4].any() else 0
                    for i in range(len(area))])
    ok &= (area < jump * np.maximum(med, 1)) & (area > np.maximum(med, 1) / jump)
    return ok


def build(frames_u8, sdir: Path, out_dir: Path, stride: int = 2, log=print) -> dict:
    """Write out_dir/<t:05d>.png (uint8 alpha) for every source frame t."""
    T, H, W = frames_u8.shape[:3]
    n_proxy = (T + stride - 1) // stride
    sam = load_sam(sdir, n_proxy, (H, W))
    ok = good_flags(sam)
    log(f"[diver] SAM proxy frames {n_proxy}, usable {int(ok.sum())}, lost {n_proxy - int(ok.sum())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    grays = {}

    def g(t):
        if t not in grays:
            grays[t] = gray_u8(np.asarray(frames_u8[t]))
            if len(grays) > 8:
                grays.pop(next(iter(grays)))
        return grays[t]

    def first_ok(rng):
        for j in rng:
            if ok[j]:
                return j
        return None

    meta = {}
    for t in range(T):
        rgb01 = np.asarray(frames_u8[t], np.float32) / 255.0
        if t % stride == 0 and ok[t // stride]:
            raw = sam[t // stride]
            src = "sam"
            a = snap(raw, rgb01, 4, 1)
        else:
            i = t // stride
            # nearest usable SAM frame at or before t, and at or after t
            lo = first_ok(range(i, -1, -1))
            hi = first_ok(range(i if t % stride == 0 else i + 1, n_proxy))
            cands = [j for j in (lo, hi) if j is not None]
            acc, wsum = np.zeros((H, W), np.float32), 0.0
            for j in cands:
                s = j * stride
                fl = flow(g(t), g(s))
                wj = 1.0 / (abs(t - s) + 1e-3) if t != s else 1e3
                acc += wj * warp(sam[j].astype(np.float32), fl)
                wsum += wj
            raw = (acc / max(wsum, 1e-6)) > 0.5 if cands else np.zeros((H, W), bool)
            src = "interp"
            a = snap(raw, rgb01, 5, 4)
        cv2.imwrite(str(out_dir / f"{t:05d}.png"), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        meta[t] = {"src": src, "area": float(a.sum())}
        if t % 50 == 0:
            log(f"[diver] frame {t}/{T} ({src}) area {a.sum() / (H * W):.3f}")
    return meta


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0
