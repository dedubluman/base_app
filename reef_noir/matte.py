"""Turn SAM 3.1 binary masks into a stable, edge-aware soft matte."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def clean(m: np.ndarray) -> np.ndarray:
    """Fill holes; keep the main body plus nearby pieces (fins, hands)."""
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    main = lab == (int(np.argmax(sizes)) + 1)
    near = ndi.binary_dilation(main, iterations=12)
    keep = main.copy()
    for k in range(1, n + 1):
        comp = lab == k
        if sizes[k - 1] >= 8 and (comp & near).any():
            keep |= comp
    return keep


def temporal_vote(masks: list[np.ndarray]) -> list[np.ndarray]:
    """3-frame majority vote: removes single-frame flicker without lag."""
    out = []
    for i in range(len(masks)):
        a = masks[max(i - 1, 0)].astype(np.uint8)
        b = masks[i].astype(np.uint8)
        c = masks[min(i + 1, len(masks) - 1)].astype(np.uint8)
        out.append((a + b + c) >= 2)
    return out


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    cIp = cv2.boxFilter(I * p, -1, k) - mI * mp
    vI = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cIp / (vI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def soft_matte(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Edge-aware alpha from a binary mask, limited to a band around it.

    Spill guard: edge pixels whose colour is indistinguishable from the
    local water (blurred background) are pushed out of the matte, so the
    diver grade never paints a warm halo onto water.
    """
    if not m.any():
        return np.zeros(m.shape, np.float32)
    ys, xs = np.nonzero(m)
    pad = 24
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
    sub = rgb[y0:y1, x0:x1]
    mf = m[y0:y1, x0:x1].astype(np.float32)
    I = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    a = np.clip(guided_filter(I, mf, 2, 2e-4), 0, 1)
    # keep the matte within 2 px of SAM's boundary
    band = cv2.dilate(mf, np.ones((5, 5), np.uint8))
    core = cv2.erode(mf, np.ones((5, 5), np.uint8))
    a = np.maximum(a * band, core)
    # spill guard
    lab = cv2.cvtColor(sub, cv2.COLOR_RGB2LAB)
    inv = 1 - mf
    s = 7
    bg = cv2.GaussianBlur(lab * inv[..., None], (0, 0), s) / (
        cv2.GaussianBlur(inv, (0, 0), s)[..., None] + 1e-4)
    d = np.linalg.norm(lab - bg, axis=-1)
    spill = np.clip((d - 4.0) / 8.0, 0, 1)
    a = np.where(core > 0.5, a, a * spill)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    out = np.zeros(m.shape, np.float32)
    out[y0:y1, x0:x1] = a
    return out
