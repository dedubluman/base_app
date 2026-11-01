"""Subtle subject/background separation driven by SAM 3.1 masks.

Diver:      a touch more exposure, warmth and local contrast.
Background: a touch less clarity plus faint depth haze toward the Abyssal
            Teal shadow anchor.
Everything is blended through a feathered mask, so there is no cut-out edge,
and the whole effect is scaled by a per-frame weight that fades out over
frames where SAM lost the diver (no pops, no flicker).
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Separation


def _smooth01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def find_lost_frames(areas: np.ndarray, frame_px: int, cfg: Separation) -> np.ndarray:
    """A frame is lost when its mask is (nearly) empty or its area jumps by
    more than `lost_area_jump` x versus the median of its +-5 neighbours."""
    n = len(areas)
    lost = areas < cfg.lost_min_area_frac * frame_px
    for i in range(n):
        nb = np.r_[areas[max(0, i - 5):i], areas[i + 1:i + 6]]
        nb = nb[nb > 0]
        if len(nb) and areas[i] > 0:
            med = float(np.median(nb))
            r = areas[i] / med
            if r > cfg.lost_area_jump or r < 1 / cfg.lost_area_jump:
                lost[i] = True
    return lost


def lost_spans(lost: np.ndarray) -> list[tuple[int, int]]:
    spans, i = [], 0
    while i < len(lost):
        if lost[i]:
            j = i
            while j + 1 < len(lost) and lost[j + 1]:
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def separation_weights(lost: np.ndarray, ramp: int) -> np.ndarray:
    """1.0 on healthy frames, eased down to 0 within `ramp` frames of any
    lost frame (and 0 on lost frames)."""
    n = len(lost)
    if not lost.any():
        return np.ones(n, np.float32)
    idx = np.flatnonzero(lost)
    dist = np.abs(np.arange(n)[:, None] - idx[None, :]).min(1)
    return _smooth01(dist / max(ramp, 1)).astype(np.float32)


def feather(mask_u8: np.ndarray, cfg: Separation, width: int) -> np.ndarray:
    """Binary mask -> soft alpha. Radii scale with frame width (1080 ref)."""
    k = width / 1080.0
    m = mask_u8
    d = int(round(cfg.mask_dilate_px * k))
    if d > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (2 * d + 1, 2 * d + 1)))
    sigma = max(cfg.mask_feather_px * k, 0.5)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sigma)
    return a[..., None]


def _blur_small(img: np.ndarray, sigma_frac: float) -> np.ndarray:
    """Large-radius Gaussian computed at 1/4 resolution (cheap, smooth)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(sigma_frac * w / 4, 0.5))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_separation(graded: np.ndarray, alpha: np.ndarray, weight: float,
                     cfg: Separation, haze_rgb: np.ndarray) -> np.ndarray:
    """graded: float32 HxWx3; alpha: HxWx1 soft mask; weight: 0..1."""
    w = weight * cfg.separation_strength
    if w <= 1e-4:
        return graded
    g = graded
    base = _blur_small(g, 0.012)             # ~13 px at 1080 wide
    detail = g - base

    # Diver: exposure, warmth, local contrast.
    subj = g * (2.0 ** cfg.subject_exposure)
    subj = subj + np.array([cfg.subject_warmth, 0.0, -cfg.subject_warmth],
                           np.float32)
    subj = subj + detail * cfg.subject_local_contrast

    # Background: soften local contrast, add faint haze toward teal.
    bg = g - detail * cfg.bg_clarity_reduction
    bg = bg + (haze_rgb - bg) * cfg.bg_haze

    out = g + w * (alpha * (subj - g) + (1 - alpha) * (bg - g))
    return np.clip(out, 0.0, 1.0)
