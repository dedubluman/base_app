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



# ---------------------------------------------------------------------------
# Skin tone: white-balance the diver against his own white clothing
# ---------------------------------------------------------------------------
# Water absorbs red long before blue, so the diver is lit by cyan light and
# his skin blends into the water. Inside the SAM mask we measure what should
# be white (the brightest unclipped subject pixels, here the white shorts) and
# apply the gains that make it neutral. Skin then shows its own warm colour,
# white stays white and dark fins stay dark, without guessing which pixel is
# skin.

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def subject_reference_gains(img: np.ndarray, mask_u8: np.ndarray,
                            cfg: Separation) -> np.ndarray | None:
    """Per-frame RGB gains (luminance-preserving) from the brightest
    unclipped pixels of `mask_u8` (the SAM 3.1 shorts mask when available:
    white fabric is the white reference). None if there are too few."""
    core = cv2.erode(mask_u8, np.ones((3, 3), np.uint8)) > 127
    px = img[core]
    if len(px) < 60:
        return None
    px = px[px.max(1) < cfg.skin_clip]
    if len(px) < 40:
        return None
    lum = px @ LUMA
    ref = px[lum >= np.percentile(lum, cfg.skin_ref_percentile)].mean(0)
    gains = ref.mean() / np.maximum(ref, 1e-3)
    gains = gains / float(gains @ LUMA)
    return np.clip(gains, cfg.skin_gain_min, cfg.skin_gain_max).astype(np.float32)


def smooth_gains(per_frame: list, window: int) -> np.ndarray:
    """Fill missing frames by interpolation, then a centred moving average,
    so the subject colour never flickers."""
    n = len(per_frame)
    have = [i for i, g in enumerate(per_frame) if g is not None]
    if not have:
        return np.ones((n, 3), np.float32)
    arr = np.array([per_frame[i] for i in have], np.float32)
    full = np.stack([np.interp(np.arange(n), have, arr[:, c]) for c in range(3)], 1)
    k = max(1, window | 1)
    pad = np.pad(full, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k, np.float32) / k
    return np.stack([np.convolve(pad[:, c], ker, mode="valid") for c in range(3)],
                    1).astype(np.float32)


def skin_alpha(mask_u8: np.ndarray, cfg: Separation, width: int) -> np.ndarray:
    """Slightly eroded, lightly feathered alpha: keeps the correction off the
    surrounding water."""
    k = width / 1080.0
    m = mask_u8
    e = int(round(cfg.skin_erode_px * k))
    if e > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * e + 1, 2 * e + 1)))
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0),
                         max(cfg.skin_feather_px * k, 0.5))
    return a[..., None]


def apply_skin_tone(img: np.ndarray, alpha: np.ndarray, gains: np.ndarray,
                    weight: float, cfg: Separation) -> np.ndarray:
    w = weight * cfg.skin_strength
    if w <= 1e-4:
        return img
    # dark gear (fins, mask, hair) keeps its own colour: the correction
    # fades out below `skin_dark_lum`
    lum0 = (img * LUMA).sum(-1, keepdims=True)
    dark = np.clip((lum0 - cfg.skin_dark_lum * 0.4) / (cfg.skin_dark_lum * 0.6), 0, 1)
    g = 1.0 + (gains - 1.0) * w * dark
    wb = img * g
    lum = (wb * LUMA).sum(-1, keepdims=True)
    # near-clipped highlights (blown-out white fabric) stay neutral, not pink
    hp = np.clip((img.max(-1, keepdims=True) - 0.85) / 0.12, 0, 1)
    wb = wb + (lum - wb) * hp
    # gentle golden warmth on mid-tones = skin; fades in shadows/highlights
    mid = np.clip((lum - 0.15) / 0.2, 0, 1) * np.clip((0.85 - lum) / 0.2, 0, 1)
    wb = wb + mid * (cfg.skin_warmth * w) * np.array([1.0, 0.35, -0.8],
                                                       np.float32) * lum
    wb = wb * (2.0 ** (cfg.skin_exposure * w))
    return np.clip(img + alpha * (wb - img), 0.0, 1.0)


def part_alpha(mask_u8: np.ndarray, feather_px: float, width: int) -> np.ndarray:
    a = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (0, 0),
                         max(feather_px * width / 1080.0, 0.5))
    return a[..., None]


def apply_shorts_color(img: np.ndarray, alpha: np.ndarray, gains: np.ndarray,
                       weight: float, cfg: Separation) -> np.ndarray:
    """Colour the SAM 3.1 shorts on their own: neutralise the cyan cast with
    the subject gains, then pull the fabric toward `shorts_hex` while keeping
    its own luminance (folds, shading) and a little of its chroma."""
    w = weight * cfg.shorts_strength
    if w <= 1e-4:
        return img
    wb = img * gains
    lum = (wb * LUMA).sum(-1, keepdims=True)
    h = cfg.shorts_hex.lstrip("#")
    tgt = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)
    t_l = float((tgt * LUMA).sum())
    lum2 = lum * (2.0 ** cfg.shorts_exposure)
    target = tgt * (lum2 / t_l) + cfg.shorts_saturation * (wb - lum)
    return np.clip(img + alpha * w * (target - img), 0.0, 1.0)
