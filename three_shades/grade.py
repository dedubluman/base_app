"""Cinematic underwater grade built FROM the palette.

grade_global() is a pure per-pixel color transform (white balance, levels,
filmic contrast, palette split-toning, sand/water separation). Because it is
per-pixel it is also exported as a .cube LUT for DaVinci Resolve / Premiere /
Final Cut.

grade_frame() adds the spatial, mask-aware part on top: subject vs background
exposure, warmth on the diver, gentle background depth-of-field, bloom on the
sun rays and a vignette.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import GradeParams, Palette

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709


def luma(img: np.ndarray) -> np.ndarray:
    return img @ LUMA


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _chroma(rgb) -> np.ndarray:
    """Color minus its own luma: a pure hue/saturation offset."""
    c = np.asarray(rgb, dtype=np.float32)
    return c - float(c @ LUMA)


@dataclass
class GradeStats:
    """Clip-wide measurements, computed once so the grade never flickers."""
    wb_gains: np.ndarray
    black: float
    white: float


def analyze(frames: list[np.ndarray], p: GradeParams) -> GradeStats:
    px = np.concatenate([f.reshape(-1, 3) for f in frames])
    y = luma(px)
    mid = px[(y > 0.15) & (y < 0.85)]
    mean = mid.mean(axis=0)
    gains = (mean.mean() / np.maximum(mean, 1e-4)) ** p.wb_strength
    gains = gains / float(gains @ LUMA)  # white balance must not change exposure
    balanced = luma(px * gains)
    return GradeStats(
        wb_gains=gains.astype(np.float32),
        black=float(np.percentile(balanced, p.black_percentile)),
        white=float(np.percentile(balanced, p.white_percentile)),
    )


def _filmic(x: np.ndarray, contrast: float, pivot: float, knee: float) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    lo = pivot * np.power(np.clip(x / pivot, 0, 1), contrast)
    hi_in = np.clip((1.0 - x) / (1.0 - pivot), 0, 1)
    hi = 1.0 - (1.0 - pivot) * np.power(hi_in, contrast)
    y = np.where(x < pivot, lo, hi)
    over = np.clip(x - 1.0, 0, None)  # anything above 1 rolls off softly
    y = y + over * 0.25
    k = knee
    return np.where(y > k, k + (1 - k) * np.tanh((y - k) / (1 - k)), y)


def grade_global(img: np.ndarray, stats: GradeStats, p: GradeParams, pal: Palette) -> np.ndarray:
    abyss, sand, crystal = (np.array(s.rgb, np.float32) for s in pal.swatches())

    # 1. Neutralize the green cast and set levels.
    x = img * stats.wb_gains
    black = stats.black * p.levels_strength
    white = 1.0 + (stats.white - 1.0) * p.levels_strength
    x = (x - black) / max(white - black, 1e-3)

    # 2. Filmic S-curve (per channel: adds density and a little saturation).
    x = _filmic(x, p.contrast, p.pivot, p.highlight_knee)

    # 3. Tonal zones.
    y = luma(x)[..., None]
    w_sh = 1.0 - smoothstep(0.0, 0.48, y)
    w_hi = smoothstep(0.55, 1.0, y)
    w_mid = np.clip(1.0 - w_sh - w_hi, 0.0, 1.0)

    # 4. Split-toning with the palette. Shadows are lifted INTO Abyssal Teal
    #    (deep and rich instead of crushed black); highlights lean to Crystal.
    x = x + w_sh * p.shadow_tint * _chroma(abyss)
    x = x + (w_sh ** 2) * p.shadow_lift * (abyss - x)
    x = x + w_mid * p.midtone_tint * _chroma(sand)
    x = x + w_hi * p.highlight_tint * _chroma(crystal)
    x = x + (w_hi ** 2) * p.highlight_mix * (crystal - x)

    # 5. Warm/cool separation: red > blue means sand or skin, otherwise water.
    #    Highlights (bokeh, surface glints) are excluded so they stay crystal.
    warm = smoothstep(-0.015, 0.06, x[..., 0:1] - x[..., 2:3]) * smoothstep(0.12, 0.5, y)
    warm = warm * (1.0 - smoothstep(0.50, 0.72, y))
    x = x + warm * p.sand_boost * _chroma(sand) * (0.4 + w_mid)
    x = x + (1.0 - warm) * w_mid * p.water_teal * _chroma(abyss)

    # 6. Saturation + vibrance (boosts muted colors more than saturated ones).
    y = luma(x)[..., None]
    sat = np.abs(x - y).max(axis=-1, keepdims=True)
    gain = p.saturation * (1.0 + p.vibrance * (1.0 - smoothstep(0.0, 0.25, sat)))
    x = y + (x - y) * gain
    return np.clip(x, 0.0, 1.0)


def soft_mask(mask: np.ndarray, p: GradeParams) -> np.ndarray:
    m = mask.astype(np.float32)
    sigma = max(p.mask_feather_frac * mask.shape[1], 0.5)
    return cv2.GaussianBlur(m, (0, 0), sigma)[..., None]


def grade_frame(img: np.ndarray, mask: np.ndarray, stats: GradeStats,
                p: GradeParams, pal: Palette) -> np.ndarray:
    """Full grade; ``mask`` is a float (H, W) subject mask in [0, 1]."""
    h, w = img.shape[:2]
    g = grade_global(img, stats, p, pal)
    m = soft_mask(mask, p)
    sand = np.array(pal.midtone.rgb, np.float32)

    # Subject: brighter, warmer skin, a touch of local contrast (clarity).
    subj = g * p.subject_exposure
    y = luma(g)[..., None]
    mids = 1.0 - np.abs(y - 0.5) * 2.0
    subj = subj + p.subject_warmth * 0.25 * mids * _chroma(sand)
    if p.subject_detail > 0:
        base = cv2.GaussianBlur(subj, (0, 0), w * 0.006)
        subj = subj + p.subject_detail * (subj - base)

    # Background: slightly darker, less saturated, gently defocused.
    bg = g * p.bg_exposure
    by = luma(bg)[..., None]
    bg = by + (bg - by) * (1.0 - p.bg_desaturate)
    if p.bg_blur_frac > 0:
        bg = cv2.GaussianBlur(bg, (0, 0), p.bg_blur_frac * w)
    out = bg * (1.0 - m) + subj * m

    # Bloom on sun rays / surface highlights, tinted Crystal Highlight.
    if p.bloom > 0:
        small = cv2.resize(out, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        bright = np.clip(luma(small) - p.bloom_threshold, 0, None)[..., None]
        glow = cv2.GaussianBlur(bright * np.array(pal.highlight.rgb, np.float32), (0, 0),
                                p.bloom_radius_frac * w / 4)
        out = out + p.bloom * cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)

    # Vignette.
    if p.vignette > 0:
        yy, xx = np.ogrid[:h, :w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.6 + ((yy - h / 2) / (h / 2)) ** 2
        out = out * (1.0 - p.vignette * smoothstep(0.35, 1.6, r2.astype(np.float32)))[..., None]
    return np.clip(out, 0.0, 1.0)


def export_cube(path: str, stats: GradeStats, p: GradeParams, pal: Palette, size: int = 33) -> None:
    """Write the global (per-pixel) part of the grade as a 3D .cube LUT."""
    r = np.linspace(0, 1, size, dtype=np.float32)
    bb, gg, rr = np.meshgrid(r, r, r, indexing="ij")  # red varies fastest
    lattice = np.stack([rr, gg, bb], axis=-1).reshape(-1, 1, 3)
    out = grade_global(lattice, stats, p, pal).reshape(-1, 3)
    with open(path, "w") as f:
        f.write('TITLE "3 Shades underwater grade"\n')
        f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
