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

import dataclasses
from dataclasses import dataclass
from typing import Optional

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
    # White balance measured on the diver only (from the SAM 3.1 mask). Water
    # tints skin cyan-green far more than the global balance can fix without
    # wrecking the water, so the subject gets its own correction.
    subject_wb_gains: Optional[np.ndarray] = None
    # Lab (mean, std) of our graded background and of the reference grade.
    src_lab: Optional[tuple[np.ndarray, np.ndarray]] = None
    ref_lab: Optional[tuple[np.ndarray, np.ndarray]] = None


def lab_stats(px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(np.clip(px, 0, 1).reshape(-1, 1, 3).astype(np.float32),
                       cv2.COLOR_RGB2Lab).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0)


def match_reference(img: np.ndarray, stats: "GradeStats", p: GradeParams) -> np.ndarray:
    """Reinhard-style Lab transfer of img toward the reference grade's statistics."""
    if stats.ref_lab is None or stats.src_lab is None:
        return img
    shape = img.shape
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    (ms, ss), (mr, sr) = stats.src_lab, stats.ref_lab
    moved = (lab - ms) / np.maximum(ss, 1e-3) * sr + mr
    w = np.array([p.ref_match_luma, p.ref_match_a, p.ref_match], np.float32)
    lab = lab + (moved - lab) * w
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(out, 0.0, 1.0).reshape(shape)


def _gray_world(px: np.ndarray, strength: float) -> np.ndarray:
    y = luma(px)
    mid = px[(y > 0.10) & (y < 0.90)]
    mean = (mid if len(mid) else px).mean(axis=0)
    gains = (mean.mean() / np.maximum(mean, 1e-4)) ** strength
    return (gains / float(gains @ LUMA)).astype(np.float32)  # keep exposure


def analyze(frames: list[np.ndarray], p: GradeParams,
            masks: Optional[list[np.ndarray]] = None) -> GradeStats:
    px = np.concatenate([f.reshape(-1, 3) for f in frames])
    y = luma(px)
    mid = px[(y > 0.15) & (y < 0.85)]
    mean = mid.mean(axis=0)
    gains = (mean.mean() / np.maximum(mean, 1e-4)) ** p.wb_strength
    gains = gains / float(gains @ LUMA)  # white balance must not change exposure
    balanced = luma(px * gains)
    subject_gains = None
    if masks is not None:
        subj = [f[m] for f, m in zip(frames, masks) if m.any()]
        if subj and sum(len(s) for s in subj) > 500:
            subject_gains = _gray_world(np.concatenate(subj), p.subject_wb_strength)
    return GradeStats(
        wb_gains=gains.astype(np.float32),
        black=float(np.percentile(balanced, p.black_percentile)),
        white=float(np.percentile(balanced, p.white_percentile)),
        subject_wb_gains=subject_gains,
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
    #    Warm pixels are PULLED toward the Golden Sand hue at a bounded
    #    saturation (never pushed past it), so skin can't turn orange.
    #    Shaded sand and highlights (bokeh, glints) are excluded.
    warm = smoothstep(-0.015, 0.06, x[..., 0:1] - x[..., 2:3]) * smoothstep(0.25, 0.55, y)
    warm = warm * (1.0 - smoothstep(0.55, 0.80, y))
    sand_c = _chroma(sand) * (p.sand_chroma / max(float(np.abs(_chroma(sand)).max()), 1e-4))
    target = y + sand_c * np.clip(y / max(float(sand @ LUMA), 1e-4), 0.0, 1.2)
    x = x + warm * p.sand_boost * (target - x)
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

    # Subject: graded with its OWN white balance and much less of the water's
    # teal toning, so skin separates from the water (computed only inside the
    # mask's bounding box).
    gs = g
    if stats.subject_wb_gains is not None and m.max() > 0.01:
        ys, xs = np.nonzero(m[..., 0] > 0.005)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        r = p.subject_teal_reduction
        sp = dataclasses.replace(p, shadow_tint=p.shadow_tint * (1 - r),
                                 shadow_lift=p.shadow_lift * (1 - r),
                                 water_teal=p.water_teal * (1 - r))
        sst = dataclasses.replace(stats, wb_gains=stats.subject_wb_gains)
        gs = g.copy()
        gs[y0:y1, x0:x1] = grade_global(img[y0:y1, x0:x1], sst, sp, pal)

    # Then brighter, warmer skin and a touch of local contrast (clarity).
    subj = gs * p.subject_exposure
    y = luma(subj)[..., None]
    mids = 1.0 - np.abs(y - 0.5) * 2.0
    subj = subj + p.subject_warmth * 0.25 * mids * _chroma(sand)

    # Skin restore: inside the mask, midtones are pulled toward skin_tone
    # (a warm peach-tan measured on the reference grade), keeping their own
    # brightness. Water strips red from skin and no white balance can put it
    # back without turning the water red too, so this happens only where SAM
    # says the diver is. Very dark parts (dive mask, snorkel, hair) and bright
    # parts (light swim shorts) are left alone.
    if p.skin_restore > 0:
        gate = smoothstep(0.08, 0.24, y) * (1.0 - smoothstep(0.56, 0.76, y))
        hx = p.skin_tone.lstrip("#")
        tone = np.array([int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
        sc = _chroma(tone) * p.skin_saturation
        target = y + sc * np.clip(y / max(float(tone @ LUMA), 1e-4), 0.0, 1.3)
        subj = subj + p.skin_restore * gate * (target - subj)
    if p.subject_detail > 0:
        base = cv2.GaussianBlur(subj, (0, 0), w * 0.006)
        subj = subj + p.subject_detail * (subj - base)

    # Background: matched to the reference grade, slightly darker, gently defocused.
    bg = match_reference(g, stats, p) * p.bg_exposure
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
    out = match_reference(grade_global(lattice, stats, p, pal), stats, p).reshape(-1, 3)
    with open(path, "w") as f:
        f.write('TITLE "3 Shades underwater grade"\n')
        f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
