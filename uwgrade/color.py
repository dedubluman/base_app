"""Colour utilities and the reference-derived target ramps.

Everything works on float32 RGB in 0..1 (display-referred Rec.709/sRGB-like
gamma, the footage is plain Rec.709 SDR) and CIELAB (D65). Lab is used for the
creative work because lightness, chroma and hue can be moved independently.

TARGET RAMPS (measured on the two reference photographs, see
README_uwgrade.md): chroma C and ab-hue as a function of Lab L* for the water
and for the warm rocks. They are *relationships*, not pixel colours.
"""
from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------
# conversions
# --------------------------------------------------------------------------


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2Lab)


def lab2rgb_raw(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_Lab2RGB)


def lab2rgb(lab: np.ndarray, iters: int = 5) -> np.ndarray:
    """Lab -> RGB with chroma-reduction gamut mapping (keeps hue and lightness,
    instead of letting a clipped channel shift the hue)."""
    lab = np.ascontiguousarray(lab, np.float32)
    rgb = lab2rgb_raw(lab)
    bad = (rgb.min(-1) < -0.002) | (rgb.max(-1) > 1.002)
    if bad.any():
        idx = np.nonzero(bad)
        L = lab[idx][:, 0]
        ab = lab[idx][:, 1:]
        lo = np.zeros(len(L), np.float32)
        hi = np.ones(len(L), np.float32)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            t = np.stack([L, ab[:, 0] * mid, ab[:, 1] * mid], -1)[None]
            r = lab2rgb_raw(t)[0]
            ok = (r.min(-1) >= -0.002) & (r.max(-1) <= 1.002)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        t = np.stack([L, ab[:, 0] * lo, ab[:, 1] * lo], -1)[None]
        rgb[idx] = lab2rgb_raw(t)[0]
    return np.clip(rgb, 0.0, 1.0)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


# --------------------------------------------------------------------------
# 1-D curves
# --------------------------------------------------------------------------


class Curve:
    """Monotone cubic (PCHIP) curve baked to a 4096-entry LUT on 0..100."""

    def __init__(self, xs, ys, lo: float = 0.0, hi: float = 100.0, n: int = 4096):
        from scipy.interpolate import PchipInterpolator

        xs = np.asarray(xs, np.float64)
        ys = np.asarray(ys, np.float64)
        self.x = np.linspace(lo, hi, n, dtype=np.float32)
        self.y = PchipInterpolator(xs, ys, extrapolate=True)(self.x).astype(np.float32)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return np.interp(v, self.x, self.y).astype(np.float32)


def ramp(xs, ys):
    """Piecewise-linear 1-D function clamped at the ends."""
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    return lambda v: np.interp(v, xs, ys).astype(np.float32)


# --------------------------------------------------------------------------
# reference-derived target ramps (mean of REF1 and REF2, see README)
# --------------------------------------------------------------------------
# water: chroma and ab-hue (degrees) versus L*. Deep water is near-black with
# little chroma; the mid water is petrol blue (hue ~255-260), the lit water
# near the surface drifts toward cyan (hue ~232) and loses chroma.
WATER_C = ramp([0, 4, 11, 17, 24, 33, 44, 60, 100], [4.0, 8.5, 13.7, 16.0, 21.0, 23.5, 20.5, 17.0, 8.0])
WATER_H = ramp([0, 10, 30, 38, 46, 60, 100], [262, 260, 255, 245, 235, 228, 225])

# rocks: warm brown. Hue goes olive-brown in the shadows -> bronze/copper in the
# lit tones; chroma stays low (muted), never orange.
ROCK_C = ramp([0, 4, 11, 17, 24, 33, 44, 60, 100], [2.5, 4.0, 6.5, 7.8, 9.0, 11.0, 13.0, 14.0, 8.0])
ROCK_H = ramp([0, 12, 25, 40, 60, 100], [74, 70, 62, 55, 52, 50])


def polar(C, H_deg):
    h = np.deg2rad(H_deg)
    return C * np.cos(h), C * np.sin(h)


def hue_deg(a, b):
    return (np.degrees(np.arctan2(b, a)) % 360.0).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
