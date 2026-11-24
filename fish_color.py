"""Small colour toolbox shared by fish_refine.py / fish_grade.py (float32, RGB 0..1).

OKLab/OKLCH (Ottosson 2020) is used for every hue / chroma operation so that
hue rotations keep perceived lightness and the per-pixel texture (chroma and
lightness variation) survives the recolour.
"""
from __future__ import annotations

import cv2
import numpy as np

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M2I = np.linalg.inv(_M2).astype(np.float32)
_M1I = np.linalg.inv(_M1).astype(np.float32)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def srgb_to_lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_oklab(rgb):
    lin = srgb_to_lin(rgb)
    lms = np.cbrt(lin @ _M1.T)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_rgb(lab):
    lms = lab @ _M2I.T
    return lin_to_srgb((lms ** 3) @ _M1I.T)


def lab_to_lch(lab):
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    return lab[..., 0], C, h


def lch_to_lab(L, C, h):
    r = np.radians(h)
    return np.stack([L, C * np.cos(r), C * np.sin(r)], -1).astype(np.float32)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def hex_to_lch(h):
    L, C, hh = lab_to_lch(rgb_to_oklab(hex_to_rgb(h)[None, None]))
    return float(L[0, 0]), float(C[0, 0]), float(hh[0, 0])


def luma(rgb):
    return rgb @ LUMA


def gblur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def guided(I, p, r, eps):
    """He et al. guided filter with a grey guide."""
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.blur(I, k), cv2.blur(p, k)
    cov = cv2.blur(I * p, k) - mI * mp
    var = cv2.blur(I * I, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.blur(a, k) * I + cv2.blur(b, k)
