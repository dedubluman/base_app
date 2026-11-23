"""sRGB <-> OKLab helpers (float32, numpy). RGB is display-referred (BT.709/sRGB-like)."""
from __future__ import annotations

import numpy as np

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1i = np.linalg.inv(_M1).astype(np.float32)
_M2i = np.linalg.inv(_M2).astype(np.float32)


def srgb_to_lin(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_oklab(rgb):
    lin = srgb_to_lin(rgb)
    lms = np.cbrt(lin @ _M1.T)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_rgb(lab, clip=True):
    lms = np.asarray(lab, np.float32) @ _M2i.T
    lin = (lms ** 3) @ _M1i.T
    out = lin_to_srgb(lin)
    return np.clip(out, 0, 1) if clip else out


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def hex_to_oklab(h: str):
    return rgb_to_oklab(hex_to_rgb(h)[None, None, :])[0, 0]


def lch(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360.0
