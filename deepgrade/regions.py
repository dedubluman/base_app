"""Mask utilities: feathering, guided-filter edge refinement, region composition."""
from __future__ import annotations

import cv2
import numpy as np


def feather(m: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return m
    return cv2.GaussianBlur(m, (0, 0), sigma)


def guided_filter(guide: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, single-channel guide (float32 0..1)."""
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mi, mp = box(guide), box(p)
    cov = box(guide * p) - mi * mp
    var = box(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box(a) * guide + box(b)


def refine_diver(raw: np.ndarray, luma: np.ndarray, feather_px: float, r: int,
                 eps: float, band_in: int = 4, band_out: int = 7) -> np.ndarray:
    """Snap a SAM soft mask to real image edges (hair, fingers, snorkel) and
    keep a natural, slightly soft alpha (no cut-out look).

    The guided filter may only decide inside a narrow band around SAM's edge
    (``band_in`` px inside, ``band_out`` px outside): the solid interior stays 1,
    the far exterior stays 0, so dark hair / gear inside the diver can never be
    punched out and thin parts can never leak.
    """
    m = np.clip(raw.astype(np.float32), 0, 1)
    m = feather(m, 1.0)
    g = np.clip(guided_filter(luma, m, r, eps), 0, 1)
    # gentle S-curve: keeps a soft alpha edge but removes the guided filter's grey halo
    g = np.clip((g - 0.5) * 1.35 + 0.5, 0, 1)
    hard = (m > 0.5).astype(np.uint8)
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_in + 1, 2 * band_in + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_out + 1, 2 * band_out + 1))
    inner = cv2.erode(hard, k_in).astype(np.float32)
    outer = cv2.dilate((m > 0.2).astype(np.uint8), k_out).astype(np.float32)
    inner = feather(inner, 1.5)
    outer = feather(outer, 1.5)
    g = np.maximum(g * outer, inner)
    return feather(np.clip(g, 0, 1), feather_px)


def compose(D, S, R, M, Wt, sand_gate: float, region_feather: float):
    """Disjoint soft region weights (sum to 1).

    D diver, S sky, M land/hills, R rocks, Wt water (all SAM 3.1 masks). Whatever SAM did
    not claim is the 'unclaimed' remainder: seabed sand while the camera looks at the
    bottom (sand_gate=1), open water otherwise (sand_gate=0).
    """
    S = feather(S, region_feather) * (1 - D)
    M = feather(M, region_feather) * (1 - D) * (1 - S)
    R = feather(R, region_feather) * (1 - D) * (1 - S) * (1 - M)
    Wt = feather(Wt, region_feather) * (1 - D) * (1 - S) * (1 - M) * (1 - R)
    rem = np.clip(1 - D - S - M - R - Wt, 0, 1)
    return {"D": D, "S": S, "R": R, "M": M,
            "W": Wt + rem * (1 - sand_gate),
            "U": rem * sand_gate}
