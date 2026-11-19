"""Parametric underwater grade (background + subject) shared by fit_grade.py and grade_dive.py.

All images are float32 RGB in 0..1 (display-referred Rec.709). No stabilisation / denoise.
"""
from __future__ import annotations

import cv2
import numpy as np

CH = "rgb"

# Defaults are overwritten by out/grade_params.json (produced by fit_grade.py).
DEFAULT = {
    "bg": {
        "a": [0.2, 0.55, 0.9], "g": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0],
        "sat": 1.0, "hue_deg": 0.0,
        "clarity_amt": 0.18, "clarity_sigma_frac": 0.04,
        "vignette": 0.35, "vignette_pow": 2.0,
        "top_dark": 0.0,
    },
    "subj": {
        "gain": 1.08, "contrast": 1.06, "sat": 1.0, 
        "luma_pivot": 0.5, "cast_neutralize": 0.9, "skin_ab": [10.4, 15.9], "water_ab": [-12.0, -33.0], "water_d": [9.0, 18.0],
        "clarity_amt": 0.12, "clarity_sigma_frac": 0.03,
    },
}


def to_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2LAB)


def from_lab(lab):
    return np.clip(cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def vignette_map(h, w, strength, power):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    return (1.0 - strength * np.clip(r, 0, 1) ** power).astype(np.float32)


def local_contrast(lab, amt, sigma_frac):
    """Large-radius, low-amount clarity on L only (no noise boost)."""
    if amt <= 0:
        return lab
    h, w = lab.shape[:2]
    sig = max(2.0, sigma_frac * max(h, w))
    L = lab[..., 0]
    lab = lab.copy()
    lab[..., 0] = np.clip(L + amt * (L - cv2.GaussianBlur(L, (0, 0), sig)), 0, 100)
    return lab


def grade_bg(rgb, P, vig=None):
    a = np.array(P["a"], np.float32); g = np.array(P["g"], np.float32); b = np.array(P["b"], np.float32)
    y = np.clip(a * np.power(np.clip(rgb, 1e-6, 1), g) + b, 0, 1)
    lab = to_lab(y)
    hr = np.deg2rad(P["hue_deg"]); c, s = np.cos(hr), np.sin(hr)
    A, B = lab[..., 1] * P["sat"], lab[..., 2] * P["sat"]
    lab[..., 1], lab[..., 2] = A * c - B * s, A * s + B * c
    lab = local_contrast(lab, P["clarity_amt"], P["clarity_sigma_frac"])
    out = from_lab(lab)
    if vig is None:
        vig = vignette_map(*rgb.shape[:2], P["vignette"], P["vignette_pow"])
    out = out * vig[..., None]
    if P.get("top_dark", 0) > 0:
        h = rgb.shape[0]
        out = out * (1 - P["top_dark"] * np.linspace(1, 0, h, dtype=np.float32) ** 2)[:, None, None]
    return np.clip(out, 0, 1)


def grade_subj(rgb, P, mask=None):
    """Keep skin hue/chroma (no blue paint). Remove the water cast partly, add light + contrast."""
    lab = to_lab(rgb)
    L = lab[..., 0]
    L = P["luma_pivot"] * 100 + (L - P["luma_pivot"] * 100) * P["contrast"]
    L = np.clip(L * P["gain"], 0, 100)
    # water cast: pull a/b toward the target skin chroma, gated so that black fins (low L)
    # and water-coloured pixels (strongly negative b, e.g. mask-edge water) stay untouched
    t = np.clip((lab[..., 0] - 25.0) / 25.0, 0, 1)
    d = np.hypot(lab[..., 1] - P["water_ab"][0], lab[..., 2] - P["water_ab"][1])   # distance from water chroma
    w = np.clip((d - P["water_d"][0]) / (P["water_d"][1] - P["water_d"][0]), 0, 1)
    w = w * w * (3 - 2 * w)
    if mask is not None:   # deep inside the mask it is certainly subject: skin may be water-tinted, so ignore the chroma gate there
        dt = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        it = np.clip((dt - 3.0) / 6.0, 0, 1)
        w = np.maximum(w, it * it * (3 - 2 * it))
    k = P["cast_neutralize"] * (t * t * (3 - 2 * t)) * w
    lab[..., 1] = (lab[..., 1] + k * (P["skin_ab"][0] - lab[..., 1])) * P["sat"]
    lab[..., 2] = (lab[..., 2] + k * (P["skin_ab"][1] - lab[..., 2])) * P["sat"]
    lab[..., 0] = L
    lab = local_contrast(lab, P["clarity_amt"], P["clarity_sigma_frac"])
    return from_lab(lab)


def composite(rgb, mask, params, vig=None):
    bg = grade_bg(rgb, params["bg"], vig)
    if mask is None or not mask.any():
        return bg
    sj = grade_subj(rgb, params["subj"], mask)
    m = mask[..., None].astype(np.float32)
    return bg * (1 - m) + sj * m
