"""Vivid aquarium look for the water/background, plus per-part recolouring
of the diver (skin, hair, dive mask, fins) inside SAM 3.1 masks.

All functions take float32 RGB 0..1 (BT.709 gamma-encoded). The background
look is a fixed per-pixel map (plus one clip-wide veil estimate), so it is
temporally stable by construction.
"""
from __future__ import annotations

import cv2
import numpy as np

import reels_grade as R

luma, smoothstep, sat = R.luma, R.smoothstep, R.sat

# ------------------------------------------------------------ background --
AQ = dict(
    dehaze=0.85,            # how much of the clip's veil (haze floor) to remove
    # channel mixer rows: the flat green-teal water -> clear turquoise/blue
    mix=np.array([[1.10, -0.05, 0.00],
                  [-0.04, 0.95, 0.06],
                  [0.00, 0.04, 1.05]], np.float32),
    gamma=0.92,             # lift the mids (bright, airy tank)
    contrast=1.18,          # S-curve strength around 0.5
    sat=1.30,               # global saturation
    vibrance=1.95,          # extra on the less-saturated water pixels
    hue_cyan=0.14,          # pull green water toward cyan (G -> B transfer)
    bed_lo=0.05, bed_hi=0.15,
    sand_lo=0.34, sand_hi=0.46,
    sand_tint=np.array([1.10, 1.02, 0.86], np.float32),  # warm, clean sand
    sand_lift=1.22,
    sand_white=0.80,
    rock_warm=0.22,         # red restored on mid-tone seabed/rocks
    clarity=0.40, clarity_sigma=14.0,
    top=0.90, top_gamma=1.4,  # gentle deeper blue toward the surface side
    top_tint=np.array([0.80, 0.95, 1.08], np.float32),
)


def clip_veil(frames):
    """Haze floor per channel: median over sample frames of the 0.5th
    percentile. One value for the whole clip (no flicker)."""
    v = [np.percentile(f.reshape(-1, 3), 0.5, axis=0) for f in frames]
    return np.median(np.stack(v), 0).astype(np.float32)


def grade_aquarium(x, veil):
    p = AQ
    h = x.shape[0]
    y0 = luma(x)[..., None]
    # seabed key from the *source* colour: the sand/rock are greener than
    # the water (G/B ~1.25 vs ~1.0), the water is often brighter than sand
    gb = (x[..., 1:2] - x[..., 2:3]) / np.maximum(x[..., 1:2], 1e-4)
    k_bed = smoothstep(p["bed_lo"], p["bed_hi"], gb)
    k_sand = k_bed * smoothstep(p["sand_lo"], p["sand_hi"], y0)
    k_rock = k_bed * (1 - smoothstep(p["sand_lo"], p["sand_hi"], y0))
    # dehaze: remove the veil, re-stretch
    v = veil * p["dehaze"]
    x = np.clip((x - v) / (1 - v), 0, None)
    x = np.clip(x @ p["mix"].T, 0, None)
    # green -> cyan: water pixels where G is the top channel give some to B
    gx = np.clip(x[..., 1:2] - np.maximum(x[..., 0:1], x[..., 2:3]), 0, None)
    x = x + np.concatenate([-0.3 * gx, -p["hue_cyan"] * 4 * gx, p["hue_cyan"] * 6 * gx], -1)
    x = np.clip(x, 0, None) ** p["gamma"]
    # rocks / mid seabed: a little red back so the seabed reads colourful
    x[..., 0:1] *= 1 + p["rock_warm"] * k_rock
    # sand: clean warm white
    y = luma(x)[..., None]
    x = x * (1 - p["sand_white"] * k_sand) + y * p["sand_tint"] * p["sand_lift"] * (p["sand_white"] * k_sand)
    # saturation + vibrance (scaled down where already saturated)
    x = sat(x, p["sat"])
    yb = luma(x)[..., None]
    mx, mn = x.max(-1, keepdims=True), x.min(-1, keepdims=True)
    chroma = (mx - mn) / np.maximum(mx, 1e-4)
    x = yb + (x - yb) * (1 + (p["vibrance"] - 1) * (1 - chroma) * (1 - k_bed))
    # surface side: slightly deeper, bluer
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    g = p["top"] + (1 - p["top"]) * t ** (1 / p["top_gamma"])
    x = x * g
    x = x * (1 + (p["top_tint"] - 1) * (1 - t) ** 2)
    # clarity: large-radius local contrast
    yl = luma(np.clip(x, 0, 1))
    blur = cv2.GaussianBlur(yl, (0, 0), p["clarity_sigma"])
    x = x + ((yl - blur) * p["clarity"])[..., None]
    # contrast S-curve
    x = np.clip(x, 0, 1)
    s = x * x * (3 - 2 * x)
    x = x + (s - x) * (p["contrast"] - 1) * 2
    return np.clip(x, 0, 1)


# ---------------------------------------------------- diver part colours --
def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def skin_hue_of(rgb) -> float:
    """BT.709 vectorscope angle (deg from +Cb) of an R:G:B skin colour."""
    rgb = np.asarray(rgb, np.float32)
    y = float(rgb @ R.LUMA)
    cb, cr = (rgb[2] - y) / 1.8556, (rgb[0] - y) / 1.5748
    return float(np.degrees(np.arctan2(cr, cb)))


def colorize(x, w, target_hex, shading=0.85, hi_keep=0.6):
    """Recolour the pixels under weight `w` (HxWx1) to `target_hex`.
    The part's median luminance is moved to the target colour's luminance;
    each pixel keeps its shading relative to that median (texture, folds),
    and specular highlights (lens glints) stay partly neutral."""
    tgt = hex_rgb(target_hex)
    ty = float(tgt @ R.LUMA)
    y = luma(x)
    sel = w[..., 0] > 0.5
    if sel.sum() < 30:
        return x
    med = max(float(np.median(y[sel])), 1e-3)
    yn = np.clip(ty * (np.maximum(y, 1e-4) / med) ** shading, 0, 0.97)
    out = yn[..., None] * (tgt / max(ty, 1e-4))
    # glints on glass / wet rubber: keep their own neutral brightness
    k_hi = smoothstep(med * 2.2, med * 3.5, y)[..., None] * hi_keep
    out = out * (1 - k_hi) + y[..., None] * k_hi
    return np.clip(out, 0, 1)


SKIN_OPTIONS = {
    # name: (label, skin colour at the skin's median brightness). Each pixel
    # keeps its shading relative to that median, so the options differ in
    # both hue and depth of tan.
    "A": ("Acik seftali", "#E9BFA0"),
    "B": ("Dogal bronz", "#D49A72"),
    "C": ("Altin bronz", "#C68B59"),
    "D": ("Koyu karamel", "#9A633E"),
}

PART_SCHEMES = {
    # name: (label, hair, dive mask, fins)
    "1": ("Dogal belirgin", "#3A2A1F", "#1C1E22", "#1E3A6E"),
    "2": ("Canli spor", "#3A2A1F", "#C8262B", "#F2C12E"),
    "3": ("Okyanus", "#3A2A1F", "#F07A24", "#1F86D6"),
    "4": ("Neon", "#141414", "#FF3FA4", "#39E66B"),
}
