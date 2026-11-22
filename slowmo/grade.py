"""Region-aware Mediterranean cinematic grade (float32, display-referred BT.709 in/out).

Source footage is tagged BT.709 / standard (non-log) profile, so no log->display input
transform is applied. Every region has its own transform; regions are composited with soft,
temporally propagated SAM 3.1 masks. All parameters are constants: no per-frame auto exposure,
so the look cannot flicker."""
from __future__ import annotations
import cv2
import numpy as np

W709 = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lum(x):
    return x @ W709


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def hsv(x):
    h = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    return h[..., 0], h[..., 1], h[..., 2]


def from_hsv(h, s, v):
    return cv2.cvtColor(np.stack([h % 360, np.clip(s, 0, 1), np.clip(v, 0, 1)], -1).astype(np.float32), cv2.COLOR_HSV2RGB)


def hue_pull(h, target, amt):
    d = (target - h + 180) % 360 - 180
    return h + amt * d


def ramp(t, stops):
    """Piecewise-linear colour ramp. stops: [(pos, (r,g,b)), ...]"""
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([c for _, c in stops], np.float32)
    return np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], -1).astype(np.float32)


def clarity(x, mask, radius, amt):
    return x + amt * (x - blur(x, radius)) * mask[..., None]


# ------------------------------------------------------------------ regions --

def g_sky(x, ys):
    L = lum(x)
    Lb = blur(L, 25)
    L2 = np.clip(L, 0, 1) ** 1.35 * 0.95 * (1 - 0.30 * (1 - ys) ** 0.8)
    t = smooth(0.18, 0.80, L2)
    col = ramp(t, [(0.0, (0.05, 0.17, 0.40)), (0.45, (0.13, 0.33, 0.60)), (1.0, (0.50, 0.66, 0.82))])
    detail = (L - blur(L, 3))[..., None]
    sky = np.clip(col * (1 + 1.2 * detail), 0, 1)
    sky = 0.80 * sky + 0.20 * x * np.array([0.9, 1.0, 1.1], np.float32)
    # sun: leave the glow warm and bright instead of tinting it blue
    sun = smooth(0.80, 0.97, Lb)[..., None]
    warm = np.clip(x * np.array([1.06, 1.0, 0.86], np.float32) + 0.03 * np.array([1, .75, .3], np.float32), 0, 1)
    return sky * (1 - sun) + warm * sun


def g_rock(x):
    h, s, v = hsv(x)
    h2 = hue_pull(h, 36.0, 0.60)
    s2 = np.clip(s * 1.25 + 0.05, 0, 0.50)
    v2 = np.clip(v, 0, 1) ** 0.92
    y = from_hsv(h2, s2, v2)
    L = lum(y)[..., None]
    sh = smooth(0.35, 0.0, L); hi = smooth(0.45, 0.9, L)
    y = y * (1 - 0.22 * sh) + np.array([0.20, 0.115, 0.07], np.float32) * 0.22 * sh * 1.0
    y = y * (1 - 0.18 * hi) + np.array([1.0, 0.82, 0.50], np.float32) * 0.18 * hi * (L / np.maximum(lum(np.array([1.0, .82, .5], np.float32)), 1e-3))
    return np.clip(y, 0, 1)


def g_veg(x):
    h, s, v = hsv(x)
    h2 = hue_pull(h, 82.0, 0.55)
    s2 = np.clip(s * 1.35 + 0.05, 0, 0.75)
    y = from_hsv(h2, s2, v * 0.97)
    hi = smooth(0.35, 0.8, lum(y))[..., None]
    return np.clip(y + hi * np.array([0.06, 0.04, -0.01], np.float32), 0, 1)


def g_water(x, ys):
    L = np.clip(lum(x) * 1.35, 0, 1) ** 0.95
    depth = np.clip(L, 0, 1)
    col = ramp(depth, [(0.0, (0.008, 0.05, 0.11)), (0.30, (0.03, 0.20, 0.27)), (0.62, (0.10, 0.42, 0.45)), (1.0, (0.55, 0.84, 0.80))])
    detail = (np.clip(lum(x), 0, 1) - blur(lum(x), 2.5))[..., None]
    g = np.clip(col * (1 + 1.6 * detail), 0, 1)
    keep = np.clip(x * 1.35, 0, 1)          # 30% of the original chroma keeps natural variety
    y = 0.58 * g + 0.42 * (0.6 * keep + 0.4 * g)
    # bright specular glints / caustics stay bright and slightly cyan-white
    spec = smooth(0.62, 0.95, lum(x))[..., None]
    return np.clip(y * (1 - spec) + np.clip(x * 1.15, 0, 1) * np.array([0.92, 1.0, 1.0], np.float32) * spec, 0, 1)


def g_seabed(x, mask):
    h, s, v = hsv(x)
    h2 = hue_pull(h, 88.0, 0.25)                # towards golden-green
    y = from_hsv(h2, np.clip(s * 1.45 + 0.05, 0, 0.8), np.clip(v * 1.35, 0, 1) ** 0.95)
    L = lum(y)[..., None]
    y = y + smooth(0.30, 0.75, L) * np.array([0.07, 0.06, -0.015], np.float32)
    y = y + smooth(0.25, 0.0, L) * np.array([-0.012, 0.025, 0.045], np.float32)
    return np.clip(clarity(y, mask, 10, 0.45), 0, 1)


def global_finish(x):
    # gentle filmic S, highlight roll-off, saturation management
    y = np.clip(x, 0, 1)
    y = 0.5 + (y - 0.5) * 1.06
    knee = 0.84
    hi = y > knee
    y = np.where(hi, knee + (1 - knee) * (1 - np.exp(-(y - knee) / (1 - knee))), y)
    L = lum(y)[..., None]
    y = L + (y - L) * 1.04
    return np.clip(y, 0, 1)


def grade(x, w: dict, strength: float = 1.0):
    """x: HxWx3 float RGB; w: soft masks {sky, rock, veg, water, seabed} (each HxW, sum <= 1)."""
    h = x.shape[0]
    ys = np.tile((np.arange(h, dtype=np.float32) / (h - 1))[:, None], (1, x.shape[1]))
    layers = {"sky": g_sky(x, ys), "rock": g_rock(x), "veg": g_veg(x),
              "water": g_water(x, ys), "seabed": g_seabed(x, w["seabed"])}
    tot = sum(w[k] for k in layers)
    out = x * np.clip(1 - tot, 0, 1)[..., None]
    for k, y in layers.items():
        out = out + y * w[k][..., None]
    out = x + strength * (out - x)
    return global_finish(out)
