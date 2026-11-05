"""Reels-style GoPro underwater grade (3 steps), diver graded apart via a mask.

Step 1  Darken          : exposure down, strongest toward the top of frame
Step 2  Skin tones      : inside the SAM 3.1 diver mask only
Step 3  White/blue sand : green cast -> blue, bright seabed -> near-white,
                          turquoise-blue shadows, more contrast

All functions take float32 RGB in 0..1 (BT.709 gamma-encoded) and are pure
per-pixel maps plus a fixed vertical gradient, so the grade is temporally
stable by construction.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(x):
    return x @ LUMA


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def sat(x, s, y=None):
    y = luma(x)[..., None] if y is None else y
    return y + (x - y) * s


# ------------------------------------------------------------ background --
BG = dict(
    # channel mixer rows (out R, G, B from in R, G, B): green cast -> blue
    mix=np.array([[1.00, 0.00, 0.00],
                  [0.00, 0.86, 0.06],
                  [0.02, 0.00, 1.02]], np.float32),
    # vertical darkening: multiplier at top -> bottom of frame
    grad_top=0.46, grad_bottom=1.0, grad_gamma=0.8,
    shadow_gamma=1.12,       # deepen darks toward navy, without crushing
    sand_lo=0.42, sand_hi=0.80,   # luma window that counts as bright seabed
    sand_white=0.78,         # how far bright sand goes toward neutral white
    sand_lift=1.18,          # exposure on bright sand
    sand_tint=np.array([0.94, 1.00, 1.04], np.float32),  # clean cool white
    shadow_tint=np.array([0.70, 0.96, 1.06], np.float32),  # blue shadows
    water_sat=0.95,
    blue_vibrance=1.30,      # extra saturation for blue/cyan hues only
    caustic_clarity=0.55,    # local contrast on the bright seabed
    caustic_sigma=5.0,
)


def vertical_gradient(h, w, top, bottom, gamma):
    t = np.linspace(0, 1, h, dtype=np.float32) ** gamma
    return (top + (bottom - top) * t)[:, None, None] * np.ones((1, w, 1), np.float32)


def grade_background(x, grad):
    p = BG
    y0 = luma(x)[..., None]
    # bright-seabed key from the *source* luma (stable, before any darkening)
    k_sand = smoothstep(p["sand_lo"], p["sand_hi"], y0)
    # Step 3a: green -> blue
    x = np.clip(x @ p["mix"].T, 0, None)
    # Step 3b: bright sand toward clean cool white
    y = luma(x)[..., None]
    white = y * p["sand_tint"] * p["sand_lift"]
    x = x * (1 - p["sand_white"] * k_sand) + white * (p["sand_white"] * k_sand)
    # Step 1: darken, strongest at the top; the sand keeps more of its light
    g = grad + (1 - grad) * 0.55 * k_sand
    x = x * g
    y = np.clip(luma(x)[..., None], 1e-5, None)
    x = x * (y ** (p["shadow_gamma"] - 1))           # crush the darks
    # cool / blue the shadows and mids (not the sand)
    tint = 1 + (p["shadow_tint"] - 1) * (1 - k_sand)
    x = x * tint
    x = sat(x, p["water_sat"])
    # Selective colour: blues/cyans (B is the top channel) get more vibrance,
    # scaled down where they are already saturated
    yb = luma(x)[..., None]
    mx, mn = x.max(-1, keepdims=True), x.min(-1, keepdims=True)
    chroma = (mx - mn) / np.maximum(mx, 1e-4)
    k_blue = smoothstep(0.0, 0.04, x[..., 2:3] - x[..., 0:1]) * (1 - chroma)
    x = yb + (x - yb) * (1 + (p["blue_vibrance"] - 1) * k_blue)
    # Caustics: local contrast on the bright seabed so the light ripples pop
    if p["caustic_clarity"] > 0:
        import cv2
        yl = luma(x)
        blur = cv2.GaussianBlur(yl, (0, 0), p["caustic_sigma"])
        x = x + ((yl - blur) * p["caustic_clarity"])[..., None] * k_sand
    return np.clip(x, 0, 1)


# ----------------------------------------------------------------- diver --
# The diver is graded only inside the SAM 3.1 mask, so none of the water's
# blue/cyan shift above reaches the skin. Steps:
#   1. per-frame white balance (diver_gains) that undoes the red loss at depth
#   2. skin key: mid-tones of the body (not the bright white shorts, not the
#      dark hair / mask / fins) are pulled toward a healthy warm flesh tone,
#      luminance preserved so the skin keeps its texture and shading
#   3. vibrance on the skin, neutral whites on the shorts
#   4. a gentle share of the scene darkening, so the diver sits in the light
DV = dict(
    gains=np.array([1.48, 1.0, 0.98], np.float32),   # fallback white balance
    # healthy, sun-tanned flesh as R:G:B (luminance comes from the pixel)
    skin_rgb=np.array([1.0, 0.84, 0.76], np.float32),
    skin_pull=0.22,          # 0 = WB only, 1 = flat skin colour
    skin_vibrance=0.92,
    dark_lo=0.10, dark_hi=0.24,    # hair, mask, fins stay as they are
    hi_lo=0.56, hi_hi=0.80,        # white shorts: neutral, no warmth
    exposure=0.88,
    contrast=1.0,
    scene_follow=0.45,
    # share of the ambient water light that falls on the body (cohesion)
    ambient=np.array([0.80, 0.94, 1.06], np.float32), ambient_mix=0.18,
    # Depth adaptation (depth 0 = surface .. 1 = deep, see diver_depth()):
    # deeper, the red is only partly restored, more ambient cyan/blue falls
    # on the skin and the body takes more of the scene darkening.
    depth_wb_keep=0.45,      # share of the WB correction given up at depth 1
    depth_ambient=0.32,      # extra ambient mix at depth 1
    depth_follow=0.30,       # extra scene_follow at depth 1
    depth_wrap=0.25,         # extra light wrap at depth 1       # how much of the background darkening it takes
)

# Average colour the diver should have after white balance (skin + white
# shorts together), as R:G:B with G = 1. Measured per frame inside the mask.
DIVER_TARGET = np.array([1.10, 1.0, 0.93], np.float32)
GAIN_LIMITS = (np.array([1.0, 1.0, 0.75], np.float32), np.array([1.8, 1.0, 1.05], np.float32))


def diver_gains(mean_rgb):
    """White-balance gains that bring the diver's mean colour to
    DIVER_TARGET (green fixed at 1). Clamped so noise never explodes."""
    m = np.maximum(np.asarray(mean_rgb, np.float32), 1e-4)
    g = DIVER_TARGET * m[1] / m
    return np.clip(g, *GAIN_LIMITS)


def diver_depth(mean_rgb):
    """Water between camera/surface and the diver, from the red loss on the
    diver itself (source colours, before any grade). Measured on this clip:
    R/G about 0.85 on the sunlit body at the surface, about 0.55 deep in
    the blue. Returns 0..1."""
    m = np.maximum(np.asarray(mean_rgb, np.float32), 1e-4)
    return float(np.clip((0.85 - m[0] / m[1]) / 0.30, 0, 1))


def grade_diver(x, grad, gains=None, depth=0.0):
    p = DV
    g = p["gains"] if gains is None else gains
    g = 1 + (g - 1) * (1 - p["depth_wb_keep"] * depth)
    x = x * g
    y = luma(x)[..., None]
    k_hi = smoothstep(p["hi_lo"], p["hi_hi"], y)
    k_dark = 1 - smoothstep(p["dark_lo"], p["dark_hi"], y)
    k_skin = (1 - k_hi) * (1 - k_dark)
    # 2. flesh tone with the pixel's own luminance
    skin = y * (p["skin_rgb"] / float(p["skin_rgb"] @ LUMA))
    x = x + (skin - x) * (p["skin_pull"] * k_skin)
    # 3. vibrance on skin; whites (shorts) to a clean, very slightly warm white
    x = sat(x, 1 + (p["skin_vibrance"] - 1) * k_skin)
    white = y * np.array([1.0, 0.985, 0.955], np.float32)
    x = x + (white - x) * (0.7 * k_hi)
    x = (x - 0.45) * p["contrast"] + 0.45
    # 4. sit in the scene: some of the cool ambient water light on the body,
    #    and a share of the scene darkening
    amb = p["ambient_mix"] + p["depth_ambient"] * depth
    x = x * (1 + (p["ambient"] - 1) * amb)
    fol = p["scene_follow"] + p["depth_follow"] * depth
    x = x * p["exposure"] * (1 - fol + fol * grad)
    return np.clip(x, 0, 1)


# -------------------------------------------------------------- finishing --
def finish(x):
    """Soft S-curve, then a navy toe: blacks are lifted a little toward deep
    blue so the water column keeps its gradient instead of going to 0."""
    x = np.clip(x, 0, 1)
    s = x * x * (3 - 2 * x)
    x = x * 0.82 + s * 0.18
    toe = np.array([0.010, 0.022, 0.034], np.float32)
    x = toe + x * (1 - toe)
    return np.clip(x, 0, 1)


def grade_frame(rgb, alpha, grad, gains=None, depth=0.0):
    """rgb float 0..1 HxWx3, alpha float 0..1 HxW (diver), grad HxWx1,
    gains: per-frame diver white balance (diver_gains), None = DV["gains"]."""
    import cv2

    bg = grade_background(rgb, grad)
    dv = grade_diver(rgb, grad, gains, depth)
    a = alpha[..., None]
    comp = bg * (1 - a) + dv * a
    if alpha.max() > 0:
        # light wrap: the blurred background light spills a little over the
        # diver's inner edge, so the body sits in the water, not on top of it
        ab = cv2.GaussianBlur(alpha, (0, 0), 4.0)
        ring = np.clip((alpha - ab) * 2.5, 0, 1)[..., None]
        bgb = cv2.GaussianBlur(bg, (0, 0), 6.0)
        comp = comp + (bgb - comp) * ((0.35 + DV["depth_wrap"] * depth) * ring)
    return finish(comp)
