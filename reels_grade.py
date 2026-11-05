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
#   2. skin key: the body minus the shorts (their own SAM 3.1 mask), minus
#      the dark hair / mask / fins and specular highlights, is pulled toward
#      the golden-tan flesh tone measured on the reference stills
#      user-picked look "C - golden warm" (more yellow-gold and a little
#      richer than the reference's ~1 : 0.81 : 0.70), luminance preserved so the
#      skin keeps its texture and shading
#   3. the shorts go to a clean, very slightly warm white
#   4. depth: a soft falloff - the skin darkens a little and takes a small
#      share of the ambient blue/cyan light, so it cools without going grey
DV = dict(
    gains=np.array([1.48, 1.0, 0.98], np.float32),   # fallback white balance
    # healthy, sun-tanned flesh as R:G:B (luminance comes from the pixel)
    skin_rgb=np.array([1.0, 0.72, 0.56], np.float32),
    skin_pull=0.72,          # 0 = WB only, 1 = flat skin colour
    skin_vibrance=1.0,
    dark_lo=0.10, dark_hi=0.24,    # hair, mask, fins stay as they are
    # ... and relative to this frame's skin luminance (black fins look grey
    # in the water, so an absolute threshold lets them into the skin key)
    dark_rel=(0.50, 0.68),
    # hair: natural dark brown (R:G:B), how far it is pulled there
    hair_rgb=np.array([1.0, 0.86, 0.76], np.float32), hair_pull=0.7,
    hi_lo=0.84, hi_hi=0.97,        # specular highlights stay neutral
    # fallback shorts key when there is no shorts mask (luminance)
    shorts_lo=0.56, shorts_hi=0.80,
    exposure=1.0,
    contrast=1.0,
    scene_follow=0.35,
    # share of the ambient water light that falls on the body (cohesion)
    ambient=np.array([0.80, 0.94, 1.06], np.float32), ambient_mix=0.06,
    # Depth adaptation (depth 0 = surface .. 1 = deep, see diver_depth()):
    # deeper, the red is only partly restored, more ambient cyan/blue falls
    # on the skin and the body takes more of the scene darkening.
    depth_wb_keep=0.25,      # share of the WB correction given up at depth 1
    depth_ambient=0.20,      # extra ambient mix at depth 1
    depth_follow=0.20,       # extra scene_follow at depth 1
    depth_wrap=0.25,         # extra light wrap at depth 1
    # Vectorscope lock (BT.709 Cb/Cr plane, angle from +Cb): the skin's hue
    # is held on the skin-tone line, so no water cyan/blue can reach it.
    # The reference stills sit at 128-132 deg (classic line ~123 deg).
    skin_hue=131.0, skin_hue_tol=5.0,
    # chroma/luma kept inside the references' spread (median 0.14,
    # IQR 0.09-0.19); depth lowers it (the only depth cooling on the skin)
    skin_chroma=(0.055, 0.23), depth_chroma=0.20,
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


def skin_line_lock(x, depth=0.0):
    """Rotate every pixel's chroma onto the skin-tone line (within
    +-skin_hue_tol) and keep its chroma/luma inside the reference range,
    reduced with depth. Luma is untouched."""
    p = DV
    y = luma(x)
    cb = (x[..., 2] - y) / 1.8556
    cr = (x[..., 0] - y) / 1.5748
    ang = np.degrees(np.arctan2(cr, cb))
    d = (ang - p["skin_hue"] + 180.0) % 360.0 - 180.0
    ang = np.radians(p["skin_hue"] + np.clip(d, -p["skin_hue_tol"], p["skin_hue_tol"]))
    lo, hi = p["skin_chroma"]
    f = 1 - p["depth_chroma"] * depth
    mag = np.clip(np.hypot(cb, cr) / np.maximum(y, 1e-4), lo * f, hi * f) * y
    cb, cr = mag * np.cos(ang), mag * np.sin(ang)
    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.stack([r, g, b], -1).astype(np.float32)


def class_weights(y, parts=None, skin_luma=None):
    """Per-pixel weights of the diver's classes, summing to 1:
    skin, hair, white (shorts + speculars), gear (fins, dive mask).
    `parts` holds SAM 3.1 part alphas (HxW, 0..1) by name; a class without
    a mask falls back to a luminance key (only on pixels no mask claims)."""
    p = DV
    parts = parts or {}
    z = np.zeros_like(y)

    def part(name):
        return parts[name][..., None] if name in parts else z

    hair = part("hair")
    # bubbles in front of the body are graded like the white fabric: neutral
    shorts = np.maximum(part("shorts"), part("bubbles"))
    gear = np.maximum(part("fins"), part("goggles"))
    claimed = np.clip(hair + shorts + gear, 0, 1)
    free = 1 - claimed
    # fallbacks for whatever no mask claims (a class missing in a chunk)
    k_hi = smoothstep(p["hi_lo"], p["hi_hi"], y)
    k_dark = 1 - smoothstep(p["dark_lo"], p["dark_hi"], y)
    if skin_luma is not None:
        lo, hi = p["dark_rel"]
        k_dark = np.maximum(k_dark, 1 - smoothstep(lo * skin_luma, hi * skin_luma, y))
    if "shorts" not in parts:
        k_hi = np.maximum(k_hi, smoothstep(p["shorts_lo"], p["shorts_hi"], y))
    w_white = shorts + free * k_hi
    w_gear = gear + free * (1 - k_hi) * k_dark
    w_hair = hair
    w_skin = np.clip(1 - w_white - w_gear - w_hair, 0, 1)
    return w_skin, w_hair, w_white, w_gear


def grade_diver(x, grad, gains=None, depth=0.0, parts=None, skin_luma=None):
    """Multi-class diver grade. Every class gets its own treatment from the
    same white-balanced pixel, and the results are blended with weights
    that sum to 1, so no class's colour spills into a neighbour."""
    p = DV
    g = p["gains"] if gains is None else gains
    g = 1 + (g - 1) * (1 - p["depth_wb_keep"] * depth)
    x = x * g
    y = luma(x)[..., None]
    w_skin, w_hair, w_white, w_gear = class_weights(y, parts, skin_luma)

    # skin: golden tan with the pixel's own luminance, locked to the
    # vectorscope skin line; depth lowers only its chroma
    skin = y * (p["skin_rgb"] / float(p["skin_rgb"] @ LUMA))
    xs = x + (skin - x) * p["skin_pull"]
    xs = sat(xs, p["skin_vibrance"])
    xs = skin_line_lock(xs, depth)
    # hair: natural dark brown, contrast intact, never the skin's warmth
    xh = y * (p["hair_rgb"] / float(p["hair_rgb"] @ LUMA))
    xh = x + (xh - x) * p["hair_pull"]
    # shorts and speculars: their true white, luminance contrast intact
    xw = y * np.array([1.0, 0.985, 0.955], np.float32)
    # fins and dive mask: neutral black rubber / glass
    xg = y * np.array([1.0, 1.0, 1.02], np.float32)
    x = xs * w_skin + xh * w_hair + xw * w_white + xg * w_gear

    x = (x - 0.45) * p["contrast"] + 0.45
    # sit in the scene: ambient water light on everything but the skin
    # (the skin's depth falloff is its chroma), and a share of the scene
    # darkening for the whole body
    amb = (p["ambient_mix"] + p["depth_ambient"] * depth) * (1 - w_skin)
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


def grade_frame(rgb, alpha, grad, gains=None, depth=0.0, parts=None):
    """rgb float 0..1 HxWx3, alpha float 0..1 HxW (diver), grad HxWx1,
    gains: per-frame diver white balance (diver_gains), None = DV["gains"],
    parts: {name: HxW 0..1 alpha} SAM 3.1 part masks (hair, shorts, fins,
    goggles); classes without a mask fall back to luminance keys."""
    import cv2

    bg = grade_background(rgb, grad)
    # this frame's skin luminance: body core, minus shorts and specular
    g0 = DV["gains"] if gains is None else gains
    yb = luma(rgb * (1 + (g0 - 1) * (1 - DV["depth_wb_keep"] * depth)))
    body = alpha > 0.9
    for m in (parts or {}).values():
        body &= m < 0.5
    if not parts or "shorts" not in parts:
        body &= yb < DV["shorts_lo"]
    body &= yb > DV["dark_lo"]
    skin_luma = float(np.percentile(yb[body], 70)) if body.sum() > 300 else None
    dv = grade_diver(rgb, grad, gains, depth, parts, skin_luma)
    a = alpha[..., None]
    comp = bg * (1 - a) + dv * a
    if alpha.max() > 0:
        # light wrap, luminance only: the surrounding water's *light* reaches
        # the diver's inner edge (no pasted-on look) but none of its cyan
        # colour does, so the skin edge stays on the skin-tone line
        ab = cv2.GaussianBlur(alpha, (0, 0), 3.0)
        ring = np.clip((alpha - ab) * 2.5, 0, 1)
        lb = cv2.GaussianBlur(luma(bg), (0, 0), 6.0)
        w = (0.30 + DV["depth_wrap"] * depth) * ring
        comp = comp + ((lb - luma(comp)) * w)[..., None]
    return finish(comp)
