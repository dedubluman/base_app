"""Underwater grade v3: one global (blue water) grade for everything, skin-only warm correction via a skin matte,
and full-frame finishing (veil, light wrap, elliptical vignette) applied after compositing.

float32 RGB 0..1, display-referred Rec.709. No stabilisation / denoise.

 layer A  (water + every non-skin part of the diver): monotone per-channel tone curves (fitted by quantile matching),
          saturation/hue, large-radius clarity, optional vertical depth falloff, slow chroma anchor
 layer B  (skin only, weighted by the skin matte): A, then a small chroma/luma pull to the measured reference skin
 finish   veil (subject shares the water haze), light wrap on the silhouette edge, elliptical vignette
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

KX = np.array([0.0, 0.06, 0.14, 0.25, 0.40, 0.58, 0.78, 1.0])   # tone-curve knot positions (input)
CH = "rgb"

DEFAULT = {
    "bg": {
        "knots_y": [[0, .002, .01, .03, .09, .22, .5, .9], [0, .01, .04, .1, .2, .35, .6, .9], [0, .03, .09, .18, .3, .5, .75, 1.0]],
        "sat": 1.0, "hue_deg": 0.0, "clarity_amt": 0.15, "clarity_sigma_frac": 0.04, "vgrad_stops": 0.0,
        "anchor_ab": [-3.1, -28.05], "anchor_strength": 1.0, "anchor_alpha": 0.03, "anchor_clip": 10.0,
    },
    "finish": {"vignette": 0.3, "vignette_pow": 2.2, "veil": 0.05, "veil_small_extra": 0.08, "wrap_px": 3.0, "wrap_amt": 0.25,
               "final_s": 0.0},
    "skin": {"skin_ab": [8.0, 16.0], "ab_strength": 0.9, "dL": 0.0, "L_restore": 0.0, "clarity_amt": 0.10,
             "clf_w": None, "clf_b": 0.0, "sharp": [0.5, 0.85], "matte_blur_px": 1.5},
}


# ----------------------------------------------------------------------------- colour helpers
def to_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2LAB)


def from_lab(lab):
    return np.clip(cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def hsl_stats(rgb_px):
    """rgb_px (N,3) 0..1 -> mean hue (deg, circular), mean saturation (HSL), mean lightness."""
    px = np.asarray(rgb_px, np.float32)
    mx, mn = px.max(1), px.min(1)
    l = (mx + mn) / 2
    d = mx - mn
    s = np.where(d < 1e-6, 0, d / np.maximum(1 - np.abs(2 * l - 1), 1e-6))
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    h = np.where(d < 1e-6, 0, np.where(mx == r, ((g - b) / np.maximum(d, 1e-6)) % 6, np.where(mx == g, (b - r) / np.maximum(d, 1e-6) + 2, (r - g) / np.maximum(d, 1e-6) + 4))) * 60
    ang = np.deg2rad(h)
    hue = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
    return float(hue), float(s.mean()), float(l.mean())


def hue_diff(a, b):
    return (a - b + 180) % 360 - 180


# ----------------------------------------------------------------------------- tone curves
def make_luts(B):
    xs = np.linspace(0, 1, 1024)
    return np.clip(np.array([PchipInterpolator(KX, np.maximum.accumulate(np.array(B["knots_y"][c], float)))(xs) for c in range(3)], np.float32), 0, 1)


def apply_luts(rgb, luts):
    idx = np.clip(rgb * 1023 + 0.5, 0, 1023).astype(np.int32)
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = luts[c][idx[..., c]]
    return out


def local_contrast(lab, amt, sigma_frac):
    if amt <= 0:
        return lab
    h, w = lab.shape[:2]
    L = lab[..., 0]
    lab = lab.copy()
    lab[..., 0] = np.clip(L + amt * (L - cv2.GaussianBlur(L, (0, 0), max(2.0, sigma_frac * max(h, w)))), 0, 100)
    return lab


def global_grade(rgb, B, luts=None):
    """Layer A: blue water grade + tone curve. Applied to the WHOLE frame."""
    luts = make_luts(B) if luts is None else luts
    y = apply_luts(rgb, luts)
    if B.get("vgrad_stops", 0):
        t = np.linspace(0, 1, rgb.shape[0], dtype=np.float32)
        y = y * (2.0 ** (B["vgrad_stops"] * (0.5 - t)))[:, None, None]
    lab = to_lab(np.clip(y, 0, 1))
    hr = np.deg2rad(B["hue_deg"])
    c, s = np.cos(hr), np.sin(hr)
    A, Bb = lab[..., 1] * B["sat"], lab[..., 2] * B["sat"]
    lab[..., 1], lab[..., 2] = A * c - Bb * s, A * s + Bb * c
    lab = local_contrast(lab, B["clarity_amt"], B["clarity_sigma_frac"])
    return from_lab(lab)


def vignette_map(h, w, strength, power):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)   # elliptical: 0 centre .. 1 corner
    return (1.0 - strength * np.clip(r, 0, 1) ** power).astype(np.float32)


# ----------------------------------------------------------------------------- skin matte (colour qualifier on the RAW frame)
def water_lab(lab, mask_bool):
    """Median Lab of the water ring around the diver (everything outside the dilated mask) of a (cropped) Lab image."""
    ring = cv2.dilate(mask_bool.astype(np.uint8), np.ones((13, 13), np.uint8)) == 0
    px = lab[ring] if ring.sum() > 50 else lab.reshape(-1, 3)
    return np.median(px, axis=0)


def skin_features(lab, wl):
    """Features relative to the local water colour: skin is lighter than the water, fins/shorts/hair darker, and the
    ab offset from the water separates warm skin from neutral/blue dark parts at every depth."""
    dL, da, db = lab[..., 0] - wl[0], lab[..., 1] - wl[1], lab[..., 2] - wl[2]
    return np.stack([dL / 20, da / 10, db / 20, (-da) / np.maximum(db, 3.0)], -1)


def skin_matte(raw_rgb, mask, S):
    """Soft skin matte inside the diver mask. Logistic qualifier (luma + hue + chroma gates relative to the local water,
    learned on SAM-labelled skin vs shorts/fins/hair pixels), slightly blurred, always limited to the diver mask."""
    H, W = mask.shape
    out = np.zeros((H, W), np.float32)
    if S["clf_w"] is None or mask.max() <= 0.004:
        return out
    ys, xs = np.where(mask > 0.004)
    y0, y1, x0, x1 = max(ys.min() - 40, 0), min(ys.max() + 41, H), max(xs.min() - 40, 0), min(xs.max() + 41, W)
    lab = to_lab(raw_rgb[y0:y1, x0:x1])
    mc = mask[y0:y1, x0:x1]
    z = skin_features(lab, water_lab(lab, mc > 0.004)) @ np.array(S["clf_w"], np.float32) + S["clf_b"]
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    t0, t1 = S.get("sharp", [0.0, 1.0])                  # sharpen the qualifier: below t0 -> 0, above t1 -> 1 (smoothstep between)
    u = np.clip((p - t0) / (t1 - t0), 0, 1)
    p = cv2.GaussianBlur((u * u * (3 - 2 * u)).astype(np.float32), (0, 0), S["matte_blur_px"])
    out[y0:y1, x0:x1] = p * np.clip(mc, 0, 1)
    return out


# ----------------------------------------------------------------------------- full frame render
def render(rgb, mask, skin, params, vig=None, dab=None, luts=None, skin_on=True, finish=True, A=None):
    """rgb raw frame; mask = soft diver mask (0..1); skin = soft skin matte (0..1, inside the mask)."""
    B, F, S = params["bg"], params["finish"], params["skin"]
    H, W = rgb.shape[:2]
    if A is None:
        A = global_grade(rgb, B, luts)
    if A is not None and dab is not None and not getattr(A, "_dab_done", False):
        lab = to_lab(A)
        lab[..., 1] += dab[0]
        lab[..., 2] += dab[1]
        A = from_lab(lab)
    out = A
    if skin_on and skin is not None and skin.max() > 0.004:
        ys, xs = np.where(skin > 0.004)
        y0, y1, x0, x1 = max(ys.min() - 2, 0), min(ys.max() + 3, H), max(xs.min() - 2, 0), min(xs.max() + 3, W)
        lab = to_lab(A[y0:y1, x0:x1])
        s = skin[y0:y1, x0:x1]
        k = S["ab_strength"] * s
        lab[..., 1] += k * (S["skin_ab"][0] - lab[..., 1])
        lab[..., 2] += k * (S["skin_ab"][1] - lab[..., 2])
        Lraw = to_lab(rgb[y0:y1, x0:x1])[..., 0]       # skin keeps (part of) its raw lighting: bright, separated from the dark water
        lab[..., 0] = np.clip(lab[..., 0] + s * (S["L_restore"] * (Lraw - lab[..., 0]) + S["dL"]), 0, 100)
        lab = local_contrast(lab, S["clarity_amt"], 0.03)
        Bk = from_lab(lab)
        out = A.copy()
        out[y0:y1, x0:x1] = A[y0:y1, x0:x1] * (1 - s[..., None]) + Bk * s[..., None]
    if finish:
        if mask is not None and mask.max() > 0.004:
            ys, xs = np.where(mask > 0.004)
            y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 9, H), max(xs.min() - 8, 0), min(xs.max() + 9, W)
            m = mask[y0:y1, x0:x1]
            water = cv2.GaussianBlur(A, (0, 0), 40)[y0:y1, x0:x1]
            area_rel = float((mask > 0.5).sum()) / 24000.0
            veil = F["veil"] + F["veil_small_extra"] * float(np.clip(1.0 - area_rel, 0, 1))   # smaller / deeper diver -> more haze
            dt = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
            edge = np.clip(1.0 - dt / max(F["wrap_px"], 1e-3), 0, 1) * (m > 0.5)
            w = np.clip(veil * m + F["wrap_amt"] * edge, 0, 0.6)[..., None]
            out = out.copy()
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - w) + water * w
        if vig is None:
            vig = vignette_map(H, W, F["vignette"], F["vignette_pow"])
        out = out * vig[..., None]
        if F.get("final_s", 0):
            out = out + F["final_s"] * (out * out * (3 - 2 * out) - out)      # mild S-curve (contrast), 0 = off
    return np.clip(out, 0, 1)
