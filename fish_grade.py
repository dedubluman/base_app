#!/usr/bin/env python3
"""Mask-driven underwater grade locked to reference stills -> H.264 1080p.

    python fish_grade.py calibrate --input inputs/fish.mov --refs REF1.jpg REF2.jpg
    python fish_grade.py stills    --input inputs/fish.mov --frames 0,60,120,... --out DIR
    python fish_grade.py render    --input inputs/fish.mov --output output/fish_graded.mp4

Layers (BT.709 display-referred RGB, float32, all colour work in OKLab/OKLCH)

    fish        matte masks/<stem>__mattes/fish        yellow/tan -> amber/burnt-orange/crimson,
                                                       espresso bars, cooled silver belly
    seabed      matte .../seabed (+ midground band)    sand bed and immediate grass/debris:
                                                       tone + tint calibrated to the reference
    background  matte .../background                   rock / dense weed / open water: heavier
                                                       pull-down, teal-slate / olive-charcoal
    atmosphere  static elliptical vignette + top-down falloff + a light pocket that follows the
                fish (smoothed centroid), luma-dependent organic grain

`calibrate` measures the reference stills once and writes grade_params.json:
tone curves (quantile matching of the clip's per-region luma onto the reference's),
per-tonal-zone colour offsets, fish chroma range and the vignette depth. `render` and
`stills` only read that file, so every frame uses the same constants (no flicker).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from fish_color import (gblur, hex_to_lch, lab_to_lch, lch_to_lab, luma, oklab_to_rgb, rgb_to_oklab,
                        smoothstep)
from fish_refine import decode_source

REPO = Path(__file__).resolve().parent
PARAMS = REPO / "grade_params.json"

# ------------------------------------------------------------------ the look
# Hand-set creative parameters (everything measured lives in grade_params.json).
LOOK = {
    # --- fish (OKLCH anchors measured on the reference: crimson h~34, burnt h~45, amber h~58)
    "fish_hue_amber": 60.0, "fish_hue_crimson": 33.0,
    "fish_L_tab": [0.10, 0.14, 0.215, 0.285, 0.36, 0.44, 0.52, 0.60, 0.70, 0.85],
    "fish_hue_tab": [38.0, 38.0, 44.0, 48.0, 47.0, 47.0, 54.0, 63.0, 72.0, 76.0],
    "fish_chroma_tab": [0.040, 0.050, 0.051, 0.064, 0.082, 0.107, 0.121, 0.123, 0.117, 0.100],
    "fish_warm_shift": 12.0, "fish_chroma_bar": 0.036,
    "crimson_depth": 0.16, "fish_hue_texture": 6.0,
    "fish_warm_lo": 0.30, "fish_warm_hi": 0.88,            # chroma percentile window -> crimson weight
    "belly_max": 0.72, "belly_hue": 245.0, "belly_chroma": 0.020, "belly_cap": 0.60,
    "fish_detail": 1.10,                                     # scale/fin-ray micro contrast
    "fish_exposure_clamp": (0.75, 1.6),
    # --- environment
    "env_lift": {"bg": 1.42, "fg": 1.22},                    # compensates the spatial falloff in the centre
    "env_norm": 0.5, "env_gain_clamp": (0.80, 1.30),         # partial per-frame exposure normalisation
    "wb_strength": 0.8,                                      # sand -> neutral before the tint
    "bg_chroma_scale": 0.80, "fg_chroma_scale": 0.85,
    "bg_clarity": 0.45, "fg_clarity": 0.20,                  # mid-frequency (sigma 12 px @1080) detail
    "particle_gain": 0.85,                                   # bright small features survive the pull-down
    "black_floor": 0.020,
    # --- atmosphere
    "vig_strength": 0.58, "vig_power": 2.1, "vig_aspect": 0.80,
    "top_fall": 0.30, "top_fall_h": 0.42,                    # extra darkening toward the top/back
    "pool_floor": 0.70, "pool_sigma": (0.60, 0.34),          # light pocket around the fish (x, y / frame)
    "grain": 0.011,
}


# ----------------------------------------------------------- reference stats
def ref_regions(path: Path):
    """OKLab pixels of the reference split into bg / fg (central columns, fish removed)."""
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    rgb = im[..., ::-1].astype(np.float32) / 255
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL).astype(np.float32) / 255
    hue = hsv[..., 0] * 360
    warm = (hsv[..., 1] > 0.35) & ((hue < 55) | (hue > 340)) & (hsv[..., 2] > 0.12)
    warm = cv2.dilate(warm.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    lab = rgb_to_oklab(rgb)
    cx = np.zeros((h, w), bool)
    cx[:, int(0.2 * w):int(0.8 * w)] = True
    bg = cx & ~warm
    bg[int(0.30 * h):] = False
    fg = cx & ~warm
    fg[:int(0.58 * h)] = False
    fg[int(0.88 * h):] = False
    return lab, bg, fg, warm


def fish_ref_lab(path: Path):
    im = cv2.imread(str(path))
    rgb = im[..., ::-1].astype(np.float32) / 255
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV_FULL).astype(np.float32) / 255
    hue = hsv[..., 0] * 360
    m = (hsv[..., 1] > 0.45) & (hsv[..., 2] > 0.12) & ((hue < 50) | (hue > 340))
    h, w = m.shape
    m[: int(0.25 * h)] = False                    # the dry-grass tips at the top are not fish
    return rgb_to_oklab(rgb), m


QS = np.array([0.5, 3, 10, 25, 40, 55, 70, 85, 95, 99.5])


def tone_curve(raw_L: np.ndarray, ref_L: np.ndarray, hi_extra=0.08):
    """Quantile-matching curve raw L -> reference L as a 1024-entry monotone table."""
    xr = np.percentile(raw_L, QS)
    yr = np.percentile(ref_L, QS)
    xs = np.concatenate([[0.0], xr, [1.0]])
    ys = np.concatenate([[0.0], yr, [min(1.0, yr[-1] + hi_extra)]])
    # strictly increasing x, non-decreasing y
    keep = np.concatenate([[True], np.diff(xs) > 1e-4])
    xs, ys = xs[keep], np.maximum.accumulate(ys[keep])
    t = np.linspace(0, 1, 1024)
    return PchipInterpolator(xs, ys)(t).clip(0, 1).tolist()


def tone_curve_env(raw_L: np.ndarray, ref_L: np.ndarray, slope_range=(0.65, 1.45), k=22.0):
    """Environment luma curve: reference median placed exactly, slope from the IQR ratio but capped
    (a narrow raw distribution, e.g. flat bright sand, must not be stretched over the reference's
    lighting gradient - that is what the vignette / light pocket are for), smooth toe and shoulder."""
    m_in, m_out = np.median(raw_L), np.median(ref_L)
    iqr = lambda x: np.subtract(*np.percentile(x, [75, 25]))
    sl = float(np.clip(iqr(ref_L) / max(iqr(raw_L), 1e-4), *slope_range))
    lo, hi = float(np.percentile(ref_L, 0.5)), float(np.percentile(ref_L, 99.5))
    x = np.linspace(0, 1, 1024)
    y = m_out + sl * (x - m_in)
    y = lo + np.log1p(np.exp(k * (y - lo))) / k                   # soft floor at the reference's black
    y = hi - np.log1p(np.exp(k * (hi - y))) / k                   # soft ceiling at the reference's white
    y = np.maximum.accumulate(y)
    y = y - (y[0] - 0.0) * (1 - x) ** 3                           # pin 0 -> 0 without a visible kink
    return np.clip(y, 0, 1).tolist(), sl


def apply_lut(x, table):
    t = np.asarray(table, np.float32)
    return np.interp(x, np.linspace(0, 1, len(t)), t).astype(np.float32)


def zone_weights(L):
    ws = 1 - smoothstep(0.20, 0.38, L)
    wh = smoothstep(0.48, 0.68, L)
    return ws, 1 - ws - wh, wh


def calibrate(a):
    src = decode_source(a.input)
    n, H, W, _ = src.shape
    n = min(n, a.limit or n)
    root = REPO / "masks" / f"{a.input.stem}__mattes"
    rd = lambda nm, t: cv2.imread(str(root / nm / f"{t:05d}.png"), 0).astype(np.float32) / 255
    frames = list(range(0, n, a.step))
    cx = np.zeros((H, W), bool)
    cx[:, int(0.2 * W):int(0.8 * W)] = True

    # --- pass 1: white balance gains from the seabed
    sand_px = []
    for t in frames:
        m = (rd("seabed", t) > 0.95) & (rd("fish", t) < 0.01) & cx
        px = src[t][m][::40].astype(np.float32) / 255
        if len(px):
            sand_px.append(px)
    med = np.median(np.concatenate(sand_px), 0)
    g = (med.mean() / med) ** LOOK["wb_strength"]
    g = g / (g @ np.array([0.2126, 0.7152, 0.0722], np.float32))     # keep luma
    print(f"[cal] sand median RGB {np.round(med * 255)}  WB gains {np.round(g, 3)}")

    # --- pass 2: raw per-region OKLab samples (after WB)
    raw = {"bg": [], "fg": [], "fish": []}
    fish_all = []
    for t in frames:
        fr = np.clip(src[t].astype(np.float32) / 255 * g, 0, 1)
        sel = {"bg": (rd("background", t) > 0.95) & cx,
               "fg": (rd("seabed", t) > 0.95) & (rd("fish", t) < 0.01) & cx}
        for k, m in sel.items():
            if m.any():
                raw[k].append(rgb_to_oklab(fr[m][::30][None])[0])
        af = rd("fish", t) > 0.9
        if af.sum() > 200:
            fish_all.append(rgb_to_oklab((src[t][af].astype(np.float32) / 255)[None])[0])
    raw = {k: np.concatenate(v) for k, v in raw.items() if v}
    fish_raw = np.concatenate(fish_all)

    # --- reference samples
    refs = {"bg": [], "fg": []}
    fref = []
    for p in a.refs:
        lab, bg, fg, warm = ref_regions(Path(p))
        refs["bg"].append(lab[bg][::6])
        refs["fg"].append(lab[fg][::6])
        flab, fm = fish_ref_lab(Path(p))
        fref.append(flab[fm])
    refs = {k: np.concatenate(v) for k, v in refs.items()}
    fref = np.concatenate(fref)

    P = {"wb_gains": g.tolist(), "tone": {}, "tint": {}, "refs": [str(p) for p in a.refs]}
    for k in ("bg", "fg"):
        P["tone"][k], slope = tone_curve_env(raw[k][:, 0], refs[k][:, 0])
        print(f"[cal] {k}: L raw p5/25/50/75/95 {np.round(np.percentile(raw[k][:, 0], [5, 25, 50, 75, 95]), 2)}  "
              f"ref {np.round(np.percentile(refs[k][:, 0], [5, 25, 50, 75, 95]), 2)}  mid slope {slope:.2f}")
        # zone colour offsets on the tone-mapped raw vs the reference
        L_after = apply_lut(raw[k][:, 0], P["tone"][k])
        cs = LOOK[f"{k}_chroma_scale"]
        P["tint"][k] = {}
        wr, wf = zone_weights(L_after), zone_weights(refs[k][:, 0])
        for zi, zn in enumerate(("shadow", "mid", "high")):
            w_raw, w_ref = wr[zi], wf[zi]
            if w_raw.sum() < 50 or w_ref.sum() < 50:
                P["tint"][k][zn] = [0.0, 0.0]
                continue
            mr = (raw[k][:, 1:] * w_raw[:, None]).sum(0) / w_raw.sum()
            mf = (refs[k][:, 1:] * w_ref[:, None]).sum(0) / w_ref.sum()
            P["tint"][k][zn] = (mf - cs * mr).tolist()
        print(f"[cal] {k}: raw L q50 {np.percentile(raw[k][:, 0], 50):.3f} -> ref {np.percentile(refs[k][:, 0], 50):.3f}; "
              f"tint {{{', '.join(f'{z}: ' + str(np.round(v, 3)) for z, v in P['tint'][k].items())}}}")
    # fish constants
    _, Cr, hr = lab_to_lch(fish_raw)
    P["fish"] = {"c_lo": float(np.percentile(Cr, 15)), "c_hi": float(np.percentile(Cr, 97)),
                 "raw_L_median": float(np.median(fish_raw[:, 0])),
                 "raw_L_q": np.percentile(fish_raw[:, 0], [2, 10, 25, 50, 75, 90, 98]).tolist(),
                 "ref_L_q": np.percentile(fref[:, 0], [2, 10, 25, 50, 75, 90, 98]).tolist(),
                 "raw_hue_q": np.percentile(hr, [5, 50, 95]).tolist()}
    Lraw = fish_raw[:, 0]
    belly = (Lraw > 0.27) & (Cr < P["fish"]["c_lo"] + 0.17 * (P["fish"]["c_hi"] - P["fish"]["c_lo"]))
    body = ~belly
    P["fish"]["tone"] = tone_curve(Lraw[body], fref[:, 0], hi_extra=0.04)
    P["fish"]["belly_frac"] = float(belly.mean())
    print(f"[cal] fish: raw hue q5/50/95 {np.round(P['fish']['raw_hue_q'])}  chroma window {P['fish']['c_lo']:.3f}-{P['fish']['c_hi']:.3f}; "
          f"L q raw {np.round(P['fish']['raw_L_q'], 2)} ref {np.round(P['fish']['ref_L_q'], 2)}")
    # vignette depth: reference corner/centre vs the raw clip's
    def cc(L):
        h, w = L.shape
        c = L[h // 3:2 * h // 3, w // 4:3 * w // 4].mean()
        e = np.mean([L[:h // 8, :w // 6].mean(), L[:h // 8, -w // 6:].mean(),
                     L[-h // 8:, :w // 6].mean(), L[-h // 8:, -w // 6:].mean()])
        return e / c
    ratios = []
    for p in a.refs:
        im = cv2.imread(str(p))[..., ::-1].astype(np.float32) / 255
        ratios.append(cc(luma(im)))
    rr = [cc(luma(src[t].astype(np.float32) / 255)) for t in frames[::4]]
    P["vignette"] = {"ref_corner_over_centre": float(np.mean(ratios)), "raw_corner_over_centre": float(np.mean(rr))}
    print(f"[cal] corner/centre luma: reference {np.mean(ratios):.2f}  raw clip {np.mean(rr):.2f}")
    PARAMS.write_text(json.dumps(P, indent=1))
    print(f"[cal] wrote {PARAMS}")


# ------------------------------------------------------------------- grading
class Grader:
    def __init__(self, P, n_frames, H, W, root: Path, centroids=None, fish_gain=None, env_gain=None):
        self.P, self.H, self.W, self.root = P, H, W, root
        self.env_gain = env_gain
        self.g = np.array(P["wb_gains"], np.float32)
        self.cent = centroids
        self.fish_gain = fish_gain
        ys = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        xs = np.linspace(-1, 1, W, dtype=np.float32)[None]
        # static vignette: elliptical, matched to the portrait frame
        r = np.sqrt((xs * LOOK["vig_aspect"]) ** 2 + (ys * 0.62) ** 2)
        self.vig = (1 - LOOK["vig_strength"] * np.clip(r, 0, 1.2) ** LOOK["vig_power"]).astype(np.float32)
        yn = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
        self.top = (1 - LOOK["top_fall"] * (1 - smoothstep(0, LOOK["top_fall_h"], yn))).astype(np.float32)
        self.xn = np.linspace(0, 1, W, dtype=np.float32)[None] * np.ones((H, 1), np.float32)
        self.yn = yn

    def read(self, kind, t):
        return cv2.imread(str(self.root / kind / f"{t:05d}.png"), 0).astype(np.float32) / 255

    # ---------------------------------------------------------- environment
    def tone(self, L, fg):
        """Reference-matched luma curve of the region (fg = 0 background, 1 seabed)."""
        P = self.P
        return apply_lut(L, P["tone"]["bg"]) * (1 - fg) + apply_lut(L, P["tone"]["fg"]) * fg

    def lift_mid(self, Lo, fg):
        """Centre-exposure lift for the mid-tones only: it fades out toward the reference's white point,
        so sunlit sand ridges keep the ceiling measured on the reference."""
        lift = LOOK["env_lift"]["bg"] * (1 - fg) + LOOK["env_lift"]["fg"] * fg
        return Lo * (1 + (lift - 1) * (1 - smoothstep(0.28, 0.52, Lo)))

    def env_colour(self, lab_ab, Lo, fg):
        """Point-wise colour of the environment: chroma scale + tonal-zone offsets measured on the
        reference (teal-slate shadows, olive-charcoal mids, bone-olive lights)."""
        P = self.P
        ws, wm, wh = zone_weights(Lo)
        ab = lab_ab * (LOOK["bg_chroma_scale"] * (1 - fg) + LOOK["fg_chroma_scale"] * fg)[..., None]
        for k, wk in (("bg", 1 - fg), ("fg", fg)):
            tint = P["tint"][k]
            off = (ws[..., None] * np.array(tint["shadow"], np.float32) + wm[..., None] * np.array(tint["mid"], np.float32)
                   + wh[..., None] * np.array(tint["high"], np.float32))
            ab = ab + (wk[..., None] * off)
        # darker pixels carry less of the source chroma (clean teal/olive blacks, no coloured noise)
        ab = ab * (0.55 + 0.45 * smoothstep(0.05, 0.30, Lo))[..., None]
        out = oklab_to_rgb(np.concatenate([Lo[..., None], ab], -1))
        f = LOOK["black_floor"]
        return np.clip(f + (1 - f) * out, 0, 1)

    def env(self, rgb, fg, bgw, t):
        x = np.clip(rgb * self.g, 0, 1)
        lab = rgb_to_oklab(x)
        L = lab[..., 0]
        if self.env_gain is not None:                      # smoothed per-frame exposure normalisation
            L = L * float(self.env_gain[t])
        Lo = self.lift_mid(self.tone(L, fg), fg)
        # mid-frequency clarity and particles: the pull-down compresses them, give them back
        det = L - gblur(L, 12)
        hp = np.clip(L - gblur(L, 2.2), 0, None)
        clar = LOOK["bg_clarity"] * (1 - fg) + LOOK["fg_clarity"] * fg
        Lo = Lo + clar * det * np.clip(1 - 0.5 * Lo, 0.4, 1) + LOOK["particle_gain"] * hp * (0.6 + 0.4 * (1 - fg))
        # light pocket that follows the fish + static vignette + top-down falloff
        cx, cy = self.cent[t]
        sx, sy = LOOK["pool_sigma"]
        pool = np.exp(-0.5 * (((self.xn - cx) / sx) ** 2 + ((self.yn - cy) / sy) ** 2))
        pool = LOOK["pool_floor"] + (1 - LOOK["pool_floor"]) * pool
        Lo = Lo * self.vig * self.top * pool
        return self.env_colour(lab[..., 1:], Lo, fg)

    # ----------------------------------------------------------------- fish
    def fish_core(self, L, C, blur=True):
        """Fish recolour from raw OKLab L / chroma (already exposure-stabilised). Pure per-pixel
        except for the optional smoothing that keeps the result on the body pattern, not codec noise."""
        F = self.P["fish"]
        sm = (lambda x, s_: gblur(x.astype(np.float32), s_)) if blur else (lambda x, s_: x)
        Cn = sm(np.clip((C - F["c_lo"]) / (F["c_hi"] - F["c_lo"]), 0, 1), 2.0)
        Lm = apply_lut(np.clip(L, 0, 1), F["tone"])
        # a light micro-contrast pass on top of the matched curve keeps scale / fin-ray texture
        Lm = Lm + (LOOK["fish_detail"] - 1) * (Lm - sm(Lm, 1.4))
        # silver belly: low chroma and lighter than the bars (the belly is not bright in absolute terms)
        belly = LOOK["belly_max"] * sm(smoothstep(0.26, 0.36, L) * (1 - smoothstep(0.06, 0.30, Cn)), 3.0)
        # warmth: higher raw chroma (tail, fin rays, flank blush) -> crimson, the rest amber / burnt orange
        w = smoothstep(LOOK["fish_warm_lo"], LOOK["fish_warm_hi"], Cn)
        Lm = Lm * (1 - LOOK["crimson_depth"] * w)                # crimson areas sit deeper, never glowing
        det = np.clip((L - sm(L, 3.0)) / 0.05, -1, 1) if blur else np.zeros_like(L)   # local scale pattern
        # hue / chroma follow the reference's own law (measured on its fish pixels): crimson in the
        # darks, burnt orange mid, amber in the lights; chroma grows with lightness
        h_out = np.interp(Lm, LOOK["fish_L_tab"], LOOK["fish_hue_tab"]) - LOOK["fish_warm_shift"] * w \
            - LOOK["fish_hue_texture"] * det
        h_out = np.maximum(h_out, LOOK["fish_hue_crimson"])
        C_out = np.interp(Lm, LOOK["fish_L_tab"], LOOK["fish_chroma_tab"]) * (0.80 + 0.45 * Cn) * (1 + 0.10 * det)
        C_out = np.maximum(C_out, LOOK["fish_chroma_bar"] * (1 - smoothstep(0.16, 0.30, Lm)))
        warm = oklab_to_rgb(lch_to_lab(Lm, C_out, h_out))
        silver = oklab_to_rgb(lch_to_lab(np.minimum(Lm + 0.04, LOOK["belly_cap"]),
                                         np.full_like(Lm, LOOK["belly_chroma"]),
                                         np.full_like(Lm, LOOK["belly_hue"])))
        return np.clip(warm * (1 - belly[..., None]) + silver * belly[..., None], 0, 1)

    def fish(self, rgb, alpha, t):
        ys, xs = np.nonzero(alpha > 0.02)
        if not len(ys):
            return None
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        lab = rgb_to_oklab(rgb[y0:y1, x0:x1])
        L, C, h = lab_to_lch(lab)
        L = L * float(np.clip(self.fish_gain[t], *LOOK["fish_exposure_clamp"]))   # exposure stabiliser
        out = np.zeros_like(rgb)
        out[y0:y1, x0:x1] = self.fish_core(L, C)
        return out, (y0, y1, x0, x1)

    def frame(self, src_u8, t, grain=True):
        rgb = src_u8.astype(np.float32) / 255
        fgm = np.maximum(self.read("seabed", t), self.read("midground", t))
        bgw = self.read("background", t)
        alpha = self.read("fish", t)
        out = self.env(rgb, fgm, bgw, t)
        r = self.fish(rgb, alpha, t)
        if r is not None:
            col, (y0, y1, x0, x1) = r
            a = alpha[y0:y1, x0:x1, None]
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + col[y0:y1, x0:x1] * a
        if grain:
            rng = np.random.default_rng(1000 + t)
            Lg = luma(out)
            amp = LOOK["grain"] * (0.35 + 0.65 * smoothstep(0.02, 0.25, Lg)) * (1 - 0.5 * smoothstep(0.6, 0.95, Lg))
            noise = gblur(rng.standard_normal(out.shape[:2]).astype(np.float32), 0.7) * 1.9
            out = out + (amp * noise)[..., None]
        return np.clip(out, 0, 1)


# ---------------------------------------------------------- clip-level stats
def clip_stats(src, root: Path, P, limit=None):
    """Smoothed fish-centroid track and fish exposure gain (median-normalised), per frame."""
    n, H, W, _ = src.shape
    n = min(n, limit or n)
    cent = np.full((n, 2), np.nan)
    med = np.full(n, np.nan)
    env_med = np.full(n, np.nan)
    gains = np.array(P["wb_gains"], np.float32)
    for t in range(n):
        sb = cv2.imread(str(root / "seabed" / f"{t:05d}.png"), 0)
        sel = (sb > 240)
        if sel.sum() > 5000:
            env_med[t] = np.median(rgb_to_oklab(np.clip(src[t][sel][::50][None].astype(np.float32) / 255 * gains, 0, 1))[0][:, 0])
        al = cv2.imread(str(root / "fish" / f"{t:05d}.png"), 0).astype(np.float32) / 255
        if al.sum() > 150:
            ys, xs = np.nonzero(al > 0.5)
            wgt = al[ys, xs]
            cent[t] = [np.average(xs, weights=wgt) / W, np.average(ys, weights=wgt) / H]
            m = al > 0.9
            if m.sum() > 100:
                med[t] = np.median(rgb_to_oklab(src[t][m][None].astype(np.float32) / 255)[0][:, 0])
    def smooth(v, s):
        ok = ~np.isnan(v)
        v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok]) if ok.any() else np.full(len(v), 0.5)
        k = int(s * 3) * 2 + 1
        ker = np.exp(-0.5 * ((np.arange(k) - k // 2) / s) ** 2)
        ker /= ker.sum()
        return np.convolve(np.pad(v, k // 2, mode="edge"), ker, "valid")
    c = np.stack([smooth(cent[:, 0], 7), smooth(cent[:, 1], 7)], 1)
    ms = smooth(med, 6)
    gain = P["fish"]["raw_L_median"] / np.maximum(ms, 1e-3)
    em = smooth(env_med, 10)
    eg = np.clip((np.nanmedian(env_med) / np.maximum(em, 1e-3)) ** LOOK["env_norm"], *LOOK["env_gain_clamp"])
    return c, gain, eg


def load_all(a):
    P = json.loads(PARAMS.read_text())
    src = decode_source(a.input)
    root = REPO / "masks" / f"{a.input.stem}__mattes"
    cent, gain, eg = clip_stats(src, root, P, getattr(a, "limit", None))
    return P, src, root, cent, gain, eg


_G = None


def _init(args):
    global _G
    P, src_path, root, cent, gain, eg, H, W = args
    from fish_refine import decode_source as ds
    src = ds(src_path)
    _G = (Grader(P, len(src), H, W, root, cent, gain, eg), src)


def _work(t):
    G, src = _G
    return t, (G.frame(np.asarray(src[t]), t) * 255 + 0.5).astype(np.uint8)


def write_cube(path: Path, table: np.ndarray, title: str, n=33):
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for r in table:
            f.write(f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")


def export_luts(a):
    """Bake the point-wise part of each layer into a 33^3 .cube (R varies fastest). The spatial parts
    (clarity, particles, vignette, light pocket, grain) are not in the LUTs; see GRADE_SPEC.md."""
    P = json.loads(PARAMS.read_text())
    G = Grader(P, 1, 8, 8, REPO, np.array([[0.5, 0.4]]), np.array([1.0]))
    n = 33
    g = np.linspace(0, 1, n, dtype=np.float32)
    B, Gc, R = np.meshgrid(g, g, g, indexing="ij")
    lat = np.stack([R, Gc, B], -1).reshape(-1, 3)          # index = r + n*g + n*n*b
    lab = rgb_to_oklab(np.clip(lat * G.g, 0, 1))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fg in (("background", 0.0), ("seabed", 1.0)):
        fgv = np.full(len(lat), fg, np.float32)
        Lo = G.lift_mid(G.tone(lab[:, 0], fgv), fgv)
        out = G.env_colour(lab[:, 1:], Lo, fgv)
        write_cube(a.out / f"fish_{name}.cube", out, f"fish clip - {name} look (point-wise)")
    L, C, h = lab_to_lch(rgb_to_oklab(lat))
    out = G.fish_core(L, C, blur=False)
    write_cube(a.out / "fish_subject.cube", out, "fish clip - subject look (point-wise, no belly/pattern smoothing)")
    print(f"[luts] wrote 3 .cube files to {a.out}")


def spec(a):
    """Print the measured grade as numbers a colourist can type into a node tree."""
    from scipy.optimize import least_squares
    from fish_color import lin_to_srgb, srgb_to_lin
    P = json.loads(PARAMS.read_text())
    G = Grader(P, 1, 8, 8, REPO, np.array([[0.5, 0.4]]), np.array([1.0]))
    v = np.linspace(0.0, 1.0, 65, dtype=np.float32)
    gray = np.repeat(v[:, None], 3, 1)
    Lin = rgb_to_oklab(gray)[:, 0]
    print("== Neutral-axis tone curves incl. the mid-tone lift (display-referred input -> output, 8-bit code values; WB gains applied first)")
    print("   in   " + "  ".join(f"{int(x * 255):4d}" for x in v[::8]))
    for name, fg in (("background", 0.0), ("seabed/midground", 1.0)):
        fgv = np.full(len(v), fg, np.float32)
        Lo = G.lift_mid(G.tone(Lin, fgv), fgv)
        out = oklab_to_rgb(np.stack([Lo, 0 * Lo, 0 * Lo], -1))[:, 0]
        print(f"   {name[:5]:5s}" + "  ".join(f"{int(x * 255 + .5):4d}" for x in out[::8]))
        def model(q):
            sl, off, pw = q
            return np.clip(sl * v + off, 0, 1) ** pw - out
        q = least_squares(model, [0.6, 0.0, 1.3], bounds=([0.05, -0.2, 0.5], [2, 0.2, 3])).x
        print(f"   ASC-CDL fit ({name}): slope {q[0]:.3f}  offset {q[1]:+.3f}  power {q[2]:.3f}   (rms {np.sqrt(np.mean(model(q) ** 2)) * 255:.1f} codes)")
    print("\n== Tonal-zone colour offsets (OKLab a,b -> RGB shift at the zone centre; wheel colour, 8-bit units)")
    for k, nm in (("bg", "background"), ("fg", "seabed/midground")):
        for zn, Lc in (("shadow", 0.20), ("mid", 0.42), ("high", 0.62)):
            a_, b_ = P["tint"][k][zn]
            base = oklab_to_rgb(np.array([[Lc, 0, 0]], np.float32))[0]
            col = oklab_to_rgb(np.array([[Lc, a_, b_]], np.float32))[0]
            d = (col - base) * 255
            print(f"   {nm[:10]:10s} {zn:6s} a,b = ({a_:+.3f},{b_:+.3f})  RGB shift R{d[0]:+.1f} G{d[1]:+.1f} B{d[2]:+.1f}")
    F = P["fish"]
    print("\n== Fish (OKLCH): hue / chroma by output lightness (Lum vs Hue / Lum vs Sat)")
    print("   L      " + "  ".join(f"{x:5.2f}" for x in LOOK["fish_L_tab"]))
    print("   hue    " + "  ".join(f"{x:5.0f}" for x in LOOK["fish_hue_tab"]) + f"   minus {LOOK['fish_warm_shift']:.0f} deg x warmth (tail/fins)")
    print("   chroma " + "  ".join(f"{x:5.3f}" for x in LOOK["fish_chroma_tab"]))
    print(f"   source fish hue p5/50/95 = {np.round(F['raw_hue_q'])}  (yellow/ochre)  ->  target {min(LOOK['fish_hue_tab']):.0f}-{max(LOOK['fish_hue_tab']):.0f} deg (crimson..amber)")
    print(f"   source chroma window {F['c_lo']:.3f}..{F['c_hi']:.3f} drives warmth; belly = low chroma + above bar luma -> hue {LOOK['belly_hue']:.0f}, chroma {LOOK['belly_chroma']}, cap L {LOOK['belly_cap']}")
    print(f"   fish luma curve source -> reference quantiles (L): raw {np.round(F['raw_L_q'], 2)}  ref {np.round(F['ref_L_q'], 2)}")
    print(f"\n== Atmosphere: vignette strength {LOOK['vig_strength']} power {LOOK['vig_power']}; top falloff {LOOK['top_fall']} over {LOOK['top_fall_h']} of the height;"
          f" light pocket floor {LOOK['pool_floor']} sigma {LOOK['pool_sigma']}; grain {LOOK['grain']}")


def stills(a):
    P, src, root, cent, gain, eg = load_all(a)
    n, H, W, _ = src.shape
    G = Grader(P, n, H, W, root, cent, gain, eg)
    a.out.mkdir(parents=True, exist_ok=True)
    for t in [int(x) for x in a.frames.split(",")]:
        o = G.frame(np.asarray(src[t]), t)
        cv2.imwrite(str(a.out / f"g{t:05d}.png"), (o[..., ::-1] * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(str(a.out / f"s{t:05d}.jpg"), np.asarray(src[t])[..., ::-1])
        print("still", t, f"fish gain {gain[t]:.2f} env gain {eg[t]:.2f} centroid {np.round(cent[t], 2)}", flush=True)


def render(a):
    P, src, root, cent, gain, eg = load_all(a)
    n, H, W, _ = src.shape
    w, h, nn, fr = (W, H, n, "30000/1001")
    audio = ["-i", str(a.input), "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", fr,
         "-i", "-", *audio,
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
         "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf), "-profile:v", "high", "-level", "4.2",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
         "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    with Pool(a.workers, initializer=_init, initargs=((P, a.input, root, cent, gain, eg, H, W),)) as pool:
        for k, (t, f) in enumerate(pool.imap(_work, range(n), chunksize=2)):
            enc.stdin.write(f.tobytes())
            if k % 30 == 0:
                print(f"[render] {k}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"[render] wrote {a.output}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--refs", nargs="+", required=True)
    c.add_argument("--step", type=int, default=12)
    c.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("stills")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--frames", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--limit", type=int, default=None)
    sub.add_parser("spec")
    e = sub.add_parser("luts")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--crf", type=int, default=15)
    r.add_argument("--preset", default="slow")
    r.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"calibrate": calibrate, "stills": stills, "render": render, "luts": export_luts, "spec": spec}[a.cmd](a)


if __name__ == "__main__":
    main()
