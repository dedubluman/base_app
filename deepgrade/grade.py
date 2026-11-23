"""Region-aware grade in OKLab.

Common idea: SAM 3.1 gives soft region weights. Each region maps the source
*luminance* through its own tone curve and takes its *colour* from a palette
ramp keyed on the graded luminance (Stratos water, Mocha seabed, Silver Sand
highlights, warm natural skin) while keeping a controlled share of the source's
high-frequency chroma. Regions blend in OKLab, so there is never a hard seam.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import (hex_to_oklab, lin_to_srgb, oklab_to_rgb, rgb_to_oklab,
                    srgb_to_lin)
from .params import Params
from .regions import feather


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def hc_to_ab(h_deg, C):
    r = np.radians(h_deg)
    return C * np.cos(r), C * np.sin(r)


def polar_mix(a1, b1, a2, b2, t):
    """Blend two chromas along the short hue arc (no grey dip at the midpoint)."""
    h1, h2 = np.arctan2(b1, a1), np.arctan2(b2, a2)
    dh = (h2 - h1 + np.pi) % (2 * np.pi) - np.pi
    h = h1 + t * dh
    c = (1 - t) * np.hypot(a1, b1) + t * np.hypot(a2, b2)
    return c * np.cos(h), c * np.sin(h)


def masked_lowpass(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(x * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    plain = cv2.GaussianBlur(x, (0, 0), sigma)
    k = np.clip(den / 0.05, 0, 1)
    return k * num / np.maximum(den, 1e-4) + (1 - k) * plain


def band(L: np.ndarray, m: np.ndarray, s_lo=1.4, s_hi=10.0, core=0.0015) -> np.ndarray:
    """Cored band-pass detail (L units); low band is region-normalised, so
    nothing leaks across region borders."""
    bp = cv2.GaussianBlur(L, (0, 0), s_lo) - masked_lowpass(L, m, s_hi)
    return np.sign(bp) * np.maximum(np.abs(bp) - core, 0)


# ------------------------------------------------------------------ palettes
def water_palette(L, p: Params):
    hue = np.interp(L, [0.17, 0.265, 0.45, 0.75], [p.water_hue_deep, p.water_hue_mid,
                                                     p.water_hue_light, 230.0])
    C = np.interp(L, [0.10, 0.17, 0.265, 0.45, 0.75],
                  [0.060, p.water_C_deep, p.water_C_mid, p.water_C_light, 0.012])
    return hc_to_ab(hue, C)


def sand_palette(L, p: Params, c_boost: float = 1.0):
    hue = np.interp(L, [0.15, 0.25, 0.36, 0.55, 0.80],
                    [p.sand_hue_shadow, 140.0, p.sand_hue_mocha, p.sand_hue_light, 90.0])
    C = np.interp(L, [0.12, 0.20, 0.36, 0.55, 0.80],
                  [0.018, p.sand_C_shadow, p.sand_C_mocha, p.sand_C_light, 0.006]) * c_boost
    return hc_to_ab(hue, C)


# ------------------------------------------------------------------ per-frame stats
def frame_stats(rgb: np.ndarray, lab0: np.ndarray, w: dict, p: Params) -> dict:
    """Anchors measured on the source frame (smoothed over time by the caller)."""
    L0 = lab0[..., 0]
    H, W = L0.shape

    def med(mask, lo=0.0, hi=1.0):
        sel = L0[mask & (L0 >= lo) & (L0 <= hi)]
        return float(np.median(sel)) if sel.size > 800 else float("nan")

    st = {"water_med": med(w["W"] > 0.8, 0.0, 0.72),
          "sand_med": med((w["U"] + w["R"]) > 0.8, 0.0, 0.72),
          "sky_med": med(w["S"] > 0.8)}
    # lit skin of the diver (upper 60 % of its box: keeps the shorts out)
    D = w["D"] > 0.95
    ys, xs = np.nonzero(D)
    if ys.size > 3000:
        y0, y1 = ys.min(), ys.max()
        band_ = np.zeros_like(D)
        band_[y0:int(y0 + 0.62 * (y1 - y0 + 1))] = True
        C0 = np.hypot(lab0[..., 1], lab0[..., 2])
        sel = D & band_ & (L0 > 0.50) & (L0 < 0.86) & (C0 < 0.10)
        if "P" in w:
            sel &= w["P"] < 0.3                                 # swimwear is not skin
        if sel.sum() > 1500:
            lin = srgb_to_lin(rgb[sel])
            m = np.median(lin, axis=0)
            Lm = float(np.median(L0[sel]))
            a_t, b_t = hc_to_ab(p.skin_hue, p.skin_C_target)
            tgt = oklab_to_rgb(np.array([[Lm, a_t, b_t]], np.float32), clip=False)[0]
            tgt = srgb_to_lin(np.clip(tgt, 0, 1))
            g = tgt / np.maximum(m, 1e-4)
            Y = np.array([0.2126, 0.7152, 0.0722], np.float32)
            g = g * (Y @ m) / max(float(Y @ (g * m)), 1e-6)
            st["skin_gains"] = np.clip(g, 0.35, 5.5).tolist()
            st["skin_lin"] = m.tolist()
            st["skin_med"] = Lm
            ms = np.median(lab0[sel][:, 1:], axis=0)             # source skin hue: the window follows it
            st["skin_hue_src"] = float(np.degrees(np.arctan2(ms[1], ms[0])) % 360)
    return st


# ------------------------------------------------------------------ the grade
def grade_frame(rgb: np.ndarray, w: dict, st: dict, p: Params,
                debug: bool = False) -> np.ndarray:
    H, W, _ = rgb.shape
    lab0 = rgb_to_oklab(rgb)
    L0, a0, b0 = lab0[..., 0], lab0[..., 1], lab0[..., 2]
    D, S, R, Wt, U = (w[k].astype(np.float32) for k in ("D", "S", "R", "W", "U"))
    yy = (np.arange(H, dtype=np.float32) / (H - 1))[:, None]
    xx = (np.arange(W, dtype=np.float32) / (W - 1))[None, :]
    r2 = np.sqrt(((xx - 0.5) * 2) ** 2 * 0.5 + ((yy - 0.5) * 2) ** 2 * 0.5)  # 0 centre .. 1 corner

    # high-frequency chroma that survives (natural variation, not the cast)
    da = a0 - cv2.GaussianBlur(a0, (0, 0), 18)
    db = b0 - cv2.GaussianBlur(b0, (0, 0), 18)

    # ---------------- WATER
    wm = st.get("water_med")
    wm = 0.53 if wm is None or np.isnan(wm) else wm
    anchor = p.water_L_mid + p.water_L_follow * (wm - 0.53)
    Lw = anchor + p.water_L_slope * (L0 - wm)
    Lw = p.water_L_floor + np.logaddexp(0, (Lw - p.water_L_floor) * 40) / 40  # soft floor
    aw, bw = water_palette(Lw, p)
    aw = aw + p.water_detail_chroma * da
    bw = bw + p.water_detail_chroma * db

    # ---------------- SEABED (sand + rocks share the Mocha palette)
    sm = st.get("sand_med")
    sm = 0.52 if sm is None or np.isnan(sm) else sm
    Ltarget = p.sand_L_bottom + (p.sand_L_top - p.sand_L_bottom) * (1 - yy) ** 1.3
    Ls = Ltarget + p.sand_L_slope * (L0 - sm)
    gate = float(st.get("gate", 1.0))   # 1 = camera on the seabed, 0 = open water
    seab = np.clip(U + R, 0, 1)
    bp = band(L0, seab)
    Lu = Ls + (p.sand_local_gain - 1) * bp
    # as the camera leaves the seabed the rocks sink into the water haze (no floating island)
    Lr = (Ls + p.rock_L_delta * (0.4 + 0.6 * gate)
          + (p.rock_local_gain - 1) * (0.55 + 0.45 * gate) * bp - p.rock_haze_dim * (1 - gate))
    Lu = p.water_L_floor + np.logaddexp(0, (Lu - p.water_L_floor) * 40) / 40
    Lr = p.water_L_floor + np.logaddexp(0, (Lr - p.water_L_floor) * 40) / 40
    au, bu = sand_palette(Lu, p)
    ar, br = sand_palette(Lr, p, p.rock_C_boost)
    # far / hazy seabed keeps the water's navy; near seabed and rocks carry the Mocha earth
    mo = p.sand_mocha_top + (p.sand_mocha_bottom - p.sand_mocha_top) * smoothstep(0.05, p.sand_mocha_span, yy)
    # distance haze over the seabed = the water's navy; near seabed carries the earth
    awu, bwu = water_palette(Lu, p)
    awr, bwr = water_palette(Lr, p)
    au, bu = polar_mix(awu, bwu, au, bu, mo)
    mo_r = np.clip(mo + p.rock_mocha_extra, 0, p.rock_mocha) * (0.6 + 0.4 * gate)   # earthier than the sand around, hazier in open water
    ar, br = polar_mix(awr, bwr, ar, br, mo_r)
    au = au + p.sand_detail_chroma * da
    bu = bu + p.sand_detail_chroma * db
    ar = ar + p.sand_detail_chroma * da
    br = br + p.sand_detail_chroma * db

    # ---------------- SKY / above water
    skm = st.get("sky_med")
    skm = 0.86 if skm is None or np.isnan(skm) else skm
    # anchored + amplified around the frame's sky brightness: keeps (and slightly deepens) the
    # natural top-to-horizon gradient instead of flattening it
    Lsky = p.sky_L_anchor + p.sky_L_slope * (L0 - skm)
    ak, bk = hc_to_ab(p.sky_hue_cool, p.sky_C_cool)
    # keep a share of the horizon glow, rotated toward silver (pastel, not saturated)
    ak = ak + p.sky_chroma_keep * a0 * 0.8
    bk = bk + p.sky_chroma_keep * b0 * 0.8

    # ---------------- DIVER
    g = np.array(st.get("skin_gains", [1, 1, 1]), np.float32)
    lin = srgb_to_lin(rgb)
    skin_like = (smoothstep(0.50, 0.58, L0) * smoothstep(0.0, 1.0, 1 - np.abs(
        (np.degrees(np.arctan2(b0, a0)) % 360 - 175) / 40))
        * smoothstep(0.015, 0.035, np.hypot(a0, b0)) * (1 - smoothstep(0.84, 0.92, L0)))
    skin_like = np.clip(skin_like, 0, 1)
    gain_img = np.where(skin_like[..., None] > 0, g ** (0.55 + 0.45 * skin_like[..., None]),
                        g ** 0.55)
    lab_d = rgb_to_oklab(lin_to_srgb(np.clip(lin * gain_img, 0, 1.4)))
    Ld0, ad, bd = lab_d[..., 0], lab_d[..., 1], lab_d[..., 2]
    sk = st.get("skin_med")
    sk = 0.66 if sk is None or np.isnan(sk) else sk
    x = Ld0 - sk
    xs = np.where(x > 0, p.skin_L_shoulder - sk, 1.0)
    # filmic shoulder above the skin anchor
    y = np.where(x > 0, (p.skin_L_slope * x) / (1 + (p.skin_L_slope * x) / np.maximum(1.05 - p.skin_target_L, 0.2)),
                 p.skin_L_slope * x)
    Ld = np.clip(p.skin_target_L + y, 0.06, 0.96)
    # local contrast, region-normalised
    Ld = Ld + (p.skin_local_gain - 1) * band(Ld0, D, 1.4, 8.0)
    # skin colour shaping: pull hue toward natural warm skin, bounded chroma
    Cd = np.hypot(ad, bd)
    hd = np.degrees(np.arctan2(bd, ad)) % 360
    dh = (p.skin_hue - hd + 180) % 360 - 180
    hd2 = hd + p.skin_hue_pull * dh * skin_like
    Cd2 = np.clip(Cd, p.skin_C_min, p.skin_C_max) * skin_like + Cd * (1 - skin_like) * (
        1 - p.gear_protect * 0.0)
    ad2, bd2 = hc_to_ab(hd2, Cd2)
    ad = ad2
    bd = bd2

    # ---------------- blend regions
    Lf = D * Ld + S * Lsky + R * Lr + U * Lu + Wt * Lw
    af = D * ad + S * ak + R * ar + U * au + Wt * aw
    bf = D * bd + S * bk + R * br + U * bu + Wt * bw

    # ---------------- highlights & particles -> Silver Sand (outside the diver)
    nd = 1 - D
    hw = smoothstep(p.highlight_start, p.highlight_end, L0) * nd * (1 - 0.9 * S)   # sky keeps its own colour
    k = int(p.particle_scale_px)
    th = L0 - cv2.morphologyEx(L0, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pw = smoothstep(p.particle_lo, p.particle_hi, th) * nd
    neutral = np.clip(np.maximum(hw * 0.85, pw * p.particle_neutral), 0, 1)
    sa, sb = hex_to_oklab(p.silver)[1:]
    af = af * (1 - neutral) + sa * neutral
    bf = bf * (1 - neutral) + sb * neutral
    # keep bright detail bright: never darker than a share of its (silver-capped) source
    floor_hi = np.minimum(L0, 0.84) * p.particle_keep - 0.03
    Lf = np.where(neutral > 0.01, Lf + neutral * np.maximum(floor_hi - Lf, 0), Lf)

    # ---------------- global highlight shoulder: roll off, never clip (Silver Sand cap)
    kn, tp = p.shoulder_knee, p.shoulder_top
    over = np.maximum(Lf - kn, 0)
    Lf = np.where(Lf > kn, kn + (tp - kn) * (1 - np.exp(-over / (tp - kn))), Lf)

    # ---------------- depth: gentle lens vignette on lows/mids (highlights untouched)
    vig = p.vignette * r2 ** p.vignette_power
    Lf = Lf * (1 - vig * (1 - smoothstep(0.55, 0.9, Lf)))

    # saturation + gamut-safe output
    if p.saturation != 1.0:
        af, bf = af * p.saturation, bf * p.saturation
    out = np.stack([Lf, af, bf], -1)
    for _ in range(3):
        raw = oklab_to_rgb(out, clip=False)
        bad = (raw.min(-1) < -0.002) | (raw.max(-1) > 1.002)
        if not bad.any():
            break
        out[..., 1:][bad] *= 0.82
    res = np.clip(oklab_to_rgb(out, clip=False), 0, 1)
    return (res, {"w": (D, S, R, U, Wt)}) if debug else res
