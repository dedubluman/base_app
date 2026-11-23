"""Grade v2: one shared underwater base look, plus a natural above-water path.

Why v2 exists (see ``diagnose.py``): the source background is rock-steady for the whole dive
(L .535 -> .521, hue 197 -> 212), but v1 graded "sand" and "water" with two different colour regimes
and cross-faded between them while the camera tilted (frames 48-64: L .318 -> .261, hue 215 -> 251,
chroma .038 -> .069). That regime switch *was* the fluctuation.

v2 rules
* water, sand and the unclaimed remainder share ONE tone map and ONE palette (``Lw`` / water palette);
* the seabed identity is added on top of that base: restrained Mocha on lit sand texture and a constant
  Mocha share on SAM-segmented rocks, never a different overall colour;
* all anchors (background brightness, skin white balance) are measured per frame and then smoothed in
  time by the caller; the grade itself is a pure function of (frame, weights, smoothed anchors);
* sky, land and the sea above the waterline use a natural path (no navy spill); the sea is only
  partly pulled toward the underwater look.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import hex_to_oklab, lin_to_srgb, oklab_to_rgb, rgb_to_oklab, srgb_to_lin
from .grade import (band, frame_stats, hc_to_ab, masked_lowpass, polar_mix,
                    sand_palette, smoothstep, water_palette)
from .params2 import Params2


def anchors(rgb: np.ndarray, lab0: np.ndarray, w: dict, p: Params2) -> dict:
    """Per-frame measurements (smoothed in time by the renderer)."""
    st = frame_stats(rgb, lab0, w, p)              # water_med, sand_med, sky_med, skin gains / L
    L0 = lab0[..., 0]
    air_frac = float(w["S"].mean() + w["M"].mean())
    st["air_frac"] = air_frac
    sel = (w["W"] + w["U"]) > 0.8
    st["bg_med"] = float("nan")
    if air_frac < 0.15 and sel.sum() > 800:        # measured on underwater frames only
        v = L0[sel]
        lo, hi = np.percentile(v, [8, 88])         # the water itself, not particles / reflections
        core = v[(v >= lo) & (v <= hi)]
        st["bg_med"] = float(np.median(core))
    if "skin_gains" in st:
        lin_after = np.clip(np.array(st["skin_lin"]) * np.array(st["skin_gains"]), 0, 1.4)
        lab_after = rgb_to_oklab(lin_to_srgb(lin_after.astype(np.float32))[None, None, :])[0, 0]
        h_after = float(np.degrees(np.arctan2(lab_after[2], lab_after[1])) % 360)
        c_after = float(np.hypot(lab_after[1], lab_after[2]))
        st["skin_dh"] = float(np.clip((p.skin_hue - h_after + 180) % 360 - 180, -60, 60))
        st["skin_kc"] = float(np.clip(p.skin_C_target / max(c_after, 1e-3), 0.6, 1.8))
    return st


def _floor(L, f):
    return f + np.logaddexp(0, (L - f) * 40) / 40


def grade_frame2(rgb: np.ndarray, w: dict, st: dict, p: Params2) -> np.ndarray:
    H, W, _ = rgb.shape
    lab0 = rgb_to_oklab(rgb)
    L0, a0, b0 = lab0[..., 0], lab0[..., 1], lab0[..., 2]
    D, S, R, M, Wt, U = (w[k].astype(np.float32) for k in ("D", "S", "R", "M", "W", "U"))
    yy = (np.arange(H, dtype=np.float32) / (H - 1))[:, None]
    xx = (np.arange(W, dtype=np.float32) / (W - 1))[None, :]
    r2 = np.sqrt(((xx - 0.5) * 2) ** 2 * 0.5 + ((yy - 0.5) * 2) ** 2 * 0.5)

    da = a0 - cv2.GaussianBlur(a0, (0, 0), 18)     # high-frequency chroma that survives
    db = b0 - cv2.GaussianBlur(b0, (0, 0), 18)
    air = float(st.get("air", 0.0))                # 0 underwater .. 1 above water (smoothed)
    rho = 1.0 - p.sea_natural_share * air          # underwater-look strength on the sea
    gate = float(st.get("gate", 1.0))              # 1 = camera on the seabed, 0 = open water

    # ================= UNDERWATER BASE (shared by water, sand, remainder)
    bm = st.get("bg_med")
    bm = 0.53 if bm is None or np.isnan(bm) else bm
    anchor = p.water_L_mid + p.water_L_follow * (bm - 0.53)
    Lw = anchor + p.water_L_slope * (L0 - bm)
    dark = 1 - smoothstep(0.35, 0.70, Lw)
    Lw = Lw - p.depth_grad * (1 - yy) ** 1.3 * dark      # constant gentle depth gradient
    Lw = _floor(Lw, p.water_L_floor)
    aw, bw = water_palette(Lw, p)
    aw = aw + p.water_detail_chroma * da
    bw = bw + p.water_detail_chroma * db

    # ================= SEABED = restrained earth on top of the base
    seab = np.clip(U + R, 0, 1)
    bp = band(L0, seab)                                            # cored micro texture
    bpm = cv2.GaussianBlur(L0, (0, 0), 2.0) - masked_lowpass(L0, seab, 24.0)   # lit ridges / rock tops
    # sand: same tone map as the water (+tiny lift) plus texture; Mocha only on lit texture
    Lu = _floor(Lw + p.sand_L_offset + (p.sand_local_gain - 1) * bp, p.water_L_floor)
    awu, bwu = water_palette(Lu, p)
    amu, bmu = sand_palette(Lu, p)
    tint = p.sand_base_earth + (p.sand_earth_max - p.sand_base_earth) * smoothstep(p.sand_earth_lo, p.sand_earth_hi, bpm)
    au, bu = polar_mix(awu, bwu, amu, bmu, tint)
    au = au + p.water_detail_chroma * da
    bu = bu + p.water_detail_chroma * db
    # rocks: constant earth share, texture contrast, hazing into the water as the seabed leaves view
    Lr = _floor(Lw + p.rock_L_delta * (0.4 + 0.6 * gate)
                + (p.rock_local_gain - 1) * (0.55 + 0.45 * gate) * bp
                - p.rock_haze_dim * (1 - gate), p.water_L_floor)
    awr, bwr = water_palette(Lr, p)
    amr, bmr = sand_palette(Lr, p, p.rock_C_boost)
    ar, br = polar_mix(awr, bwr, amr, bmr, p.rock_mocha * (0.6 + 0.4 * gate))
    ar = ar + p.sand_detail_chroma * da
    br = br + p.sand_detail_chroma * db

    # ================= NATURAL PATH (sky, land, sea above water)
    Ln = p.nat_black + (1 - p.nat_black) * np.clip(L0, 0, 1) ** p.nat_gamma
    an, bn = a0 * p.nat_sat, b0 * p.nat_sat
    Lm = Ln + (p.nat_local_gain - 1) * band(L0, M, 1.4, 10.0)      # land micro-contrast

    # ================= DIVER (local WB from lit skin, bounded warm skin shaping, gear protected)
    g = np.array(st.get("skin_gains", [1, 1, 1]), np.float32)
    lin = srgb_to_lin(rgb)
    hs = st.get("skin_hue_src")
    hs = 175.0 if hs is None or np.isnan(hs) else hs               # hue window follows the measured skin
    Pw = np.clip(w.get("P", 0 * L0), 0, 1)                           # SAM 3.1 swimwear mask
    skin_like = (smoothstep(0.30, 0.44, L0)
                 * smoothstep(0.0, 1.0, 1 - np.abs(((np.degrees(np.arctan2(b0, a0)) % 360 - hs + 180) % 360 - 180) / 60))
                 * smoothstep(0.012, 0.030, np.hypot(a0, b0)) * (1 - smoothstep(0.84, 0.92, L0))
                 * (1 - Pw))
    skin_like = np.clip(skin_like, 0, 1)
    gain_img = g ** skin_like[..., None]                           # skin only: shorts / gear keep their own colour
    lab_d = rgb_to_oklab(lin_to_srgb(np.clip(lin * gain_img, 0, 1.4)))
    Ld0, ad, bd = lab_d[..., 0], lab_d[..., 1], lab_d[..., 2]
    sk = st.get("skin_med")
    sk = 0.66 if sk is None or np.isnan(sk) else sk
    x = Ld0 - sk
    y = np.where(x > 0, (p.skin_L_slope * x) / (1 + (p.skin_L_slope * x) / np.maximum(1.05 - p.skin_target_L, 0.2)),
                 p.skin_L_slope * x)
    Ld = np.clip(p.skin_target_L + y, 0.06, 0.96)
    Ld = Ld + (p.skin_local_gain - 1) * band(Ld0, D, 1.4, 8.0)
    Cd = np.hypot(ad, bd)
    hd = np.degrees(np.arctan2(bd, ad)) % 360
    dh = (p.skin_hue - hd + 180) % 360 - 180
    hd1 = hd + float(st.get("skin_dh", 0.0)) * skin_like           # closed loop: lands on the target hue
    dh1 = (p.skin_hue - hd1 + 180) % 360 - 180
    hd2 = hd1 + 0.30 * dh1 * skin_like                             # small pull keeps natural variation
    kc = float(st.get("skin_kc", 1.0))
    Cd2 = np.clip(Cd * kc, p.skin_C_min, p.skin_C_max) * skin_like + Cd * (1 - skin_like) * (
        1 - 0.30 * Pw)                                              # swimwear keeps its colour, a touch calmer
    ad, bd = hc_to_ab(hd2, Cd2)

    # ================= BLEND (soft region weights; sea blends underwater <-> natural by `rho`)
    Lf = D * Ld + S * Ln + M * Lm + R * Lr + U * Lu + Wt * (rho * Lw + (1 - rho) * Ln)
    af = D * ad + S * an + M * an + R * ar + U * au + Wt * (rho * aw + (1 - rho) * an)
    bf = D * bd + S * bn + M * bn + R * br + U * bu + Wt * (rho * bw + (1 - rho) * bn)

    # ================= HIGHLIGHTS & PARTICLES -> Silver Sand (underwater-look pixels only)
    nat_w = np.clip(S + M + (1 - rho) * Wt, 0, 1)                  # share of natural look at the pixel
    uwm = 1 - nat_w
    nd = 1 - D
    hw = smoothstep(p.highlight_start, p.highlight_end, L0) * nd * uwm
    k = int(p.particle_scale_px)
    th = L0 - cv2.morphologyEx(L0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pw = smoothstep(p.particle_lo, p.particle_hi, th) * nd * uwm
    neutral = np.clip(np.maximum(hw * 0.85, pw * p.particle_neutral), 0, 1)
    sa, sb = hex_to_oklab(p.silver)[1:]
    af = af * (1 - neutral) + sa * neutral
    bf = bf * (1 - neutral) + sb * neutral
    floor_hi = np.minimum(L0, 0.84) * p.particle_keep - 0.03
    Lf = np.where(neutral > 0.01, Lf + neutral * np.maximum(floor_hi - Lf, 0), Lf)

    # ================= global highlight shoulder (silver cap underwater, higher cap in air)
    kn = p.shoulder_knee * (1 - nat_w) + p.nat_knee * nat_w
    tp = p.shoulder_top * (1 - nat_w) + p.nat_top * nat_w
    over = np.maximum(Lf - kn, 0)
    Lf = np.where(Lf > kn, kn + (tp - kn) * (1 - np.exp(-over / (tp - kn))), Lf)

    # ================= light lens vignette on lows/mids; saturation; gamut-safe output
    vig = p.vignette * r2 ** p.vignette_power
    Lf = Lf * (1 - vig * (1 - smoothstep(0.55, 0.9, Lf)))
    if p.saturation != 1.0:
        af, bf = af * p.saturation, bf * p.saturation
    out = np.stack([Lf, af, bf], -1)
    for _ in range(3):
        raw = oklab_to_rgb(out, clip=False)
        bad = (raw.min(-1) < -0.002) | (raw.max(-1) > 1.002)
        if not bad.any():
            break
        out[..., 1:][bad] *= 0.82
    return np.clip(oklab_to_rgb(out, clip=False), 0, 1)
