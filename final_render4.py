#!/usr/bin/env python3
"""Skin grade v4 + head/gear neutralisation + final 1080p render. The water grade is not touched.

G  No fixed gain. Every frame: measure the SKIN median, pull it to the target with a 5-frame moving
   average of the correction (Lab hue 48 +-2, C* 19 +-2, R/G 1.28-1.36); L* follows the source.
H  Soft knee above L* 75 on the skin, no channel above 245.
I  No blur, no denoise: the skin keeps the source's fine luminance texture.
F  Skin colour is computed on the mask dilated by 3 px (edge pixels carry skin chroma), then merged
   with a 1 px-in, 2 px-soft alpha.
A  HEAD (hair + diving mask), FINS and GEAR (watch) lose the water tint: |R-G|, |G-B| <= 6, luminance
   of the source; hair keeps a little warmth (R >= G >= B, spread <= 10); the snorkel tip stays yellow-green.
SHORTS neutral white with the folds.

    python final_render4.py --input inputs/diver.mov --output output/diver_graded_1080p.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import matte as M
import reel_match as R
from diver_classes import dil, hand_tips, read_mask
from final_render3 import Y2L, delin, hue_of, lin, smoothstep, thickness, to_lab, to_rgb, wrap
from shades.io import iter_frames, probe

REPO = Path(__file__).resolve().parent
N = 90
SHEET = [2, 10, 33, 45, 75, 87]
CLASSES = ["SKIN", "SHORTS", "FINS", "MASK", "HAIR", "WATCH"]
H_T = 48.0                                           # target hue angle (deg), +-2
RG_T = 1.32                                          # target R/G (1.28-1.36)
KNEE = 75.0
CAP = 245.0 / 255.0
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def c_of_L(rg=RG_T, h=H_T):
    """C* that gives the target R/G (gamma sRGB) for a skin colour of lightness L*, within 17.2-20.8."""
    Ls = np.arange(30, 90, 2.5)
    out = []
    for L in Ls:
        lo, hi = 8.0, 40.0
        for _ in range(30):
            c = (lo + hi) / 2
            rgb = cv2.cvtColor(np.array([[[L, c * np.cos(np.radians(h)), c * np.sin(np.radians(h))]]], np.float32), cv2.COLOR_LAB2RGB)[0, 0]
            if rgb[0] / max(rgb[1], 1e-6) < rg:
                lo = c
            else:
                hi = c
        out.append(np.clip((lo + hi) / 2, 17.2, 20.8))
    return lambda L: np.interp(L, Ls, out)


def ma5(v):
    v = np.asarray(v, np.float64)
    return np.convolve(np.pad(v, 2, mode="edge"), np.ones(5) / 5, "valid")


def knee(L):
    return np.where(L > KNEE, KNEE + 4.0 * np.tanh((L - KNEE) / 4.0), L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--sheet", type=Path, default=REPO / "output" / "contact_sheet_final.jpg")
    ap.add_argument("--test", type=int, default=0)
    a = ap.parse_args()
    info = probe(a.input)
    w, h = info.width, info.height
    stem = a.input.stem
    cdir = REPO / "masks" / f"{stem}__cls"
    nfr = a.test or N
    rng_f = range(nfr)
    frames = []
    for k, x in enumerate(iter_frames(a.input, info)):
        if k >= nfr:
            break
        frames.append(x)

    def rd(nm, i):
        im = cv2.imread(str(cdir / nm / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return np.zeros((h, w), np.uint8) if im is None else im

    A8 = {c: {i: rd(c, i) for i in rng_f} for c in CLASSES}
    Af = lambda c, i: A8[c][i].astype(np.float32) / 255
    person = {i: rd("PERSON", i) > 127 for i in rng_f}
    sk_bin = {i: Af("SKIN", i) > 0.5 for i in rng_f}
    e3 = np.ones((3, 3), np.uint8)

    # ---- diver alpha: 1 px in, 2 px soft ---------------------------------------------------------------------
    alpha8 = {i: np.clip(M.snap_diver_alpha(person[i], frames[i]) * 255 + 0.5, 0, 255).astype(np.uint8) for i in rng_f}
    med = {i: np.median(np.stack([alpha8[max(0, i - 1)], alpha8[i], alpha8[min(nfr - 1, i + 1)]]), 0).astype(np.uint8) for i in rng_f}
    al_final = {}
    for i in rng_f:
        b = cv2.erode((med[i] > 127).astype(np.uint8), e3) > 0
        al = smoothstep(0.30, 0.65, cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.15))
        # F: the skin grade lives on the SKIN mask dilated by 3 px; its alpha is that region eroded by 1 px and
        # softened by 2 px (not next to the other classes, so no skin-coloured rim around shorts / fins / head)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        oth_b = ((Af("MASK", i) + Af("FINS", i) + Af("SHORTS", i) + Af("HAIR", i) + Af("WATCH", i)) > 0.5)
        reg = (cv2.dilate(sk_bin[i].astype(np.uint8), k7) > 0) & ~(cv2.dilate(oth_b.astype(np.uint8), k7) > 0)
        ext = smoothstep(0.30, 0.65, cv2.GaussianBlur((cv2.erode(reg.astype(np.uint8), e3) > 0).astype(np.float32), (0, 0), 1.15))
        al_final[i] = np.maximum(np.maximum(al, ext), np.maximum(Af("FINS", i), Af("MASK", i)))

    # ---- strict skin core, extremities, bbox ---------------------------------------------------------------------
    core, bbox, tips = {}, {}, {}
    for i in rng_f:
        oth = np.maximum.reduce([Af(c, i) for c in ("SHORTS", "FINS", "MASK", "HAIR", "WATCH")])
        c = cv2.erode(((Af("SKIN", i) > 0.98) & (oth < 0.01) & (al_final[i] > 0.98)).astype(np.uint8), e3, iterations=2) > 0
        if c.sum() < 300:
            c = cv2.erode((Af("SKIN", i) > 0.9).astype(np.uint8), e3) > 0
        core[i] = c
        ys, xs = np.nonzero(dil(person[i], 24))
        bbox[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        avoid = (Af("FINS", i) > 0.5) | (Af("SHORTS", i) > 0.5) | (Af("MASK", i) > 0.5) | (Af("HAIR", i) > 0.5)
        tips[i] = hand_tips(person[i], avoid)

    # ---- global exposure only (L* follows the source): reference = the clip's skin luminance ---------------------------
    Yo_all = [(lin(frames[f])[core[f]] @ LUMA).mean() for f in (0, 30, 60, 89) if f < nfr]
    Y_t = float(lin(to_rgb(np.array([[[59.0, 0, 0]]], np.float32)))[0, 0, 0])
    gE = Y_t / max(float(np.mean(Yo_all)), 1e-6)
    Ct = c_of_L()
    print(f"[skin] exposure x{gE:.2f}; no fixed channel gain (per-frame pull)", flush=True)

    # ---- per-frame source quantities for the skin pixels (core) and the whole bbox ------------------------------------------
    prep = {}
    for i in rng_f:
        x = frames[i]
        x0, y0, x1, y1 = bbox[i]
        xl = lin(x[y0:y1, x0:x1])
        sbin = sk_bin[i][y0:y1, x0:x1]
        thick = thickness(sbin)
        tipw = np.zeros(sbin.shape, np.float32)
        for tx, ty in tips[i]:
            cv2.circle(tipw, (tx - x0, ty - y0), 55, 1.0, -1)
        tipw = cv2.GaussianBlur(tipw, (0, 0), 10) * sbin
        Yo = xl @ LUMA
        L = knee(Y2L(np.clip(Yo * gE, 0, 1.2))).astype(np.float32)             # H: soft knee above L* 75
        # the pixel's own hue / chroma variation, from a provisional (median-based) red gain; damped below
        c = core[i][y0:y1, x0:x1]
        med_lin = np.median(xl[c], 0)
        gprov = (lin(to_rgb(np.array([[[np.median(L[c]), 12.7, 14.1]]], np.float32)))[0, 0] / np.maximum(med_lin, 1e-6)).astype(np.float32)
        lab_c = to_lab(delin(np.clip(xl * gprov, 0, 1)))
        hc = hue_of(lab_c[..., 1], lab_c[..., 2])
        Cc = np.hypot(lab_c[..., 1], lab_c[..., 2])
        hm = float(hue_of(np.median(lab_c[..., 1][c]), np.median(lab_c[..., 2][c])))
        Cm = float(np.median(Cc[c]))
        dh = np.clip(wrap(hc - hm), -15, 15)
        dC = np.clip(Cc / max(Cm, 1e-3) - 1.0, -0.4, 0.4)
        limb = 1.0 - smoothstep(24.0, 48.0, thick)
        prep[i] = dict(L=L, dh=dh, dC=dC, limb=limb, tip=tipw, sbin=sbin, core=c, Lmed=float(np.median(L[c])), x0=x0, y0=y0)

    def skin_ab(i, kH, kC):
        P = prep[i]
        L = P["L"]
        w_sh = 1.0 - smoothstep(40.0, 50.0, L)
        w_hi = smoothstep(66.0, 74.0, L)
        Ctar = float(Ct(P["Lmed"]))
        h = H_T + kH + (-3.0 * w_sh + 3.0 * w_hi) + 0.4 * P["dh"]
        C = Ctar * kC * (1.0 + 0.3 * P["dC"]) * (0.95 + 0.05 * (1.0 - w_sh))
        a_ = C * np.cos(np.radians(h)) + 1.5 * P["limb"] + 0.5 * P["tip"]
        b_ = C * np.sin(np.radians(h))
        return L, a_.astype(np.float32), b_.astype(np.float32)

    def skin_rgb_core(i, kH, kC):
        P = prep[i]
        L, a_, b_ = skin_ab(i, kH, kC)
        c = P["core"]
        rgb = to_rgb(np.stack([L[c], a_[c], b_[c]], -1)[None])[0]
        mx = rgb.max(1, keepdims=True)
        rgb = rgb * np.minimum(1.0, CAP / np.maximum(mx, 1e-6))                 # no channel above 245
        lab = to_lab(rgb[None])[0]
        ma, mb = float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))
        r255 = np.median(rgb, 0) * 255
        return dict(hue=float(hue_of(ma, mb)), C=float(np.hypot(ma, mb)), L=float(np.median(lab[:, 0])),
                    rg=float(r255[0] / max(r255[1], 1e-6)))

    # ---- G: measure every frame, pull to the target, 5-frame moving average of the correction ------------------------------
    kH = np.zeros(nfr)
    kC = np.ones(nfr)
    for it in range(4):
        meas = [skin_rgb_core(i, kH[i], kC[i]) for i in rng_f]
        dH = np.array([H_T - m["hue"] for m in meas])
        rg = np.array([m["rg"] for m in meas])
        Cm_ = np.array([m["C"] for m in meas])
        Ctar = np.array([float(Ct(prep[i]["Lmed"])) for i in rng_f])
        kH_new = kH + dH
        kC_new = kC * np.where(np.abs(rg - RG_T) > 0.015, (RG_T - 1.0) / np.maximum(rg - 1.0, 0.05), Ctar / np.maximum(Cm_, 1e-3))
        kC_new = np.clip(kC_new, 0.6, 1.5)
        kH, kC = ma5(kH_new), ma5(kC_new)                                        # 5-frame moving average
    for _ in range(3):                                   # C* must stay inside 19 +-2 (ceiling 20.9)
        meas = [skin_rgb_core(i, kH[i], kC[i]) for i in rng_f]
        Cm_ = np.array([m["C"] for m in meas])
        if (Cm_ <= 20.9).all():
            break
        kC = ma5(kC * np.minimum(1.0, 20.9 / np.maximum(Cm_, 1e-3)))
    meas = [skin_rgb_core(i, kH[i], kC[i]) for i in rng_f]
    hues = np.array([m["hue"] for m in meas])
    print(f"[skin] hue per frame {hues.min():.2f}-{hues.max():.2f}, max step {np.abs(np.diff(hues)).max():.2f} deg; "
          f"C* {min(m['C'] for m in meas):.1f}-{max(m['C'] for m in meas):.1f}; R/G {min(m['rg'] for m in meas):.3f}-{max(m['rg'] for m in meas):.3f}", flush=True)

    # ---- water colour, neutral head/gear, edge colour, render ----------------------------------------------------------------
    proc = None
    if not a.test:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        oh = a.height
        ow = int(round(w * oh / h / 2)) * 2
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
               "-r", str(info.fps), "-i", "-",
               "-vf", f"scale={ow}:{oh}:flags=lanczos+accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", "libx264", "-profile:v", "high", "-preset", a.preset, "-crf", str(a.crf),
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", "-an", str(a.output)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(1234)
    g_col = R.vertical_weight(h, w)
    rows, outs, class_px = [], {}, {"HEAD": [], "FINS": [], "GEAR": []}
    for i in rng_f:
        x = frames[i]
        bg = R.grade_background(x)                      # the water grade is not touched
        P = prep[i]
        x0, y0, x1, y1 = bbox[i]
        L, a_, b_ = skin_ab(i, kH[i], kC[i])
        sbin = P["sbin"]
        # F: skin colour on the mask dilated by 3 px; colour of the edge band (and the 3 px outside) from the interior
        interior = cv2.erode(sbin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        skin_ext = cv2.dilate(sbin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        if interior.sum() > 200:
            _, labels = cv2.distanceTransformWithLabels((~interior).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            coords = np.argwhere(interior)
            lb = np.clip(labels - 1, 0, len(coords) - 1)
            an, bn = a_[coords[lb, 0], coords[lb, 1]], b_[coords[lb, 0], coords[lb, 1]]
            wgt = cv2.GaussianBlur(((~interior) & skin_ext).astype(np.float32), (0, 0), 1.2) * skin_ext
            a_, b_ = a_ * (1 - wgt) + an * wgt, b_ * (1 - wgt) + bn * wgt
        rgb = to_rgb(np.stack([L, a_, b_], -1))
        rgb = rgb * np.minimum(1.0, CAP / np.maximum(rgb.max(-1, keepdims=True), 1e-6))
        skin_full = np.zeros_like(x)
        skin_full[y0:y1, x0:x1] = rgb
        # SHORTS neutral white, folds kept
        xb = np.clip(x * R.DV["gains"], 0, 1.5)
        yb = R.luma(xb)
        scale = 1 + (g_col[..., None] - 1) * R.DV["depth_mix"]
        shorts = np.clip(yb[..., None] * R.DV["lift"] * np.ones(3, np.float32) * scale, 0, 1)
        # A: HEAD / FINS / GEAR without the water tint, luminance of the source
        Yg = (x @ LUMA)[..., None]
        gray = np.repeat(Yg, 3, 2)
        warm = np.clip(gray + np.array([4, 1, -4], np.float32) / 255.0, 0, 1)         # hair: R >= G >= B, spread 8
        hsv = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        tip = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 55) & (hsv[..., 1] > 80) & (hsv[..., 2] > 70)).astype(np.float32)[..., None]   # yellow-green
        sharp = lambda c: np.clip((Af(c, i) - 0.3) / 0.4, 0, 1)                      # hard class interfaces, no chroma mixing
        w_mask, w_hair = sharp("MASK")[..., None], sharp("HAIR")[..., None]
        w_fins, w_gear = sharp("FINS")[..., None], sharp("WATCH")[..., None]
        head = gray * (1 - tip) + x * tip                                              # snorkel tip keeps its own colour
        w_sh, w_sk = sharp("SHORTS"), sharp("SKIN")
        w_other = np.clip(w_mask[..., 0] + w_hair[..., 0] + w_fins[..., 0] + w_gear[..., 0], 0, 1)
        w_sk_eff = np.clip(1 - w_sh - w_other, 0, 1)
        dv = (skin_full * w_sk_eff[..., None] + shorts * w_sh[..., None] + head * w_mask + warm * w_hair
              + gray * w_fins + gray * w_gear)
        al = al_final[i]
        dist = cv2.distanceTransform((al > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        wrap_w = (0.10 * np.clip(1 - dist / 4.0, 0, 1) * (al > 0.02) * (1 - w_sh))[..., None]
        dv = dv * (1 - wrap_w) + bg * wrap_w
        out = bg * (1 - al[..., None]) + dv * al[..., None]
        # ---- numeric controls on the finished frame ----
        c = core[i]
        lab_o = to_lab(out)
        ac, bc = lab_o[..., 1][c], lab_o[..., 2][c]
        r255 = np.median(out[c], 0) * 255
        ring_in = (dist == 0) & dil(al > 0.5, 5) & ~dil(al > 0.5, 1)
        outside = ~(dil(al > 0.5, 1))
        d_out = cv2.distanceTransform(outside.astype(np.uint8), cv2.DIST_L2, 3)
        ringA = outside & (d_out >= 2) & (d_out <= 5)
        ringB = outside & (d_out >= 20) & (d_out <= 40)
        dLring = float(np.median(lab_o[..., 0][ringA]) - np.median(lab_o[..., 0][ringB])) if ringA.sum() > 100 and ringB.sum() > 100 else float("nan")
        lab_bg = to_lab(bg)
        dLvsbg = float(np.median(lab_o[..., 0][ringA] - lab_bg[..., 0][ringA])) if ringA.sum() > 100 else float("nan")
        sbin_full = sk_bin[i]
        ring_s = sbin_full & ~(cv2.erode(sbin_full.astype(np.uint8), e3) > 0)
        int_s = cv2.erode(sbin_full.astype(np.uint8), e3, iterations=4) > 0
        lab_sk = to_lab(np.ascontiguousarray(skin_full))
        Cring = np.hypot(lab_sk[..., 1][ring_s], lab_sk[..., 2][ring_s]).mean() if ring_s.sum() > 30 else float("nan")
        Cint_l = np.hypot(lab_sk[..., 1][int_s], lab_sk[..., 2][int_s]).mean() if int_s.sum() > 100 else float("nan")
        layer_ratio = float(Cring / max(Cint_l, 1e-6))
        a05 = al_final[i] > 0.5
        inner1 = a05 & ~(cv2.erode(a05.astype(np.uint8), e3) > 0) & (Af("SKIN", i) > 0.5)
        Cin = np.hypot(lab_o[..., 1][inner1], lab_o[..., 2][inner1]).mean() if inner1.sum() > 30 else float("nan")
        Cint = np.hypot(ac, bc).mean()
        sk_mask = Af("SKIN", i) > 0.5
        over250 = float((out[sk_mask].max(-1) * 255 > 250).mean()) if sk_mask.any() else 0.0
        rec = dict(frame=i, hue=float(hue_of(np.median(ac), np.median(bc))), C=float(np.hypot(np.median(ac), np.median(bc))),
                   L=float(np.median(lab_o[..., 0][c])), rg=float(r255[0] / max(r255[1], 1e-6)), dLring=dLring, dLvsbg=dLvsbg,
                   inner_chroma=float(Cin / max(Cint, 1e-6)), inner_layer=layer_ratio, over250=over250, skin_area=int(sk_bin[i].sum()))
        out255 = out * 255
        for key, wm in (("HEAD", np.maximum(Af("HAIR", i), Af("MASK", i))), ("FINS", Af("FINS", i)), ("GEAR", Af("WATCH", i))):
            m = (wm > 0.98) & (al_final[i] > 0.98)
            if key == "HEAD":
                m &= (tip[..., 0] < 0.5)                 # the snorkel tip is meant to stay coloured
            if m.sum() > 20:
                px = out255[m]
                class_px[key].append((np.abs(px[:, 0] - px[:, 1]), np.abs(px[:, 1] - px[:, 2]), (px[:, 0] > px[:, 2] + 10)))
        rows.append(rec)
        if i in SHEET:
            outs[i] = out.copy()
        if proc is not None:
            n_ = (rng.random(out.shape[:2] + (1,), np.float32) - rng.random(out.shape[:2] + (1,), np.float32)) / 255.0
            proc.stdin.write((np.clip(out + n_, 0, 1) * 65535.0 + 0.5).astype(np.uint16).tobytes())
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    # ---- control report ----------------------------------------------------------------------------------------------------------
    hh = np.array([r["hue"] for r in rows])
    sa = np.array([r["skin_area"] for r in rows])
    ctrl = dict(
        skin_hue_min=float(hh.min()), skin_hue_max=float(hh.max()), skin_hue_max_step=float(np.abs(np.diff(hh)).max()),
        rg_min=float(min(r["rg"] for r in rows)), rg_max=float(max(r["rg"] for r in rows)),
        chroma_min=float(min(r["C"] for r in rows)), chroma_max=float(max(r["C"] for r in rows)),
        skin_area_max_step_pct=float(100 * np.abs(np.diff(sa) / sa[:-1]).max()),
        ring_dL_max=float(np.nanmax(np.abs([r["dLring"] for r in rows]))),
        ring_dL_vs_graded_bg_max=float(np.nanmax(np.abs([r["dLvsbg"] for r in rows]))),
        inner_chroma_min=float(np.nanmin([r["inner_chroma"] for r in rows])),
        inner_layer_chroma_min=float(np.nanmin([r["inner_layer"] for r in rows])),
        skin_over250_max=float(max(r["over250"] for r in rows)),
        L_range=[float(min(r["L"] for r in rows)), float(max(r["L"] for r in rows))])
    for key, lst in class_px.items():
        if lst:
            rg_ = np.concatenate([t[0] for t in lst])
            gb_ = np.concatenate([t[1] for t in lst])
            wm_ = np.concatenate([t[2] for t in lst])
            ctrl[f"{key}_median_RG"], ctrl[f"{key}_median_GB"], ctrl[f"{key}_warm_pct"] = float(np.median(rg_)), float(np.median(gb_)), float(100 * wm_.mean())
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    (qc / "skin_lab.csv").write_text(",".join(keys) + "\n" + "\n".join(",".join(f"{r[k]:.4f}" for k in keys) for r in rows) + "\n")
    (qc / "controls.json").write_text(json.dumps(ctrl, indent=1))
    print("[controls]", json.dumps(ctrl), flush=True)

    # ---- the one contact sheet: class overlay (top), grade (bottom) -------------------------------------------------------------------
    col = {"SKIN": (255, 170, 60), "HAIR": (170, 110, 40), "SHORTS": (60, 120, 255), "FINS": (255, 60, 255),
           "MASK": (60, 255, 60), "WATCH": (255, 40, 40)}
    th = 440
    top, bot = [], []
    for i in [f for f in SHEET if f < nfr]:
        ys, xs = np.nonzero(person[i])
        x0, x1 = max(0, xs.min() - 40), min(w, xs.max() + 40)
        y0, y1 = max(0, ys.min() - 40), min(h, ys.max() + 40)
        ov = (frames[i] * 255).astype(np.float32)
        for c in ("SKIN", "HAIR", "SHORTS", "FINS", "MASK", "WATCH"):
            m = Af(c, i) > 0.5
            ov[m] = 0.45 * ov[m] + 0.55 * np.array(col[c], np.float32)
        for lst, src in ((top, ov), (bot, outs[i] * 255)):
            tile = np.clip(src[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            tile = cv2.resize(tile, (int(tile.shape[1] * th / tile.shape[0]), th), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"f{i}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            lst.append(tile)
    sheet = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    leg = np.zeros((26, sheet.shape[1], 3), np.uint8)
    x_ = 8
    for c, nm in (("SKIN", "SKIN"), ("HAIR", "HAIR"), ("MASK", "MASK"), ("SHORTS", "SHORTS"), ("FINS", "FINS"), ("WATCH", "GEAR(watch)")):
        cv2.rectangle(leg, (x_, 6), (x_ + 16, 20), col[c], -1)
        cv2.putText(leg, nm, (x_ + 22, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_ += 130
    sheet = np.concatenate([sheet, leg], 0)
    cv2.imwrite(str(a.sheet), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[done] {a.output if not a.test else '(test)'}  sheet {a.sheet}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
