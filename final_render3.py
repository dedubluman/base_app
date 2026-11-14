#!/usr/bin/env python3
"""Part 2 (v3) - skin colour from the ORIGINAL footage, inside the SKIN mask; final 1080p render.

Nothing is equalised to a per-frame target:
 1. ONE correction for the whole clip: channel gains (red back, cyan down) solved jointly from
    f0, f30, f60, f89 and applied to every frame as is; one global exposure gain.
 2. Targets are for the clip average, as ranges (L* 52-66, a* 13-19, b* 16-24); the frames keep
    their natural variation. Only sudden steps are limited: L* <= 1.5, a* and b* <= 0.8 per frame.
 3. Lightness comes from the original luminance (no blur, no denoise); the fine texture is kept
    (texture ratio reported, >= 0.9 of the original).
 4. Tonal variety: shadows (L* < 45) hue 42-48 deg / chroma 22-26 with ~12 % water colour,
    mid tones 48-54 / 24-28, highlights (L* > 70) 54-60 / 14-20; hue spread >= 8 deg.
 5. Hands, fingers, knees, elbows, feet a little redder (a* +2-3), torso calmer.
 6. 6 px edge band: blue cleaned, colour carried from the interior skin, lightness untouched.
 7. Edge: 1 px in, 3 px feather, 10 % water colour over the outer 4 px.
 8. SHORTS neutral white with the folds; FINS, MASK, HAIR, WATCH are left as the original.
 9. Background mid tones +10-15 %, one setting for the whole clip.

    python final_render3.py --input inputs/diver.mov --output output/diver_graded_1080p.mp4
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
from diver_reels import drop_specks
from shades.io import iter_frames, probe

REPO = Path(__file__).resolve().parent
N = 90
SHEET = [0, 12, 20, 28, 36, 46, 60, 70, 89]
CLASSES = ["SKIN", "SHORTS", "FINS", "MASK", "HAIR", "WATCH"]
TARGET = dict(L=59.0, a=16.0, b=20.0)             # centre of the allowed clip-average ranges
LIM = dict(L=1.5, a=0.8, b=0.8)                    # max frame-to-frame step
GLOBAL_F = [0, 30, 60, 89]
K_H, K_C = 0.45, 0.35                              # share of the pixel's own hue / chroma variation kept


def lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def delin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def to_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2LAB)


def to_rgb(lab):
    return cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_LAB2RGB)


def Y2L(Y):
    f = np.where(Y > 0.008856, np.cbrt(np.maximum(Y, 1e-9)), 7.787 * Y + 16 / 116)
    return 116 * f - 16


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def hue_of(a, b):
    return np.degrees(np.arctan2(b, a))


class Skin:
    """Global skin pipeline: gains + exposure are fixed for the clip."""

    def __init__(self, g, gE, mean_hc, mean_Cc):
        self.g, self.gE, self.mean_hc, self.mean_Cc = g, gE, mean_hc, mean_Cc

    def lab(self, xl, thick, tipw, water_ab):
        """xl: linear original RGB (..., 3) of skin pixels. thick: local limb thickness (px).
        tipw: extremity weight 0..1. water_ab: (a, b) of the water colour around the diver."""
        Yo = xl @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        L = Y2L(np.clip(Yo * self.gE, 0, 1.2)).astype(np.float32)           # lightness: original luminance
        xc = np.clip(xl * self.g, 0, 1)
        lab_c = to_lab(delin(xc)[None] if xc.ndim == 2 else delin(xc))
        lab_c = lab_c[0] if xc.ndim == 2 else lab_c
        hc = hue_of(lab_c[..., 1], lab_c[..., 2])
        Cc = np.hypot(lab_c[..., 1], lab_c[..., 2])
        # tone anchors by lightness
        w_sh = 1.0 - smoothstep(40.0, 50.0, L)
        w_hi = smoothstep(66.0, 74.0, L)
        w_mid = np.clip(1.0 - w_sh - w_hi, 0, 1)
        h_t = w_sh * 45.0 + w_mid * 51.0 + w_hi * 57.0
        C_t = w_sh * 24.0 + w_mid * 26.0 + w_hi * 17.0
        # natural variation from the corrected pixel (damped)
        dh = np.clip(wrap(hc - self.mean_hc), -18, 18)
        dC = np.clip(Cc / max(self.mean_Cc, 1e-3) - 1.0, -0.4, 0.4)
        h = np.clip(h_t + K_H * dh, w_sh * 42 + w_mid * 48 + w_hi * 54, w_sh * 48 + w_mid * 54 + w_hi * 60)
        C = np.clip(C_t * (1 + K_C * dC), w_sh * 22 + w_mid * 24 + w_hi * 14, w_sh * 26 + w_mid * 28 + w_hi * 20)
        a = C * np.cos(np.radians(h))
        b = C * np.sin(np.radians(h))
        # cool shadows: ~12 % water colour in the shadows (the target vector already contains it)
        w = 0.12 * w_sh
        aw, bw = water_ab
        a = a * (1 - w) + (aw if aw is not None else 0.0) * w
        b = b * (1 - w) + (bw if bw is not None else 0.0) * w
        # redder extremities and thin limbs (+2..3 a*), calm torso
        limb = 1.0 - smoothstep(24.0, 48.0, thick)
        a = a + 2.4 * limb + 0.6 * tipw
        return L, a.astype(np.float32), b.astype(np.float32)


def thickness(skin_bin):
    return cv2.distanceTransform(skin_bin.astype(np.uint8), cv2.DIST_L2, 5) * 2.0


def edge_chroma(a, b, skin_bin, band=6):
    """6 px edge band: take (a*, b*) from the nearest interior skin pixel."""
    interior = cv2.erode(skin_bin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))) > 0
    if interior.sum() < 200:
        return a, b
    inp = (~interior).astype(np.uint8)
    dist, labels = cv2.distanceTransformWithLabels(inp, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    coords = np.argwhere(interior)
    lab_ = np.clip(labels - 1, 0, len(coords) - 1)
    ny, nx = coords[lab_, 0], coords[lab_, 1]
    an, bn = a[ny, nx], b[ny, nx]
    w = (~interior).astype(np.float32)
    w = cv2.GaussianBlur(w, (0, 0), 1.2) * skin_bin
    return a * (1 - w) + an * w, b * (1 - w) + bn * w


def limit_steps(m, lim):
    """Smallest change that keeps |m[i]-m[i-1]| <= lim (forward then backward clamp)."""
    m = np.asarray(m, np.float64).copy()
    for i in range(1, len(m)):
        m[i] = np.clip(m[i], m[i - 1] - lim, m[i - 1] + lim)
    for i in range(len(m) - 2, -1, -1):
        m[i] = np.clip(m[i], m[i + 1] - lim, m[i + 1] + lim)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--sheet", type=Path, default=REPO / "output" / "contact_sheet_final.jpg")
    ap.add_argument("--test", type=int, default=0, help="debug: only the first N frames, no encode")
    a = ap.parse_args()
    info = probe(a.input)
    w, h = info.width, info.height
    stem = a.input.stem
    cdir = REPO / "masks" / f"{stem}__cls"
    pdir = REPO / "masks" / f"{stem}__fwd" / "subject"
    nfr = a.test or N
    frames = []
    for k, x in enumerate(iter_frames(a.input, info)):
        if k >= nfr:
            break
        frames.append(x)
    rng_f = range(nfr)

    def rd(nm, i):
        im = cv2.imread(str(cdir / nm / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return np.zeros((h, w), np.uint8) if im is None else im

    A8 = {c: {i: rd(c, i) for i in rng_f} for c in CLASSES}
    Af = lambda c, i: A8[c][i].astype(np.float32) / 255
    person = {i: drop_specks(read_mask(pdir, i, (h, w))) for i in rng_f}
    sk_bin = {i: Af("SKIN", i) > 0.5 for i in rng_f}

    # ---- diver alpha: 1 px in, 3 px feather ------------------------------------------------
    alpha8 = {i: np.clip(M.snap_diver_alpha(person[i], frames[i]) * 255 + 0.5, 0, 255).astype(np.uint8) for i in rng_f}
    med = {i: np.median(np.stack([alpha8[max(0, i - 1)], alpha8[i], alpha8[min(nfr - 1, i + 1)]]), 0).astype(np.uint8) for i in rng_f}
    e3 = np.ones((3, 3), np.uint8)
    al_final = {}
    for i in rng_f:
        b = cv2.erode((med[i] > 127).astype(np.uint8), e3) > 0
        al = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.5)
        al_final[i] = np.maximum(al, np.maximum(Af("FINS", i), Af("MASK", i)))

    # ---- strict skin core + extremities ------------------------------------------------------------
    core, bbox, tips = {}, {}, {}
    for i in rng_f:
        oth = np.maximum.reduce([Af(c, i) for c in ("SHORTS", "FINS", "MASK", "HAIR", "WATCH")])
        c = cv2.erode(((Af("SKIN", i) > 0.98) & (oth < 0.01) & (al_final[i] > 0.98)).astype(np.uint8), e3, iterations=2) > 0
        if c.sum() < 300:
            c = cv2.erode((Af("SKIN", i) > 0.9).astype(np.uint8), e3) > 0
        core[i] = c
        ys, xs = np.nonzero(dil(person[i], 20))
        bbox[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        avoid = (Af("FINS", i) > 0.5) | (Af("SHORTS", i) > 0.5) | (Af("MASK", i) > 0.5) | (Af("HAIR", i) > 0.5)
        tips[i] = hand_tips(person[i], avoid)

    # ---- 1. ONE global correction from f0, f30, f60, f89 -------------------------------------------------
    gf = [f for f in GLOBAL_F if f < nfr] or [0]
    m_lin = np.mean([lin(frames[f])[core[f]].mean(0) for f in gf], 0)             # mean linear RGB of the skin
    t_lab = np.array([[[TARGET["L"], TARGET["a"], TARGET["b"]]]], np.float32)
    t_lin = lin(to_rgb(t_lab))[0, 0]
    g = (t_lin / np.maximum(m_lin, 1e-6)).astype(np.float32)                       # red up, cyan (G, B) down
    Yo_mean = float(np.mean([(lin(frames[f])[core[f]] @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean() for f in gf]))
    Y_t = float(lin(to_rgb(np.array([[[TARGET["L"], 0, 0]]], np.float32)))[0, 0, 0])
    gE = Y_t / max(Yo_mean, 1e-6)                                                  # one exposure gain
    print(f"[skin] global gains R {g[0]:.2f} G {g[1]:.2f} B {g[2]:.2f}, exposure x{gE:.2f} (from f{gf})", flush=True)
    # reference hue / chroma of the corrected skin (for the damped natural variation)
    allc = np.concatenate([to_lab(delin(np.clip(lin(frames[f])[core[f]] * g, 0, 1))[None])[0] for f in gf])
    sk = Skin(g, gE, float(hue_of(allc[:, 1].mean(), allc[:, 2].mean())), float(np.hypot(allc[:, 1], allc[:, 2]).mean()))

    # water colour next to the diver (for the cool shadows)
    def water_ab(i, bg):
        ring = dil(person[i], 40) & ~dil(person[i], 14)
        lab = to_lab(bg[ring][None])[0]
        return float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))

    # ---- pass A: skin colour per frame, core statistics for the step limiter ---------------------------------
    bgs = {}
    stats0 = []
    cache_skin = {}
    for i in rng_f:
        x = frames[i]
        bg = R.grade_background(x)                      # rule 9: one fixed setting for the whole clip
        bgs[i] = bg
        x0, y0, x1, y1 = bbox[i]
        xl = lin(x[y0:y1, x0:x1])
        sbin = sk_bin[i][y0:y1, x0:x1]
        thick = thickness(sbin)
        tipw = np.zeros(sbin.shape, np.float32)
        for tx, ty in tips[i]:
            cv2.circle(tipw, (tx - x0, ty - y0), 55, 1.0, -1)
        tipw = cv2.GaussianBlur(tipw, (0, 0), 10) * sbin
        wab = water_ab(i, bg)
        L, a_, b_ = sk.lab(xl, thick, tipw, wab)
        a_, b_ = edge_chroma(a_, b_, sbin)             # 6 px band: blue out, colour from the interior
        cache_skin[i] = (L, a_, b_)
        c = core[i][y0:y1, x0:x1]
        stats0.append([float(L[c].mean()), float(a_[c].mean()), float(b_[c].mean())])
    stats0 = np.array(stats0)
    # sudden steps only: L* 1.5, a* and b* 0.8 per frame
    adj = np.stack([limit_steps(stats0[:, k], lim) for k, lim in enumerate((LIM["L"], LIM["a"], LIM["b"]))], 1)
    off = adj - stats0
    print(f"[skin] clip mean before limiter L {stats0[:, 0].mean():.2f} a {stats0[:, 1].mean():.2f} b {stats0[:, 2].mean():.2f}; "
          f"largest step L {np.abs(np.diff(stats0[:, 0])).max():.2f} a {np.abs(np.diff(stats0[:, 1])).max():.2f} "
          f"b {np.abs(np.diff(stats0[:, 2])).max():.2f}; frames adjusted {int((np.abs(off) > 1e-6).any(1).sum())}", flush=True)

    # ---- encoder -----------------------------------------------------------------------------------------------
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

    # ---- render -----------------------------------------------------------------------------------------------------
    outs, rows = {}, []
    g_col = R.vertical_weight(h, w)
    hf = lambda Lc: Lc - cv2.GaussianBlur(Lc, (0, 0), 1.5)
    for i in rng_f:
        x = frames[i]
        bg = bgs[i]
        x0, y0, x1, y1 = bbox[i]
        L, a_, b_ = cache_skin[i]
        L = L + off[i, 0]
        a_ = a_ + off[i, 1]
        b_ = b_ + off[i, 2]
        skin_full = np.zeros_like(x)
        skin_full[y0:y1, x0:x1] = to_rgb(np.stack([L, a_, b_], -1))
        # SHORTS: neutral white, folds kept (luminance of the original, no colour)
        xb = np.clip(x * R.DV["gains"], 0, 1.5)
        yb = R.luma(xb)
        scale = 1 + (g_col[..., None] - 1) * R.DV["depth_mix"]
        shorts = np.clip(yb[..., None] * R.DV["lift"] * np.ones(3, np.float32) * scale, 0, 1)
        w_sk, w_sh = Af("SKIN", i), Af("SHORTS", i)
        w_orig = np.clip(Af("FINS", i) + Af("MASK", i) + Af("HAIR", i) + Af("WATCH", i), 0, 1)
        w_sk_eff = np.clip(1 - w_sh - w_orig, 0, 1)      # edge pixels not covered by a class are skin
        dv = skin_full * w_sk_eff[..., None] + shorts * w_sh[..., None] + x * w_orig[..., None]
        al = al_final[i]
        # rule 7: ~10 % water colour over the outer 4 px of the diver
        dist = cv2.distanceTransform((al > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        wrap_w = (0.10 * np.clip(1 - dist / 4.0, 0, 1) * (al > 0.02))[..., None]
        dv = dv * (1 - wrap_w) + bg * wrap_w
        out = bg * (1 - al[..., None]) + dv * al[..., None]
        # measurements on the finished frame, strict skin core
        c = core[i]
        lab_o = to_lab(out)
        Lc, ac, bc = lab_o[..., 0][c], lab_o[..., 1][c], lab_o[..., 2][c]
        hh = hue_of(ac, bc)
        Lorig = Y2L(lin(x) @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
        tex = float(np.std(hf(lab_o[..., 0])[c]) / max(np.std(hf(Lorig)[c]), 1e-6))
        sh_m, hi_m = Lc < 45, Lc > 70
        sm = cv2.erode((Af("SHORTS", i) > 0.9).astype(np.uint8), e3) > 0
        rows.append(dict(frame=i, L=float(Lc.mean()), a=float(ac.mean()), b=float(bc.mean()),
                         spread=float(np.percentile(hh, 95) - np.percentile(hh, 5)),
                         hue_shadow=float(hue_of(ac[sh_m].mean(), bc[sh_m].mean())) if sh_m.sum() > 30 else float("nan"),
                         hue_mid=float(hue_of(ac[~sh_m & ~hi_m].mean(), bc[~sh_m & ~hi_m].mean())) if (~sh_m & ~hi_m).sum() > 30 else float("nan"),
                         hue_high=float(hue_of(ac[hi_m].mean(), bc[hi_m].mean())) if hi_m.sum() > 30 else float("nan"),
                         chroma_shadow=float(np.hypot(ac[sh_m], bc[sh_m]).mean()) if sh_m.sum() > 30 else float("nan"),
                         chroma_high=float(np.hypot(ac[hi_m], bc[hi_m]).mean()) if hi_m.sum() > 30 else float("nan"),
                         frac_shadow=float(sh_m.mean()), frac_high=float(hi_m.mean()),
                         texture=tex,
                         shorts_a=float(lab_o[..., 1][sm].mean()) if sm.sum() > 50 else float("nan"),
                         shorts_b=float(lab_o[..., 2][sm].mean()) if sm.sum() > 50 else float("nan")))
        if i in SHEET:
            outs[i] = out.copy()
        if proc is not None:
            n_ = (rng.random(out.shape[:2] + (1,), np.float32) - rng.random(out.shape[:2] + (1,), np.float32)) / 255.0
            proc.stdin.write((np.clip(out + n_, 0, 1) * 65535.0 + 0.5).astype(np.uint16).tobytes())
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    # ---- tables -------------------------------------------------------------------------------------------------------
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    (qc / "skin_lab.csv").write_text(",".join(keys) + "\n" + "\n".join(",".join(f"{r[k]:.3f}" for k in keys) for r in rows) + "\n")
    Ls, As, Bs = (np.array([r[k] for r in rows]) for k in ("L", "a", "b"))
    summary = dict(
        gains=[float(v) for v in g], exposure=float(gE),
        clip_mean=dict(L=float(Ls.mean()), a=float(As.mean()), b=float(Bs.mean())),
        max_step=dict(L=float(np.abs(np.diff(Ls)).max()), a=float(np.abs(np.diff(As)).max()), b=float(np.abs(np.diff(Bs)).max())),
        spread_min=float(min(r["spread"] for r in rows)),
        texture_min=float(min(r["texture"] for r in rows)),
        shorts_ab_max=float(np.nanmax([max(abs(r["shorts_a"]), abs(r["shorts_b"])) for r in rows])))
    (qc / "skin_summary.json").write_text(json.dumps(summary, indent=1))
    print("[skin] summary", json.dumps(summary), flush=True)

    # ---- the one contact sheet: overlay (top), result (bottom) --------------------------------------------------------
    col = {"SKIN": (255, 170, 60), "SHORTS": (60, 120, 255), "FINS": (255, 60, 255), "MASK": (60, 255, 60),
           "HAIR": (170, 110, 40), "WATCH": (255, 40, 40)}
    th = 340
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
            cv2.putText(tile, f"f{i}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            lst.append(tile)
    sheet = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    leg = np.zeros((26, sheet.shape[1], 3), np.uint8)
    x_ = 8
    for c in ("SKIN", "HAIR", "SHORTS", "FINS", "MASK", "WATCH"):
        cv2.rectangle(leg, (x_, 6), (x_ + 16, 20), col[c], -1)
        cv2.putText(leg, c, (x_ + 22, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_ += 110
    cv2.putText(leg, "MASK > WATCH > FINS > SHORTS > HAIR > SKIN", (x_ + 20, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
    sheet = np.concatenate([sheet, leg], 0)
    cv2.imwrite(str(a.sheet), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[done] {a.output if not a.test else '(test)'}  sheet {a.sheet}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
