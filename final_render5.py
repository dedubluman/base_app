#!/usr/bin/env python3
"""Skin / face grade v5 + head, shorts, gear treatment + final 1080p render. The water grade is not touched.

B1  Luminance and texture come from the source; only hue / chroma are written (no flat fill, no per-frame mean pull).
B2  Tone-dependent colour: mid-tones warm peach-beige (hue ~53 deg, C* ~15), shadows cooler / less saturated with a
    little cyan ambient, highlights neutral and soft-kneed (no channel above 245); hue stays on the skin line.
B3  Regional warmth: fingertips / thin limbs, cheeks redder-warm; thick torso (belly) paler.  The source's own local
    redness is kept (damped) so knees, elbows, ears stay a little redder than the rest.
B4  Depth harmony: bright / warm near the surface, darker / cooler / less saturated when the water around the diver is
    deep (measured on the graded background next to the diver, smoothed over 9 frames).
B5  Water light: source caustics / top light stay in the luminance; a slight blue water reflection on the lower and edge
    parts of the body.
B6  FACE: same tone family, own adjustment (a touch redder, slightly more chroma, features kept); MASK glass dark,
    reflective, no skin colour; the snorkel tip keeps its yellow-green.
B7  SHORTS toned down (p95 ~ 0.84), folds kept, slight blue ambient in the shadows.  HAIR wet dark brown, varied.
B8  After the grade the source's grain (high-pass residual) is matched inside the masks.

    python final_render5.py --input inputs/diver.mov --output output/diver_graded_1080p.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import reel_match as R
from final_render3 import Y2L, delin, hue_of, lin, smoothstep, thickness, to_lab, to_rgb, wrap
from shades.io import iter_frames, probe

REPO = Path(__file__).resolve().parent
N = 90
SHEET = [2, 10, 33, 45, 75, 87]
ORDER = ["MASK", "GEAR", "FINS", "SHORTS", "HAIR", "FACE", "SKIN"]
KNEE = 75.0
CAP = 245.0 / 255.0
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
e3 = np.ones((3, 3), np.uint8)


def knee(L):
    return np.where(L > KNEE, KNEE + 4.0 * np.tanh((L - KNEE) / 4.0), L)


def ma(v, k):
    v = np.asarray(v, np.float64)
    p = k // 2
    return np.convolve(np.pad(v, p, mode="edge"), np.ones(k) / k, "valid")


def blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--sheet", type=Path, default=REPO / "output" / "contact_sheet_v5.jpg")
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--dump", type=Path, default=None)
    a = ap.parse_args()
    info = probe(a.input)
    w, h = info.width, info.height
    stem = a.input.stem
    cdir = REPO / "masks" / f"{stem}__cls5"
    nfr = a.test or N
    rng_f = range(nfr)
    frames = []
    for k, x in enumerate(iter_frames(a.input, info)):
        if k >= nfr:
            break
        frames.append(x)

    def rd(nm, i):
        im = cv2.imread(str(cdir / nm / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return np.zeros((h, w), np.float32) if im is None else im.astype(np.float32) / 255

    Wc = lambda c, i: rd(c, i)
    ALPHA = {i: rd("ALPHA", i) for i in rng_f}
    sbin = {i: (Wc("SKIN", i) + Wc("FACE", i)) > 0.5 for i in rng_f}
    union = {i: ALPHA[i] > 0.5 for i in rng_f}
    core = {}
    for i in rng_f:
        c = cv2.erode(((Wc("SKIN", i) > 0.98) & (ALPHA[i] > 0.98)).astype(np.uint8), e3, iterations=2) > 0
        if c.sum() < 300:
            c = cv2.erode((Wc("SKIN", i) > 0.9).astype(np.uint8), e3) > 0
        core[i] = c

    # ---- exposure: one global gain (L* follows the source), reference = the clip's skin luminance ---------------------
    Yo_all = [(lin(frames[f])[core[f]] @ LUMA).mean() for f in (0, 30, 60, 89) if f < nfr]
    Y_t = float(lin(to_rgb(np.array([[[59.0, 0, 0]]], np.float32)))[0, 0, 0])
    gE = Y_t / max(float(np.mean(Yo_all)), 1e-6)

    # ---- B4: depth from the graded background next to the diver, smoothed ----------------------------------------------
    Lbg = np.zeros(nfr)
    bg_cache = {}
    for i in rng_f:
        bg = R.grade_background(frames[i])
        d_out = cv2.distanceTransform((~dil_u(union[i], 1)).astype(np.uint8), cv2.DIST_L2, 3)
        ring = (d_out >= 25) & (d_out <= 70)
        Lbg[i] = float(np.median(to_lab(bg)[..., 0][ring])) if ring.sum() > 200 else np.nan
        if i in SHEET:
            bg_cache[i] = None
    Lbg = np.where(np.isnan(Lbg), np.nanmedian(Lbg), Lbg)
    Ls = ma(Lbg, 9)
    depth = np.clip((Ls.max() - Ls) / max(Ls.max() - Ls.min(), 1e-3), 0, 1)
    depth = ma(depth, 5)
    # per-frame exposure: the skin's median L* follows a depth target (bright near the surface, darker in the deep),
    # luminance structure inside the frame is the source's; smoothed over 7 frames so the change is gradual
    Yc = np.array([float(np.median((lin(frames[i])[core[i]] @ LUMA))) for i in rng_f])
    L_t = 58.5 - 8.0 * depth
    Y_tar = ((L_t + 16.0) / 116.0) ** 3
    gEf = np.exp(ma(np.log(np.clip(Y_tar / np.maximum(Yc, 1e-6), 0.3, 3.0)), 7))
    print(f"[depth] background L* {Lbg.min():.1f}-{Lbg.max():.1f}; depth 0..1 at f2 {depth[2]:.2f} f33 {depth[min(33, nfr - 1)]:.2f} "
          f"f75 {depth[min(75, nfr - 1)]:.2f} f87 {depth[min(87, nfr - 1)]:.2f}", flush=True)

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
    outs, rows = {}, []
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    only = {int(t) for t in a.only.split(",") if t.strip()}
    picks = pick_closeups(frames, cdir, h, w, nfr)
    print(f"[sheet] close-up frames {picks}", flush=True)
    for i in rng_f:
        if only and i not in only and i not in picks:
            continue
        x = frames[i]
        bg = R.grade_background(x)                               # the water grade is not touched
        A = ALPHA[i]
        ys, xs = np.nonzero(A > 0.01)
        x0, y0, x1, y1 = max(xs.min() - 30, 0), max(ys.min() - 30, 0), min(xs.max() + 31, w), min(ys.max() + 31, h)
        xc = x[y0:y1, x0:x1]
        xl = lin(xc)
        Yo = xl @ LUMA
        dep = float(depth[i])
        # B1/B4: luminance follows the source; the clip-wide exposure and the depth (slightly darker when deep)
        Lsk = knee(Y2L(np.clip(Yo * float(gEf[i]), 0, 1.2))).astype(np.float32)
        Lsk = Lsk + 9.0 * (1.0 - smoothstep(0.0, 46.0, Lsk))                    # shadows read as shaded skin, not black
        wsk = Wc("SKIN", i)[y0:y1, x0:x1]
        wfc = Wc("FACE", i)[y0:y1, x0:x1]
        sb = (wsk + wfc) > 0.5
        thick = thickness(sb)
        # source's own local hue / chroma variation (from a provisional red-gain correction), damped
        c = core[i][y0:y1, x0:x1]
        med_lin = np.median(xl[c], 0)
        gprov = (lin(to_rgb(np.array([[[np.median(Lsk[c]), 12.7, 14.1]]], np.float32)))[0, 0] / np.maximum(med_lin, 1e-6)).astype(np.float32)
        lab_c = to_lab(delin(np.clip(xl * gprov, 0, 1)))
        hc = hue_of(lab_c[..., 1], lab_c[..., 2])
        Cc = np.hypot(lab_c[..., 1], lab_c[..., 2])
        hm = float(hue_of(np.median(lab_c[..., 1][c]), np.median(lab_c[..., 2][c])))
        Cm = float(np.median(Cc[c]))
        dh = blur(np.clip(wrap(hc - hm), -20, 20).astype(np.float32), 4.0)
        dC = blur(np.clip(Cc / max(Cm, 1e-3) - 1.0, -0.5, 0.5).astype(np.float32), 4.0)

        # ---- B2 tone curve: warm peach-beige mids, cooler / quieter shadows, neutral soft highlights -------------------
        w_sh = 1.0 - smoothstep(36.0, 52.0, Lsk)
        w_hi = smoothstep(68.0, 80.0, Lsk)
        bell = np.exp(-(((Lsk - 58.0) / 17.0) ** 2))
        C = (6.0 + 9.2 * bell) * (1.0 - 0.30 * w_sh) * (1.0 - 0.45 * w_hi)
        hh = 55.0 + 7.0 * w_sh - 4.0 * w_hi
        # B3 regional warmth
        tips = np.zeros(sb.shape, np.float32)
        from diver_classes import hand_tips
        pers = union[i]
        avoid = (Wc("FINS", i) > 0.5) | (Wc("SHORTS", i) > 0.5) | (Wc("MASK", i) > 0.5) | (Wc("HAIR", i) > 0.5)
        for tx, ty in hand_tips(pers, avoid):
            cv2.circle(tips, (tx - x0, ty - y0), 55, 1.0, -1)
        tips = blur(tips, 9.0) * sb
        thin = 1.0 - smoothstep(22.0, 46.0, thick)                        # thin limbs / fingers / elbows / knees
        pale = smoothstep(34.0, 62.0, thick)                              # torso interior (belly) paler
        hh = hh + 0.7 * dh - 5.0 * tips - 2.0 * thin
        C = C * (1.0 + 0.5 * dC) * (1.0 + 0.30 * tips + 0.10 * thin) * (1.0 - 0.14 * pale)
        # B4 depth: warm near the surface, cooler / quieter in the deep
        hh = hh + 4.5 * dep - 1.5 * (1 - dep)
        C = C * (1.0 + 0.06 * (1 - dep) - 0.14 * dep)
        # B6 face: own adjustment, a touch redder and more chromatic (cheeks), features stay (luminance is the source's)
        fc = np.clip(wfc / np.maximum(wsk + wfc, 1e-3), 0, 1)
        fin = cv2.distanceTransform((wfc > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        cheek = smoothstep(2.0, 9.0, fin)
        hh = hh - fc * (2.0 + 2.5 * cheek)
        C = C * (1.0 + fc * (0.05 + 0.10 * cheek))
        a_ = C * np.cos(np.radians(hh))
        b_ = C * np.sin(np.radians(hh))
        # shadows: slight cyan-blue ambient (B2); B5: blue water reflection on the lower / edge parts
        a_ = a_ - 1.6 * w_sh
        b_ = b_ - 1.8 * w_sh
        din = cv2.distanceTransform(union[i][y0:y1, x0:x1].astype(np.uint8), cv2.DIST_L2, 3)
        yy = np.linspace(0.0, 1.0, y1 - y0, dtype=np.float32)[:, None]
        rim = smoothstep(16.0, 1.5, din) * (0.40 + 0.60 * yy) * (0.85 + 0.30 * dep)
        a_ = a_ - 2.2 * rim
        b_ = b_ - 5.0 * rim
        # edge band colour from the interior (no grey / blue fringe from mixed pixels)
        interior = cv2.erode(sb.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        sk_ext = cv2.dilate(sb.astype(np.uint8), k7) > 0
        if interior.sum() > 200:
            _, lbl = cv2.distanceTransformWithLabels((~interior).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            coords = np.argwhere(interior)
            lb = np.clip(lbl - 1, 0, len(coords) - 1)
            an, bn = a_[coords[lb, 0], coords[lb, 1]], b_[coords[lb, 0], coords[lb, 1]]
            ww = blur(((~interior) & sk_ext).astype(np.float32), 1.2) * sk_ext
            a_, b_ = a_ * (1 - ww) + an * ww, b_ * (1 - ww) + bn * ww
        rgb = to_rgb(np.stack([Lsk, a_, b_], -1))
        rgb = rgb * np.minimum(1.0, CAP / np.maximum(rgb.max(-1, keepdims=True), 1e-6))
        skin_full = np.zeros_like(x)
        skin_full[y0:y1, x0:x1] = rgb

        # ---- B7 SHORTS: toned down, folds kept, slight blue ambient in the shadows ---------------------------------------
        xb = np.clip(x * R.DV["gains"], 0, 1.5)
        yb = R.luma(xb)
        scale = 1 + (g_col[..., None] - 1) * R.DV["depth_mix"]
        sh = np.clip(yb[..., None] * R.DV["lift"] * np.ones(3, np.float32) * scale, 0, 1)
        shw = Wc("SHORTS", i) > 0.5
        if shw.sum() > 200:
            p95 = float(np.percentile(sh[..., 1][shw], 95))
            sh = sh * (0.80 / max(p95, 1e-3))
        Ysh = sh[..., 1:2]
        amb = (1.0 - smoothstep(0.35, 0.75, Ysh))                          # shadows / folds
        sh = np.clip(sh + amb * np.array([-0.020, 0.000, 0.040], np.float32) + np.array([-0.006, 0.0, 0.012], np.float32), 0, 1)

        # ---- head, gear, fins ---------------------------------------------------------------------------------------------
        Yg = (x @ LUMA)
        Yl = lin(np.repeat(Yg[..., None], 3, 2))[..., 0]
        # HAIR: wet dark brown, not one tone (luminance from the source, hue / chroma vary with it and with the local detail)
        Lh = np.clip(Y2L(np.clip(Yl * gE * 0.50, 0, 1)), 5, 30).astype(np.float32)
        detail = Lh - blur(Lh, 3.0)
        Ch = (1.6 + 2.8 * smoothstep(7.0, 20.0, Lh) * (1.0 - smoothstep(26.0, 34.0, Lh))).astype(np.float32) * (1.0 + 0.5 * np.clip(detail / 6.0, -0.6, 0.8))
        hhh = (30.0 + 12.0 * (blur(Lh, 6.0) - 20.0) / 20.0).astype(np.float32)
        hair = to_rgb(np.stack([Lh, Ch * np.cos(np.radians(hhh)), Ch * np.sin(np.radians(hhh))], -1))
        # MASK glass / strap: dark, reflective, no skin colour (neutral, slightly cool; specular stays)
        Lm = np.clip(Y2L(np.clip(Yl * gE * 0.75, 0, 1)), 4, 70).astype(np.float32)
        Lm = Lm + 5.0 * (1.0 - smoothstep(0.0, 30.0, Lm))
        glass = to_rgb(np.stack([Lm, -0.8 * np.ones_like(Lm), -3.0 * np.ones_like(Lm)], -1))
        hsv = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        tip = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 55) & (hsv[..., 1] > 80) & (hsv[..., 2] > 70)).astype(np.float32)[..., None]
        glass = glass * (1 - tip) + x * tip                                # the snorkel tip keeps its own colour
        gray = np.repeat(Yg[..., None], 3, 2)                              # fins / watch: source luminance, no water tint
        fins = np.clip(gray + np.array([-1.0, 0.0, 2.0], np.float32) / 255.0, 0, 1)
        wmask = lambda c: Wc(c, i)[..., None]
        wsk_f = Wc("SKIN", i) + Wc("FACE", i)
        dv = (skin_full * wsk_f[..., None] + sh * wmask("SHORTS") + hair * wmask("HAIR") + glass * wmask("MASK")
              + fins * wmask("FINS") + gray * wmask("GEAR"))
        wsum = np.maximum(sum(Wc(c, i) for c in ORDER), 1e-4)[..., None]
        dv = dv / wsum
        # light blue wrap at the very edge (the diver sits in the water, B5)
        dist = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        wrap_w = (0.08 * np.clip(1 - dist / 4.0, 0, 1) * (A > 0.02) * (1 - Wc("SHORTS", i)))[..., None]
        dv = dv * (1 - wrap_w) + bg * wrap_w
        out = bg * (1 - A[..., None]) + dv * A[..., None]

        # ---- B8: grain match inside the masks ------------------------------------------------------------------------------
        body = cv2.erode((A > 0.95).astype(np.uint8), e3, iterations=3) > 0
        hf = lambda im: im - np.stack([blur(im[..., k], 1.3) for k in range(3)], -1)
        if body.sum() > 500:
            hs, ho = hf(x)[body], hf(out)[body]
            s_src = 1.4826 * np.median(np.abs(hs - np.median(hs, 0)), 0)
            s_out = 1.4826 * np.median(np.abs(ho - np.median(ho, 0)), 0)
            add = np.sqrt(np.maximum(s_src ** 2 - s_out ** 2, 0)).astype(np.float32)
            nz = np.stack([blur(rng.standard_normal((h, w)).astype(np.float32), 0.6) for _ in range(3)], -1)
            nz = nz / max(float(nz.std()), 1e-6)
            out = np.clip(out + nz * add[None, None, :] * A[..., None], 0, 1)
            ho2 = hf(out)[body]
            s_out2 = 1.4826 * np.median(np.abs(ho2 - np.median(ho2, 0)), 0)
        else:
            s_src = s_out = s_out2 = np.zeros(3)

        # ---- numbers ----------------------------------------------------------------------------------------------------------
        lab_o = to_lab(out)
        cc = core[i]
        sel = {"sh": cc & (lab_o[..., 0] < 45), "mid": cc & (lab_o[..., 0] >= 45) & (lab_o[..., 0] < 66), "hi": cc & (lab_o[..., 0] >= 66)}
        rec = dict(frame=i, depth=dep, L=float(np.median(lab_o[..., 0][cc])))
        for k_, m_ in sel.items():
            if m_.sum() > 80:
                rec[f"h_{k_}"] = float(hue_of(np.median(lab_o[..., 1][m_]), np.median(lab_o[..., 2][m_])))
                rec[f"C_{k_}"] = float(np.hypot(np.median(lab_o[..., 1][m_]), np.median(lab_o[..., 2][m_])))
            else:
                rec[f"h_{k_}"] = rec[f"C_{k_}"] = float("nan")
        r255 = np.median(out[cc], 0) * 255
        rec["rg"] = float(r255[0] / max(r255[1], 1e-6))
        rec["over250"] = float((out[cc].max(-1) * 255 > 250).mean())
        rec["grain_src"] = float(np.mean(s_src) * 255)
        rec["grain_before"] = float(np.mean(s_out) * 255)
        rec["grain_after"] = float(np.mean(s_out2) * 255)
        rows.append(rec)
        if i in SHEET or i in picks:
            outs[i] = out.copy()
            if a.dump is not None:
                a.dump.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(a.dump / f"g{i:05d}.png"), cv2.cvtColor((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if proc is not None:
            n_ = (rng.random(out.shape[:2] + (1,), np.float32) - rng.random(out.shape[:2] + (1,), np.float32)) / 255.0
            proc.stdin.write((np.clip(out + n_, 0, 1) * 65535.0 + 0.5).astype(np.uint16).tobytes())
        if i % 10 == 0:
            print(f"[render] f{i}", flush=True)
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    (qc / "skin_lab_v5.csv").write_text(",".join(keys) + "\n" + "\n".join(",".join(f"{r[k]:.4f}" for k in keys) for r in rows) + "\n")
    def stat(k):
        v = np.array([r[k] for r in rows], float)
        return [float(np.nanmin(v)), float(np.nanmedian(v)), float(np.nanmax(v))]
    ctrl = {k: stat(k) for k in ("h_sh", "h_mid", "h_hi", "C_sh", "C_mid", "C_hi", "rg", "L", "over250", "grain_src", "grain_before", "grain_after")}
    (qc / "controls_v5.json").write_text(json.dumps(ctrl, indent=1))
    print("[controls]", json.dumps(ctrl), flush=True)

    sheet_tiles(frames, outs, cdir, h, w, a.sheet, nfr, picks)
    print(f"[done] {a.output if not a.test else '(test)'}  sheet {a.sheet}", flush=True)
    return 0


def dil_u(m, r):
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def pick_closeups(frames, cdir, h, w, nfr):
    """The two frames where the face is clearest: largest FACE area x sharpness, at least 3 frames apart."""
    cand = []
    for i in range(70, min(nfr, 90)):
        im = cv2.imread(str(cdir / "FACE" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        fm = im > 127
        if fm.sum() < 300:
            continue
        g = cv2.cvtColor((frames[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_32F)[fm].var()
        cand.append((fm.sum() * lap, i))
    cand.sort(reverse=True)
    pick = []
    for _, i in cand:
        if all(abs(i - j) >= 3 for j in pick):
            pick.append(i)
        if len(pick) == 2:
            break
    return sorted(pick)


def sheet_tiles(frames, outs, cdir, h, w, path, nfr, picks):
    """Top: class masks (SKIN, FACE, HAIR, MASK, SHORTS, FINS, GEAR), bottom: the grade; f2 f10 f33 f45 f75 f87 + the two
    clearest face frames as close-ups."""
    col = {"SKIN": (255, 170, 60), "FACE": (255, 255, 40), "HAIR": (170, 110, 40), "MASK": (60, 255, 60),
           "SHORTS": (60, 120, 255), "FINS": (255, 60, 255), "GEAR": (255, 40, 40)}

    def rdw(c, i):
        im = cv2.imread(str(cdir / c / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return np.zeros((h, w), np.float32) if im is None else im.astype(np.float32) / 255

    th = 440
    top, bot = [], []
    for i in [f for f in SHEET if f < nfr and f in outs]:
        A = rdw("ALPHA", i)
        ys, xs = np.nonzero(A > 0.5)
        x0, x1 = max(0, xs.min() - 40), min(w, xs.max() + 40)
        y0, y1 = max(0, ys.min() - 40), min(h, ys.max() + 40)
        ov = (frames[i] * 255).astype(np.float32)
        for c in ("SKIN", "FACE", "HAIR", "MASK", "SHORTS", "FINS", "GEAR"):
            m = rdw(c, i) > 0.5
            ov[m] = 0.45 * ov[m] + 0.55 * np.array(col[c], np.float32)
        for lst, src in ((top, ov), (bot, outs[i] * 255)):
            tile = np.clip(src[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            tile = cv2.resize(tile, (int(tile.shape[1] * th / tile.shape[0]), th), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"f{i}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            lst.append(tile)
    pick = picks
    for i in pick:
        if i not in outs:
            continue
        hd = (rdw("FACE", i) + rdw("HAIR", i) + rdw("MASK", i)) > 0.5
        ys, xs = np.nonzero(hd)
        x0, x1 = max(0, xs.min() - 30), min(w, xs.max() + 30)
        y0, y1 = max(0, ys.min() - 30), min(h, ys.max() + 30)
        ov = (frames[i] * 255).astype(np.float32)
        for c in ("SKIN", "FACE", "HAIR", "MASK", "SHORTS", "FINS", "GEAR"):
            m = rdw(c, i) > 0.5
            ov[m] = 0.45 * ov[m] + 0.55 * np.array(col[c], np.float32)
        for lst, src in ((top, ov), (bot, outs[i] * 255)):
            tile = np.clip(src[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            tile = cv2.resize(tile, (int(tile.shape[1] * th / tile.shape[0]), th), interpolation=cv2.INTER_CUBIC)
            cv2.putText(tile, f"f{i} close-up", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            lst.append(tile)
    sheet = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    leg = np.zeros((26, sheet.shape[1], 3), np.uint8)
    x_ = 8
    for c, nm in (("SKIN", "SKIN"), ("FACE", "FACE"), ("HAIR", "HAIR"), ("MASK", "MASK"), ("SHORTS", "SHORTS"), ("FINS", "FINS"), ("GEAR", "GEAR (watch)")):
        cv2.rectangle(leg, (x_, 6), (x_ + 16, 20), col[c], -1)
        cv2.putText(leg, nm, (x_ + 22, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_ += 130
    sheet = np.concatenate([sheet, leg], 0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    raise SystemExit(main())
