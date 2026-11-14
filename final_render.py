#!/usr/bin/env python3
"""Part 2 - skin colour in CIELAB inside the SKIN mask, then the final 1080p render.

Targets (SKIN mean, CIELAB): a* 15-17, b* 19-22, saturation C*/sqrt(C*^2+L*^2)
40-45 %, hue angle 50-54 deg, L* mean 58-62, shadows (p5) 38-42, highlights <= 76.

Per frame, inside SKIN only, never a flat fill:
 1. red gain by depth (solved from the skin's own R/G ratio), then saturation;
    hands get 15-20 % extra red gain and their blue cast is removed
 2. L* keeps the original texture: percentiles p5/p50/p95 -> 40/60/74, mid gamma
    0.90, a light S curve, soft cap at 76
 3. hue locked near 52 deg (35 % of the pixel's own hue variation kept), chroma
    set so the mean hits the target (50 % of its variation kept)
 4. every gain/anchor is smoothed over 7 frames; the per-frame L*/C*/hue
    offsets are solved so the means sit on target (frame-to-frame L* <= 1.5)
SHORTS stay neutral white, FINS and MASK are not touched.

    python final_render.py --input inputs/diver.mov --output output/diver_graded_1080p.mp4
"""
from __future__ import annotations

import argparse
import colorsys
import dataclasses
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import matte as M
import reel_match as R
from diver_classes import hand_tips, read_mask, dil
from diver_reels import drop_specks
from shades.io import iter_frames, probe

REPO = Path(__file__).resolve().parent
N = 90
FR = [0, 12, 28, 46, 60, 70, 89]                      # contact-sheet frames
TABLE_FR = [0, 28, 54, 70, 89]
T = dict(L=59.5, a=16.6, b=21.2, p5=40.0, p50=60.0, p95=74.0, cap=76.0)
T["C"] = float(np.hypot(T["a"], T["b"]))
T["h"] = float(np.degrees(np.arctan2(T["b"], T["a"])))
GAMMA, SCURVE = 0.90, 0.60
HAND_R, HAND_R_STRONG, HAND_B = 1.15, 1.20, 0.85
BG_MID_LIFT = 0.13                                     # rule 7: +10-15 % in the mid tones


def lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def delin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def to_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2LAB)


def to_rgb(lab):
    return cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_LAB2RGB)


def sat_ab(L, a, b):
    c = np.hypot(a, b)
    return c / np.sqrt(c * c + L * L)


T_RGB = to_rgb(np.array([[[T["L"], T["a"], T["b"]]]], np.float32))[0, 0]
T_LIN = lin(T_RGB)
T_RG = float(T_LIN[0] / T_LIN[1])


def smooth7(v):
    v = np.asarray(v, np.float64)
    k = np.ones(7) / 7
    return np.convolve(np.pad(v, 3, mode="edge"), k, "valid")


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def map_lab(Lc, ac, bc, P):
    """Corrected pixel Lab -> graded Lab for frame params P."""
    L1 = np.interp(Lc, [P["p5"], P["p50"], P["p95"]], [T["p5"], T["p50"], T["p95"]])
    lo, hi = Lc < P["p5"], Lc > P["p95"]
    L1 = np.where(lo, T["p5"] + (Lc - P["p5"]) * 0.8, L1)
    L1 = np.where(hi, T["p95"] + (Lc - P["p95"]) * 0.6, L1)
    t = np.clip((L1 - T["p5"]) / (T["p95"] - T["p5"]), 0, 1)
    tg = t ** GAMMA                                         # mid-tone gamma 0.90
    ts = tg + SCURVE * tg * (1 - tg) * (2 * tg - 1)         # light S curve
    inside = (L1 >= T["p5"]) & (L1 <= T["p95"])
    L2 = np.where(inside, T["p5"] + (T["p95"] - T["p5"]) * ts, L1) + P["kL"]
    over = L2 > 72.0                                       # soft cap, never above 76
    L3 = np.where(over, 72.0 + 4.0 * np.tanh((L2 - 72.0) / 4.0), L2)
    L3 = np.where(L3 < 34.0, 34.0 - 3.0 * np.tanh((34.0 - L3) / 3.0), L3)
    Cc = np.hypot(ac, bc)
    hc = np.degrees(np.arctan2(bc, ac))
    cvar = np.clip((Cc / max(P["Cm"], 1e-3)) ** 0.5, 0.6, 1.5)
    hvar = np.clip(wrap(hc - P["hm"]), -20, 20) * 0.35
    tt = np.clip((L3 - T["p5"]) / (T["p95"] - T["p5"]), 0, 1)
    C = P["kC"] * T["C"] * cvar * (0.92 + 0.08 * tt)
    h = np.radians(T["h"] + P["kH"] + np.clip(hvar, -8, 8))
    return L3, C * np.cos(h), C * np.sin(h)


def correct(xl, g_r, hw, strong):
    """Linear RGB -> red-gain corrected, blue cast removed in the hands."""
    gain = g_r * (1.0 + (HAND_R_STRONG if strong else HAND_R) * hw - hw)   # +15-20 % on the hands
    xc = xl.copy()
    xc[..., 0] = xc[..., 0] * gain
    xc[..., 2] = xc[..., 2] * (1.0 - (1.0 - HAND_B) * hw)
    return np.clip(xc, 0, 1)


def solve(core_lab, P0):
    """kL, kC, kH so that the core's mean L*, chroma and hue sit on target."""
    P = dict(P0)
    P["kL"], P["kC"], P["kH"] = 0.0, 1.0, 0.0
    for _ in range(5):
        L, a, b = map_lab(core_lab[:, 0], core_lab[:, 1], core_lab[:, 2], P)
        P["kL"] += T["L"] - float(L.mean())
        ma, mb = float(a.mean()), float(b.mean())
        P["kC"] *= T["C"] / max(float(np.hypot(ma, mb)), 1e-3)
        P["kH"] += T["h"] - float(np.degrees(np.arctan2(mb, ma)))
    return P


def measure(L, a, b):
    ang = float(np.degrees(np.arctan2(b.mean(), a.mean())))
    return dict(L=float(L.mean()), a=float(a.mean()), b=float(b.mean()),
                sat=float(sat_ab(L.mean(), a.mean(), b.mean())) * 100, hue=ang,
                p5=float(np.percentile(L, 5)), Lmax=float(L.max()))


def in_target(m):
    return (15 <= m["a"] <= 17 and 19 <= m["b"] <= 22 and 40 <= m["sat"] <= 45
            and 50 <= m["hue"] <= 54 and 58 <= m["L"] <= 62 and 38 <= m["p5"] <= 42 and m["Lmax"] <= 76)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--sheet", type=Path, default=REPO / "output" / "contact_sheet_final.jpg")
    a = ap.parse_args()
    info = probe(a.input)
    w, h = info.width, info.height
    stem = a.input.stem
    cdir = REPO / "masks" / f"{stem}__cls"
    pdir = REPO / "masks" / f"{stem}__fwd" / "subject"
    frames = []
    for k, x in enumerate(iter_frames(a.input, info)):
        if k >= N:
            break
        frames.append(x)

    def rd(nm, i):
        im = cv2.imread(str(cdir / nm / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return np.zeros((h, w), np.float32) if im is None else im.astype(np.float32) / 255

    person = {i: drop_specks(read_mask(pdir, i, (h, w))) for i in range(N)}
    A = {c: {i: rd(c, i) for i in range(N)} for c in ("SKIN", "SHORTS", "FINS", "MASK", "HAIR", "WATCH")}
    bin_ = {c: {i: A[c][i] > 0.5 for i in range(N)} for c in A}

    # ---- diver alpha: SAM person mask snapped to the edge, 1 px in, 3 px feather ----
    alpha8 = {i: np.clip(M.snap_diver_alpha(person[i], frames[i]) * 255 + 0.5, 0, 255).astype(np.uint8) for i in range(N)}
    med = {i: np.median(np.stack([alpha8[max(0, i - 1)], alpha8[i], alpha8[min(N - 1, i + 1)]]), 0).astype(np.uint8)
           for i in range(N)}
    e3 = np.ones((3, 3), np.uint8)
    al_final = {}
    for i in range(N):
        b = cv2.erode((med[i] > 127).astype(np.uint8), e3) > 0
        al = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.5)
        al_final[i] = np.maximum(al, A["FINS"][i])        # fins outside the person mask are the diver too

    # diver luma reference (as before: median balanced luma, smoothed)
    yd = []
    for i in range(N):
        core = cv2.erode((med[i] > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        yd.append(float(np.median(R.luma(np.clip(frames[i] * R.DV["gains"], 0, 1.5))[core])) if core.sum() > 200 else np.nan)
    yd = np.array(yd)
    t = np.arange(N)
    yd = np.interp(t, t[~np.isnan(yd)], yd[~np.isnan(yd)])
    kk = np.exp(-0.5 * (np.arange(-18, 19) / 6.0) ** 2)
    yref = np.convolve(np.pad(yd, 18, mode="edge"), kk / kk.sum(), "valid")

    # ---- hands: disks around the hull extremities of the diver, inside SKIN ----
    hands = {}
    for i in range(N):
        avoid = bin_["FINS"][i] | bin_["SHORTS"][i] | bin_["MASK"][i] | bin_["HAIR"][i]
        d = np.zeros((h, w), np.uint8)
        for x, y in hand_tips(person[i], avoid):
            cv2.circle(d, (x, y), 75, 1, -1)
        hands[i] = cv2.GaussianBlur(d.astype(np.float32), (0, 0), 8) * (A["SKIN"][i] > 0.3)

    # ---- pass A: red gain by depth (solved from the skin's own R/G), smoothed 7 ----
    core, bbox = {}, {}
    gr_raw = []
    for i in range(N):
        sk = A["SKIN"][i]
        c = cv2.erode((sk > 0.9).astype(np.uint8), e3) > 0
        core[i] = c
        ys, xs = np.nonzero(dil(person[i], 20))
        bbox[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        xl = lin(frames[i])[c]
        gr_raw.append(float(np.clip(T_RG / (xl[:, 0].mean() / max(xl[:, 1].mean(), 1e-6)), 1.0, 3.5)))
    g_r = smooth7(gr_raw)
    strong = {i: 66 <= i <= 74 for i in range(N)}

    # ---- pass B: anchors, solved offsets, 7-frame smoothing ----
    core_lab, anchor = {}, []
    for i in range(N):
        xl = lin(frames[i])
        xc = correct(xl[core[i]][None], g_r[i], hands[i][core[i]][None, :, None] * 1.0, strong[i])[0]
        lab = to_lab(delin(xc)[None])[0]
        core_lab[i] = lab
        hh = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
        mh = float(np.degrees(np.arctan2(np.sin(np.radians(hh)).mean(), np.cos(np.radians(hh)).mean())))
        anchor.append([np.percentile(lab[:, 0], 5), np.percentile(lab[:, 0], 50), np.percentile(lab[:, 0], 95),
                       float(np.hypot(lab[:, 1], lab[:, 2]).mean()), mh])
    anchor = np.array(anchor)
    sm = np.stack([smooth7(anchor[:, k]) for k in range(5)], 1)
    sm[:, 1] = np.maximum(sm[:, 1], sm[:, 0] + 2)          # keep p5 < p50 < p95
    sm[:, 2] = np.maximum(sm[:, 2], sm[:, 1] + 2)
    params, stats = [], []
    for i in range(N):
        P0 = dict(p5=sm[i, 0], p50=sm[i, 1], p95=sm[i, 2], Cm=sm[i, 3], hm=sm[i, 4])
        P = solve(core_lab[i], P0)
        params.append(P)
    for k in ("kL", "kC", "kH"):                           # offsets smoothed too (7 frames)
        v = smooth7([p[k] for p in params])
        for i in range(N):
            params[i][k] = v[i]
    for i in range(N):
        L, a_, b_ = map_lab(core_lab[i][:, 0], core_lab[i][:, 1], core_lab[i][:, 2], params[i])
        m = measure(L, a_, b_)
        if not in_target(m):                               # off target: re-solve this frame alone
            params[i] = solve(core_lab[i], {k: params[i][k] for k in ("p5", "p50", "p95", "Cm", "hm")})
            L, a_, b_ = map_lab(core_lab[i][:, 0], core_lab[i][:, 1], core_lab[i][:, 2], params[i])
            m = measure(L, a_, b_)
        stats.append(m)
    dL = np.abs(np.diff([m["L"] for m in stats]))
    print(f"[skin] frame-to-frame L* max {dL.max():.2f}; in target {sum(in_target(m) for m in stats)}/{N}", flush=True)

    # ---- encoder (1080p H.264) ----------------------------------------------------
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

    # ---- render --------------------------------------------------------------------
    final_stats, outs, bg_ratio = {}, {}, []
    for i in range(N):
        x = frames[i]
        bg = R.grade_background(x)
        x0, y0, x1, y1 = bbox[i]
        xl = lin(x[y0:y1, x0:x1])
        hw = hands[i][y0:y1, x0:x1][..., None]
        xc = correct(xl, g_r[i], hw, strong[i])
        lab = to_lab(delin(xc))
        L, a_, b_ = map_lab(lab[..., 0], lab[..., 1], lab[..., 2], params[i])
        skin_full = np.zeros_like(x)
        skin_full[y0:y1, x0:x1] = to_rgb(np.stack([L, a_, b_], -1))
        parts = {"shorts": A["SHORTS"][i], "hair": A["HAIR"][i],
                 "fins": np.clip(np.maximum(np.maximum(A["FINS"][i], A["MASK"][i]), A["WATCH"][i]), 0, 1)}
        dv = R.grade_diver(x, yref[i], depth=g_col[..., None], parts=parts, strict=True, skin_rgb=skin_full)
        al = al_final[i][..., None]
        out = bg * (1 - al) + dv * al
        # measure the finished frame inside SKIN (alpha > 0.9, one pixel in)
        mk = core[i]
        lab_o = to_lab(out)
        final_stats[i] = measure(lab_o[..., 0][mk], lab_o[..., 1][mk], lab_o[..., 2][mk])
        # shorts neutrality, for the report
        sm_ = cv2.erode((A["SHORTS"][i] > 0.9).astype(np.uint8), e3) > 0
        if sm_.sum() > 50:
            final_stats[i]["shorts_a"] = float(lab_o[..., 1][sm_].mean())
            final_stats[i]["shorts_b"] = float(lab_o[..., 2][sm_].mean())
        free = ~dil(person[i], 30)
        bgm = free & (R.luma(bg) > 0.10) & (R.luma(bg) < 0.45)
        bg0 = R.grade_background(x, dict(R.BG, mid_lift=0.0)) if i in (0, 28, 54, 89) else None
        if bg0 is not None and bgm.sum() > 1000:
            bg_ratio.append(float(np.median(R.luma(bg)[bgm]) / max(np.median(R.luma(bg0)[bgm]), 1e-6)))
        if i in FR:
            outs[i] = out.copy()
        n_ = (rng.random(out.shape[:2] + (1,), np.float32) - rng.random(out.shape[:2] + (1,), np.float32)) / 255.0
        proc.stdin.write((np.clip(out + n_, 0, 1) * 65535.0 + 0.5).astype(np.uint16).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")

    # ---- tables ------------------------------------------------------------------------
    rows = ["frame,L,a,b,sat_pct,hue_deg,p5,Lmax,shorts_a,shorts_b"]
    for i in range(N):
        s = final_stats[i]
        rows.append(f"{i},{s['L']:.2f},{s['a']:.2f},{s['b']:.2f},{s['sat']:.1f},{s['hue']:.1f},{s['p5']:.1f},{s['Lmax']:.1f},"
                    f"{s.get('shorts_a', float('nan')):.2f},{s.get('shorts_b', float('nan')):.2f}")
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "skin_lab.csv").write_text("\n".join(rows) + "\n")
    Ls = [final_stats[i]["L"] for i in range(N)]
    summary = dict(
        in_target=int(sum(in_target(final_stats[i]) for i in range(N))),
        frame_to_frame_L_max=float(np.abs(np.diff(Ls)).max()),
        L_range=[min(Ls), max(Ls)],
        shorts_ab_max=float(np.nanmax([max(abs(final_stats[i].get("shorts_a", 0)), abs(final_stats[i].get("shorts_b", 0))) for i in range(N)])),
        bg_mid_ratio=float(np.mean(bg_ratio)) if bg_ratio else None,
        off_target=[i for i in range(N) if not in_target(final_stats[i])])
    (qc / "skin_summary.json").write_text(json.dumps(summary, indent=1))
    print("[skin] summary", json.dumps(summary), flush=True)

    # ---- the one contact sheet: mask overlay (top), graded result (bottom) -----------
    col = {"SKIN": (255, 170, 60), "SHORTS": (60, 120, 255), "FINS": (255, 60, 255), "MASK": (60, 255, 60)}
    th = 400
    top, bot = [], []
    for i in FR:
        ys, xs = np.nonzero(person[i])
        x0, x1 = max(0, xs.min() - 50), min(w, xs.max() + 50)
        y0, y1 = max(0, ys.min() - 50), min(h, ys.max() + 50)
        ov = (frames[i] * 255).astype(np.float32)
        for c in ("SKIN", "SHORTS", "FINS", "MASK"):
            m = bin_[c][i]
            ov[m] = 0.45 * ov[m] + 0.55 * np.array(col[c], np.float32)
        for fn, src in ((top, ov), (bot, outs[i] * 255)):
            tile = np.clip(src[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            tile = cv2.resize(tile, (int(tile.shape[1] * th / tile.shape[0]), th), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"f{i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            fn.append(tile)
    rowt, rowb = np.concatenate(top, 1), np.concatenate(bot, 1)
    sheet = np.concatenate([rowt, rowb], 0)
    cv2.imwrite(str(a.sheet), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[done] {a.output}  sheet {a.sheet}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
