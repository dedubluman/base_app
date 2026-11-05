#!/usr/bin/env python3
"""Backlit-diver fix (prompts/diver-silhouette-grade.md).

Reference reel "How to recover bad visibility": 1. Curves, 2. White balance,
3. Colour separation. Applied here, in this order, in float RGB:

  a. global curves     black point + gentle S, highlights (sun, rays) kept
  b. global WB         green cast out, water to deep cyan-blue
  c. diver lift        SAM 3.1 diver matte; shadow-weighted gain, "balanced"
  d. skin colour       skin = diver matte minus SAM 3.1 part masks (hair,
                       shorts, fins) and the dive mask; surface skin #E8B896
                       seen through the estimated water path (red absorbed
                       first), applied as an a*b*/L offset so texture stays
  e. separation        diver a little warmer/more saturated, water a little
                       less saturated

Masks come from the repo's SAM 3.1 tools (diver_masks.py, diver_parts.py)
and are refined/feathered/temporally smoothed by diver_reels.build_mattes.

    python diver_silhouette.py --input inputs/diver.mov --output output/diver_graded.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

import diver_reels as DR
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

REPO = Path(__file__).resolve().parent
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

SKIN_HEX = "#E8B896"          # light-warm skin, as it would look at the surface
LIFT_TARGET = 0.30            # "balanced": median diver luma after the lift
LIFT_MAX = 2.6
SKIN_CHROMA = 0.8             # share of the (depth target - frame median) a*/b* shift
SKIN_L_NUDGE = 0.30           # share of the L difference
# Deep end of the skin in the *corrected* image (relative to the surface skin):
# red/warmth go first (a* and b* shrink, b* faster), and it darkens. A raw
# physical model (exp(-k d) + water veil) lands on green-cyan once the global
# white balance has already taken the water out, so it double-counts the water.
DEEP_L, DEEP_A, DEEP_B = -14.0, 0.45, 0.35
# water path from the red loss on the diver (source R/G), calibrated on the
# clip itself: the 95th-percentile R/G frame is D_MIN, the 5th is D_MAX
D_MIN, D_MAX = 1.0, 5.0

# a. curves (same points as the approved ffmpeg preview)
_MASTER = PchipInterpolator([0, .06, .30, .70, 1], [0, .02, .30, .76, 1])
_BLUE = PchipInterpolator([0, .5, 1], [.02, .55, 1])
# b. white balance as a channel mix (rows = output R, G, B)
_MIX = np.array([[1.10, 0.00, 0.04],
                 [0.00, 0.90, 0.00],
                 [0.00, 0.08, 1.05]], np.float32)


def luma(x):
    return x @ LUMA


def sat(x, s):
    y = luma(x)[..., None]
    return y + (x - y) * s


def global_grade(x):
    y = np.clip(x @ _MIX.T, 0, 1)                       # b. WB (mix first, as previewed)
    y[..., 2] = _BLUE(y[..., 2])                         # a. curves: blue, then master
    y = _MASTER(y).astype(np.float32)
    y = np.clip(sat(y, 1.08), 0, 1) ** (1 / 1.12)        # light saturation + gamma
    # sun core: fade the channel mix back to neutral so it stays white, not pink
    yl = luma(y)[..., None]
    w = np.clip((yl - 0.82) / 0.15, 0, 1)
    y = y * (1 - w) + yl * w
    return y.astype(np.float32)


def lift(y, g):
    """Shadow-weighted gain: g at black, 1 at white (keeps the backlit rim)."""
    return y * g / (1 + (g - 1) * y)


def to_lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def to_rgb(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def hex_lab(h):
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)[None, None] / 255
    return to_lab(c)[0, 0]


TARGET = hex_lab(SKIN_HEX)
SKIN_RGB = np.array([int(SKIN_HEX[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def depth_m(rg, rg_shallow, rg_deep):
    """Water path (m) from the diver's source R/G ratio (more red lost = deeper)."""
    t = np.clip((rg_shallow - rg) / max(rg_shallow - rg_deep, 1e-3), 0, 1)
    return D_MIN + (D_MAX - D_MIN) * t


def skin_at_depth(d):
    """Lab target for surface skin seen d metres deep, in the corrected image."""
    t = float(np.clip((d - D_MIN) / (D_MAX - D_MIN), 0, 1))
    t = 1 - np.exp(-2.2 * t)                 # most of the change in the first metres
    t /= 1 - np.exp(-2.2)
    L, a, b = TARGET
    return np.array([L + DEEP_L * t, a * (1 - (1 - DEEP_A) * t),
                     b * (1 - (1 - DEEP_B) * t)], np.float32)


def goggles(x, dm):
    """Dive-mask lens/strap: dark, low-chroma pixels in the head (top 30% of the
    diver's box). No SAM 3.1 text prompt finds it on this clip."""
    ys, xs = np.nonzero(dm)
    if len(ys) < 500:
        return np.zeros(dm.shape, bool)
    top = ys.min() + 0.30 * (ys.max() - ys.min())
    head = dm.copy()
    head[int(top):] = False
    if head.sum() < 100:
        return np.zeros(dm.shape, bool)
    L = luma(x)
    dark = L < 0.6 * np.median(L[head])
    g = cv2.morphologyEx((dark & head).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(g, np.ones((5, 5), np.uint8)) > 0


def gsmooth(v, sigma):
    n = len(v)
    ok = ~np.isnan(v)
    t = np.arange(n)
    v = np.interp(t, t[ok], v[ok]) if ok.any() else np.zeros(n)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(v, r, mode="edge"), k / k.sum(), "valid").astype(np.float32)


def grade(x, a_div, a_skin, g, l_off, ab_off=(0.0, 0.0)):
    y = global_grade(x)
    d = a_div[..., None]
    yl = lift(y, g)                                       # c. lift inside the diver
    lab = to_lab(yl)
    s = a_skin[..., None]
    hi = np.clip((lab[..., :1] - 80) / 15, 0, 1)          # keep speculars
    lab_s = lab.copy()
    lab_s[..., 1:] = lab[..., 1:] + np.asarray(ab_off, np.float32) * (1 - hi)   # d. skin chroma shift
    lab_s[..., :1] = lab[..., :1] + l_off * (1 - hi)                   #    L nudge
    ys = to_rgb(lab_s)
    yd = ys * s + yl * (1 - s)
    yd = np.clip(sat(yd, 1.08), 0, 1)                     # e. diver a touch richer
    yb = np.clip(sat(y, 0.95), 0, 1)                      # e. water a touch calmer
    return yd * d + yb * (1 - d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=REPO / "output" / "diver_graded.mp4")
    a = ap.parse_args()
    info = probe(a.input)
    w, h, n = info.width, info.height, info.n_frames
    stem = a.input.stem
    qc = REPO / "output" / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    raw, report = DR.merge_masks(stem, n, (h, w))
    al, part_al, _ = DR.build_mattes(a.input, info, stem, raw, log=print)
    part_al.pop("bubbles", None)
    print(f"[mask] parts: {sorted(part_al)}", flush=True)

    def skin_alpha(i, x):
        s = al[i].astype(np.float32) / 255
        for v in part_al.values():
            s = s * (1 - v[i].astype(np.float32) / 255)
        s = s * (1 - goggles(x, al[i] > 127))
        # ~3 px wider feather than the diver matte: skin melts into the water
        return cv2.GaussianBlur(s, (0, 0), 3.0) * (al[i].astype(np.float32) / 255)

    # pass 1: per frame - lift gain, water path d (red loss on the source
    # skin), water colour around the diver; all smoothed over time
    gains = np.full(n, np.nan, np.float32)
    depth = np.full(n, np.nan, np.float32)
    water = np.full((n, 3), np.nan, np.float32)
    k41 = np.ones((41, 41), np.uint8)
    for i, x in enumerate(iter_frames(a.input, info)):
        core = al[i] > 200
        if core.sum() < 500:
            continue
        xs, c2 = x[::2, ::2], core[::2, ::2]
        y = global_grade(xs)
        med = float(np.median(luma(y)[c2]))
        gains[i] = float(np.clip(LIFT_TARGET / max(med, 1e-3), 1.0, LIFT_MAX))
        ring = (cv2.dilate(c2.astype(np.uint8), k41) > 0) & ~c2
        water[i] = np.median(y[ring], 0)
        sk = skin_alpha(i, x)[::2, ::2] > 0.8
        if sk.sum() > 200:
            m = np.median(xs[sk], 0)
            depth[i] = m[0] / max(m[1], 1e-4)          # R/G for now
    ok = ~np.isnan(depth)
    rg_sh, rg_dp = np.percentile(depth[ok], 95), np.percentile(depth[ok], 5)
    depth[ok] = depth_m(depth[ok], rg_sh, rg_dp)
    print(f"[depth] source skin R/G {rg_dp:.2f} (deep, {D_MAX} m) .. {rg_sh:.2f} (shallow, {D_MIN} m)", flush=True)
    gains, depth = gsmooth(gains, 8.0), gsmooth(depth, 8.0)
    water = np.stack([gsmooth(water[:, c], 8.0) for c in range(3)], 1)
    print(f"[lift] gain {gains.min():.2f}-{gains.max():.2f}; water path "
          f"{depth.min():.2f}-{depth.max():.2f} m", flush=True)

    # pass 2: per-frame skin target at that depth -> offsets on the lifted skin
    loffs = np.full(n, np.nan, np.float32)
    aoffs = np.full((n, 2), np.nan, np.float32)
    targets = np.zeros((n, 3), np.float32)
    for i, x in enumerate(iter_frames(a.input, info)):
        targets[i] = skin_at_depth(float(depth[i]))
        sk = skin_alpha(i, x)[::2, ::2] > 0.8
        if sk.sum() > 200:
            lab = to_lab(lift(global_grade(x[::2, ::2]), float(gains[i])))[sk]
            med = np.median(lab, 0)
            loffs[i] = SKIN_L_NUDGE * (targets[i][0] - med[0])
            aoffs[i] = SKIN_CHROMA * (targets[i][1:] - med[1:])
    loffs = gsmooth(loffs, 8.0)
    aoffs = np.stack([gsmooth(aoffs[:, c], 8.0) for c in range(2)], 1)
    print(f"[skin] target Lab shallow {np.round(targets[int(depth.argmin())], 1).tolist()} "
          f"deep {np.round(targets[int(depth.argmax())], 1).tolist()}", flush=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.input, a.output, info, Encode(crf=16, preset="slow", profile="high", dither=True))
    checks = sorted({0, n // 3, 2 * n // 3, int(0.95 * n)})
    skin_means, tiles = [], []
    for i, x in enumerate(iter_frames(a.input, info)):
        s = skin_alpha(i, x)
        y = grade(x, al[i].astype(np.float32) / 255, s, float(gains[i]), float(loffs[i]), aoffs[i])
        enc.write(y)
        m = s > 0.8
        if m.sum() > 200:
            skin_means.append((i, float(depth[i]), np.median(to_lab(y)[m], 0)))
        if i in checks:
            src8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1]
            out8 = (np.clip(y, 0, 1) * 255).astype(np.uint8)[..., ::-1]
            ov = src8.copy()
            ov[m] = (0.5 * ov[m] + 0.5 * np.array([60, 140, 255])).astype(np.uint8)
            dm = al[i] > 127
            cs, _ = cv2.findContours(dm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
            tiles.append(cv2.resize(np.hstack([src8, ov, out8]), (3 * w // 3, h // 3)))
        if i % 50 == 0:
            print(f"[render] {i}/{n}", flush=True)
    enc.close()
    cv2.imwrite(str(qc / f"{stem}_silhouette_check.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])

    ds = np.array([d for _, d, _ in skin_means])
    labs = np.array([l for _, _, l in skin_means])
    lo, hi_ = ds <= np.percentile(ds, 20), ds >= np.percentile(ds, 80)
    corr_b = float(np.corrcoef(ds, labs[:, 2])[0, 1]) if ds.std() > 1e-3 else float("nan")
    jumps = np.abs(np.diff(labs, axis=0)).max(0)
    o = probe(a.output)
    rep = {"skin_surface_hex": SKIN_HEX, "deep_end_L_a_b": [DEEP_L, DEEP_A, DEEP_B],
           "water_path_m": [round(float(depth.min()), 2), round(float(depth.max()), 2)],
           "skin_lab_shallow": np.round(labs[lo].mean(0), 1).tolist(),
           "skin_lab_deep": np.round(labs[hi_].mean(0), 1).tolist(),
           "corr_depth_vs_b": round(corr_b, 3),
           "max_frame_jump_lab": np.round(jumps, 2).tolist(),
           "lift_gain_range": [round(float(gains.min()), 2), round(float(gains.max()), 2)],
           "frames_with_skin": len(skin_means), "parts": sorted(part_al),
           "out": {"w": o.width, "h": o.height, "fps": str(o.fps), "frames": o.n_frames, "audio": o.has_audio},
           "src": {"w": w, "h": h, "fps": str(info.fps), "frames": n, "audio": info.has_audio}}
    (qc / f"{stem}_silhouette_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
