#!/usr/bin/env python3
"""Backlit-diver fix (prompts/diver-silhouette-grade.md).

Reference reel "How to recover bad visibility": 1. Curves, 2. White balance,
3. Colour separation. Applied here, in this order, in float RGB:

  a. global curves     black point + gentle S, highlights (sun, rays) kept
  b. global WB         green cast out, water to deep cyan-blue
  c. diver lift        SAM 3.1 diver matte; shadow-weighted gain, "balanced"
  d. skin colour       skin = diver matte minus SAM 3.1 part masks (hair,
                       shorts, fins); chroma to Mediterranean tan #C68B59,
                       luminance texture kept, L nudged
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

SKIN_HEX = "#C68B59"          # Mediterranean tan
LIFT_TARGET = 0.30            # "balanced": median diver luma after the lift
LIFT_MAX = 2.6
SKIN_CHROMA = 0.85            # how far skin a*/b* move to the target
SKIN_L_NUDGE = 0.35           # how far the skin's mean L moves to the target

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


def gsmooth(v, sigma):
    n = len(v)
    ok = ~np.isnan(v)
    t = np.arange(n)
    v = np.interp(t, t[ok], v[ok]) if ok.any() else np.zeros(n)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(v, r, mode="edge"), k / k.sum(), "valid").astype(np.float32)


def grade(x, a_div, a_skin, g, l_off):
    y = global_grade(x)
    d = a_div[..., None]
    yl = lift(y, g)                                       # c. lift inside the diver
    lab = to_lab(yl)
    s = a_skin[..., None]
    hi = np.clip((lab[..., :1] - 80) / 15, 0, 1)          # keep speculars
    k = SKIN_CHROMA * (1 - hi)
    lab_s = lab.copy()
    lab_s[..., 1:] = lab[..., 1:] + (TARGET[1:] - lab[..., 1:]) * k   # d. skin chroma
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

    def skin_alpha(i):
        s = al[i].astype(np.float32) / 255
        for v in part_al.values():
            s = s * (1 - v[i].astype(np.float32) / 255)
        return s

    # pass 1: per-frame lift gain and skin L offset, smoothed over time
    gains = np.full(n, np.nan, np.float32)
    loffs = np.full(n, np.nan, np.float32)
    for i, x in enumerate(iter_frames(a.input, info)):
        core = al[i] > 200
        if core.sum() < 500:
            continue
        y = global_grade(x[::2, ::2])
        c2 = core[::2, ::2]
        med = float(np.median(luma(y)[c2]))
        g = float(np.clip(LIFT_TARGET / max(med, 1e-3), 1.0, LIFT_MAX))
        gains[i] = g
        sk = skin_alpha(i)[::2, ::2] > 0.8
        if sk.sum() > 200:
            L = to_lab(lift(y, g))[..., 0][sk]
            loffs[i] = SKIN_L_NUDGE * (TARGET[0] - float(np.median(L)))
    gains, loffs = gsmooth(gains, 8.0), gsmooth(loffs, 8.0)
    print(f"[lift] gain {gains.min():.2f}-{gains.max():.2f}; skin L offset "
          f"{loffs.min():.1f}-{loffs.max():.1f}; target Lab {np.round(TARGET, 1).tolist()}", flush=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.input, a.output, info, Encode(crf=16, preset="slow", profile="high", dither=True))
    checks = sorted({0, n // 3, 2 * n // 3, int(0.95 * n)})
    skin_means, tiles = [], []
    for i, x in enumerate(iter_frames(a.input, info)):
        s = skin_alpha(i)
        y = grade(x, al[i].astype(np.float32) / 255, s, float(gains[i]), float(loffs[i]))
        enc.write(y)
        m = s > 0.8
        if m.sum() > 200:
            skin_means.append(np.median(to_lab(y)[m], 0))
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

    sm = np.median(np.array(skin_means), 0) if skin_means else np.full(3, np.nan)
    de = float(np.linalg.norm(sm - TARGET))
    de_ab = float(np.linalg.norm(sm[1:] - TARGET[1:]))
    o = probe(a.output)
    rep = {"skin_target_hex": SKIN_HEX, "skin_target_lab": np.round(TARGET, 2).tolist(),
           "skin_measured_lab": np.round(sm, 2).tolist(), "deltaE76": round(de, 2),
           "deltaE76_chroma_only": round(de_ab, 2),
           "lift_gain_range": [round(float(gains.min()), 2), round(float(gains.max()), 2)],
           "frames_with_skin": len(skin_means), "parts": sorted(part_al),
           "out": {"w": o.width, "h": o.height, "fps": str(o.fps), "frames": o.n_frames, "audio": o.has_audio},
           "src": {"w": w, "h": h, "fps": str(info.fps), "frames": n, "audio": info.has_audio}}
    (qc / f"{stem}_silhouette_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
