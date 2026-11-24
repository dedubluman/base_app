#!/usr/bin/env python3
"""Fit the global deep-abyss look (abyss_look.Look) to a reference still.

The reference must show the SAME frame as a source video frame (graded version of it, any
resolution with the same aspect ratio). The fit is done on low-frequency, subject-free
pixels only (divers / fish are excluded via their SAM masks, or --exclude-png):

    1. tone + vignette  : least squares of  Y_ref ~ tone(base luma) * vignette(x, y)
    2. chroma           : linear regression of the reference's log(R/G), log(B/G) on the
                          source's log ratios and the output luma
    3. detail gain      : ratio of mid-frequency luma detail (reference / source, after tone)

    python abyss_fit.py --video inputs/dive.mov --frame 241 \
        --reference /path/to/reference_still.jpg --out abyss_look.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from abyss_common import gblur, luma, probe, read_frames
from abyss_look import Look, nblur, src_ratios, tone, vignette

REPO = Path(__file__).resolve().parent


def load_frame(video: Path, idx: int) -> np.ndarray:
    info = probe(video)
    for i, f in enumerate(read_frames(video, info["w"], info["h"])):
        if i == idx:
            return f
    raise SystemExit(f"frame {idx} not found")


def subject_mask(args, h: int, w: int) -> np.ndarray:
    """1 = subject (diver / fish) pixel to ignore, dilated a little."""
    m = np.zeros((h, w), np.uint8)
    if args.exclude_png:
        e = cv2.imread(str(args.exclude_png), 0)
        m |= cv2.resize(e, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    else:
        mdir = args.masks_dir or REPO / "masks" / f"{args.video.stem}__abyss"
        for job in ("diver", "fish"):
            ks = sorted((mdir / job).glob("kf_*.png"))
            if not ks:
                continue
            near = min(ks, key=lambda p: abs(int(p.stem[3:]) - args.frame))
            m |= cv2.imread(str(near), 0) > 127
        if not m.any():
            print("  [fit] WARNING: no subject masks found - fitting on all pixels")
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True, help="video frame index the reference shows")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "abyss_look.json")
    ap.add_argument("--masks-dir", type=Path)
    ap.add_argument("--exclude-png", type=Path, help="subject mask image (white = ignore) instead of SAM masks")
    ap.add_argument("--base-sigma", type=float, default=30.0)
    args = ap.parse_args()

    f = load_frame(args.video, args.frame)
    h, w = f.shape[:2]
    ref = cv2.imread(str(args.reference))[..., ::-1].astype(np.float32) / 255
    ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC)
    excl = subject_mask(args, h, w)
    look = Look(base_sigma=args.base_sigma)

    L, R = luma(f), luma(ref)
    Lb, Rb = nblur(L, excl, look.base_sigma), nblur(R, excl, look.base_sigma)
    ok = excl[::10, ::10] < 0.5
    Lb_s, Rb_s = Lb[::10, ::10][ok], Rb[::10, ::10][ok]
    yy, xx = np.mgrid[0:h:10, 0:w:10]
    yy, xx = (yy / h).astype(np.float32)[ok], (xx / w).astype(np.float32)[ok]

    # ---- 1. tone + vignette ------------------------------------------------------------
    names = ["black", "gain", "gamma", "cx", "cy", "vig_x", "vig_y", "vig_p"]
    p0 = [look.black, look.gain, look.gamma, look.cx, look.cy, look.vig_x, look.vig_y, look.vig_p]
    lo = [0.25, 0.3, 0.8, 0.3, 0.3, 0.0, 0.0, 1.2]
    hi = [0.45, 6.0, 1.6, 0.7, 0.7, 3.0, 3.0, 4.0]

    def model(p):
        k = Look(**{n: v for n, v in zip(names, p)}, toe=look.toe)
        ux = np.abs(xx - k.cx) / max(k.cx, 1 - k.cx)
        uy = np.abs(yy - k.cy) / max(k.cy, 1 - k.cy)
        v = np.exp(-(k.vig_x * ux ** k.vig_p + k.vig_y * uy ** k.vig_p))
        return tone(Lb_s, k) * v + look.floor

    # residual in a perceptual-ish (sqrt) domain so the dark half counts as much as the bright
    res = lambda p: np.sqrt(np.maximum(model(p), 0) + 0.01) - np.sqrt(Rb_s + 0.01)
    sol = least_squares(res, p0, bounds=(lo, hi))
    for n, v in zip(names, sol.x):
        setattr(look, n, float(v))
    err = np.abs(model(sol.x) - Rb_s)
    print("  [fit] tone/vignette: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, sol.x)))
    print(f"  [fit] low-freq luma error: mean {err.mean():.4f}, p95 {np.percentile(err, 95):.4f}")

    # ---- 2. chroma ---------------------------------------------------------------------
    s_rg, s_bg = src_ratios(f, excl, look.chroma_sigma)
    r_rg, r_bg = src_ratios(ref, excl, look.chroma_sigma)
    sl = lambda a: a[::10, ::10][ok]
    for key, s, r in (("rg", s_rg, r_rg), ("bg", s_bg, r_bg)):
        A = np.stack([np.ones_like(sl(s)), sl(s), Rb_s], 1)
        coef, *_ = np.linalg.lstsq(A, sl(r), rcond=None)
        pred = A @ coef
        print(f"  [fit] chroma {key} (linear ratio): coef {np.round(coef, 3).tolist()}  rms {np.sqrt(np.mean((pred - sl(r)) ** 2)):.4f}"
              f"  (src std {sl(s).std():.3f}, ref std {sl(r).std():.3f}, ref mean {sl(r).mean():.3f})")
        setattr(look, key, [float(c) for c in coef])

    # ---- 3. detail gain ----------------------------------------------------------------
    band = lambda x: gblur(x, 2.0) - nblur(x, excl, look.base_sigma)
    sel = (excl < 0.5)
    sel[: int(h * 0.55)] = False                       # seabed texture is what carries the detail
    dL, dR = band(L)[sel], band(R)[sel]
    k0 = Look(**{n: float(v) for n, v in zip(names, sol.x)}, toe=look.toe)
    eps = 0.003
    slope = ((tone(Lb + eps, k0) - tone(Lb, k0)) / eps * vignette(h, w, k0))[sel]
    look.detail = float(np.clip(dR.std() / max((slope * dL).std(), 1e-6), 1.0, 2.5))
    print(f"  [fit] detail gain {look.detail:.2f} (ref band std {dR.std():.4f}, "
          f"source x slope {(slope * dL).std():.4f})")

    look.save(args.out)
    print(f"  [fit] saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
