#!/usr/bin/env python3
"""Fit the reference look numerically.

The reference still is a graded version of video frame 528. Pairing the two
pixel for pixel lets us fit, on the background only, a compact colourist model:

    out = vignette(x, y) * saturate( (slope*in + offset)^power )     (per channel CDL)

and, on the fish only, a second CDL (no vignette). The fitted numbers seed the
defaults in fish_grade.py; they are a starting point, not the final look.

    python fit_reference.py --orig still_orig.jpg --ref still_ref.jpg [--fish-mask m.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

LUMA = np.array([0.2126, 0.7152, 0.0722])


def load(path, size):
    x = cv2.imread(str(path))[..., ::-1]
    return cv2.resize(x, size, interpolation=cv2.INTER_AREA).astype(np.float64) / 255


def cdl(x, p):
    gain, off, pw, sat = p[0:3], p[3:6], p[6:9], p[9]
    t = np.clip(x * gain + off, 1e-4, None) ** pw
    L = (t @ LUMA)[:, None]
    return L + sat * (t - L)


def vignette(rr, yy, p):
    kr, qr, kt, kb = p
    v = np.exp(-kr * rr ** qr)
    v = v * (1 - kt * np.clip(1 - yy / 0.35, 0, 1) ** 1.5)       # darker toward the top
    v = v * (1 - kb * np.clip((yy - 0.75) / 0.25, 0, 1) ** 1.5)  # and the bottom
    return v[:, None]


def lab(x):
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--fish-mask", type=Path, help="full-frame 0/255 mask of the fish in the original")
    ap.add_argument("--out", type=Path, default=Path("output/reference_fit.json"))
    a = ap.parse_args()

    W, H = 192, 341                      # 1/4 of the reference size
    O, R = load(a.orig, (W, H)), load(a.ref, (W, H))
    if a.fish_mask:
        fm = cv2.resize(cv2.imread(str(a.fish_mask), 0), (W, H), interpolation=cv2.INTER_AREA) > 127
    else:
        fm = np.zeros((H, W), np.uint8)
        cv2.ellipse(fm, (88, 177), (50, 18), -12, 0, 360, 1, -1)
        fm = fm > 0
    near = cv2.dilate(fm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bg = ~near
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot((xx / (W - 1) - .5) * 2, (yy / (H - 1) - .5) * 2)[bg]
    yn = (yy / (H - 1))[bg]
    Ob, Rb = O[bg], R[bg]

    def model(p):
        return vignette(rr, yn, p[10:14]) * cdl(Ob, p[:10])

    def resid(p):
        m = model(p)
        # perceptual-ish: compare in a sqrt (gamma .5) domain so shadows count
        return (np.sqrt(np.clip(m, 0, 1)) - np.sqrt(Rb)).ravel()

    p0 = np.r_[0.4, 0.7, 0.9, 0, 0, 0, 1.2, 1.2, 1.2, 1.0, 0.5, 2.0, 0.2, 0.1]
    lo = np.r_[0.05, 0.05, 0.05, -0.2, -0.2, -0.2, 0.6, 0.6, 0.6, 0.3, 0.0, 1.0, 0, 0]
    hi = np.r_[3, 3, 3, 0.2, 0.2, 0.2, 3, 3, 3, 2.5, 3.0, 5.0, 0.9, 0.9]
    s = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.05)
    p = s.x
    m = model(p)
    dE = np.linalg.norm(lab(m) - lab(Rb), axis=1)
    res = {"bg_cdl": {"slope": p[0:3].tolist(), "offset": p[3:6].tolist(),
                      "power": p[6:9].tolist(), "saturation": float(p[9])},
           "bg_vignette": {"k_radial": float(p[10]), "q_radial": float(p[11]),
                           "k_top": float(p[12]), "k_bottom": float(p[13])},
           "bg_fit_dE76_median": float(np.median(dE)), "bg_fit_dE76_p90": float(np.percentile(dE, 90)),
           "bg_mean_rgb_model": m.mean(0).tolist(), "bg_mean_rgb_ref": Rb.mean(0).tolist()}

    # fish: CDL only (the vignette seen at the fish position is divided out)
    rf = np.hypot((xx / (W - 1) - .5) * 2, (yy / (H - 1) - .5) * 2)[fm]
    yf = (yy / (H - 1))[fm]
    vf = vignette(rf, yf, p[10:14])
    Of, Rf = O[fm], np.clip(R[fm] / vf, 0, 1.2)

    def residf(q):
        return (np.sqrt(np.clip(cdl(Of, q), 0, 1)) - np.sqrt(np.clip(Rf, 0, 1))).ravel()

    q0 = np.r_[1, 1, 1, 0, 0, 0, 1, 1, 1, 1.1]
    lof = np.r_[0.3, 0.3, 0.3, -0.2, -0.2, -0.2, 0.5, 0.5, 0.5, 0.3]
    hif = np.r_[3, 3, 3, 0.2, 0.2, 0.2, 3, 3, 3, 2.5]
    sf = least_squares(residf, q0, bounds=(lof, hif), loss="soft_l1", f_scale=0.05)
    q = sf.x
    mf = cdl(Of, q)
    dEf = np.linalg.norm(lab(mf) - lab(Rf), axis=1)
    res["fish_cdl"] = {"slope": q[0:3].tolist(), "offset": q[3:6].tolist(),
                       "power": q[6:9].tolist(), "saturation": float(q[9])}
    res["fish_fit_dE76_median"] = float(np.median(dEf))
    res["fish_mean_rgb_model"] = mf.mean(0).tolist()
    res["fish_mean_rgb_ref_devignetted"] = Rf.mean(0).tolist()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
