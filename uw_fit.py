#!/usr/bin/env python3
"""Fit the reference grade from an aligned source/reference pair.

The reference image is a graded still of the clip's first frame, so
(source, reference) pixel pairs are available. Model:

    ref(x, y) = V(x, y) * C(src(x, y))

C is a cubic polynomial colour map RGB->RGB (20 terms), V a smooth luminance
gain field (the reference's vignette / light falloff). They are fitted
alternately. Optional masks give a separate C for metal and environment.
Output: uw_fit.npz with the coefficients, the vignette field, and diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def feats(x: np.ndarray) -> np.ndarray:
    """x: (N,3) in 0..1 -> cubic polynomial features (N,20)."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    cols = [np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b,
            r ** 3, g ** 3, b ** 3, r * r * g, r * r * b, g * g * r, g * g * b,
            b * b * r, b * b * g, r * g * b]
    return np.stack(cols, 1).astype(np.float64)


def luma(x):
    return x @ np.array([0.2126, 0.7152, 0.0722])


QUAD = list(range(10))   # 1, r, g, b, rr, gg, bb, rg, rb, gb


def fit_poly(src, dst, w=None, lam=1e-3, cols=None):
    F = feats(src)
    if cols is not None:
        keep = np.zeros(F.shape[1], bool); keep[cols] = True
        F = F * keep
    if w is not None:
        F = F * w[:, None]
        dst = dst * w[:, None]
    A = F.T @ F + lam * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ dst)  # unused columns are zero -> zero coefficients


def apply_poly(coef, x):
    return np.clip(feats(x) @ coef, 0, 1)


def smooth_field(ratio, sigma_frac=0.08):
    h, w = ratio.shape
    small = cv2.resize(ratio, (64, int(64 * h / w)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metal-mask", default=None)
    ap.add_argument("--out", default="uw_fit.npz")
    a = ap.parse_args()

    s = cv2.cvtColor(cv2.imread(a.src), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    r = cv2.cvtColor(cv2.imread(a.ref), cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    h, w = s.shape[:2]
    # Small blur on both: the reference was resampled, so pixel-exact pairs are noisy.
    sb = cv2.GaussianBlur(s, (0, 0), 1.2)
    rb = cv2.GaussianBlur(r, (0, 0), 1.2)

    metal = None
    if a.metal_mask:
        metal = cv2.imread(a.metal_mask, 0) > 127
        metal = cv2.erode(metal.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    V = np.ones((h, w))
    S_ = sb.reshape(-1, 3)
    R_ = rb.reshape(-1, 3)
    # emphasise central pixels (vignette-free) in the first colour fit
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    for it in range(6):
        target = (rb / V[..., None]).reshape(-1, 3)
        wgt = np.clip(1.2 - 0.6 * rad.reshape(-1), 0.3, 1)
        coef = fit_poly(S_, np.clip(target, 0, 1.5), wgt)
        pred = apply_poly(coef, S_).reshape(h, w, 3)
        ratio = (luma(rb) + 0.01) / (luma(pred) + 0.01)
        ratio = np.clip(ratio, 0.05, 2.0)
        V = smooth_field(ratio)
        # normalise: V is relative to its own median
        V /= np.median(V[h // 3:2 * h // 3, w // 3:2 * w // 3])
        err = np.abs(pred * V[..., None] - rb).mean()
        print(f"iter {it}: mean abs err {err * 255:.2f}")
    coef_all = coef

    coef_metal = None
    if metal is not None and metal.sum() > 2000:
        m = metal.reshape(-1)
        target = np.clip((rb / V[..., None]).reshape(-1, 3), 0, 1.5)
        coef_metal = fit_poly(S_[m], target[m], None, lam=3e-2, cols=QUAD)
        pm = (apply_poly(coef_metal, S_[m]) * V.reshape(-1)[m, None])
        pa = (apply_poly(coef_all, S_[m]) * V.reshape(-1)[m, None])
        print("metal err: separate", np.abs(pm - rb.reshape(-1, 3)[m]).mean() * 255,
              "global", np.abs(pa - rb.reshape(-1, 3)[m]).mean() * 255)
    np.savez(a.out, coef=coef_all, coef_metal=coef_metal if coef_metal is not None else np.zeros(0),
             V=V.astype(np.float32))
    full = np.clip(apply_poly(coef_all, S_).reshape(h, w, 3) * V[..., None], 0, 1)
    cv2.imwrite(str(Path(a.out).with_suffix(".fit.png")),
                cv2.cvtColor((full * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(a.out).with_suffix(".V.png")), (np.clip(V, 0, 2) * 127).astype(np.uint8))


if __name__ == "__main__":
    main()
