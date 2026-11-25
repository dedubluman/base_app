#!/usr/bin/env python3
"""Reference-matched deep-ocean grade: LUT builders + trilinear LUT application.

Environment grade = cubic RGB polynomial fitted to the reference/original pairs
(fit_reference_grade.py) followed by a luma tone curve, baked to a 3D LUT. The diver gets
a separate, much gentler LUT, blended through the SAM 3.1 mask (render_grade.py).
Everything is a static function of colour, so the grade cannot flicker by construction."""
import itertools
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
LUMA = np.array([.2126, .7152, .0722])


def _feats(X):
    cols = [np.ones(len(X))]
    for d in (1, 2, 3):
        for c in itertools.combinations_with_replacement(range(3), d):
            cols.append(np.prod([X[:, k] for k in c], axis=0))
    return np.stack(cols, 1)


def grid(n):
    g = np.linspace(0, 1, n)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)  # [r,g,b] order, r slowest


def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def build_env_lut(n=33, p=None):
    p = {**dict(contrast=0.0, hi_gain=1.0, sat=1.0, green_trim=0.0, warm_kill=0.0), **(p or {})}
    X = grid(n)
    Y = np.clip(_feats(X) @ np.load(HERE / "grade_data/poly_coef.npy"), 0, 1)
    xp, yq = np.load(HERE / "grade_data/lumacurve.npy")
    y = Y @ LUMA
    y2 = np.interp(y, xp, yq)
    # highlight shoulder: lift the brightest caustics/reflections a bit further, smooth roll-off
    y2 = y2 + (p["hi_gain"] - 1) * smoothstep((y2 - 0.25) / 0.5) * y2 * 0.6
    ratio = (y2 / np.maximum(y, 1e-4))[:, None]
    Y = np.where(y[:, None] > 1e-3, Y * ratio, Y)
    # mild S-curve contrast around the shadow/mid pivot
    if p["contrast"]:
        yy = Y @ LUMA; yc = np.clip(yy, 1e-4, 1)
        s = 0.5 + 0.5 * np.tanh((yc - 0.18) * (4 + 6 * p["contrast"])) / np.tanh((1 - 0.18) * (4 + 6 * p["contrast"])) * 0 + 0
        # luma-only contrast via pivoted power curve, chroma preserved
        t = np.where(yc < 0.18, 0.18 * (yc / 0.18) ** (1 + p["contrast"]), 1 - (1 - 0.18) * ((1 - yc) / (1 - 0.18)) ** (1 - 0.35 * p["contrast"]))
        Y = Y * (t / yc)[:, None]
    # cool silver highlights: pull orange/beige reflections toward the blue-cyan side
    if p["warm_kill"]:
        yy = Y @ LUMA; cool = (Y[:, 1] + Y[:, 2]) / 2
        warm = np.clip(Y[:, 0] - cool, 0, None)
        Y[:, 0] = Y[:, 0] - p["warm_kill"] * warm
        Y[:, 2] = Y[:, 2] + p["warm_kill"] * 0.35 * warm
    Y = Y - np.array([0, p["green_trim"], 0]) * (Y @ LUMA)[:, None]
    gray = (Y @ LUMA)[:, None]; Y = gray + (Y - gray) * p["sat"]
    # soft clip toward 1 so highlights never hard-clip
    Y = np.where(Y > 0.88, 0.88 + 0.12 * np.tanh((Y - 0.88) / 0.12), Y)
    return np.clip(Y, 0, 1).reshape(n, n, n, 3).astype(np.float32)


def build_diver_lut(n=33, p=None):
    """Gentle correction for the diver: remove the green water cast, sit the exposure in the
    darker scene, keep skin hue. WB gains neutralise the cast (shorts become neutral/cool-white)."""
    p = {**dict(gains=(1.22, 0.90, 1.0), exposure=0.80, gamma=1.12, sat=0.95, warm=0.0), **(p or {})}
    X = grid(n)
    Y = X * np.array(p["gains"])
    Y = np.clip(Y, 0, 1) ** p["gamma"] * p["exposure"]
    gray = (Y @ LUMA)[:, None]; Y = gray + (Y - gray) * p["sat"]
    return np.clip(Y, 0, 1).reshape(n, n, n, 3).astype(np.float32)


def apply_lut(img, lut):
    """img float32 HxWx3 (0..1, RGB). Tetra-free trilinear lookup."""
    n = lut.shape[0]
    f = np.clip(img, 0, 1) * (n - 1)
    i0 = np.minimum(f.astype(np.int32), n - 2); d = f - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    dr, dg, db = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    out = 0
    for a, wa in ((0, 1 - dr), (1, dr)):
        for b, wb in ((0, 1 - dg), (1, dg)):
            for c, wc in ((0, 1 - db), (1, db)):
                out = out + lut[r0 + a, g0 + b, b0 + c] * (wa * wb * wc)
    return out.astype(np.float32)


def write_cube(path, lut, title="underwater"):
    n = lut.shape[0]
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        # .cube: red fastest
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    f.write("%.6f %.6f %.6f\n" % tuple(lut[r, g, b]))
