#!/usr/bin/env python3
"""Fit ONE shared parameter set (2 reference pairs) for the background grade, numerically.

after_* has a different crop than before_*, so we match STATISTICS (per-channel
percentiles, Lab mean/std, vignette edge/centre ratio), never pixels.

Usage: python fit_grade.py --after A1.jpg A2.jpg --before-frames f1.png f2.png ... --out out/grade_params.json
Each --pair is  before.png:before_mask.png|box  ->  after.jpg:box   (see main()).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

import gradelib as G

PCT = [1, 5, 25, 50, 75, 95, 99]


def load_small(path, width=320):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    s = width / w
    return cv2.resize(im, (width, round(h * s)), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255, s


def stats(rgb, m):
    """rgb float 0..1 HxWx3, m bool bg mask. Returns dict of target statistics."""
    lab = G.to_lab(rgb)
    px, lp = rgb[m], lab[m]
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    Lc, Le = lab[..., 0][m & (r < .35)].mean(), lab[..., 0][m & (r > .75)].mean()
    return {
        "pct": np.stack([np.percentile(px[:, c], PCT) for c in range(3)]) * 255,
        "mean": lp.mean(0), "std": lp.std(0), "vig": Le / max(Lc, 1e-3),
    }


def vec(st):
    return np.concatenate([st["pct"].ravel() / 4.0, st["mean"] * 1.0, st["std"] * 2.0, [st["vig"] * 40]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", nargs=4, metavar=("BEFORE", "BEFORE_MASK", "AFTER", "AFTER_BOX"), required=True,
                    help="MASK = png (255=diver) or x0,y0,x1,y1 box (full-res px)")
    ap.add_argument("--out", default="out/grade_params.json")
    a = ap.parse_args()

    def bgmask(spec, shape, scale):
        h, w = shape
        m = np.ones((h, w), bool)
        if spec.endswith(".png"):
            d = cv2.resize(cv2.imread(spec, 0), (w, h), interpolation=cv2.INTER_AREA) > 40
            d = cv2.dilate(d.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0   # keep subject edge out of bg stats
            m &= ~d
        elif spec != "none":
            x0, y0, x1, y1 = [int(float(v) * scale) for v in spec.split(",")]
            m[y0:y1, x0:x1] = False
        return m

    pairs = []
    for bf, bm, af, ab in a.pair:
        b, sb = load_small(bf)
        t, sa = load_small(af)
        mb, ma = bgmask(bm, b.shape[:2], sb), bgmask(ab, t.shape[:2], sa)
        pairs.append((b, mb, vec(stats(t, ma)), stats(t, ma)))

    names = [f"a{c}" for c in G.CH] + [f"g{c}" for c in G.CH] + [f"b{c}" for c in G.CH] + ["sat", "hue", "vig"]
    x0 = np.array([0.15, 0.55, 0.9, 1.0, 1.0, 1.0, 0, 0, 0, 1.0, 0.0, 0.35])
    lo = np.array([0.01, 0.1, 0.2, 0.4, 0.4, 0.4, -0.2, -0.2, -0.2, 0.5, -30, 0.0])
    hi = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 2.5, 30, 0.9])

    def params(x):
        P = json.loads(json.dumps(G.DEFAULT["bg"]))
        P.update(a=list(x[0:3]), g=list(x[3:6]), b=list(x[6:9]), sat=float(x[9]), hue_deg=float(x[10]), vignette=float(x[11]))
        return P

    def resid(x):
        P = params(x)
        out = []
        for b, mb, tv, _ in pairs:
            g = G.grade_bg(b, P)
            out.append(vec(stats(g, mb)) - tv)
        return np.concatenate(out)

    r0 = resid(x0)
    sol = least_squares(resid, x0, bounds=(lo, hi), diff_step=1e-2, max_nfev=120)
    P = params(sol.x)
    rep = {"rms_before": float(np.sqrt(np.mean(r0 ** 2))), "rms_after": float(np.sqrt(np.mean(sol.fun ** 2)))}
    per = []
    for b, mb, tv, ts in pairs:
        s = stats(G.grade_bg(b, P), mb)
        per.append({"dLab_mean": (s["mean"] - ts["mean"]).round(2).tolist(), "dLab_std": (s["std"] - ts["std"]).round(2).tolist(),
                    "dPct_max_abs": float(np.abs(s["pct"] - ts["pct"]).max().round(1)),
                    "vig_got": float(s["vig"]), "vig_target": float(ts["vig"])})
    rep["per_pair"] = per
    out = Path(a.out)
    prev = json.loads(out.read_text()) if out.exists() else {"bg": G.DEFAULT["bg"], "subj": G.DEFAULT["subj"]}
    prev["bg"] = P
    prev["fit_report"] = rep
    out.write_text(json.dumps(prev, indent=1))
    print(json.dumps({"params": dict(zip(names, np.round(sol.x, 4).tolist())), **rep}, indent=1))


if __name__ == "__main__":
    main()
