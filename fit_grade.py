#!/usr/bin/env python3
"""Fit ONE shared background grade (tone curves + sat/hue + vignette [+ vertical falloff]) to BOTH reference pairs.

after_* is cropped differently from before_*, so only statistics are compared (never pixels):
per-channel percentiles p1/p5/p25/p50/p75/p95/p99, Lab mean/std, corner/centre L* ratio, L* of top/mid/bottom thirds.
Init = pooled, smoothed per-channel CDF matching evaluated at the curve knots; then bounded least squares
(monotone knots) on all pairs jointly, with the vignette inside the forward model.

  python fit_grade.py --pair raw.png diver_mask.png after.jpg after_parts.npz  [--pair ...] [--vgrad]
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
# R p95/p99 of after_head come from surface light shafts (spatial structure a global curve cannot make) -> low weight
PW = np.array([[1, 1, 1, 1, 1, .15, .15], [1] * 7, [1] * 7], float)
W_SMALL = 320


def load_rgb(path, width=W_SMALL):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    return cv2.resize(im, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255


def bg_mask(mask_full, shape, dil=3):
    m = cv2.resize(mask_full.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return ~(cv2.dilate(m.astype(np.uint8), np.ones((2 * dil + 1,) * 2, np.uint8)) > 0)


def stats(rgb, m):
    lab = G.to_lab(rgb)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    L = lab[..., 0]
    th = [L[(m & (yy >= h * i // 3) & (yy < h * (i + 1) // 3))].mean() for i in range(3)]
    px = rgb[m]
    return {"pct": np.stack([np.percentile(px[:, c], PCT) for c in range(3)]) * 255,
            "mean": lab[m].mean(0), "std": lab[m].std(0),
            "vig": float(L[m & (r > .75)].mean() / max(L[m & (r < .35)].mean(), 1e-3)),
            "thirds": np.array(th) / max(L[m].mean(), 1e-3)}


def vec(st, vgrad):
    v = [(st["pct"] * PW).ravel() / 5.0, st["mean"] * np.array([1.5, 1.5, 1.5]), np.log(st["std"] + 1e-3) * 30, [st["vig"] * 40]]
    if vgrad:
        v.append(st["thirds"] * 20)
    return np.concatenate(v)


def unpack(x, vgrad):
    P = json.loads(json.dumps(G.DEFAULT["bg"]))
    P.update(knots_y=np.maximum.accumulate(x[:24].reshape(3, 8), axis=1).tolist(), sat=float(x[24]), hue_deg=float(x[25]),
             vgrad_stops=float(x[29]) if vgrad else 0.0, anchor_ab=[float(x[27]), float(x[28])])
    return P, float(x[26])


def run_bg(rgb, P, vs, luts=None, mb=None):
    """global grade -> chroma anchor (what the temporal EMA converges to) -> elliptical vignette."""
    out = G.global_grade(rgb, P, luts)
    if mb is not None:
        lab = G.to_lab(out)
        dab = np.array(P["anchor_ab"], np.float32) - lab[..., 1:][mb].mean(0)
        lab[..., 1] += dab[0]; lab[..., 2] += dab[1]
        out = G.from_lab(lab)
    return out * G.vignette_map(*rgb.shape[:2], vs, 2.2)[..., None]


def cdf_init(befores, afters):
    """pooled per-channel CDF matching, evaluated at the knots (before-domain x -> after-domain y)."""
    ky = np.zeros((3, 8))
    for c in range(3):
        b = np.concatenate([x[..., c][m] for x, m in befores])
        a = np.concatenate([x[..., c][m] for x, m in afters])
        for k, xk in enumerate(G.KX):
            f = (b <= xk).mean() if 0 < xk < 1 else (0.0 if xk == 0 else 1.0)
            ky[c, k] = np.quantile(a, min(max(f, 0), 1))
    ky = np.maximum.accumulate(ky, axis=1)
    return ky


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", nargs=4, metavar=("RAW", "DIVER_MASK", "AFTER", "AFTER_PARTS_NPZ"), required=True)
    ap.add_argument("--out", default="out/grade_params.json")
    ap.add_argument("--vgrad", action="store_true", help="fit a vertical depth falloff too")
    ap.add_argument("--nfev", type=int, default=80)
    a = ap.parse_args()

    pairs, befores, afters = [], [], []
    for raw, mk, aft, npz in a.pair:
        b = load_rgb(raw)
        mb = bg_mask(cv2.imread(mk, 0) > 40, b.shape[:2], 3)
        t = load_rgb(aft)
        person = np.load(npz)["person"]
        mt = bg_mask(person, t.shape[:2], 3)
        st = stats(t, mt)
        pairs.append((b, mb, vec(st, a.vgrad), st))
        befores.append((b, mb)); afters.append((t, mt))

    # init: the v1 power-law fit (a*x^g+b) sampled at the knots, a good smooth monotone start
    old = json.loads(Path("/tmp/gp_old.json").read_text())["bg"] if Path("/tmp/gp_old.json").exists() else None
    if old and "a" in old:
        ky0 = np.clip(np.array([np.array(old["a"][c]) * np.power(np.clip(G.KX, 1e-6, 1), old["g"][c]) + old["b"][c] for c in range(3)]), 0, 1)
        ky0 = np.maximum.accumulate(ky0, axis=1)
    else:
        ky0 = cdf_init(befores, afters)
    mean_ab = np.mean([t[3]["mean"][1:] for t in pairs], axis=0)
    x0 = np.concatenate([ky0.ravel(), [1.0, 0.0, 0.25, mean_ab[0], mean_ab[1]] + ([0.0] if a.vgrad else [])])
    lo = np.concatenate([np.zeros(24), [0.5, -30, 0.0, -30, -60] + ([-1.5] if a.vgrad else [])])
    hi = np.concatenate([np.ones(24), [2.5, 30, 0.8, 30, 10] + ([1.5] if a.vgrad else [])])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    def resid(x):
        P, vs = unpack(x, a.vgrad)
        luts = G.make_luts(P)
        return np.concatenate([vec(stats(run_bg(b, P, vs, luts, mb), mb), a.vgrad) - tv for b, mb, tv, _ in pairs])

    r0 = resid(x0)
    sol = least_squares(resid, x0, bounds=(lo, hi), diff_step=5e-3, x_scale=np.concatenate([np.full(24, 0.1), [0.2, 5, 0.1, 3, 3] + ([0.3] if a.vgrad else [])]), max_nfev=a.nfev)
    print("status", sol.status, "nfev", sol.nfev, "rms", np.sqrt(np.mean(sol.fun ** 2)))
    P, vs = unpack(sol.x, a.vgrad)

    rep = {"rms_init(cdf)": float(np.sqrt(np.mean(r0 ** 2))), "rms_fit": float(np.sqrt(np.mean(sol.fun ** 2))), "pairs": []}
    luts = G.make_luts(P)
    for (b, mb, tv, ts), name in zip(pairs, [Path(p[0]).stem for p in a.pair]):
        s = stats(run_bg(b, P, vs, luts, mb), mb)
        rep["pairs"].append({"raw": name, "dLab_mean": (s["mean"] - ts["mean"]).round(2).tolist(),
                             "std_ratio": (s["std"] / ts["std"]).round(2).tolist(), "pct_max_abs_diff": float(np.abs(s["pct"] - ts["pct"]).max().round(1)),
                             "vig_out/target": [round(s["vig"], 3), round(ts["vig"], 3)],
                             "thirds_out": s["thirds"].round(3).tolist(), "thirds_target": ts["thirds"].round(3).tolist()})
    out = Path(a.out)
    d = json.loads(out.read_text()) if out.exists() else {}
    d.setdefault("finish", G.DEFAULT["finish"]); d.setdefault("skin", G.DEFAULT["skin"])
    d["bg"] = {**G.DEFAULT["bg"], **P}
    d["finish"]["vignette"] = vs
    d["fit_report_bg"] = rep
    out.write_text(json.dumps(d, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
