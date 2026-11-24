"""Calibrate the regional colour model against the reference stills.

For each scene (stand / entry / under) the user supplied a graded reference
still and the matching original still. They are pixel-aligned (luma corr 0.99),
so a joint ridge regression finds, per scene, a shared cubic RGB polynomial
plus a per-region delta (top / body / subject), weighted by the SAM-derived
soft region masks of that exact frame:

    ref(x) ~ poly_shared(orig(x)) + sum_r w_r(x) * poly_delta_r(orig(x))

Both images are blurred (sigma 2 px) before fitting so the regression learns
colour only, not the reference's grain/halation (those are modelled
separately in uw_look.finish). Reports dE2000 in-sample and on spatial
hold-out halves (fit on one half, test on the other) so over-fitting is visible.

    python uw_fit.py --masks masks_uw/clip --out look.json \
        --pair stand:0:ref1.jpg:orig1.jpg --pair entry:141:... --pair under:255:...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from uw_look import (DEG, REGIONS, Look, feat_degrees, mono, n_feat, refine_mask, region_weights)

import os
HI_DEG_PEN = float(os.environ.get('UW_HIPEN', 100))
DEFAULT_POST = {"knee": 0.86, "halation": 0.30, "halation_thr": 0.78,
                "halation_tint": [1.0, 0.58, 0.26], "grain": 0.009}


def de2000(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = rgb2lab(np.clip(a, 0, 1).reshape(-1, 1, 3))
    lb = rgb2lab(np.clip(b, 0, 1).reshape(-1, 1, 3))
    return deltaE_ciede2000(la, lb).ravel()


def load_rgb(p: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def design(x: np.ndarray, w: dict) -> np.ndarray:
    """[N,3] pixels + dict of [N] weights -> [N, 4n] regression design matrix."""
    f = mono(x)
    return np.concatenate([f] + [f * w[r][:, None] for r in REGIONS], 1)


def solve(X: np.ndarray, Y: np.ndarray, lam_shared: float, lam_delta: float,
          hipen: float | None = None) -> np.ndarray:
    n = X.shape[1] // (1 + len(REGIONS))
    # degree-aware penalty: affine terms are almost free, curvature costs more
    # and cubic terms most, so the model stays smooth outside the training gamut
    dscale = 1.0 + (HI_DEG_PEN if hipen is None else hipen) * np.maximum(feat_degrees(DEG) - 1, 0) ** 2
    pen = np.concatenate([lam_shared * dscale] + [lam_delta * dscale] * len(REGIONS))
    A = X.T.astype(np.float64) @ X.astype(np.float64) + np.diag(pen)
    return np.linalg.solve(A, X.T.astype(np.float64) @ Y.astype(np.float64)).astype(np.float32)


def split_coef(c: np.ndarray) -> dict:
    n = c.shape[0] // (1 + len(REGIONS))
    shared = c[:n]
    return {r: shared + c[n * (i + 1): n * (i + 2)] for i, r in enumerate(REGIONS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pair", action="append", required=True,
                    help="scene:frame_index:reference.jpg:original.jpg")
    ap.add_argument("--lam-shared", type=float, default=1e-4)
    ap.add_argument("--lam-delta", type=float, default=3e-3)
    ap.add_argument("--samples", type=int, default=250_000)
    a = ap.parse_args()

    full = {c: np.load(a.masks / f"{c}_full.npy", mmap_mode="r")
            for c in ("subject", "water", "sky") if (a.masks / f"{c}_full.npy").exists()}
    coef, fb, support, report = {}, {}, {}, {}
    rng = np.random.default_rng(0)
    for spec in a.pair:
        scene, fi, ref_p, org_p = spec.split(":", 3)
        fi = int(fi)
        ref = load_rgb(ref_p)
        org = cv2.resize(load_rgb(org_p), (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        ref_b = cv2.GaussianBlur(ref, (0, 0), 2.0)
        org_b = cv2.GaussianBlur(org, (0, 0), 2.0)
        m = {c: refine_mask(np.asarray(full[c][fi]), org) if c in full else None
             for c in ("subject", "water", "sky")}
        w = region_weights(scene, org, m["subject"], m["water"], m["sky"])
        h, wd = org.shape[:2]
        X_all = org_b.reshape(-1, 3)
        Y_all = ref_b.reshape(-1, 3)
        W_all = {r: w[r].reshape(-1) for r in REGIONS}
        cov = {r: float(W_all[r].mean()) for r in REGIONS}
        idx = rng.choice(len(X_all), min(a.samples, len(X_all)), replace=False)
        sel = lambda ix: ({r: W_all[r][ix] for r in REGIONS})  # noqa: E731
        c = solve(design(X_all[idx], sel(idx)), Y_all[idx], a.lam_shared, a.lam_delta)
        coef[scene] = {r: v.tolist() for r, v in split_coef(c).items()}
        # safe affine fallback (higher-order terms effectively off)
        c1 = solve(design(X_all[idx], sel(idx)), Y_all[idx], a.lam_shared, a.lam_delta, hipen=1e8)
        fb[scene] = {r: v.tolist() for r, v in split_coef(c1).items()}
        # colour support: how densely the reference pair covers each input colour
        from scipy.ndimage import gaussian_filter
        support[scene] = {}
        for r in REGIONS:
            cnt, _ = np.histogramdd(X_all[::2], bins=Look.GRID, range=[(0, 1)] * 3,
                                    weights=W_all[r][::2])
            cnt = gaussian_filter(cnt * 2.0, 0.9)
            sup = np.clip((cnt - 3.0) / (60.0 - 3.0), 0, 1)
            support[scene][r] = (sup * sup * (3 - 2 * sup)).tolist()
        pred = np.clip(design(X_all[idx], sel(idx)) @ c, 0, 1)
        e = de2000(pred, Y_all[idx])
        e0 = de2000(X_all[idx], Y_all[idx])
        per = {r: float((e * W_all[r][idx]).sum() / max(W_all[r][idx].sum(), 1e-6)) for r in REGIONS}
        # spatial hold-out halves (top/bottom)
        yy = (np.arange(len(X_all)) // wd) < h // 2
        ho = []
        for tr, te in ((yy, ~yy), (~yy, yy)):
            itr = np.flatnonzero(tr)[:: max(1, tr.sum() // 120_000)]
            ite = np.flatnonzero(te)[:: max(1, te.sum() // 60_000)]
            cc = solve(design(X_all[itr], {r: W_all[r][itr] for r in REGIONS}), Y_all[itr],
                       a.lam_shared, a.lam_delta)
            pp = np.clip(design(X_all[ite], {r: W_all[r][ite] for r in REGIONS}) @ cc, 0, 1)
            ho.append(float(de2000(pp, Y_all[ite]).mean()))
        report[scene] = {"region_coverage": {k: round(v, 3) for k, v in cov.items()},
                         "dE_identity": round(float(e0.mean()), 2),
                         "dE_fit_mean": round(float(e.mean()), 2),
                         "dE_fit_p95": round(float(np.percentile(e, 95)), 2),
                         "dE_per_region": {k: round(v, 2) for k, v in per.items()},
                         "dE_holdout_halves": [round(x, 2) for x in ho]}
        print(scene, json.dumps(report[scene]), flush=True)

    Look(coef, DEFAULT_POST, fb, support).save(a.out)
    (a.out.with_suffix(".report.json")).write_text(json.dumps(report, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
