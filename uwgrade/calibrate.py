#!/usr/bin/env python3
"""Measure the source-side statistics the grade needs and write params JSON.

For every region class (water, rock, surface, skin, shorts) it records, on the keyframes:
    L_q      lightness quantiles (after node 1) -> quantile-matched to the reference quantiles
    ab_ramp  mean (a*, b*) versus L*           -> the "typical" colour at each lightness; the node
                                                  replaces it with the reference ramp and keeps a
                                                  share of the local deviation from it
Node-1 levels come from the clip's non-diver pixels. Nothing is per-frame, so the grade is constant.

    python -m uwgrade.calibrate --video inputs/player_export.mov --out uwgrade/params_player_export.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from uwgrade import env, video  # noqa: E402
from uwgrade.build_masks import KF, paths, read_a  # noqa: E402
from uwgrade.color import rgb2lab  # noqa: E402
from uwgrade.grade import Q_LEVELS, Params, save_params  # noqa: E402


def ab_ramp(L, a, b, lo, hi, width=3.0, min_n=150):
    edges = np.arange(np.floor(lo), np.ceil(hi) + width, width)
    Lk, ak, bk = [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (L >= e0) & (L < e1)
        if m.sum() >= min_n:
            Lk.append(float(L[m].mean()))
            ak.append(float(a[m].mean()))
            bk.append(float(b[m].mean()))
    ak, bk = np.array(ak), np.array(bk)
    if len(Lk) >= 3:   # light smoothing of the ramp
        k = np.array([0.25, 0.5, 0.25])
        ak = np.convolve(np.pad(ak, 1, mode="edge"), k, mode="valid")
        bk = np.convolve(np.pad(bk, 1, mode="edge"), k, mode="valid")
    return [Lk, ak.tolist(), bk.tolist()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--l1-strength", type=float, default=0.85)
    a = ap.parse_args()
    stem = a.video.stem
    base, kfdir, _ = paths(stem)
    frames = video.frame_cache(a.video, REPO / ".cache" / "frames_u8.npy")
    rng = np.random.default_rng(0)

    env_px, per = [], []
    data = {k: [] for k in ("water", "rock", "surf", "skin", "shorts")}
    # pass 1: node-1 levels from non-diver pixels
    for t in KF:
        rgb = np.asarray(frames[t], np.float32)[::2, ::2] / 255.0
        D = read_a(base / "diver", t)[::2, ::2] > 0.2
        env_px.append(rgb[~D])
    E = np.concatenate(env_px)
    bp = np.percentile(E, 0.5, axis=0)
    wp = np.percentile(E, 99.8, axis=0)
    p = Params(l1_black=[float(x) for x in bp], l1_white=[float(x) for x in wp], l1_strength=a.l1_strength)
    print("node-1 levels black", np.round(bp, 3), "white", np.round(wp, 3))
    from uwgrade.grade import Grader
    g0 = Grader(p)
    # pass 2: region statistics after node 1
    for t in KF:
        rgb = np.asarray(frames[t], np.float32) / 255.0
        lab = rgb2lab(g0.node1(rgb))[::2, ::2]
        D = read_a(base / "diver", t)[::2, ::2]
        R = read_a(base / "rock", t)[::2, ::2]
        sd = env.seeds(kfdir, t, frames.shape[1:3])
        S = sd["surf"][::2, ::2]
        envm = D < 0.05
        sets = {"rock": envm & (R > 0.85), "water": envm & (R < 0.15), "surf": envm & S & (R < 0.5)}
        for k in ("skin", "shorts"):
            P = read_a(base / k, t)[::2, ::2]
            sets[k] = (D > 0.9) & (P > 0.8)
        for k, m in sets.items():
            if m.sum() > 50:
                px = lab[m]
                if len(px) > 60000:
                    px = px[rng.choice(len(px), 60000, replace=False)]
                data[k].append(px)
    cal = {}
    for k, v in data.items():
        if not v:
            print("WARNING: no pixels for", k)
            continue
        px = np.concatenate(v)
        Lq = np.percentile(px[:, 0], Q_LEVELS)
        lo, hi = np.percentile(px[:, 0], [1, 99])
        cal[k] = {"n": int(len(px)), "L_q": [float(x) for x in Lq], "ab_ramp": ab_ramp(px[:, 0], px[:, 1], px[:, 2], lo, hi)}
        print(f"{k:7s} n={len(px):7d} L_q={np.round(Lq, 1).tolist()}  mean ab=({px[:, 1].mean():.1f}, {px[:, 2].mean():.1f})")
    p.cal = cal
    save_params(p, a.out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
