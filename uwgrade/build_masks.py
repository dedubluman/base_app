#!/usr/bin/env python3
"""Build every per-frame mask from the SAM 3.1 outputs.

    python -m uwgrade.build_masks --video inputs/player_export.mov --stage env
    python -m uwgrade.build_masks --video inputs/player_export.mov --stage diver
    python -m uwgrade.build_masks --video inputs/player_export.mov --stage parts

Layout (uint8 PNG alphas):
    masks/<stem>__final/diver/<t>.png    SAM 3.1 diver (interpolated + edge-snapped on odd frames)
    masks/<stem>__final/rock/<t>.png     rocks / seabed probability, 1/4 resolution, temporally smoothed
    masks/<stem>__final/{skin,gear,shorts}/<t>.png   parts of the diver, full resolution
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from uwgrade import diver as dv  # noqa: E402
from uwgrade import env, parts, video  # noqa: E402
from uwgrade.color import smoothstep  # noqa: E402

KF = list(range(0, 451, 30))
MIN_BLOB = 600   # quarter-resolution px (~9600 full-resolution px)


def paths(stem: str):
    base = REPO / "masks" / f"{stem}__final"
    return base, REPO / "masks" / f"{stem}__keyframes", REPO / "masks" / f"{stem}__person_s2"


def parse_range(spec: str) -> tuple[int, int]:
    a, b = spec.split(":")[:2]
    return int(a), int(b)


def read_a(d: Path, t: int) -> np.ndarray:
    return cv2.imread(str(d / f"{t:05d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def write_a(d: Path, t: int, a: np.ndarray):
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{t:05d}.png"), (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def stage_env(frames, base: Path, kfdir: Path, log=print):
    T = len(frames)
    dd = base / "diver"
    have_diver = all((dd / f"{t:05d}.png").exists() for t in KF)
    ex = (lambda t: cv2.dilate((read_a(dd, t) > 0.3).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) if have_diver else None
    clf, (X, y, g) = env.train(frames, kfdir, KF, exclude=ex, log=log)
    env.cv_auc(X, y, g, log=log)
    pickle.dump(clf, open(base / "env_clf.pkl", "wb")) if base.exists() else None
    t0 = time.time()
    h4, w4 = frames.shape[1] // env.DS, frames.shape[2] // env.DS
    P = np.zeros((T, h4, w4), np.float32)
    for t in range(T):
        P[t] = env.p_rock(clf, np.asarray(frames[t]))
        if t % 100 == 0:
            log(f"[env] p_rock {t}/{T} ({time.time() - t0:.0f}s)")
    np.save(base / "p_rock_raw_q.npy", (P * 255).astype(np.uint8))
    env_alpha(base, P, log=log)


def env_alpha(base: Path, P: np.ndarray | None = None, lo: float = 0.12, hi: float = 0.88, sigma_q: float = 3.0, log=print):
    """Raw rock probability (T,h,w) -> soft rock alpha. Wide transition + blur (12 px at full res) because the
    real water/rock boundary is haze, not an edge: distant rocks fade into the water column."""
    from scipy.ndimage import convolve1d

    if P is None:
        P = np.load(base / "p_rock_raw_q.npy").astype(np.float32) / 255.0
    T = len(P)
    k = np.array([1, 2, 3, 2, 1], np.float32)
    k /= k.sum()
    Ps = convolve1d(P, k, axis=0, mode="nearest")
    A = smoothstep(lo, hi, Ps)
    (base / "rock").mkdir(parents=True, exist_ok=True)
    removed = 0
    for t in range(T):
        # isolated rock specks in the water (reflection / particle blobs) and tiny water holes in the rock are
        # classifier noise, not structure: drop components under MIN_BLOB quarter-resolution pixels
        b = (Ps[t] > 0.5).astype(np.uint8)
        for val in (1, 0):
            n, lab, st, _ = cv2.connectedComponentsWithStats(b if val else 1 - b)
            small = np.isin(lab, [i for i in range(1, n) if st[i, 4] < MIN_BLOB])
            if small.any():
                removed += int(small.any())
                A[t][small] = float(1 - val)
        write_a(base / "rock", t, cv2.GaussianBlur(A[t], (0, 0), sigma_q))
    log(f"[env] wrote {T} rock alphas (smoothstep {lo}-{hi}, blur {sigma_q} px at 1/{env.DS} res); "
        f"mean raw frame-to-frame |dP| {np.abs(np.diff(P, axis=0)).mean():.4f}, smoothed {np.abs(np.diff(Ps, axis=0)).mean():.4f}")


def stage_diver(frames, base: Path, sdir: Path, log=print, only=None):
    sdirs = {0: sdir}
    odd = sdir.parent / (sdir.name + "o1")
    if odd.exists():
        sdirs[1] = odd
    pdir = sdir.parent / (sdir.name.split("__")[0] + "__patch")
    return dv.build(frames, sdirs, base / "diver", stride=2, log=log, only=only, patch_dir=pdir if pdir.exists() else None)


def stage_parts(frames, base: Path, kfdir: Path, log=print, only=None):
    """Classify skin / gear / shorts inside the diver for every frame, then smooth the probabilities over time
    (3-tap [.25 .5 .25], rolling window): the parts move with the diver, a 1-2 frame blotch is a classifier glitch."""
    T = len(frames)
    dd = base / "diver"
    dfn = lambda t: read_a(dd, t)

    def sfn(t):
        return env.seeds(kfdir, t, frames.shape[1:3])["shorts"]

    clf, (X, y, g) = parts.train(frames, dfn, sfn, KF, log=log)
    parts.loo_accuracy(X, y, g, log=log)
    order = list(range(T)) if only is None else list(only)
    consecutive = all(b - a == 1 for a, b in zip(order, order[1:]))
    win: dict = {}

    def emit(t, nbrs):
        w = {i: (0.5 if i == t else 0.25) for i in nbrs}
        tot = sum(w.values())
        acc = sum(w[i] / tot * win[i] for i in nbrs)
        for k, name in enumerate(("skin", "gear", "shorts")):
            write_a(base / name, t, acc[k])

    for n, t in enumerate(order):
        D = dfn(t)
        win[t] = parts.classify(clf, np.asarray(frames[t]), D) * D[None]
        if not consecutive:
            emit(t, [t])
        else:
            if n >= 1:
                c = order[n - 1]
                emit(c, [i for i in (c - 1, c, c + 1) if i in win])
                win.pop(c - 1, None)
        if n % 100 == 0:
            log(f"[parts] {n}/{len(order)}")
    if consecutive:
        c = order[-1]
        emit(c, [i for i in (c - 1, c) if i in win])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--stage", choices=["env", "diver", "parts", "all"], default="all")
    ap.add_argument("--frames", default=None, help="subset for diver/parts, e.g. 0:452:6 (default: every frame)")
    a = ap.parse_args()
    only = None
    if a.frames:
        from uwgrade.sam_keyframes import parse_frames
        only = parse_frames(a.frames)
    stem = a.video.stem
    base, kfdir, sdir = paths(stem)
    base.mkdir(parents=True, exist_ok=True)
    frames = video.frame_cache(a.video, REPO / ".cache" / "frames_u8.npy")
    if a.stage in ("diver", "all"):
        stage_diver(frames, base, sdir, only=only)
    if a.stage in ("env", "all"):
        stage_env(frames, base, kfdir)
    if a.stage in ("parts", "all"):
        stage_parts(frames, base, kfdir, only=only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
