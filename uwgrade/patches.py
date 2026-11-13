#!/usr/bin/env python3
"""Continuity repair of dark accessories (fins, snorkel) that SAM 3.1's "person" tracking dropped for a few frames.

Found by scanning the 453 diver masks for temporal discontinuities: exactly one, 87 -> 88 (area x0.70). The black fin
is still visible on frames 88-90 (it leaves the frame by ~90) and so is the snorkel tube, but the "person" mask no
longer encloses them. SAM 3.1 itself is prompted for them ("swim fin", "fin" -> uwgrade.sam_keyframes --tag fin_fix)
and a candidate is merged into the diver mask only if it is
    * dark   (mean lightness < DARK_REL x the diver's median), and
    * big enough (>= MIN_AREA px), and
    * either touches the diver mask, or sits where the previous frame's mask had material this frame lost.

    python -m uwgrade.patches --video inputs/player_export.mov --frames 88:90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from uwgrade import color as C  # noqa: E402
from uwgrade import diver as dv  # noqa: E402
from uwgrade import env, video  # noqa: E402
from uwgrade.build_masks import parse_range  # noqa: E402

DARK_REL = 0.60
MIN_AREA = 300
P_MIN = 0.06


def pick(frames, kfdir: Path, t: int, sam_t: np.ndarray, prev: np.ndarray | None, log=print):
    rgb = np.asarray(frames[t]).astype(np.float32) / 255.0
    L = C.rgb2lab(rgb)[..., 0]
    med = float(np.median(L[sam_t])) if sam_t.sum() > 50 else 40.0
    near = cv2.dilate(sam_t.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    lost = np.zeros_like(sam_t)
    if prev is not None:
        lost = cv2.dilate((prev & ~sam_t).astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
    out = np.zeros_like(sam_t)
    for prompt in ("swim fin", "fin"):
        ms, ps = env.load_kf(kfdir, t, prompt)
        for m, p in zip(ms, ps):
            a = int(m.sum())
            if a < MIN_AREA or p < P_MIN or (m & ~sam_t).sum() < 0.3 * a:
                continue
            dark = float(L[m].mean() / med)
            touch = float((m & near).sum() / a)
            cont = float((m & lost).sum() / a)
            ok = dark < DARK_REL and (touch >= 0.5 or cont >= 0.5)
            log(f"[patch] f{t} '{prompt}' p={p:.2f} area={a} dark={dark:.2f} touch={touch:.2f} continuity={cont:.2f} -> {'MERGE' if ok else 'skip'}")
            if ok:
                out |= m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", required=True, help="e.g. 88:90 (consecutive frames after the last good one)")
    ap.add_argument("--kf-tag", default="fin_fix")
    a = ap.parse_args()
    stem = a.video.stem
    frames = video.frame_cache(a.video, REPO / ".cache" / "frames_u8.npy")
    kfdir = REPO / "masks" / f"{stem}__{a.kf_tag}"
    sdirs = {0: REPO / "masks" / f"{stem}__person_s2", 1: REPO / "masks" / f"{stem}__person_s2o1"}
    sam = dv.load_all(sdirs, len(frames), 2)
    lo, hi = parse_range(a.frames)
    pdir = REPO / "masks" / f"{stem}__patch"
    pdir.mkdir(parents=True, exist_ok=True)
    prev = sam[lo - 1]
    for t in range(lo, hi + 1):
        add = pick(frames, kfdir, t, sam[t], prev)
        cv2.imwrite(str(pdir / f"{t:05d}.png"), add.astype(np.uint8) * 255)
        prev = sam[t] | add
        print(f"[patch] f{t}: +{int(add.sum())} px merged (mask {int(sam[t].sum())} -> {int(prev.sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
