#!/usr/bin/env python3
"""Fingers beyond the person mask at f0-f14: colour region-grow around the user's two points
(source (528,878) at f0 -> (678,878) at f14, linear in between), seeded from the skin colour of the
person-mask pixels next to the hand, attached to the diver mask only."""
import pickle
from pathlib import Path
import cv2
import numpy as np
from diver_classes import dil, luma01, read_mask

REPO = Path(__file__).resolve().parent
frames = REPO / ".cache" / "sam_frames_diver_bwd"
pdir = REPO / "masks" / "diver__fwd" / "subject"


def grow(f, verbose=False):
    img = cv2.imread(str(frames / f"{f:05d}.jpg"))
    p = read_mask(pdir, f)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy = int(528 + (678 - 528) * f / 14), 878
    disk = np.zeros(p.shape, np.uint8)
    cv2.circle(disk, (cx, cy), 85, 1, -1)
    near = (disk > 0) & p
    seed_px = lab[(np.zeros(p.shape, np.uint8) + cv2.circle(np.zeros(p.shape, np.uint8), (cx, cy), 22, 1, -1) > 0) & p]
    if len(seed_px) < 30:
        seed_px = lab[near]
    seed = np.median(seed_px, 0) if len(seed_px) else np.median(lab[p], 0)
    g = luma01(img)
    d = np.linalg.norm(lab - seed, axis=2)
    out, used = np.zeros_like(p), None
    for thr in (12, 16, 20, 24):
        cand = (d < thr) & (disk > 0) & ~p & (g > 0.28)
        attached = np.zeros_like(p)
        n, cc, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
        for k in range(1, n):
            comp = cc == k
            if st[k, cv2.CC_STAT_AREA] >= 50 and (dil(comp, 3) & p).any():
                attached |= comp
        attached = cv2.morphologyEx(attached.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        if 200 <= attached.sum() <= 3500:
            out, used = attached & ~p, thr
            break
    return out, used, seed


if __name__ == "__main__":
    res = {}
    for f in range(15):
        m, thr, seed = grow(f)
        res[f] = m
        print(f"f{f:2d} thr {thr} extra {int(m.sum()):5d}  seed Lab {np.round(seed,0)}", flush=True)
    pickle.dump(res, open(REPO / "masks" / "diver__cls" / "hand_grow.pkl", "wb"))
