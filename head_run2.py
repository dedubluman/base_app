#!/usr/bin/env python3
"""Head labels for every frame -> .cache/head_labels/{MASK,FACE,HAIR}/%05d.png

pass 1  seeds tracked from the key frames -> random walker (f0-f9: SAM hair blob only, the head is behind the arm)
pass 2+ temporal consistency: a class whose area changes by more than 15 % against a neighbouring frame is
        re-segmented on that frame with the (flow-warped, eroded) masks of the neighbours as seeds.
"""
import pickle, sys
from pathlib import Path
import cv2, numpy as np
from head_split import *

REPO = Path(__file__).resolve().parent
v4 = REPO / ".cache" / "cls_v4"
out = REPO / ".cache" / "head_labels"
CLS = ("MASK", "FACE", "HAIR")
LIM = 0.15
RIM_END = 73


def warp_mask(frames, m, f_from, f_to, cx, cy):
    """mask of frame f_from -> frame f_to (backward flow f_to -> f_from sampled on a crop)."""
    g = lambda f: cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)
    fl, x0, y0 = local_flow(g(f_to), g(f_from), cx, cy, R=160)
    h, w = fl.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sub = m[y0:y0 + h, x0:x0 + w].astype(np.float32)
    wm = cv2.remap(sub, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR, borderValue=0)
    res = np.zeros((H, W), bool)
    res[y0:y0 + h, x0:x0 + w] = wm > 0.5
    return res


def flagged(areas):
    bad = {}
    for c in CLS:
        a = areas[c]
        for f in range(1, N):
            if f < 10:
                continue
            lo, hi = sorted((a[f], a[f - 1]))
            if hi >= 40 and (hi - lo) / hi > LIM:
                bad.setdefault(f, set()).add(c)
            if f + 1 < N:
                lo, hi = sorted((a[f], a[f + 1]))
                if hi >= 40 and (hi - lo) / hi > LIM:
                    bad.setdefault(f, set()).add(c)
    return bad


def main():
    tr, A, B = pickle.load(open(REPO / ".cache" / "head_tracks.pkl", "rb"))
    frames = {f: cv2.imread(str(REPO / ".cache" / "sam_frames_diver_bwd" / f"{f:05d}.jpg")) for f in range(N)}
    for c in CLS:
        (out / c).mkdir(parents=True, exist_ok=True)
    lab = {}
    for f in range(N):
        if f < 10:
            lab[f] = {"MASK": np.zeros((H, W), bool), "FACE": np.zeros((H, W), bool), "HAIR": B[f].copy()}
        else:
            lab[f] = head_labels(frames[f], A[f], seeds_for(tr, f), f, blob=B[f]) if f > RIM_END or any(a <= f <= b for a, b in FACE_WINDOWS) else {c: np.zeros((H, W), bool) for c in CLS}
            if f <= RIM_END:                                   # head behind the arm: hair core + outer band
                mk, hr = rim_split(B[f], A[f])
                lab[f]["MASK"], lab[f]["HAIR"] = mk, hr
                if not any(a <= f <= b for a, b in FACE_WINDOWS):
                    lab[f]["FACE"] = np.zeros((H, W), bool)
    area = lambda: {c: [int(lab[f][c].sum()) for f in range(N)] for c in CLS}
    a0 = area()
    log = [f"pass1 flagged frames (>15 %): {sorted(flagged(a0))}"]
    # pass 2: motion-compensated majority vote over f-2..f+2 (flow-warped neighbour labels), twice
    W5 = {0: 2.0, 1: 1.5, -1: 1.5, 2: 1.0, -2: 1.0}
    for it in range(2):
        new = {}
        for f in range(N):
            if f < 10:
                new[f] = lab[f]
                continue
            ys, xs = np.nonzero(B[f] | lab[f]["FACE"] | lab[f]["HAIR"])
            if len(xs) == 0:
                new[f] = lab[f]
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            votes = {c: np.zeros((H, W), np.float32) for c in CLS}
            tot = 0.0
            for d, w in W5.items():
                g = f + d
                if g < 10 or g >= N:
                    continue
                tot += w
                for c in CLS:
                    votes[c] += w * (lab[f][c] if d == 0 else warp_mask(frames, lab[g][c], g, f, cx, cy))
            none = tot - sum(votes.values())
            stack = np.stack([votes[c] for c in CLS] + [none])
            win = stack.argmax(0)
            Bd = cv2.dilate(B[f].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            cur = {}
            for i, c in enumerate(CLS):
                if c == "FACE" and f < 74:                      # the ear at f44-f50 is too small to vote on
                    cur[c] = lab[f][c]
                    continue
                m = (win == i)
                if c != "FACE":
                    m &= Bd
                m &= (A[f] | Bd)
                cur[c] = m
            # exclusivity (a pixel keeps the class with the most votes)
            for c in CLS:
                num, cc, st, _ = cv2.connectedComponentsWithStats(cur[c].astype(np.uint8), 8)
                keep = np.zeros_like(cur[c])
                for k in range(1, num):
                    if st[k, cv2.CC_STAT_AREA] >= (20 if c == "FACE" else 25):
                        keep |= cc == k
                cur[c] = keep
            new[f] = cur
        lab = new
        log.append(f"pass{it + 2} (vote) flagged: {sorted(flagged(area()))}")
    for f in range(N):
        for c in CLS:
            cv2.imwrite(str(out / c / f"{f:05d}.png"), lab[f][c].astype(np.uint8) * 255)
    a = area()
    rows = ["frame,MASK,FACE,HAIR"] + [f"{f},{a['MASK'][f]},{a['FACE'][f]},{a['HAIR'][f]}" for f in range(N)]
    (out / "areas.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(log))
    print("\n".join(rows[::3]))


if __name__ == "__main__":
    main()
