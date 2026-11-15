#!/usr/bin/env python3
"""Head split: MASK (glass + frame + strap) / FACE / HAIR (+ SNORKEL), per frame.

SAM 3.1 gives one head blob (text + box prompts return the same blob whatever the word: "hair", "face",
"diving mask", "goggles", ... -- see probe in the report), so the parts are decided here:

  1. seed points on a few key frames (read on 6x grid crops of the source, source px),
  2. every seed is tracked to the other frames with DIS dense optical flow (forward + backward),
     a seed that leaves the head region or changes luminance is dropped,
  3. a marker watershed on the (slightly blurred) source colour allocates the head region to the seeds,
     so the part borders follow the real image edges.

    from head_split import split_head
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
N, W, H = 90, 1232, 1078
LAB = {"BODY": 1, "FACE": 2, "HAIR": 3, "MASK": 4, "SNORKEL": 5, "WATER": 6}
INV = {v: k for k, v in LAB.items()}

# key-frame seeds, source px (x, y).  Read on grid crops (6x) of the source frames.
# MASK includes the snorkel tube/tip that is clipped to the strap.
KEY = {
    10: {
        "MASK": [(583, 760), (584, 795), (587, 815), (594, 829)],
        "HAIR": [(604, 795), (606, 815), (600, 781), (618, 825)],
        "BODY": [(624, 765), (636, 805)],
    },
    33: {
        "MASK": [(825, 777), (842, 802), (846, 826), (849, 842)],
        "HAIR": [(896, 842), (908, 852), (900, 826), (916, 860)],
        "BODY": [(864, 802), (876, 842), (912, 762)],
    },
    45: {
        "FACE": [(814, 809)],
        "HAIR": [(802, 793), (797, 803), (815, 815), (817, 823)],
        "MASK": [(778, 774), (780, 782), (784, 798), (788, 812), (796, 823), (804, 834), (788, 827), (792, 838),
                 (802, 855), (815, 865), (830, 869)],
        "BODY": [(832, 780), (839, 810)],
    },
    55: {
        "HAIR": [(778, 715), (777, 727), (780, 742)],
        "MASK": [(756, 688), (762, 732), (770, 739), (777, 744), (768, 759), (777, 769), (787, 777)],
        "BODY": [(805, 722), (797, 755)],
    },
    65: {
        "HAIR": [(753, 597), (755, 610), (805, 627), (812, 643), (802, 660)],
        "MASK": [(733, 571), (740, 598), (745, 615), (748, 623), (740, 627), (743, 640)],
        "BODY": [(765, 610), (770, 643)],
    },
    75: {
        "HAIR": [(770, 583), (775, 593), (823, 588)],
        "MASK": [(793, 593), (803, 602), (805, 612), (817, 618), (792, 608)],
        "FACE": [(807, 582), (808, 595), (839, 598), (844, 602)],
        "BODY": [(758, 553), (793, 555), (826, 553)],
    },
    80: {
        "HAIR": [(803, 543), (793, 552), (815, 553), (792, 563)],
        "MASK": [(808, 573), (800, 582), (820, 595), (840, 602), (853, 598)],
        "FACE": [(828, 565), (825, 575), (845, 573), (862, 585), (837, 585)],
        "BODY": [(828, 527), (845, 540), (878, 562)],
    },
    87: {
        "MASK": [(794, 435), (799, 445), (803, 465), (809, 485), (814, 494),
                 (833, 495), (835, 502), (847, 530), (839, 525), (859, 534)],
        "HAIR": [(819, 460), (827, 455), (814, 472), (805, 492), (812, 509), (822, 520)],
        "FACE": [(834, 484), (850, 502), (855, 512), (844, 515)],
        "BODY": [(879, 472), (872, 455), (887, 492), (889, 509)],
    },
}


def gray(f, frames):
    return cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)


_dis = None


def local_flow(ga, gb, cx, cy, R=130):
    """DIS flow on a crop around (cx, cy); returns (flow, x0, y0)."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    x0 = int(min(max(cx - R, 0), W - 2 * R))
    y0 = int(min(max(cy - R, 0), H - 2 * R))
    fl = _dis.calc(np.ascontiguousarray(ga[y0:y0 + 2 * R, x0:x0 + 2 * R]),
                   np.ascontiguousarray(gb[y0:y0 + 2 * R, x0:x0 + 2 * R]), None)
    return fl, x0, y0


def sample(fl, x, y, x0, y0):
    xi, yi = int(round(x - x0)), int(round(y - y0))
    h, w = fl.shape[:2]
    xi, yi = min(max(xi, 0), w - 1), min(max(yi, 0), h - 1)
    r = 2
    p = fl[max(yi - r, 0):yi + r + 1, max(xi - r, 0):xi + r + 1].reshape(-1, 2)
    return float(np.median(p[:, 0])), float(np.median(p[:, 1]))


def track(seeds, k, frames, region, lum_tol=38):
    """seeds {cls: [(x, y)]} at frame k -> {frame: {cls: [(x, y)]}}; a seed that leaves `region` or changes
    luminance by more than lum_tol grey levels is dropped."""
    grays = {}
    G = lambda f: grays.setdefault(f, gray(f, frames))
    out = {k: {c: list(p) for c, p in seeds.items()}}
    g0 = {c: [float(G(k)[int(y), int(x)]) for x, y in p] for c, p in seeds.items()}
    for step, rng in ((1, range(k, N - 1)), (-1, range(k, 0, -1))):
        cur = {c: [(x, y, i) for i, (x, y) in enumerate(p)] for c, p in seeds.items()}
        for f in rng:
            nf = f + step
            allp = [(x, y) for pts in cur.values() for x, y, _ in pts]
            if not allp:
                out[nf] = {c: [] for c in seeds}
                continue
            cx, cy = np.mean([p[0] for p in allp]), np.mean([p[1] for p in allp])
            fl, x0, y0 = local_flow(G(f), G(nf), cx, cy)
            nxt = {}
            for c, pts in cur.items():
                keep = []
                for x, y, i in pts:
                    dx, dy = sample(fl, x, y, x0, y0)
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < W - 1 and 0 <= ny < H - 1):
                        continue
                    if not region[nf][int(ny), int(nx)]:
                        continue
                    if abs(float(G(nf)[int(ny), int(nx)]) - g0[c][i]) > lum_tol:
                        continue
                    keep.append((nx, ny, i))
                nxt[c] = keep
            cur = nxt
            out[nf] = {c: [(x, y) for x, y, _ in pts] for c, pts in cur.items()}
    return out


def rw_segment(frame, region, seeds, beta=90.0, pad=6, seed_masks=None):
    """Random-walker allocation of `region` to the seed classes on the source colour (Lab), on a crop."""
    from skimage.segmentation import random_walker
    ys, xs = np.nonzero(region)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    reg = region[y0:y1, x0:x1]
    lab = np.zeros(reg.shape, np.int32)
    lab[~reg] = -1
    for c, pts in seeds.items():
        for x, y in pts:
            xi, yi = int(round(x)) - x0, int(round(y)) - y0
            if 0 <= xi < lab.shape[1] and 0 <= yi < lab.shape[0] and reg[yi, xi]:
                cv2.circle(lab, (xi, yi), 1, LAB[c], -1)
    if seed_masks:
        for c, m in seed_masks.items():
            lab[(m[y0:y1, x0:x1]) & reg] = LAB[c]
    lab[~reg] = -1
    used = [v for v in np.unique(lab) if v > 0]
    out = np.zeros((H, W), np.int32)
    if len(used) == 0:
        return out
    if len(used) == 1:
        out[y0:y1, x0:x1][reg] = used[0]
        return out
    im = cv2.cvtColor(cv2.GaussianBlur(frame, (0, 0), 0.8), cv2.COLOR_BGR2LAB).astype(np.float32)[y0:y1, x0:x1]
    im = im / np.array([8.0, 3.0, 3.0], np.float32)             # luminance dominates the edge weights
    res = random_walker(im, lab, beta=beta, mode="cg_j", tol=1e-3, channel_axis=-1)
    res = np.where(res > 0, np.array([0] + sorted(used))[np.clip(res, 0, len(used))], 0)   # skimage returns 1..K
    res = np.where(lab == -1, 0, res)
    out[y0:y1, x0:x1] = res
    return out


FACE_WINDOWS = [(44, 50), (74, 89)]            # frames where an ear / face part is actually visible


def seeds_for(tr, f, keys=None):
    """Tracked seeds for frame f: the nearest key frame plus the nearest one on the other side (<= 8 frames away)."""
    keys = sorted(tr) if keys is None else keys
    ks = sorted(keys, key=lambda k: abs(k - f))
    out = {}
    k1 = ks[0]
    for c, p in tr[k1][f].items():
        out.setdefault(c, []).extend(p)
    for k2 in ks[1:]:
        if (k2 - f) * (k1 - f) < 0 and abs(k2 - f) <= 14:
            for c, p in tr[k2][f].items():
                out.setdefault(c, []).extend(p)
            break
    if not any(a <= f <= b for a, b in FACE_WINDOWS):
        out.pop("FACE", None)
    return out


def head_labels(frame, region, seeds, f, blob=None, min_face=20, seed_masks=None):
    """-> dict MASK/FACE/HAIR boolean masks for one frame."""
    lab = rw_segment(frame, region, seeds, seed_masks=seed_masks)
    out = {c: lab == LAB[c] for c in ("MASK", "FACE", "HAIR")}
    if blob is not None:                                       # hair / mask stay inside the SAM head blob
        bd = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        out["HAIR"] &= bd
        out["MASK"] &= bd
    for c in out:                                              # drop specks
        num, cc, st, _ = cv2.connectedComponentsWithStats(out[c].astype(np.uint8), 8)
        keep = np.zeros_like(out[c])
        for k in range(1, num):
            if st[k, cv2.CC_STAT_AREA] >= (min_face if c == "FACE" else 25):
                keep |= cc == k
        out[c] = keep
    return out


MASK_FRAC = 0.65                                 # share of the lone head blob that is the outer band (strap / frame)


def rim_split(blob, region):
    """Head behind the arm (f10-f73): the dark blob = hair core + an outer band (mask strap / frame / snorkel keeper).
    * two or more big blobs (the arm hides the middle of the head): the left blob (with the snorkel tip) is mask
      strap / keeper, the far side of the head (right of the arm) is hair;
    * one big blob: the band is the outer MASK_FRAC of the blob by distance to the water (a fixed share, so the two
      classes change smoothly in time); the core next to the body is hair.
    Small pieces (the snorkel tip) belong to the mask."""
    d = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    num, cc, st, cen = cv2.connectedComponentsWithStats(blob.astype(np.uint8), 8)
    comps = [k for k in range(1, num) if st[k, cv2.CC_STAT_AREA] >= 12]
    big = [k for k in comps if st[k, cv2.CC_STAT_AREA] >= 300]
    mask = np.zeros_like(blob)
    hair = np.zeros_like(blob)
    if not comps:
        return mask, hair
    if len(big) >= 2:
        left = min(big, key=lambda k: cen[k][0])
        for k in comps:
            if k == left or k not in big:
                mask |= cc == k
            else:
                hair |= cc == k
        return mask, hair
    main = max(comps, key=lambda k: st[k, cv2.CC_STAT_AREA])
    for k in comps:
        m = cc == k
        if k != main:
            mask |= m
            continue
        r = np.percentile(d[m], 100 * MASK_FRAC)
        band = m & (d <= r)
        mask |= band
        hair |= m & ~band
    return mask, hair
