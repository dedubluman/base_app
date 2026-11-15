#!/usr/bin/env python3
"""Mask round 5 (on top of the v4 class masks in .cache/cls_v4 and the head split in .cache/head_labels).

classes (priority high -> low): MASK, GEAR, FINS, SHORTS, HAIR, FACE, SKIN
  MASK   diving-mask glass + frame + strap (+ the snorkel keeper / tip clipped to the strap)   <- head_split.py
  FACE   forehead, cheeks, ear, jaw: only the visible part                                     <- head_split.py
  HAIR   all hair                                                                              <- head_split.py
  GEAR   the wrist watch: anchored on the real strap (one wrist), nothing else                  <- ANCHORS
  FINS   paddles: only while they are in the water (f0-f32); out of frame / above the surface after
  SHORTS fabric only: foam / surface reflection removed, edge snapped to the fabric folds
  SKIN   the rest of the body; water between the fingers removed by the red-share matte; ghost of the leg
         above the surface (f45) removed.
all masks: 1 px erosion + 2.5 px soft edge, one soft alpha for the whole diver (no halo, no gaps).

    python diver_masks5.py --input inputs/diver.mov
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from diver_classes import dil, hand_tips, luma01
from diver_fix import centroid, fill_holes, sdf
from head_split import local_flow

REPO = Path(__file__).resolve().parent
N, W, H = 90, 1232, 1078
V4 = REPO / ".cache" / "cls_v4"
HL = REPO / ".cache" / "head_labels"
FR = REPO / ".cache" / "sam_frames_diver_bwd"
ORDER = ["MASK", "GEAR", "FINS", "SHORTS", "HAIR", "FACE", "SKIN"]
FIN_LAST = 32                      # paddles are out of the water / frame after this frame

# the strap of the watch, read on 3x crops of the source (one wrist: the left-in-image arm), source px
ANCHORS = {41: (900, 912), 42: (894, 909), 45: (885, 930), 48: (862, 935), 51: (864, 927), 54: (858, 912), 63: (743, 812), 64: (743, 810), 66: (724, 793), 70: (691, 726), 72: (700, 679), 74: (719, 623),
           78: (797, 572), 80: (857, 556), 82: (893, 566), 84: (905, 563), 86: (905, 553), 89: (900, 533)}
GEAR_V4_LAST = 40                  # v4 (anchored at f2 / f33) is right up to here; f43-f54 re-anchored on the strap
GEAR_HIDDEN = list(range(55, 63)) + [75, 76, 77]   # hand points down / the other hand covers the strap


def rd(root, c, f):
    m = cv2.imread(str(root / c / f"{f:05d}.png"), 0)
    return np.zeros((H, W), bool) if m is None else m > 127


def comps(m, lo=0):
    num, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return [(k, int(st[k, cv2.CC_STAT_AREA]), cen[k], lab == k) for k in range(1, num) if st[k, cv2.CC_STAT_AREA] >= lo]


def smooth_bin(m, sigma=1.4):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) > 0.5


# ---------------------------------------------------------------------------------------------------- GEAR
def gear_masks(imgs, person, skin_v4, v4_watch, head):
    """Watch strap per frame. f0-f56: v4 (anchored at f2/f33/f45, checked); f57-f62 hidden; f63-f89: the strap blob
    nearest to the interpolated anchor path (dark, skin on both sides)."""
    out = {f: np.zeros((H, W), bool) for f in range(N)}
    for f in range(GEAR_V4_LAST + 1):
        if v4_watch[f].sum() >= 40:
            out[f] = v4_watch[f]
    for f in range(GEAR_V4_LAST + 1):                          # short gaps: hold the neighbour
        if out[f].sum() < 40:
            for g in (f - 1, f + 1, f - 2, f + 2, f - 3, f + 3):
                if 0 <= g <= GEAR_V4_LAST and out[g].sum() >= 40:
                    out[f] = out[g].copy()
                    break
    afs = sorted(ANCHORS)
    last = None
    for f in list(range(41, 55)) + list(range(63, N)):
        if f in GEAR_HIDDEN:
            continue
        if f in ANCHORS:
            p = ANCHORS[f]
        else:
            lo = max([a for a in afs if a < f], default=None)
            hi = min([a for a in afs if a > f], default=None)
            if lo is None or hi is None:
                p = ANCHORS[hi if lo is None else lo]
            else:
                t = (f - lo) / (hi - lo)
                p = (ANCHORS[lo][0] * (1 - t) + ANCHORS[hi][0] * t, ANCHORS[lo][1] * (1 - t) + ANCHORS[hi][1] * t)
        g = luma01(cv2.GaussianBlur(imgs[f], (0, 0), 0.8))
        skin_med = float(np.median(g[cv2.erode(skin_v4[f].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0])) \
            if skin_v4[f].sum() > 500 else 0.45
        reg = np.zeros((H, W), np.uint8)
        cv2.circle(reg, (int(p[0]), int(p[1])), 19 if f >= 76 else 42, 1, -1)
        reg = (reg > 0) & dil(person[f], 5)
        best = None
        if f >= 76:                                              # strap partly hidden / next to shadow: oriented ellipse
            ys, xs = np.nonzero(skin_v4[f] & (cv2.circle(np.zeros((H, W), np.uint8), (int(p[0]), int(p[1])), 70, 1, -1) > 0))
            ang = 0.0
            if len(xs) > 200:
                pts = np.stack([xs, ys], 1).astype(np.float32)
                _, ev = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
                ax = ev[:, -1]                                    # forearm axis; the strap is perpendicular to it
                ang = float(np.degrees(np.arctan2(ax[1], ax[0]))) + 90.0
            el = np.zeros((H, W), np.uint8)
            cv2.ellipse(el, (int(p[0]), int(p[1])), (23, 10), ang, 0, 360, 1, -1)
            m = (el > 0) & dil(person[f], 3) & (g < 0.42) & ~dil(head[f], 3)
            if m.sum() >= 120:
                best = (0.0, m)
        else:
            for thr in (min(0.25, 0.55 * skin_med), min(0.28, 0.62 * skin_med), 0.31, 0.34, 0.37):
                cand = (g < thr) & reg & ~dil(head[f], 5)
                cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
                for k, a_, cen, m in comps(cand, 60):
                    if a_ > 1100:
                        continue
                    d = float(np.hypot(cen[0] - p[0], cen[1] - p[1]))
                    if best is None or d < best[0]:
                        best = (d, m)
                if best is not None:
                    break
        if best is not None:
            out[f] = best[1]
            last = (f, best[1])
        elif last is not None and f - last[0] <= 3:            # lost for a moment: hold, shifted to the path
            c0 = centroid(last[1])
            dx, dy = int(round(p[0] - c0[0])), int(round(p[1] - c0[1]))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out[f] = cv2.warpAffine(last[1].astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    return out


# ---------------------------------------------------------------------------------------------------- SHORTS
def foam_zone(img, person, top_rows=45):
    """Bright, textured surface foam / reflection that touches the top of the diver."""
    g = luma01(img)
    gs = cv2.GaussianBlur(g, (0, 0), 1.0)
    mu = cv2.blur(gs, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(gs * gs, (7, 7)) - mu * mu, 0))
    foam = (sd > 0.035) & (gs > 0.50)
    foam = cv2.morphologyEx(foam.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(person)
    if len(ys) == 0:
        return np.zeros((H, W), bool)
    top = ys.min()
    band = np.zeros((H, W), bool)
    band[max(0, top - 80):top + top_rows, :] = True
    foam &= band
    out = np.zeros((H, W), bool)
    for k, a, cen, m in comps(foam, 150):
        if cen[1] < top + 0.8 * top_rows:
            out |= m
    return dil(out, 2)


def snap_shorts(img, shorts, person, skin_like):
    """Edge of the shorts follows the fabric: inside a +-6 px band around the boundary a pixel is fabric when it is
    brighter than the midpoint between the fabric core and the skin next to it (folds / hems, not a polygon)."""
    if shorts.sum() < 300:
        return shorts
    g = cv2.GaussianBlur(luma01(img), (0, 0), 1.0)
    core = cv2.erode(shorts.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = dil(shorts, 14) & ~dil(shorts, 6) & skin_like
    if core.sum() < 100 or ring.sum() < 100:
        return shorts
    lf, ls = float(np.median(g[core])), float(np.median(g[ring]))
    if lf - ls < 0.06:                                          # no luminance contrast: keep SAM's edge
        return shorts
    t = 0.5 * (lf + ls)
    band = dil(shorts, 4) & ~(cv2.erode(shorts.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & person
    new = (shorts & ~band) | (band & (g > t))
    new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    new = fill_holes(new, 400)
    # keep the main piece(s)
    out = np.zeros_like(new)
    for k, a, cen, m in comps(new, 80):
        out |= m
    return smooth_bin(out, 1.6) if out.sum() else shorts


# ---------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    a = ap.parse_args()
    stem = a.input.stem
    out_dir = REPO / "masks" / f"{stem}__cls5"
    binr = REPO / ".cache" / "cls5_bin"
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    imgs = {f: cv2.imread(str(FR / f"{f:05d}.jpg")) for f in range(N)}
    z = np.zeros((H, W), bool)

    v4 = {c: {f: rd(V4, c, f) for f in range(N)} for c in ("SKIN", "SHORTS", "FINS", "HAIR", "MASK", "WATCH", "PERSON")}
    hd = {c: {f: rd(HL, c, f) for f in range(N)} for c in ("MASK", "FACE", "HAIR")}
    person = {f: v4["PERSON"][f] | v4["SKIN"][f] | v4["SHORTS"][f] | v4["FINS"][f] | v4["HAIR"][f] | v4["MASK"][f]
              | v4["WATCH"][f] for f in range(N)}
    head = {f: hd["MASK"][f] | hd["FACE"][f] | hd["HAIR"][f] for f in range(N)}

    # ---- head: new MASK / FACE / HAIR; leftovers of the old head blob go to the nearest new head class ------------
    mask, face, hair = {}, {}, {}
    for f in range(N):
        mk, fc, hr = hd["MASK"][f].copy(), hd["FACE"][f].copy(), hd["HAIR"][f].copy()
        left = (v4["HAIR"][f] | v4["MASK"][f]) & ~(mk | fc | hr)
        if left.any() and (mk | hr).any():
            lab = np.zeros((H, W), np.uint8)
            lab[mk] = 1
            lab[hr] = 2
            _, idx = cv2.distanceTransformWithLabels((lab == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            # nearest labelled pixel's class: via a lookup of the closest label
            ys, xs = np.nonzero(lab > 0)
            tree_lab = lab[ys, xs]
            from scipy.spatial import cKDTree
            tr = cKDTree(np.stack([xs, ys], 1))
            ly, lx = np.nonzero(left)
            _, nn = tr.query(np.stack([lx, ly], 1))
            cl = tree_lab[nn]
            mk[ly[cl == 1], lx[cl == 1]] = True
            hr[ly[cl == 2], lx[cl == 2]] = True
        mask[f], face[f], hair[f] = mk, fc, hr
    head = {f: mask[f] | face[f] | hair[f] for f in range(N)}

    # ---- GEAR -----------------------------------------------------------------------------------------------------
    gear = gear_masks(imgs, person, v4["SKIN"], v4["WATCH"], head)
    log(f"[v5] GEAR frames with a strap: {sum(1 for f in range(N) if gear[f].sum() >= 40)}/{N}; hidden {GEAR_HIDDEN}")

    # ---- FINS: only while in the water -------------------------------------------------------------------------------
    fins = {}
    for f in range(N):
        if f > FIN_LAST:
            fins[f] = z.copy()
            continue
        m = cv2.morphologyEx(v4["FINS"][f].astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        fins[f] = smooth_bin(m, 3.0 if f <= 8 else 2.0) if m.sum() else m

    # ---- SKIN / SHORTS / person edge cleanup ---------------------------------------------------------------------------
    shorts, personx = {}, {}
    foam_log = {}
    for f in range(N):
        px = person[f] | head[f]
        px = fill_holes(px, 2500)
        ghost = z.copy()
        if f == 45:                                              # skin piece above the shorts = the leg above the surface
            ghost[465:556, 895:1012] = True
            ghost &= v4["SKIN"][f]
        sh = v4["SHORTS"][f].copy()
        if f <= 40:
            fz = foam_zone(imgs[f], px)
            foam_log[f] = int(fz.sum())
            sh &= ~fz
        sk_like = v4["SKIN"][f] & ~ghost
        sh = snap_shorts(imgs[f], sh, px, sk_like)
        sh = sh & ~mask[f] & ~hair[f] & ~face[f]
        shorts[f] = sh
        personx[f] = px & ~ghost
    log(f"[v5] foam removed from SHORTS (px, f<=40): {dict(list((f, foam_log[f]) for f in (0, 2, 6, 10, 14, 20, 30, 40) if f in foam_log))}")

    # ---- label map (priority) ----------------------------------------------------------------------------------------
    lab = {}
    for f in range(N):
        L = np.zeros((H, W), np.uint8)                           # 0 water; index into ORDER + 1
        classes = {"MASK": mask[f], "GEAR": gear[f], "FINS": fins[f], "SHORTS": shorts[f], "HAIR": hair[f], "FACE": face[f]}
        pxd = dil(personx[f], 2)
        for i in range(len(ORDER) - 1, -1, -1):                  # low priority first, high priority overwrites
            c = ORDER[i]
            if c == "SKIN":
                m = personx[f] & ~(mask[f] | gear[f] | fins[f] | shorts[f] | hair[f] | face[f])
            else:
                m = classes[c] & pxd
            L[m] = i + 1
        lab[f] = L

    # ---- islands: a class piece < 120 px that is not GEAR / FACE goes to the neighbouring class -----------------------
    for f in range(N):
        L = lab[f]
        for i, c in enumerate(ORDER, 1):
            if c in ("GEAR", "FACE"):
                continue
            for k, ar, cen, m in comps(L == i):
                if ar >= 120:
                    continue
                ring = dil(m, 2) & ~m
                vals = L[ring]
                vals = vals[(vals > 0) & (vals != i)]
                if len(vals):
                    L[m] = np.bincount(vals).argmax()
        lab[f] = L

    # ---- temporal consistency: SHORTS may not vanish / jump by more than 15 % between frames ------------------------------
    from head_run2 import warp_mask
    iS = ORDER.index("SHORTS") + 1

    def series(i):
        return [int((lab[f] == i).sum()) for f in range(N)]

    def flagged(i, lo=300):
        a_ = series(i)
        bad = []
        for f in range(1, N):
            hi_, lo_ = max(a_[f], a_[f - 1]), min(a_[f], a_[f - 1])
            if hi_ >= lo and (hi_ - lo_) / hi_ > 0.15:
                bad.append(f)
        return bad

    log(f"[v5] SHORTS area change > 15 % before consistency pass: {flagged(iS)}")
    for it in range(2):
        bad = flagged(iS)
        for f in bad:
            nb = [g for g in (f - 1, f + 1) if 0 <= g < N and (lab[g] == iS).sum() >= 200]
            if not nb:
                continue
            ys, xs = np.nonzero(personx[f])
            cx, cy = float(xs.mean()), float(ys.mean())
            votes = (lab[f] == iS).astype(np.float32)
            for g in nb:
                votes += warp_mask(imgs, lab[g] == iS, g, f, cx, cy)
            new = votes >= (1 + len(nb)) / 2 + 0.01
            new = smooth_bin(new, 1.2) & personx[f]
            allowed = (lab[f] == iS) | (lab[f] == ORDER.index("SKIN") + 1) | (lab[f] == ORDER.index("FACE") + 1)
            new &= allowed
            lab[f][(lab[f] == iS) & ~new] = ORDER.index("SKIN") + 1
            lab[f][new & (lab[f] == ORDER.index("SKIN") + 1)] = iS
    log(f"[v5] SHORTS area change > 15 % after consistency pass: {flagged(iS)} (real exposure / occlusion changes)")

    # ---- a class present in a frame may not vanish: fill gaps by flow-warping the nearest frames on both sides -----------
    def fill_gaps(c, dark_only=False, min_good=150):
        i = ORDER.index(c) + 1
        a_ = series(i)
        good = [f for f in range(N) if a_[f] >= min_good]
        filled = []
        for f in range(N):
            if a_[f] >= 0.4 * min_good:
                continue
            lo = max([g for g in good if g < f], default=None)
            hi = min([g for g in good if g > f], default=None)
            if lo is None or hi is None or hi - lo > 12:
                continue
            ys, xs = np.nonzero(personx[f])
            cx, cy = float(xs.mean()), float(ys.mean())
            t = (f - lo) / (hi - lo)
            v = (1 - t) * warp_mask(imgs, lab[lo] == i, lo, f, cx, cy).astype(np.float32) \
                + t * warp_mask(imgs, lab[hi] == i, hi, f, cx, cy).astype(np.float32)
            new = smooth_bin(v >= 0.45, 1.0) & personx[f]
            if dark_only:
                new &= luma01(imgs[f]) < 0.30
            new &= (lab[f] == ORDER.index("SKIN") + 1) | (lab[f] == i) | (lab[f] == ORDER.index("FACE") + 1)
            if new.sum() >= 40:
                lab[f][new] = i
                filled.append(f)
        return filled

    log(f"[v5] gap-filled SHORTS frames {fill_gaps('SHORTS')}, HAIR {fill_gaps('HAIR', dark_only=True, min_good=300)}, "
        f"MASK {fill_gaps('MASK', dark_only=True, min_good=300)}")

    # ---- hands + edges: red-share matte (water between fingers, halo), soft alpha ----------------------------------------
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    iSK, iFC = ORDER.index("SKIN") + 1, ORDER.index("FACE") + 1
    for c in ORDER + ["ALPHA"]:
        (out_dir / c).mkdir(parents=True, exist_ok=True)
        (binr / c).mkdir(parents=True, exist_ok=True)
    rep = {}
    for f in range(N):
        L = lab[f]
        union = L > 0
        rgb = imgs[f][..., ::-1].astype(np.float32) / 255
        r = rgb[..., 0] / np.maximum(rgb.sum(-1), 1e-3)
        r = cv2.GaussianBlur(r, (0, 0), 0.8)
        skin_like = (L == iSK) | (L == iFC)
        dout = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, 3)
        din = cv2.distanceTransform(union.astype(np.uint8), cv2.DIST_L2, 3)
        fz = foam_zone(imgs[f], union) if f <= 75 else z
        foam_rm = fz if f <= 24 else z                 # near the surface the back meets the foam: foam is not skin
        ring = (dout >= 8) & (dout <= 26) & ~dil(fz, 6)
        core = cv2.erode(skin_like.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        fac = np.ones((H, W), np.float32)
        dlt = 0.0
        if ring.sum() > 300 and core.sum() > 300:
            r_w, r_s = float(np.median(r[ring])), float(np.median(r[core]))
            dlt = r_s - r_w
            if dlt >= 0.03:
                m = np.clip((r - (r_w + 0.30 * dlt)) / (0.35 * dlt), 0, 1)
                m = m * m * (3 - 2 * m)
                tips = hand_tips(personx[f], (L == ORDER.index("FINS") + 1) | (L == ORDER.index("SHORTS") + 1), n=2)
                zone = np.zeros((H, W), np.uint8)
                for tx, ty in tips:
                    cv2.circle(zone, (tx, ty), 75, 1, -1)
                if 30 <= f <= 50:                                  # the hanging hand
                    cv2.circle(zone, (887, 975), 70, 1, -1)
                # thin structures only (fingers: every pixel is within ~9 px of the outside); forearm / palm shadows are
                # skin whatever their colour (shadowed skin and water have the same red share), only their rim is matted
                zone = (zone > 0) & skin_like & (din <= 9.0)
                band = skin_like & (din <= 2.5) & ~dil(fz, 6)
                use = zone | band
                fac = np.where(use, m, 1.0).astype(np.float32)
        rep[f] = dlt
        keep = (fac > 0.12) & ~(foam_rm & ((L == iSK) | (L == iFC) | (L == ORDER.index('SHORTS') + 1)))
        U = union & keep
        U = cv2.erode(U.astype(np.uint8), k1) > 0                   # 1 px in
        A0 = cv2.GaussianBlur(U.astype(np.float32), (0, 0), 1.15)  # ~2.5 px soft edge
        A = A0 * cv2.GaussianBlur(fac, (0, 0), 1.1)
        A = np.clip(A, 0, 1)
        wts = {}
        for i, c in enumerate(ORDER, 1):
            wts[c] = cv2.GaussianBlur(((L == i) & keep).astype(np.float32), (0, 0), 1.15)
        ssum = np.maximum(sum(wts.values()), 1e-4)
        for c in ORDER:
            cv2.imwrite(str(out_dir / c / f"{f:05d}.png"), (np.clip(wts[c] / ssum, 0, 1) * 255 + 0.5).astype(np.uint8))
            cv2.imwrite(str(binr / c / f"{f:05d}.png"), (((L == ORDER.index(c) + 1) & keep) * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / "ALPHA" / f"{f:05d}.png"), (A * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(str(binr / "ALPHA" / f"{f:05d}.png"), (A * 255 + 0.5).astype(np.uint8))
    pickle.dump(lab, open(REPO / ".cache" / "cls5_lab.pkl", "wb"))
    areas = {c: [int(((lab[f] == i)).sum()) for f in range(N)] for i, c in enumerate(ORDER, 1)}
    json.dump(areas, open(qc / "areas_v5.json", "w"))
    for c in ORDER:
        a_ = areas[c]
        bad = [f for f in range(1, N) if max(a_[f], a_[f - 1]) >= 300 and abs(a_[f] - a_[f - 1]) / max(a_[f], a_[f - 1]) > 0.15]
        log(f"[v5] {c:6s} frames with area change > 15 %: {bad}")
    log(f"[v5] r-share contrast skin-water (per 15 frames): {[round(rep[f], 3) for f in range(0, N, 15)]}")
    rows = ["frame," + ",".join(ORDER)] + [f"{f}," + ",".join(str(areas[c][f]) for c in ORDER) for f in range(N)]
    (qc / "class_areas_v5.csv").write_text("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
