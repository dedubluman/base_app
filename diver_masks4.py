#!/usr/bin/env python3
"""Mask corrections on top of the v3 class masks (preview coordinates -> source x1.9935, y1.9963).

B  Wrist watch = GEAR class (never SKIN): dark strap blob grown from the reported spots
   (f2 (496,838), f33 (885,918), f45 (885,930)) and tracked frame to frame.
C  The hand beyond the watch (f0 (528,878) -> f14 (678,878)): SAM positive points, forward + backward
   (hand_sam.py); the part outside the person mask joins the diver (person_x) and SKIN.
D  f32-f45 hand (887,968) and limb tip f45 (929,1016) are SKIN; class islands < 30 preview px
   (= 120 source px) go to the neighbouring class.
E  Foot pocket (source x764-817, y729-783), f0-f8: by the source, dark = paddle (FINS), bright = SKIN,
   the same rule every frame.
SKIN area may change < 5 % between frames: the other classes are smoothed over time (signed-distance).

    python diver_masks4.py --input inputs/diver.mov
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

from diver_classes import dil, luma01, read_mask
from diver_fix import centroid, fill_holes, sdf

REPO = Path(__file__).resolve().parent
N, W, H = 90, 1232, 1078
PRIORITY = ["MASK", "WATCH", "FINS", "SHORTS", "HAIR", "SKIN"]
ANCHORS = {2: (496, 838), 33: (885, 918), 45: (885, 930)}          # the watch, source px
POCKET = (764, 729, 817, 783)                                       # x0, y0, x1, y1, source px
ISLAND_MIN = 120                                                    # 30 preview px^2 = ~120 source px^2
STRIP = int(round(0.06 * 1078))


def comps(m, lo=0):
    num, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return [(k, int(st[k, cv2.CC_STAT_AREA]), cen[k], lab == k) for k in range(1, num) if st[k, cv2.CC_STAT_AREA] >= lo]


def watch_blob(g, region, near, thr=0.24, lo=60, hi=700, r=50):
    cand = (g < thr) & region
    best, bd = None, 1e9
    for k, a, c, m in comps(cand, lo):
        if a > hi:
            continue
        d = float(np.hypot(c[0] - near[0], c[1] - near[1]))
        if d < r and d < bd:
            best, bd = m, d
    return best


def track_watch(imgs, region, other, anchor_f, anchor_pt, direction):
    """Dark strap blob grown at the anchor, then followed frame by frame."""
    out, cur, c = {}, None, anchor_pt
    for f in (range(anchor_f, N) if direction > 0 else range(anchor_f, -1, -1)):
        g = luma01(imgs[f])
        reg = region[f] & ~dil(other[f], 4)
        m = None
        for thr in (0.22, 0.26, 0.30):
            m = watch_blob(g, reg, c, thr=thr, r=16 if f == anchor_f else 55)
            if m is not None:
                break
        if m is None and cur is not None:                      # lost for a frame: hold the strap's dark pixels near the last place
            m = dil(cur, 6) & reg & (g < 0.30)
            m = m if 40 <= m.sum() <= 800 else None
        if m is not None:
            cur, c = m, tuple(centroid(m))
        out[f] = m
    return out


def merge_islands(cls, person_x, f):
    """Class islands smaller than ISLAND_MIN go to the neighbouring class (not GEAR's watch)."""
    lab = np.zeros((H, W), np.uint8)
    for i, c in enumerate(PRIORITY, 1):
        lab[cls[c]] = i
    changed = True
    for _ in range(2):
        for i, c in enumerate(PRIORITY, 1):
            m = lab == i
            for k, a, cen, comp in comps(m):
                if a >= ISLAND_MIN:
                    continue
                if c == "WATCH" and a >= 20:                   # the strap itself may be small
                    continue
                ring = dil(comp, 2) & ~comp
                vals = lab[ring & (lab != i)]
                vals = vals[vals > 0]
                if len(vals):
                    lab[comp] = np.bincount(vals).argmax()
    return {c: lab == i for i, c in enumerate(PRIORITY, 1)}


def temporal_smooth(masks, person_x, weights=(1, 2, 1)):
    out = {}
    wsum = sum(weights)
    for f in range(N):
        if f in (0, N - 1):
            out[f] = masks[f]
            continue
        d = (weights[0] * sdf(masks[f - 1]) + weights[1] * sdf(masks[f]) + weights[2] * sdf(masks[f + 1])) / wsum
        m = (d > 0) & dil(person_x[f], 3)
        out[f] = m if (m.sum() > 0 or masks[f].sum() == 0) else masks[f]
    return out


def temporal_smooth_plain(masks, weights=(1, 2, 1)):
    out = {}
    wsum = sum(weights)
    for f in range(N):
        if f in (0, N - 1):
            out[f] = masks[f]
            continue
        d = (weights[0] * sdf(masks[f - 1]) + weights[1] * sdf(masks[f]) + weights[2] * sdf(masks[f + 1])) / wsum
        m = d > 0
        out[f] = m if m.sum() > 0 else masks[f]
    return out


def sm5_plain(masks):
    out = {}
    for f in range(N):
        idx = [g for g in range(f - 2, f + 3) if 0 <= g < N]
        d = sum(sdf(masks[g]) for g in idx) / len(idx)
        out[f] = (d > 0) if (d > 0).sum() > 0 else masks[f]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    a = ap.parse_args()
    stem = a.input.stem
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    cdir = REPO / "masks" / f"{stem}__cls"
    qc = REPO / "output" / "qc_classes"
    v3 = cache / "cls_v3"
    if not v3.exists():                                          # keep the v3 masks as the input of this stage
        shutil.copytree(cdir, v3, ignore=shutil.ignore_patterns("*.pkl"))
    log = lambda s: print(s, flush=True)
    person = {f: read_mask(REPO / "masks" / f"{stem}__fwd" / "subject", f) for f in range(N)}
    imgs = {f: cv2.imread(str(frames / f"{f:05d}.jpg")) for f in range(N)}
    v3m = lambda c, f: read_mask(v3 / c, f)
    shorts = {f: v3m("SHORTS", f) for f in range(N)}
    fins = {f: v3m("FINS", f) for f in range(N)}
    mask = {f: v3m("MASK", f) for f in range(N)}
    hair = {f: v3m("HAIR", f) for f in range(N)}
    z = np.zeros((H, W), bool)

    # ---- C: the hand (SAM points) joins the diver ------------------------------------------------------------
    hp = cdir / "hand_sam.pkl"
    hand = pickle.load(open(hp, "rb")) if hp.exists() else {}
    person_x = {}
    for f in range(N):
        px = person[f].copy()
        hm = hand.get(f)
        if hm is not None and f <= 14 and 300 <= hm.sum() <= 20000:
            cx_ = int(528 + (678 - 528) * f / 14)               # the hand's path between the two clicks
            disk = np.zeros((H, W), np.uint8)
            cv2.circle(disk, (cx_, 878), 120, 1, -1)
            extra = hm & ~px & (disk > 0)
            ok = z.copy()                                       # only pieces touching the diver (the wrist side)
            for k, ar, cen, comp in comps(extra, 40):
                if (dil(comp, 10) & px).any() or f == 14:
                    ok |= comp
            ok = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            px = px | ok
        person_x[f] = fill_holes(px, 2500)                      # holes in the diver (cyan blotches) are skin
    # fingertips / limb tips the person mask misses: skin has a clearly higher red share than the water
    from diver_classes import hand_tips
    ext_log = {}
    for f in range(N):
        img = imgs[f].astype(np.float32) / 255
        rgb = img[..., ::-1]
        r = rgb[..., 0] / np.maximum(rgb.sum(-1), 1e-3)
        px = person_x[f]
        avoid = fins[f] | shorts[f] | mask[f] | hair[f]
        d = cv2.distanceTransform((~px).astype(np.uint8), cv2.DIST_L2, 3)
        ring = (d >= 15) & (d <= 40)
        wm = np.median(r[ring]) if ring.sum() > 200 else np.median(r[~px])
        zones = [(x_, y_, 100, 0.045) for x_, y_ in hand_tips(px, avoid)]
        if f <= 14:
            zones.append((int(528 + (678 - 528) * f / 14), 878, 90, 0.03))
        if 32 <= f <= 45:
            zones.append((887, 968, 60, 0.02))
        if f == 45:
            zones.append((929, 1016, 45, 0.02))
        ok = z.copy()
        for x_, y_, rr, mg in zones:
            disk = np.zeros((H, W), np.uint8)
            cv2.circle(disk, (x_, y_), rr, 1, -1)
            cand = dil(px, 7) & ~px & (disk > 0) & (r > wm + mg)
            for k, ar, cen, comp in comps(cand, 12):
                if (dil(comp, 2) & px).any():
                    ok |= comp
        ok = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        ok &= ~px
        if ok.sum() > 1500:                                      # safety: never grow the diver by a blob
            ok = z.copy()
        ext_log[f] = int(ok.sum())
        person_x[f] = px | ok
    log(f"[v4] extremity extension (red share): px added per frame, f0-14 {[ext_log[f] for f in range(15)]}, "
        f"f32-45 {[ext_log[f] for f in range(32, 46)]}, total frames touched {sum(1 for v in ext_log.values() if v)}")
    added = {f: int((person_x[f] & ~person[f]).sum()) for f in range(N) if (person_x[f] & ~person[f]).any()}
    log(f"[v4] diver mask extended (hand pieces + filled holes) on {len(added)} frames; px added at f0-14: "
        f"{[added.get(f, 0) for f in range(15)]}")

    # ---- B/D: the watch is its own class (GEAR) ----------------------------------------------------------------
    region = {f: dil(person_x[f], 6) for f in range(N)}
    other = {f: hair[f] | fins[f] | mask[f] | shorts[f] for f in range(N)}
    tracks = {af: (track_watch(imgs, region, other, af, pt, +1), track_watch(imgs, region, other, af, pt, -1))
              for af, pt in ANCHORS.items()}
    afs = sorted(ANCHORS)
    watch = {}
    for f in range(N):
        af = min(afs, key=lambda k: abs(k - f))
        fw, bw = tracks[af]
        m = (fw if f >= af else bw).get(f)
        if m is None:                                           # nearest anchor lost it: try the others
            for k in afs:
                m = (tracks[k][0] if f >= k else tracks[k][1]).get(f)
                if m is not None:
                    break
        watch[f] = m if m is not None else z.copy()
    v3w = {f: read_mask(v3 / "WATCH", f) for f in range(N)}
    for f in range(N):
        if watch[f].sum() < 40:
            if 60 <= v3w[f].sum() <= 700:
                watch[f] = v3w[f]
            elif f > 0 and watch[f - 1].sum() >= 40:           # hold: the dark strap pixels around the last place
                g = luma01(imgs[f])
                h = dil(watch[f - 1], 8) & dil(person_x[f], 6) & (g < 0.28)
                watch[f] = h if 40 <= h.sum() <= 800 else z.copy()
    for f in range(N):                                          # a frame without strap takes the neighbour's
        if watch[f].sum() < 40:
            for g in (f - 1, f + 1):
                if 0 <= g < N and watch[g].sum() >= 40:
                    watch[f] = watch[g].copy()
                    break
    log(f"[v4] GEAR (watch) found on {sum(1 for f in range(N) if watch[f].sum() >= 40)}/{N} frames; "
        f"area f2 {int(watch[2].sum())} f33 {int(watch[33].sum())} f45 {int(watch[45].sum())}")

    # ---- E: foot pocket, one rule for every frame --------------------------------------------------------------
    x0, y0, x1, y1 = POCKET
    pocket = np.zeros((H, W), bool)
    pocket[y0:y1, x0:x1] = True
    for f in range(0, 9):
        g = luma01(imgs[f])
        R = pocket & dil(person_x[f], 3)
        fins[f] = (fins[f] & ~R) | (R & (g < 0.34))
        # bright pixels in the pocket are bare skin: they leave FINS (SKIN is the remainder below)

    # ---- classes, priority, islands, skin area stability --------------------------------------------------------
    def assemble(sh, fn, mk, hr, wt):
        out = {}
        for f in range(N):
            Pd = dil(person_x[f], 3)
            m = {"MASK": mk[f] & Pd, "WATCH": wt[f] & Pd, "FINS": fn[f], "SHORTS": sh[f] & Pd,
                 "HAIR": hr[f] & Pd}
            if 40 <= f <= 52:                                   # top strip stays excluded
                for c in m:
                    m[c][:STRIP] = False
            taken = z.copy()
            for c in PRIORITY[:-1]:
                m[c] = m[c] & ~taken
                taken |= m[c]
            oth = (m["MASK"] | m["FINS"] | m["SHORTS"] | m["HAIR"]) | dil(m["WATCH"], 2)
            skin = person_x[f] & ~oth & ~dil(m["WATCH"], 2)
            if 40 <= f <= 52:
                skin[:STRIP] = False
            m["SKIN"] = skin
            # D: the hand at f32-f45 and the limb tip at f45 are skin whatever else was there
            for ff, (px_, py_), rr in ((range(32, 46), (887, 968), 42), ((45,), (929, 1016), 34)):
                if f in ff:
                    disk = np.zeros((H, W), np.uint8)
                    cv2.circle(disk, (px_, py_), rr, 1, -1)
                    force = (disk > 0) & dil(person_x[f], 6) & ~(m["MASK"] | m["FINS"] | m["SHORTS"] | m["HAIR"])
                    m["SKIN"] = m["SKIN"] | (force & ~dil(m["WATCH"], 2))
                    m["WATCH"] = m["WATCH"] & ~(force & ~m["WATCH"])
            out[f] = m
        return out

    # SHORTS has two window-boundary jumps (f9->f10, f15->f16) that are tracker artefacts, not motion:
    # a light signed-distance smoothing only there (frames where it jumps by > 15 % of the shorts area)
    sa = [int(shorts[f].sum()) for f in range(N)]
    jump = [f for f in range(1, N) if abs(sa[f] - sa[f - 1]) > 0.15 * max(sa[f - 1], 1500) and f < 40]
    log(f"[v4] SHORTS area jumps > 15 % (artefacts, f < 40): {jump}")
    for _ in range(2):
        sm = temporal_smooth(shorts, person_x, (1, 2, 1))
        for f in jump:
            for g in (f - 2, f - 1, f, f + 1):
                if 0 < g < N - 1:
                    shorts[g] = sm[g]
    cls = assemble(shorts, fins, mask, hair, watch)
    # islands (D), then a last exclusivity pass
    for f in range(N):
        cls[f] = merge_islands(cls[f], person_x[f], f)
        taken = z.copy()
        for c in PRIORITY:
            cls[f][c] = cls[f][c] & ~taken
            taken |= cls[f][c]
    for f in range(N):                                          # second island pass (the first exclusivity pass can split classes again)
        cls[f] = merge_islands(cls[f], person_x[f], f)
        taken = z.copy()
        for c in PRIORITY:
            cls[f][c] = cls[f][c] & ~taken
            taken |= cls[f][c]
    sk = [int(cls[f]["SKIN"].sum()) for f in range(N)]
    steps = [(f, 100 * (sk[f] - sk[f - 1]) / max(sk[f - 1], 1)) for f in range(1, N)]
    over = [(f, round(s, 1)) for f, s in steps if abs(s) >= 5]
    log(f"[v4] SKIN area steps >= 5 %: {over}; max {max(abs(s) for _, s in steps):.2f} %")

    # ---- write -----------------------------------------------------------------------------------------------------------
    for c in PRIORITY + ["PERSON"]:
        (cdir / c).mkdir(parents=True, exist_ok=True)
    e3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for f in range(N):
        ys = np.nonzero(person_x[f])[0]
        top = int(ys.min()) if len(ys) else 0
        al = {}
        for c in PRIORITY:
            b = cls[f][c]
            if c == "SKIN":                                     # 1 px in, 2 px soft
                b = cv2.erode(b.astype(np.uint8), e3) > 0
            a_ = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1.0)
            if 56 <= f <= 64:                                   # water line: 4 px feather
                a2 = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 2.0)
                wgt = np.zeros(a_.shape, np.float32)
                wgt[top:top + 60, :] = 1
                a_ = a_ * (1 - wgt) + a2 * wgt
            al[c] = a_
        s_ = np.maximum(sum(al.values()), 1.0)
        for c in PRIORITY:
            cv2.imwrite(str(cdir / c / f"{f:05d}.png"), (np.clip(al[c] / s_, 0, 1) * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(str(cdir / "PERSON" / f"{f:05d}.png"), person_x[f].astype(np.uint8) * 255)
    areas = {c: [int(cls[f][c].sum()) for f in range(N)] for c in PRIORITY}
    overlap = sum(int((sum(cls[f][c].astype(np.uint8) for c in PRIORITY) > 1).sum()) for f in range(N))
    lines = ["frame,MASK,WATCH,FINS,SHORTS,HAIR,SKIN"]
    for f in range(N):
        lines.append(f"{f}," + ",".join(str(areas[c][f]) for c in PRIORITY))
    (qc / "class_areas.csv").write_text("\n".join(lines) + "\n")
    log(f"[v4] overlap pixels {overlap}; GEAR zero frames {[f for f in range(N) if areas['WATCH'][f] == 0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
