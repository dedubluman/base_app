#!/usr/bin/env python3
"""Part 1 - corrected SAM 3.1 class masks (SKIN, SHORTS, FINS, MASK), 90 frames.

Priority MASK > FINS > SHORTS > SKIN; no pixel is in two classes.

A  SHORTS: f46 mask is the seed, tracked BACKWARD to f0 in short chained
   windows. Positive points on the shorts at f12 and f28. Any frame whose
   SHORTS area is zero is re-prompted from its nearest valid frame.
B  MASK: f46 mask tracked backward to f0, point on the dark head / goggle area
   at f12. Hair is cut out of it, the two f60 parts are joined into one object.
C  FINS: every paddle is its own SAM object (one box-sized dark seed each) over
   f0-f30, merged into one class; 3 px closing; paddle skin leaves SKIN.
D  f40-f52: top 6 % strip removed from every class (fixed exclusion zone).
E  f56-f64: the leg edge at the water line is rounded and feathered 4 px.
F  Wrist watch: never SKIN, never classed.
G  f0: negative points on the foam at the top end of SHORTS.
H  Specks < 150 px removed, holes < 100 px filled.

    python diver_fix.py --input inputs/diver.mov [--reuse]
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

import diver_classes as D
from diver_classes import (CLASSES, N, POST, PRIORITY, Sam, clean, dil, foam_points,
                           grow_dark, hand_tips, interior_point, luma01, read_mask, refine_dark,
                           seed_events, valid_mask)

D.OID.update({"FIN1": 107, "FIN2": 108})
REPO = Path(__file__).resolve().parent
STRIP = int(round(0.06 * 1078))                     # D: top 6 % of the frame
POINTS = {                                           # frame -> object -> [(x, y, label)]
    ("SHORTS", 12): [(735, 680, 1)],
    ("SHORTS", 28): [(945, 600, 1), (1030, 578, 1)],
    ("MASK", 12): [(633, 793, 1)],
}


class SamX(Sam):
    def propagate2(self, sid, cs, start, direction):
        got = {k: {} for k in D.OID}
        inv = {v: k for k, v in D.OID.items()}
        for resp in self.pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": direction, "start_frame_index": start - cs}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                c = inv.get(int(oid))
                if c:
                    got[c][fi] = np.asarray(mm) > 0
        self.pred.handle_request({"type": "close_session", "session_id": sid})
        return got


def run_window(sam, a, b, ev, seed_frame, tag):
    cs, ce = a, b + 1
    sid = sam.start(cs, ce, tag)
    sam.text(sid, cs, (a + b) // 2, "person")       # the session needs one text prompt
    for (f, name), pl in ev.items():
        if cs <= f < ce and any(l == 1 for *_, l in pl):
            sam.points(sid, cs, f, name, [(x, y) for x, y, _ in pl], [l for *_, l in pl])
    d = "forward" if seed_frame == cs else "backward" if seed_frame == ce - 1 else "both"
    return sam.propagate2(sid, cs, seed_frame, d)


def sdf(m):
    m8 = m.astype(np.uint8)
    return cv2.distanceTransform(m8, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)


def ok_mask(name, m, person, img):
    if name in ("FIN1", "FIN2"):
        return 150 <= m.sum() <= 0.25 * max(int(person.sum()), 1)
    return valid_mask(name, m, person, img)


def tidy(name, m, img, hair):
    base = "FINS" if name in ("FIN1", "FIN2") else name
    m = refine_dark(base, m, img)
    if name == "MASK":
        m = m & ~dil(hair, 1)
    return m


def blob_near(img, person, centre, exclude, lo=200, hi=9000, r=160, thr=0.20):
    g = luma01(img)
    cand = (g < thr) & dil(person, 10) & ~dil(exclude, 4)
    num, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    best, bd = None, 1e9
    for k in range(1, num):
        a = st[k, cv2.CC_STAT_AREA]
        d = float(np.hypot(cen[k][0] - centre[0], cen[k][1] - centre[1]))
        if lo <= a <= hi and d < r and d < bd:
            best, bd = k, d
    return (lab == best) if best is not None else np.zeros_like(person)


def centroid(m):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)


def join_parts(m, person):
    """B: several pieces of one object -> one connected object (bridge the gaps)."""
    m = clean(m)
    num, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if num <= 2:
        return m
    out = m.astype(np.uint8).copy()
    ids = sorted(range(1, num), key=lambda k: -st[k, cv2.CC_STAT_AREA])
    main = ids[0]
    for k in ids[1:]:
        ys, xs = np.nonzero(lab == k)
        ym, xm = np.nonzero(lab == main)
        # closest pair of pixels between the two parts
        sub = np.linspace(0, len(xs) - 1, min(len(xs), 200)).astype(int)
        subm = np.linspace(0, len(xm) - 1, min(len(xm), 400)).astype(int)
        d = (xs[sub][:, None] - xm[subm][None]) ** 2 + (ys[sub][:, None] - ym[subm][None]) ** 2
        i, j = np.unravel_index(np.argmin(d), d.shape)
        cv2.line(out, (int(xs[sub][i]), int(ys[sub][i])), (int(xm[subm][j]), int(ym[subm][j])), 1, 5)
    return (out > 0) & dil(person, 6)


def smooth_spikes(masks, person, names=("SHORTS", "MASK"), limit=0.08):
    """A frame whose area leaves +-8 % of its neighbours' mean while both
    neighbours agree is a tracker spike: replace it by the shape halfway
    between its neighbours (signed-distance interpolation)."""
    fixed = []
    for c in names:
        fr = masks[c]
        for f in range(1, N - 1):
            if f not in fr or f - 1 not in fr or f + 1 not in fr:
                continue
            a0, a, a1 = fr[f - 1].sum(), fr[f].sum(), fr[f + 1].sum()
            ref = (a0 + a1) / 2
            if ref > 0 and abs(a - ref) / ref > limit and abs(a0 - a1) / ref < limit * 1.5:
                m = ((sdf(fr[f - 1]) + sdf(fr[f + 1])) / 2 > 0) & dil(person[f], 3)
                fr[f] = m
                fixed.append((c, f))
    return fixed


def find_watch2(img, person, avoid, tips):
    g = luma01(img)
    cand = person & (g < 0.24) & ~dil(avoid, 5)
    num, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    best, bd = None, 1e9
    for k in range(1, num):
        a = st[k, cv2.CC_STAT_AREA]
        if not 25 <= a <= 1400:
            continue
        for t in tips:
            d = float(np.hypot(cen[k][0] - t[0], cen[k][1] - t[1]))
            if d < 230 and d < bd:
                best, bd = k, d
    return (lab == best) if best is not None else np.zeros_like(person)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    stem = a.input.stem
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    cdir = REPO / "masks" / f"{stem}__cls"
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    pdir = REPO / "masks" / f"{stem}__fwd" / "subject"
    person = {f: read_mask(pdir, f) for f in range(N)}
    imgs = {f: cv2.imread(str(frames / f"{f:05d}.jpg")) for f in range(N)}
    log = lambda s: print(s, flush=True)
    if not (cache / "cls_prev").exists():
        shutil.copytree(cdir, cache / "cls_prev", ignore=shutil.ignore_patterns("sam_raw*.pkl"))
    prev = cache / "cls_prev"
    cur = {c: {f: read_mask(prev / c, f) for f in range(N)} for c in CLASSES + ["HAIR", "WATCH"]}
    hair = cur["HAIR"]
    pkl = cdir / "sam_raw_fix.pkl"

    if a.reuse and pkl.exists():
        tracked = pickle.loads(pkl.read_bytes())
    else:
        sam = SamX(frames, cache)
        tracked = {"SHORTS": {}, "MASK": {}, "FINS": {}}
        for f in range(46, N):                       # f46 onward: the clean masks already there
            for c in ("SHORTS", "MASK"):
                if cur[c][f].sum() > 0:
                    tracked[c][f] = cur[c][f]
        # ---- A + B: backward from f46 in chained windows ------------------
        t0 = time.time()
        for (wa, wb) in [(34, 46), (22, 34), (10, 22), (0, 10)]:
            ev = {}
            for c in ("SHORTS", "MASK"):
                src = next((tracked[c][g] for g in range(wb, 47) if g in tracked[c] and tracked[c][g].sum() > 30), None)
                if src is not None:
                    ev[(wb, c)] = seed_events(src)
            for (c, f), pts in POINTS.items():
                if wa <= f < wb:
                    ev.setdefault((f, c), []).extend(pts)
            res = run_window(sam, wa, wb, ev, wb, f"bk{wa}")
            for c in ("SHORTS", "MASK"):
                got = 0
                for f in range(wa, wb):
                    m = res[c].get(f)
                    if m is None:
                        continue
                    m = tidy(c, m, imgs[f], hair[f])
                    if m.sum() > 0 and ok_mask(c, m, person[f], imgs[f]):
                        tracked[c][f] = m
                        got += 1
                log(f"[fix] window {wa}-{wb} {c}: {got}/{wb - wa} frames valid ({time.time() - t0:.0f}s)")
        # ---- A + B: no frame may stay empty -> re-prompt from the nearest valid frame
        for c in ("SHORTS", "MASK"):
            for rnd in range(5):
                miss = [f for f in range(N) if f not in tracked[c]]
                if not miss:
                    break
                f = miss[0]
                v = min(tracked[c], key=lambda g: abs(g - f))
                wa, wb = (v, min(N - 1, v + 10)) if f > v else (max(0, v - 10), v)
                ev = {(v, c): seed_events(tracked[c][v])}
                res = run_window(sam, wa, wb, ev, v, f"gap{c}{rnd}")
                for g in range(wa, wb + 1):
                    m = res[c].get(g)
                    if m is not None and g not in tracked[c]:
                        m = tidy(c, m, imgs[g], hair[g])
                        if m.sum() > 0 and ok_mask(c, m, person[g], imgs[g]):
                            tracked[c][g] = m
                log(f"[fix] re-prompt round {rnd} {c}: still empty {[g for g in range(N) if g not in tracked[c]][:12]}")
        # ---- C: FINS, one SAM object per paddle, f0-f30 --------------------
        f0m = cur["FINS"][0]
        num, lab, st, cen = cv2.connectedComponentsWithStats(f0m.astype(np.uint8), 8)
        parts = sorted([k for k in range(1, num) if st[k, cv2.CC_STAT_AREA] > 800],
                       key=lambda k: cen[k][0])[:2]
        seeds0 = {"FIN1": lab == parts[0], "FIN2": lab == parts[-1]} if len(parts) >= 2 else {}
        log(f"[fix] FINS f0 paddles: {[int((lab == k).sum()) for k in parts]}")
        blobs = {}
        for f, pts in ((12, {"FIN1": (890, 775), "FIN2": (1010, 760)}),
                       (20, {"FIN1": (964, 630), "FIN2": (1108, 640)}),
                       (28, {"FIN1": (1160, 560), "FIN2": (1063, 568)})):
            for nm, p in pts.items():
                b = None
                for lo, hi in ((1500, 12000), (250, 12000)):
                    b = grow_dark(imgs[f], person[f], [p], lo=lo, hi=hi)
                    if b.sum() > 0:
                        break
                blobs[(f, nm)] = b
                log(f"  [box] f{f} {nm} at {p}: {int(b.sum())} px")
        fin_tr = {"FIN1": {}, "FIN2": {}}
        for (wa, wb, sf) in [(0, 15, 0), (15, 30, 15)]:
            ev = {}
            for nm in ("FIN1", "FIN2"):
                src = seeds0.get(nm) if sf == 0 else fin_tr[nm].get(15)
                if src is None or src.sum() < 150:
                    src = blobs.get((12 if sf == 0 else 20, nm))
                if src is not None and src.sum() > 0:
                    ev[(sf, nm)] = seed_events(src)
                for (bf, bn), bm in blobs.items():
                    if bn == nm and wa < bf <= wb and bm.sum() > 0:
                        ev[(bf, nm)] = seed_events(bm)
            res = run_window(sam, wa, wb, ev, sf, f"fin{wa}")
            for nm in ("FIN1", "FIN2"):
                for f in range(wa, wb + 1):
                    m = res[nm].get(f)
                    if m is not None:
                        m = tidy(nm, m, imgs[f], hair[f])
                        if m.sum() > 0 and ok_mask(nm, m, person[f], imgs[f]):
                            fin_tr[nm][f] = m
            log(f"[fix] fins {wa}-{wb}: FIN1 {len(fin_tr['FIN1'])} FIN2 {len(fin_tr['FIN2'])} frames valid")
        for f in range(0, 31):
            m = np.zeros_like(person[f])
            for nm in ("FIN1", "FIN2"):
                if f in fin_tr[nm]:
                    m |= fin_tr[nm][f]
            for nm in ("FIN1", "FIN2"):                 # the user's boxes are authoritative
                if (f, nm) in blobs:
                    m |= blobs[(f, nm)]
            tracked["FINS"][f] = m
        # frames with no paddle at all: dark blob next to the previous paddles
        for f in range(1, 31):
            if tracked["FINS"][f].sum() == 0 and tracked["FINS"][f - 1].sum() > 0:
                excl = tracked["SHORTS"].get(f, np.zeros_like(person[f])) | tracked["MASK"].get(f, np.zeros_like(person[f]))
                tracked["FINS"][f] = blob_near(imgs[f], person[f], centroid(tracked["FINS"][f - 1]), excl, lo=150, hi=12000, r=200)
        pkl.write_bytes(pickle.dumps(tracked))
        del sam

    # ---- the rest of the classes, helpers and post-processing -------------
    raw = {"SHORTS": dict(tracked["SHORTS"]), "MASK": dict(tracked["MASK"]), "FINS": dict(tracked["FINS"])}
    for c in ("SHORTS", "MASK", "FINS"):                 # a held mask only when nothing was ever found
        for f in range(N):
            if f not in raw[c] or raw[c][f].sum() == 0:
                v = min((g for g in raw[c] if raw[c][g].sum() > 0), key=lambda g: abs(g - f), default=None)
                if v is not None and c != "FINS":
                    raw[c][f] = raw[c][v] & dil(person[f], 3)
                    log(f"[fix] {c} f{f}: held from f{v} (no valid SAM mask)")
    spikes = smooth_spikes(raw, person)
    log(f"[fix] area spikes replaced by interpolation: {spikes}")

    watch = {}
    tipsd = {}
    for f in range(N):
        avoid = hair[f].copy()
        for c in ("SHORTS", "MASK", "FINS"):
            if f in raw[c]:
                avoid |= raw[c][f]
        tipsd[f] = hand_tips(person[f], avoid)
        w = cur["WATCH"][f]
        watch[f] = w if w.sum() >= 10 else find_watch2(imgs[f], person[f], avoid, tipsd[f])
    for f in range(N):                                    # gaps of up to 3 frames: neighbour's strap, dark pixels here
        if watch[f].sum() < 10:
            for g in (f - 1, f + 1, f - 2, f + 2, f - 3, f + 3):
                if 0 <= g < N and watch[g].sum() >= 10:
                    cand = dil(watch[g], 14) & person[f] & (luma01(imgs[f]) < 0.26)
                    if cand.sum() >= 25:
                        watch[f] = cand
                        break
    wcount = sum(1 for f in range(N) if watch[f].sum() >= 10)
    log(f"[fix] wrist watch classless on {wcount}/{N} frames")

    cls = {c: {} for c in CLASSES}
    soft = {}
    for f in range(N):
        P = dil(person[f], 3)
        z = np.zeros_like(P)
        m = {c: raw[c].get(f, z) & P for c in ("SHORTS", "MASK", "FINS")}
        oth = m["MASK"] | m["FINS"] | m["SHORTS"]
        m["SKIN"] = person[f] & ~dil(oth, 1) & ~dil(hair[f], 1) & ~dil(watch[f], 3)
        # fingertips stay in SKIN (extremities of the diver mask)
        avoid = oth | hair[f]
        for x, y in tipsd[f]:
            disk = np.zeros(P.shape, np.uint8)
            cv2.circle(disk, (x, y), 24, 1, -1)
            m["SKIN"] |= (disk > 0) & person[f] & ~avoid & ~dil(watch[f], 3)
        if 40 <= f <= 52:                                 # D: top 6 % strip, every class
            for c in m:
                m[c][:STRIP] = False
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m["FINS"] = cv2.morphologyEx(m["FINS"].astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0     # C: 3 px closing
        m["MASK"] = join_parts(m["MASK"], P)               # B: one object
        for _ in range(2):
            taken = z.copy()
            ex = {}
            for c in PRIORITY:
                ex[c] = m[c] & ~taken
                taken |= ex[c]
            m = {c: clean(ex[c]) for c in CLASSES}
        taken = z.copy()
        e3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ys = np.nonzero(person[f])[0]
        top = int(ys.min()) if len(ys) else 0
        for c in PRIORITY:
            b = m[c] & ~taken
            taken |= b
            b = cv2.erode(b.astype(np.uint8), e3, iterations=POST["erode"]) > 0
            if 56 <= f <= 64:                              # E: round the water-line cut
                band = np.zeros_like(b)
                band[top:top + 60, :] = True
                sm = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 3.0) > 0.5
                b = np.where(band, sm & dil(person[f], 3), b)
            cls[c][f] = b
    for f in range(N):
        ys = np.nonzero(person[f])[0]
        top = int(ys.min()) if len(ys) else 0
        a = {}
        for c in CLASSES:
            sg = 1.5 if c == "SKIN" else 1.0                   # SKIN: 3 px, others 2 px
            al = cv2.GaussianBlur(cls[c][f].astype(np.float32), (0, 0), sg)
            if 56 <= f <= 64:                                  # E: 4 px at the water line
                al2 = cv2.GaussianBlur(cls[c][f].astype(np.float32), (0, 0), 2.0)
                w = np.zeros(al.shape, np.float32)
                w[top:top + 60, :] = 1.0
                al = al * (1 - w) + al2 * w
            a[c] = al
        s = sum(a.values())
        k = np.maximum(s, 1.0)
        soft[f] = {c: np.clip(a[c] / k, 0, 1) for c in CLASSES}

    # ---- write masks, tables ------------------------------------------------
    for c in CLASSES:
        (cdir / c).mkdir(parents=True, exist_ok=True)
    for f in range(N):
        for c in CLASSES:
            cv2.imwrite(str(cdir / c / f"{f:05d}.png"), (soft[f][c] * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(str(cdir / "WATCH" / f"{f:05d}.png"), watch[f].astype(np.uint8) * 255)
    areas = {c: [int(cls[c][f].sum()) for f in range(N)] for c in CLASSES}
    overlap = sum(int((sum(cls[c][f].astype(np.uint8) for c in CLASSES) > 1).sum()) for f in range(N))
    flags = {}
    for c in CLASSES:
        for f in range(1, N):
            p, q = areas[c][f - 1], areas[c][f]
            if p == 0 and q == 0:
                continue
            ch = (q - p) / max(p, 1)
            if abs(ch) > POST["jump"]:
                flags.setdefault(f, []).append(f"{c} {ch * 100:+.0f}%")
    lines = ["frame," + ",".join(CLASSES) + ",watch,flags"]
    for f in range(N):
        lines.append(f"{f}," + ",".join(str(areas[c][f]) for c in CLASSES) +
                     f",{int(watch[f].sum())}," + ";".join(flags.get(f, [])))
    (qc / "class_areas.csv").write_text("\n".join(lines) + "\n")
    zeros = {c: [f for f in range(N) if areas[c][f] == 0] for c in CLASSES}
    log(f"[fix] overlap pixels {overlap}; empty frames {zeros}")
    log("[fix] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
