#!/usr/bin/env python3
"""Part 1 (v3) - six classes, priority MASK > WATCH > FINS > SHORTS > HAIR > SKIN.

SAM 3.1 stage (box prompts, no point prompts):
  SHORTS  f0 mask tracked forward, f46 mask tracked backward, 6-frame windows, the box of
          each window is the bounding box of the previous frame's SHORTS mask; a frame
          below 50 % of the f0 area is a failure and is re-prompted once.
  HAIR    same chain over the gap f42-f55 (existing SAM hair masks elsewhere).
  FINS    one box per paddle, windows f0-10 / 10-20 / 20-30, plus both paddles at f28.
  WATCH   "wristwatch" tried on a few frames; colour band elsewhere.
Post stage (rules B-G): MASK parts must be >= 300 px and within 15 px of HAIR, thin
extensions (< 5 px, snorkel tube excepted) go, no joining lines; FINS get 5 px closing +
contour smoothing, skin pieces enclosed by FINS go to FINS; water-line / strip rules;
SKIN holes < 400 px filled, SKIN specks < 150 px removed.

    python diver_masks3.py --input inputs/diver.mov [--post-only]
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
from diver_classes import Sam, clean, dil, hand_tips, luma01, read_mask
from diver_fix import centroid, fill_holes, find_watch2

REPO = Path(__file__).resolve().parent
N, W, H = 90, 1232, 1078
STRIP = int(round(0.06 * 1078))
PRIORITY = ["MASK", "WATCH", "FINS", "SHORTS", "HAIR", "SKIN"]
CLASSES = PRIORITY


def bbox(m, margin=10):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(W - 1, xs.max() + margin), min(H - 1, ys.max() + margin))


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


class SamB(Sam):
    def __init__(self, frames_dir, cache):
        super().__init__(frames_dir, cache)
        # hotstart (delay 15, unmatch 8) deleted objects the tracker still held whenever the text
        # detector did not re-find them in the next frames (shorts above the water line, f46 seed)
        self.pred.model.hotstart_delay = 0

    def text_box(self, sid, cs, f, text, boxes):
        norm = [[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H] for x0, y0, x1, y1 in boxes]
        r = self.pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": f - cs,
                                      "text": text, "bounding_boxes": norm,
                                      "bounding_box_labels": [1] * len(norm)})
        o = r["outputs"]
        return [int(i) for i in o["out_obj_ids"]], [np.asarray(m) > 0 for m in o["out_binary_masks"]]

    def text_only(self, sid, cs, f, text):
        r = self.pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": f - cs, "text": text})
        o = r["outputs"]
        return [int(i) for i in o["out_obj_ids"]], [np.asarray(m) > 0 for m in o["out_binary_masks"]]

    def propagate_ids(self, sid, cs, start, direction):
        got: dict[int, dict[int, np.ndarray]] = {}
        for resp in self.pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": direction, "start_frame_index": start - cs}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                got.setdefault(int(oid), {})[fi] = np.asarray(mm) > 0
        self.pred.handle_request({"type": "close_session", "session_id": sid})
        return got


def box_track(sam, a, b, seed_f, text, boxes, targets, tag, log):
    """One window: text + positive box exemplar(s) on the seed frame, instance(s)
    matched to `targets` by IoU, tracked to the rest of the window. Returns
    [ {frame: mask} per target ] or None if nothing was detected."""
    sid = sam.start(a, b + 1, tag)
    ids, masks = sam.text_box(sid, a, seed_f, text, boxes)
    chosen = []
    for t in targets:
        best, bi = 0.0, None
        for i, m in zip(ids, masks):
            v = iou(m, t) if t is not None and t.any() else 0.0
            if v > best:
                best, bi = v, i
        chosen.append(bi if best > 0.12 else None)
    log(f"    [{tag}] '{text}' f{seed_f}: {len(ids)} inst, match {chosen}")
    if all(c is None for c in chosen):
        sam.pred.handle_request({"type": "close_session", "session_id": sid})
        return None
    d = "forward" if seed_f == a else "backward" if seed_f == b else "both"
    res = sam.propagate_ids(sid, a, seed_f, d)
    return [res.get(c, {}) if c is not None else {} for c in chosen]


def chain(sam, name, text_alts, seed_f, seed_m, end_f, person, A_ok, area_bounds, log, raw_min=0, rawlog=None):
    """Chain 6-frame windows from seed_f toward end_f. A_ok(clipped mask)->bool is the success
    test (SHORTS: >= 50 % of the f0 area). The box and the swelling guard use the RAW SAM mask
    (the whole object, also its part above the water line); the result is clipped to the
    underwater part. Returns {frame: clipped mask}, failed frames."""
    fwd = end_f > seed_f
    out = {seed_f: seed_m}
    fails = []
    cur_f, cur_m = seed_f, seed_m                       # raw mask used for the box
    while cur_f != end_f:
        nxt = min(cur_f + 6, end_f) if fwd else max(cur_f - 6, end_f)
        a, b = (cur_f, nxt) if fwd else (nxt, cur_f)
        best = None
        for att, (margin, text) in enumerate(text_alts):
            bb = bbox(cur_m, margin)
            if bb is None:
                break
            r = box_track(sam, a, b, cur_f, text, [bb], [cur_m], f"{name}{cur_f}", log)
            if r is None:
                continue
            frames = {}
            for f, m in r[0].items():
                if f == cur_f:
                    continue
                raw_a = int(m.sum())
                if rawlog is not None:
                    rawlog.setdefault(f, []).append(raw_a)
                if raw_a == 0 or raw_a > area_bounds[1] * max(int(cur_m.sum()), 800):
                    continue                                       # empty / swollen: tracker lost it
                frames[f] = (m, m & dil(person[f], 3))             # water line: only the underwater part
            score = sum(int(c.sum()) for _, c in frames.values())
            if best is None or score > best[0]:
                best = (score, frames)
            lost = len(frames) != abs(nxt - cur_f) or any(m.sum() < raw_min for m, _ in frames.values())
            if frames and not lost:
                break                                              # tracked every frame, whole object seen
        frames = best[1] if best else {}
        for f in range(min(cur_f, nxt), max(cur_f, nxt) + 1):
            if f == cur_f:
                continue
            if f in frames:
                out[f] = frames[f][1]
            if f not in frames or not A_ok(frames[f][1]):
                fails.append(f)
        if nxt in frames and frames[nxt][0].sum() > 0:
            cur_m = frames[nxt][0]
        cur_f = nxt
    return out, sorted(set(fails))


def sam_stage(a_input: Path, log):
    stem = a_input.stem
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    prev = cache / "cls_prev"
    person = {f: read_mask(REPO / "masks" / f"{stem}__fwd" / "subject", f) for f in range(N)}
    imgs = {f: cv2.imread(str(frames / f"{f:05d}.jpg")) for f in range(N)}
    pv = lambda c, f: read_mask(prev / c, f)
    sam = SamB(frames, cache)
    raw: dict = {}
    t0 = time.time()

    # ---- SHORTS: f0 forward, f46 backward ------------------------------------
    s0, s46 = pv("SHORTS", 0), pv("SHORTS", 46)
    A0 = int(s0.sum())
    log(f"[v3] SHORTS f0 area {A0} (failure below {A0 // 2}), f46 area {int(s46.sum())}")
    A_ok = lambda m: m.sum() >= 0.5 * A0
    alts = [(10, "shorts"), (24, "white swim shorts")]
    rawA = {"F": {}, "B": {}}
    fw, ff = chain(sam, "shF", alts, 0, s0, 46, person, A_ok, (0, 2.4), log, raw_min=0.5 * A0, rawlog=rawA["F"])
    log(f"[v3] SHORTS forward done ({time.time() - t0:.0f}s); failed frames {ff}")
    bw, fb = chain(sam, "shB", alts, 46, s46, 0, person, A_ok, (0, 2.4), log, raw_min=0.5 * A0, rawlog=rawA["B"])
    log(f"[v3] SHORTS backward done ({time.time() - t0:.0f}s); failed frames {fb}")
    raw["SHORTS_fwd"], raw["SHORTS_bwd"], raw["SHORTS_rawA"] = fw, bw, rawA

    # ---- HAIR gap f42-f55: f41 forward, f56 backward -----------------------------
    h41, h56 = pv("HAIR", 41), pv("HAIR", 56)
    hok = lambda m: m.sum() >= 600
    hw, _ = chain(sam, "haF", [(10, "hair"), (24, "hair")], 41, h41, 56, person, hok, (0, 2.2), log)
    hb, _ = chain(sam, "haB", [(10, "hair"), (24, "hair")], 56, h56, 41, person, hok, (0, 2.2), log)
    raw["HAIR_fwd"], raw["HAIR_bwd"] = hw, hb
    log(f"[v3] HAIR chains done ({time.time() - t0:.0f}s)")

    # ---- FINS: one box per paddle, f0-f30 ------------------------------------------
    f0m = pv("FINS", 0)
    num, lab, st, cen = cv2.connectedComponentsWithStats(f0m.astype(np.uint8), 8)
    comps = sorted([k for k in range(1, num) if st[k, cv2.CC_STAT_AREA] > 800], key=lambda k: cen[k][0])[:2]
    paddles = [lab == comps[0], lab == comps[-1]] if len(comps) >= 2 else [f0m, np.zeros_like(f0m)]
    fins = {0: [paddles[0], paddles[1]]}
    cur = [paddles[0], paddles[1]]
    for (a_, b_) in [(0, 10), (10, 20), (20, 30)]:
        boxes = [bbox(m, 12) for m in cur if m.any()]
        tg = [m for m in cur if m.any()]
        done = None
        for text in ("swim fins", "flipper"):
            r = box_track(sam, a_, b_, a_, text, boxes, tg, f"fin{a_}", log)
            if r is not None and any(len(x) > 1 for x in r):
                done = r
                break
        if done is None:
            continue
        for k, res in enumerate(done):
            for f, m in res.items():
                if m.sum() > 100 and m.sum() <= 2.4 * max(int(cur[k].sum()), 1):
                    fins.setdefault(f, [np.zeros_like(f0m), np.zeros_like(f0m)])[k] = m
        end = fins.get(b_)
        if end is not None:
            cur = [end[0] if end[0].any() else cur[0], end[1] if end[1].any() else cur[1]]
    raw["FINS"] = fins
    # f28: both paddles, the second one's whole underwater part
    sid = sam.start(26, 31, "fin28")
    ids, ms = sam.text_box(sid, 26, 28, "swim fins", [(1105, 505, 1215, 625), (1040, 545, 1095, 595)])
    got28 = {}
    if ids:
        res = sam.propagate_ids(sid, 26, 28, "both")
        for i in ids:
            for f, m in res.get(i, {}).items():
                if f == 28 and m.sum() > 60:
                    got28[i] = m
    else:
        sam.pred.handle_request({"type": "close_session", "session_id": sid})
    raw["FINS28"] = got28
    log(f"[v3] FINS windows done ({time.time() - t0:.0f}s); f28 instances {[int(m.sum()) for m in got28.values()]}")

    old = REPO / "masks" / f"{stem}__cls" / "sam_raw_v3.pkl"
    raw["WATCH"] = pickle.load(open(old, "rb")).get("WATCH", {}) if old.exists() else {}
    pickle.dump(raw, open(REPO / "masks" / f"{stem}__cls" / "sam_raw_v4.pkl", "wb"))
    log(f"[v3] SAM stage finished in {time.time() - t0:.0f}s")
    return raw


# ------------------------------------------------------------------ post stage
def thin_free(m, minw=5):
    """Rule C: drop MASK extensions thinner than `minw` px; a long thin piece
    attached to the main body (the snorkel tube) stays."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (minw, minw))
    core = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    thin = m & ~core
    num, lab, st, _ = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    keep = core.copy()
    for j in range(1, num):
        comp = lab == j
        a = st[j, cv2.CC_STAT_AREA]
        length = max(st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT])
        touches = (dil(comp, 2) & core).any()
        if touches and length >= 30 and a >= 60:               # snorkel tube
            keep |= comp
    return keep


def smooth_contour(m, sigma=2.0):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) > 0.5


def post_stage(a_input: Path, log):
    stem = a_input.stem
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    prev = cache / "cls_prev"
    cdir = REPO / "masks" / f"{stem}__cls"
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    person = {f: read_mask(REPO / "masks" / f"{stem}__fwd" / "subject", f) for f in range(N)}
    imgs = {f: cv2.imread(str(frames / f"{f:05d}.jpg")) for f in range(N)}
    pv = lambda c, f: read_mask(prev / c, f)
    v4 = cdir / "sam_raw_v4.pkl"
    raw = pickle.load(open(v4 if v4.exists() else cdir / "sam_raw_v3.pkl", "rb"))
    z = np.zeros((H, W), bool)

    A0 = int(pv("SHORTS", 0).sum())
    sh, hr, fn, wt, mk = {}, {}, {}, {}, {}
    mk_prior = {f: pv("MASK", f) for f in range(N)}
    deleted_mask = []
    for f in range(N):
        Pd = dil(person[f], 3)
        # SHORTS (A): f0 and f46+ masks as they are, 1..45 the union of both chains
        if f == 0 or f >= 46:
            sh[f] = pv("SHORTS", f) & Pd
        else:                                              # forward U backward U the earlier SAM masks
            sh[f] = (raw["SHORTS_fwd"].get(f, z) | raw["SHORTS_bwd"].get(f, z) | pv("SHORTS", f)) & Pd
        # HAIR: existing SAM masks where they exist, the chains in the gap f42-f55
        h = pv("HAIR", f)
        if 42 <= f <= 55:
            h = h | raw["HAIR_fwd"].get(f, z) | raw["HAIR_bwd"].get(f, z)
        hr[f] = h & dil(person[f], 6)
    # FINS (E): per-paddle SAM masks, f28 both paddles
    from diver_fix import blob_near, grow_dark
    fins_src = {}
    for f in range(0, 31):
        base = pv("FINS", f)
        ps = raw["FINS"].get(f)
        v3 = (ps[0] | ps[1]) if ps is not None else z.copy()
        both = ps is not None and ps[0].sum() >= 150 and ps[1].sum() >= 150
        m = base.copy()
        src = "SAM (earlier)" if base.sum() >= 150 else ""
        if v3.sum() >= 150:                               # the per-paddle box chain is the primary mask
            m = m | v3
            src = "SAM (box chain, both paddles)" if both else "SAM (box chain, one paddle)"
        if m.sum() < 150 and f >= 1:                       # no SAM mask: the dark paddle next to the previous one
            pc = centroid(fn[f - 1]) if fn[f - 1].sum() else (1100, 560)
            for thr in (0.20, 0.24, 0.28, 0.32):
                b = blob_near(imgs[f], person[f], pc, sh.get(f, z) | mk_prior.get(f, z), lo=120, hi=12000, r=220, thr=thr)
                if b.sum() >= 120:
                    m, src = b, "dark blob"
                    break
        fn[f] = m
        fins_src[f] = src
    # f28: the second paddle's whole underwater part (both paddles, dark blobs at the two spots)
    for pt in ((1160, 560), (1063, 568)):
        for thr in (0.22, 0.26, 0.30):
            gdark = (luma01(imgs[28]) < thr) & dil(person[28], 45)
            num, cc, _, _ = cv2.connectedComponentsWithStats(gdark.astype(np.uint8), 8)
            win = cc[pt[1] - 12:pt[1] + 13, pt[0] - 12:pt[0] + 13]
            ids = win[win > 0]
            if len(ids):
                comp = cc == np.bincount(ids).argmax()
                if 100 <= comp.sum() <= 14000:
                    fn[28] = fn[28] | comp
                    break
    fins_src[28] = (fins_src.get(28, "") + " + dark blobs").strip()
    log("[v3] FINS sources: " + ", ".join(f"{k}: {sum(1 for v in fins_src.values() if v == k)}" for k in sorted(set(fins_src.values()))))
    for f in range(N):
        fn.setdefault(f, z.copy())
    # MASK (B, C): no point prompts, empty f0-f40; f41+ the SAM masks, parts cleaned
    for f in range(N):
        m = pv("MASK", f) if f >= 41 else z.copy()
        if m.any():
            num, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            keep = z.copy()
            for j in range(1, num):
                comp = lab == j
                a = int(st[j, cv2.CC_STAT_AREA])
                hair_here = hr[f].sum() >= 300
                near = (dil(comp, 15) & hr[f]).any() if hair_here else True
                if a < 300:
                    deleted_mask.append((f, a, "< 300 px"))
                elif not near:
                    deleted_mask.append((f, a, "> 15 px from HAIR"))
                else:
                    keep |= comp
            m2 = thin_free(keep)
            gone = int((keep & ~m2).sum())
            if gone:
                deleted_mask.append((f, gone, "thin extension < 5 px"))
            m = m2
        mk[f] = m
    # WATCH (D): SAM where it found one, otherwise the dark strap band next to the hand
    tipsd, wt = {}, {}
    for f in range(N):
        avoid = hr[f] | fn[f] | sh[f] | mk[f]
        tipsd[f] = hand_tips(person[f], avoid)
        w = raw["WATCH"].get(f)
        wt[f] = w if w is not None else find_watch2(imgs[f], person[f], avoid, tipsd[f])
    for f in range(N):
        if not 60 <= wt[f].sum() <= 500:
            wt[f] = z.copy()
    for f in range(N):
        if wt[f].sum() < 60:
            for g in (f - 1, f + 1, f - 2, f + 2, f - 3, f + 3):
                if 0 <= g < N and 60 <= wt[g].sum() <= 500:
                    cand = dil(wt[g], 6) & person[f] & (luma01(imgs[f]) < 0.26)
                    if 60 <= cand.sum() <= 500:
                        wt[f] = cand
                        break

    for f in range(N):                                    # still no strap: widen the search to +-6 frames
        if wt[f].sum() < 60:
            for d in range(1, 7):
                for g in (f - d, f + d):
                    if 0 <= g < N and 60 <= wt[g].sum() <= 500:
                        cand = dil(wt[g], 8 + 2 * d) & person[f] & (luma01(imgs[f]) < 0.27)
                        if 60 <= cand.sum() <= 500:
                            wt[f] = cand
                            break
                if wt[f].sum() >= 60:
                    break
    cls = {c: {} for c in CLASSES}
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for f in range(N):
        P = person[f]
        Pd = dil(P, 3)
        m = {"MASK": mk[f], "WATCH": wt[f] & dil(P, 6), "FINS": fn[f], "SHORTS": sh[f], "HAIR": hr[f]}
        # E: FINS = 5 px closing + contour smoothing; solid paddles
        if m["FINS"].any():
            fm = cv2.morphologyEx(m["FINS"].astype(np.uint8), cv2.MORPH_CLOSE, k5) > 0
            fm = fill_holes(fm)
            m["FINS"] = smooth_contour(fm, 2.0)
        # SKIN = diver mask minus every other class
        oth = m["MASK"] | m["WATCH"] | m["FINS"] | m["SHORTS"] | m["HAIR"]
        skin = P & ~dil(oth, 1)
        # E: skin pieces enclosed by FINS (>50 % of their outline) and < 6 px away go to FINS; only the
        #    largest piece (the one attached to the leg) may remain SKIN
        if m["FINS"].any():
            near = dil(m["FINS"], 6)
            num, lab, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
            if num > 2:
                main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                for j in range(1, num):
                    if j == main:
                        continue
                    comp = lab == j
                    if not (comp & near).any():
                        continue
                    ring = dil(comp, 2) & ~comp
                    if ring.sum() and (ring & m["FINS"]).sum() / ring.sum() > 0.5:
                        m["FINS"] = m["FINS"] | comp
                        skin &= ~comp
        if 40 <= f <= 52:                                    # strip stays exactly as before
            for c in list(m):
                m[c][:STRIP] = False
            skin[:STRIP] = False
        m["SKIN"] = skin
        # priority MASK > WATCH > FINS > SHORTS > HAIR > SKIN, nothing in two classes
        taken = z.copy()
        for c in PRIORITY:
            m[c] = m[c] & ~taken
            taken |= m[c]
        # G: unclassed holes < 400 px inside the diver become SKIN; SKIN specks < 150 px go
        unclassed = P & ~taken
        num, lab, st, _ = cv2.connectedComponentsWithStats(unclassed.astype(np.uint8), 4)
        for j in range(1, num):
            if st[j, cv2.CC_STAT_AREA] < 400:
                m["SKIN"] = m["SKIN"] | (lab == j)
        m["SKIN"] = clean(m["SKIN"])
        for c in ("SHORTS", "FINS", "HAIR"):
            m[c] = clean(m[c]) if m[c].sum() >= 150 else m[c]
        cls_f = m
        # F: f56-f64 water-line edge rounded, 4 px feather applied at the soft-alpha step
        if 56 <= f <= 64:
            ys = np.nonzero(P)[0]
            top = int(ys.min()) if len(ys) else 0
            band = np.zeros((H, W), bool)
            band[top:top + 60, :] = True
            for c in ("SKIN", "SHORTS"):
                sm = smooth_contour(cls_f[c], 3.0) & Pd
                cls_f[c] = np.where(band, sm, cls_f[c])
        for c in CLASSES:
            cls[c][f] = cls_f[c]
    # final exclusivity pass (the smoothing above may have overlapped classes)
    for f in range(N):
        taken = z.copy()
        for c in PRIORITY:
            cls[c][f] = cls[c][f] & ~taken
            taken |= cls[c][f]

    # ---- write masks + tables --------------------------------------------------------
    for c in CLASSES:
        (cdir / c).mkdir(parents=True, exist_ok=True)
    e3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for f in range(N):
        ys = np.nonzero(person[f])[0]
        top = int(ys.min()) if len(ys) else 0
        al = {}
        for c in CLASSES:
            b = cls[c][f]
            sg = 1.5 if c == "SKIN" else 1.0
            a = cv2.GaussianBlur(b.astype(np.float32), (0, 0), sg)
            if 56 <= f <= 64:                                  # F: 4 px feather at the water line
                a2 = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 2.0)
                w = np.zeros(a.shape, np.float32)
                w[top:top + 60, :] = 1
                a = a * (1 - w) + a2 * w
            al[c] = a
        s = np.maximum(sum(al.values()), 1.0)
        for c in CLASSES:
            cv2.imwrite(str(cdir / c / f"{f:05d}.png"), (np.clip(al[c] / s, 0, 1) * 255 + 0.5).astype(np.uint8))
    areas = {c: [int(cls[c][f].sum()) for f in range(N)] for c in CLASSES}
    overlap = sum(int((sum(cls[c][f].astype(np.uint8) for c in CLASSES) > 1).sum()) for f in range(N))
    shorts_fail = [f for f in range(N) if areas["SHORTS"][f] < 0.5 * A0]
    flags = {}
    for c in CLASSES:
        for f in range(1, N):
            p, q = areas[c][f - 1], areas[c][f]
            if p == 0 and q == 0:
                continue
            if abs(q - p) / max(p, 1) > 0.08:
                flags.setdefault(f, []).append(f"{c} {100 * (q - p) / max(p, 1):+.0f}%")
    lines = ["frame," + ",".join(CLASSES) + ",flags"]
    for f in range(N):
        lines.append(f"{f}," + ",".join(str(areas[c][f]) for c in CLASSES) + "," + ";".join(flags.get(f, [])))
    (qc / "class_areas.csv").write_text("\n".join(lines) + "\n")
    (qc / "shorts_failed.txt").write_text(" ".join(map(str, shorts_fail)))
    # which of the failures are a lost track and which are only the water line (whole shorts seen, underwater part small)
    rawA = raw.get("SHORTS_rawA", {"F": {}, "B": {}})
    lost, wl = [], []
    for f in shorts_fail:
        vals = rawA["F"].get(f, []) + rawA["B"].get(f, [])
        if f == 0 or f >= 46:
            vals = vals + [int(pv("SHORTS", f).sum())]
        (wl if vals and max(vals) >= 0.5 * A0 else lost).append(f)
    (qc / "shorts_failed_split.txt").write_text(f"lost: {lost}\nwaterline: {wl}\n")
    log(f"[v3] SHORTS failures: {len(lost)} lost track {lost}; {len(wl)} water-line limited {wl}")
    (qc / "mask_deleted.txt").write_text("\n".join(f"f{f}: {a} px ({why})" for f, a, why in deleted_mask))
    log(f"[v3] overlap pixels {overlap}; SHORTS f0 area {A0}; failed frames (<50 %): {shorts_fail}")
    log(f"[v3] deleted MASK parts: {len(deleted_mask)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--post-only", action="store_true")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    if not a.post_only:
        sam_stage(a.input, log)
    return post_stage(a.input, log)


if __name__ == "__main__":
    raise SystemExit(main())
