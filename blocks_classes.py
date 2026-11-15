#!/usr/bin/env python3
"""Four SAM 3.1 classes for inputs/blocks_dive.mov frames 210-299.

    SKIN   "bare skin of the diver"
    FINS   "black swim fins"        (both fins and the foot pockets)
    SHORTS "dark swimsuit"
    HEAD   "hair and dive mask"
Helpers (tracked, never graded): WATCH "wrist watch", WAIST "white belt" -
these stay classless.  Priority FINS > SHORTS > HEAD > SKIN, no pixel in two
classes.

Stages (python blocks_classes.py <stage>):
    detect   text prompts on the prompt frames (one-frame sessions, cached)
    track    one tracker session per chunk: every class is its own object,
             the user's point prompts and the gap negatives are added on the
             prompted frames, propagation runs both ways
    thin     tight 2x crop around the thin side-on left fin (f223, f274 +-5)
    finish   exclusive, cleaned, edge-ruled class alphas + numeric checks
    grade    class-driven grade of the 90 frames + the contact sheet

All SAM 3.1 calls go through the CPU port (shades/sam31_cpu.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from diver_masks_bwd import prompts_from_mask
from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent
CACHE = REPO / ".cache"
FRAMES = CACHE / "sam_frames_blocks_dive_bwd"
OUT = REPO / "work" / "classes"
W, H = 1232, 1078
F0, F1 = 210, 300

CLASSES = ["SKIN", "FINS", "SHORTS", "HEAD"]
HELPERS = ["WATCH", "WAIST"]
PRIORITY = ["FINS", "SHORTS", "HEAD", "SKIN"]
OID = {c: 101 + k for k, c in enumerate(CLASSES + HELPERS + ["HAND"])}
TEXT = {"SKIN": "bare skin of the diver", "FINS": "black swim fins",
        "SHORTS": "dark swimsuit", "HEAD": "hair and dive mask",
        "WATCH": "wrist watch", "WAIST": "white belt"}
COLOR = {"SKIN": (255, 170, 60), "FINS": (255, 60, 255), "SHORTS": (60, 120, 255),
         "HEAD": (60, 255, 60)}                                   # RGB

# the user's phrase first; shorter phrases that SAM 3.1 actually answers follow
# Measured on f210/226/235/274/290: "bare skin of the diver", "hair and dive mask",
# "dark swimsuit", "shorts", "wrist watch", "watch", "white belt" find nothing;
# "bare skin", "black swim fins" and "head" are answered on every frame.
PHRASES = {"SKIN": ["bare skin of the diver", "bare skin"], "FINS": ["black swim fins"],
           "SHORTS": ["dark swimsuit"], "HEAD": ["hair and dive mask", "head"],
           "WATCH": ["wrist watch"], "WAIST": ["white belt"]}
RUN_PHRASES = ["bare skin", "black swim fins", "head"]

# the user's anchors for the LEFT fin (x, y in frame pixels): FINS +, SKIN -
LEFT_FIN = {248: (292, 266), 286: (247, 201), 289: (198, 177), 292: (175, 171),
            295: (169, 176), 299: (189, 198)}
RIGHT_POCKET_FRAMES = list(range(280, 287))          # right fin foot pocket: FINS +, SKIN -
LEG_GAP_FRAMES = [235, 286]                          # negative for every class
ARM_GAP_FRAMES = [210, 235]
INTER_FRAMES = [214, 216, 218, 248, 250, 252]        # extra prompt frames, both directions
THIN = [(218, 229), (269, 280)]                      # +-5 frames around f223 and f274
CHUNKS = [(210, 242), (238, 270), (266, 300)]
SPLITS = [210, 240, 268, 300]                        # frames owned by each chunk
# frames on which the text detector runs (every class and helper, cached)
PROMPT_FRAMES = [210, 214, 216, 218, 226, 235, 248, 250, 252, 254, 274, 280, 281, 282, 283, 284,
                 285, 286, 289, 292, 295, 299]
# text detection on every second frame plus every prompt frame; the frames in between
# are filled from both neighbours with optical flow
DETECT_FRAMES = sorted(set(range(210, 300, 2)) | set(PROMPT_FRAMES))


def dil(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def ero(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(m.astype(np.uint8), k) > 0


def luma01(f):
    x = cv2.cvtColor(cv2.imread(str(FRAMES / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return x, x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def envelope():
    """The diver's silhouette on every frame (plan (a) person mask, SAM 3.1)."""
    import blocks_grade as B
    P, _ = B.merged_masks(list(range(F0 - 6, F1 + 6)), (H, W))
    return P


# ------------------------------------------------------------------ SAM ---
class Sam:
    def __init__(self):
        self.pred, on_gpu = _build_predictor(Masking(parts={}, max_num_objects=8), REPO, CACHE)

        def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
            o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
            o.update(refined_obj_id_to_mask or {})
            return o

        self.pred.model._build_sam2_output = _build_sam2_output
        print(f"[cls] SAM 3.1 checkpoint sam3.1_multiplex.pt on {'CUDA' if on_gpu else 'CPU (patched port)'}", flush=True)

    def start(self, frames: list[int], tag: str, src: Path = FRAMES):
        d = CACHE / f"sam_chunk_cls_{tag}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        for k, i in enumerate(frames):
            (d / f"{k:05d}.jpg").symlink_to(src / f"{i:05d}.jpg")
        return self.pred.handle_request({"type": "start_session", "resource_path": str(d)})["session_id"]

    def text(self, sid, local_f, text):
        r = self.pred.handle_request({"type": "add_prompt", "session_id": sid,
                                      "frame_index": local_f, "text": text})
        return [np.asarray(m) > 0 for m in r["outputs"]["out_binary_masks"]]

    def points(self, sid, local_f, cls, pts, lab, w=W, h=H):
        self.pred.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": local_f,
            "points": [[x / w, y / h] for x, y in pts],
            "point_labels": [int(v) for v in lab], "obj_id": OID[cls]})

    def propagate(self, sid, cs, start):
        got = {c: {} for c in OID}
        inv = {v: k for k, v in OID.items()}
        for resp in self.pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "both", "start_frame_index": start}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                c = inv.get(int(oid))
                if c:
                    got[c][fi] = np.asarray(mm) > 0
        self.pred.handle_request({"type": "close_session", "session_id": sid})
        return got


# ---------------------------------------------------------------- detect ---
def stage_detect(a):
    """Text detections on the prompt frames, cached by phrase: det[frame][phrase] = masks."""
    OUT.mkdir(parents=True, exist_ok=True)
    pk = OUT / "det.pkl"
    det = pickle.loads(pk.read_bytes()) if pk.exists() else {}
    frames = [int(v) for v in a.frames] if a.frames else DETECT_FRAMES
    phrases = a.texts or RUN_PHRASES
    sam = Sam()
    for f in frames:
        det.setdefault(f, {})
        todo = [t for t in phrases if t not in det[f]]
        if not todo:
            continue
        sid = sam.start([f], f"det{f}")
        for t in todo:
            t0 = time.time()
            ms = sam.text(sid, 0, t)
            det[f][t] = [np.packbits(m) for m in ms]
            print(f"[detect] f{f} '{t}': {len(ms)} inst, areas {[int(m.sum()) for m in ms]}, "
                  f"{time.time() - t0:.1f}s", flush=True)
        sam.pred.handle_request({"type": "close_session", "session_id": sid})
        pk.write_bytes(pickle.dumps(det))
    return 0


def get_det(det, f, phrase):
    return [np.unpackbits(p)[:W * H].reshape(H, W).astype(bool) for p in det.get(f, {}).get(phrase, [])]


# ----------------------------------------------------------- geometry ---
def interior_point(m):
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return int(x), int(y)


def gap_points(P, kind):
    """Water enclosed by the silhouette (between the legs / between the arms).
    Found as closing(P) - P, no hole filling is ever applied to a mask. Legs are
    at the top of the silhouette, arms at the bottom (he is head down)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    gap = (cv2.morphologyEx(P.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0) & ~P
    n, lab, st, cen = cv2.connectedComponentsWithStats(gap.astype(np.uint8), 8)
    ys = np.nonzero(P)[0]
    cy = float(ys.mean())
    out = []
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < 250:
            continue
        m = lab == c
        up = cen[c][1] < cy
        if (kind == "legs") == up:
            x, y = interior_point(m)
            out.append((x, y, int(st[c, cv2.CC_STAT_AREA])))
    return out


def fin_blobs(f, P):
    """Dark connected pieces (>=150 px) on/next to the silhouette, used only to
    place the right fin's foot-pocket point."""
    _, l = luma01(f)
    dk = (l < 0.34) & dil(P[f], 4)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dk.astype(np.uint8), 8)
    return [(int(st[k, 4]), lab == k) for k in range(1, n) if st[k, 4] >= 150]


def right_pocket_point(f, P):
    ys, xs = np.nonzero(P[f])
    cx, cy = xs.mean(), ys.mean()
    blobs = [(a, m) for a, m in fin_blobs(f, P) if a >= 700 and np.nonzero(m)[1].mean() > cx]
    if not blobs:
        return None
    a, m = max(blobs, key=lambda t: t[0])
    yy, xx = np.nonzero(m)
    d = np.hypot(xx - cx, yy - cy)
    near = d <= np.quantile(d, 0.25)
    sub = np.zeros_like(m)
    sub[yy[near], xx[near]] = True
    return interior_point(sub)


# --------------------------------------------------------- seeds / events ---
LIMS = {"SKIN": (3000, 0.85), "SHORTS": (300, 0.40), "FINS": (150, 0.45), "HEAD": (40, 0.10),
        "WATCH": (10, 0.02), "WAIST": (30, 0.05)}


def pick_class(cls, ms, Pf):
    pa = max(int(Pf.sum()), 1)
    lo, hi = LIMS[cls]
    near = dil(Pf, 25 if cls == "FINS" else 4)
    cand = [m for m in ms if lo <= m.sum() <= hi * pa and (m & near).sum() / max(m.sum(), 1) >= 0.8]
    if not cand:
        return None
    if cls == "FINS":
        cand = sorted(cand, key=lambda m: -m.sum())[:2]
        return np.any(np.stack(cand), 0)
    return max(cand, key=lambda m: m.sum())


def derive_shorts(f, P, seeds):
    """No text prompt finds the shorts: take what is left of the person mask
    after SKIN, FINS and HEAD, keep the dark part (white belt / watch dial stay
    out), the biggest piece."""
    used = np.zeros((H, W), bool)
    for c in ("SKIN", "FINS", "HEAD"):
        if (f, c) in seeds:
            used |= seeds[(f, c)]
    _, l = luma01(f)
    rem = P[f] & ~dil(used, 2) & (l < 0.52)
    rem = cv2.morphologyEx(rem.astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    rem = cv2.morphologyEx(rem.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(rem.astype(np.uint8), 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = lab == k
    return m if m.sum() >= 300 else None


def warp_chain(m, g, f):
    """Carry a seed mask from frame g to frame f one frame at a time with optical flow."""
    cur, step = m, (1 if f > g else -1)
    for k in range(g, f, step):
        cur = flow_warp(k + step, k, cur) > 0.5
    return cur


def seed_masks(det, P, frames, log=print):
    """{(f, cls): mask}: cached text detections (user phrase first); a class the
    detector misses on a frame is carried from the nearest detected frame (<= 6
    frames, optical flow); SKIN is last derived from the person mask; SHORTS has
    no text answer at all and is derived from what the other classes leave."""
    seeds, how = {}, {}
    for f in frames:
        for cls in CLASSES + HELPERS:
            for ph in PHRASES[cls]:
                m = pick_class(cls, get_det(det, f, ph), P[f])
                if m is not None:
                    seeds[(f, cls)], how[(f, cls)] = m, "text"
                    break
    for cls in ("FINS", "HEAD", "SKIN"):
        have = sorted(f for (f, c) in seeds if c == cls)
        for f in frames:
            if (f, cls) in seeds:
                continue
            near = [g for g in have if abs(g - f) <= 6]
            if near:
                g = min(near, key=lambda g: abs(g - f))
                m = warp_chain(seeds[(g, cls)], g, f)
                if m.sum() >= LIMS[cls][0]:
                    seeds[(f, cls)], how[(f, cls)] = m, f"flow from f{g}"
    for f in frames:
        if (f, "SKIN") not in seeds and any((f, c) in seeds for c in ("FINS", "HEAD")):
            used = np.zeros((H, W), bool)
            for c in ("FINS", "HEAD"):
                if (f, c) in seeds:
                    used |= seeds[(f, c)]
            m = P[f] & ~dil(used, 2)
            seeds[(f, "SKIN")], how[(f, "SKIN")] = clean_part(m), "person minus fins/head"
        if (f, "SHORTS") not in seeds and (f, "SKIN") in seeds:
            m = derive_shorts(f, P, seeds)
            if m is not None:
                seeds[(f, "SHORTS")], how[(f, "SHORTS")] = m, "person minus skin/fins/head, dark part"
    cnt = {}
    for (f, c), h in how.items():
        cnt.setdefault((c, h.split(" from")[0]), 0)
        cnt[(c, h.split(" from")[0])] += 1
    log("  [seeds] " + "; ".join(f"{c}:{h}={n}" for (c, h), n in sorted(cnt.items())))
    return seeds


def fps_points(mask, n, min_dt=3.0):
    """n points spread over the whole mask (farthest-point sampling), at least
    min_dt px inside it: a point prompt then reproduces the seed instead of a blob."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dt.max() < 1:
        return []
    cand = np.argwhere(dt >= min(min_dt, max(1.0, 0.5 * dt.max())))
    start = np.unravel_index(np.argmax(dt), dt.shape)
    pts = [start]
    d = np.hypot(cand[:, 0] - start[0], cand[:, 1] - start[1])
    for _ in range(max(0, min(n, len(cand)) - 1)):
        p = cand[int(np.argmax(d))]
        pts.append((int(p[0]), int(p[1])))
        d = np.minimum(d, np.hypot(cand[:, 0] - p[0], cand[:, 1] - p[1]))
    return [(int(x), int(y)) for y, x in pts]


N_POS = {"SKIN": 16, "FINS": 7, "SHORTS": 6, "HEAD": 4, "WATCH": 2, "WAIST": 2}


def seed_events(cls, m, others):
    """Positives spread over the seed (each connected piece gets its own share),
    negatives on a ring 6-14 px outside it and inside every other class's seed."""
    ev = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    pieces = [lab == k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 100] or [m]
    per = max(2, N_POS[cls] // len(pieces)) if cls == "FINS" else N_POS[cls]
    for pc in (pieces if cls == "FINS" else [m]):
        ev += [(x, y, 1) for x, y in fps_points(pc, per)]
    ring = dil(m, 14) & ~dil(m, 6)
    ev += [(x, y, 0) for x, y in fps_points(ring, 6, min_dt=1.5)]
    for o in others:
        ev += [(x, y, 0) for x, y in fps_points(o, 3)]
    return ev


def build_events(cs, ce, seeds, P, log):
    """{(frame, class): [(x, y, label)]} for one chunk."""
    ev = {}
    for (f, cls), m in seeds.items():
        if cs <= f < ce:
            ev[(f, cls)] = seed_events(cls, m, [seeds[(f, o)] for o in CLASSES if o != cls and (f, o) in seeds])
    def add(f, cls, pt, lab):
        """A negative point is only valid for an object that has a positive on that
        frame (a negative-only prompt would blank the object there)."""
        if lab == 0 and not any(l == 1 for *_, l in ev.get((f, cls), [])):
            log(f"  [skip] f{f} {cls}: negative {pt} without a positive prompt on that frame")
            return
        ev.setdefault((f, cls), []).append((pt[0], pt[1], lab))
    for f, pt in LEFT_FIN.items():                        # A: left fin = FINS +, not SKIN/SHORTS/HEAD
        if cs <= f < ce:
            add(f, "FINS", pt, 1)
            for o in ("SKIN", "SHORTS", "HEAD"):
                add(f, o, pt, 0)
    for f in RIGHT_POCKET_FRAMES:                         # A: right fin foot pocket
        if cs <= f < ce:
            pt = right_pocket_point(f, P)
            log(f"  [user] f{f} right fin pocket point {pt}")
            if pt:
                add(f, "FINS", pt, 1)
                for o in ("SKIN", "SHORTS", "HEAD"):
                    add(f, o, pt, 0)
    for f, kind in [(f, "legs") for f in LEG_GAP_FRAMES] + [(f, "arms") for f in ARM_GAP_FRAMES]:
        if cs <= f < ce:                                  # C: water between legs / arms: no class
            g = gap_points(P[f], kind)
            log(f"  [user] f{f} {kind} gap points {[(x, y) for x, y, _ in g]}")
            for x, y, _ in g:
                for cls in CLASSES:
                    add(f, cls, (x, y), 0)
    return ev


def stage_track(a):
    pk = OUT / "det.pkl"
    det = pickle.loads(pk.read_bytes())
    P = envelope()
    sam = Sam()
    raw_pk = OUT / "raw.pkl"
    raw = pickle.loads(raw_pk.read_bytes()) if raw_pk.exists() else {}
    log = lambda s: print(s, flush=True)
    chunks = [int(v) for v in a.chunks] if a.chunks else list(range(len(CHUNKS)))
    for ci in chunks:
        cs, ce = CHUNKS[ci]
        frames = [f for f in sorted({f for f in det} | set(INTER_FRAMES)) if cs <= f < ce]
        seeds = seed_masks(det, P, frames, log)
        log(f"[track] chunk {ci} {cs}-{ce - 1}: seeds " + ", ".join(f"f{f}:{c}={int(m.sum())}" for (f, c), m in sorted(seeds.items())))
        ev = build_events(cs, ce, seeds, P, log)
        classes_seeded = {c for (_, c) in ev if c in CLASSES}
        log(f"[track] classes with prompts: {sorted(classes_seeded)}; prompted frames {sorted({f for f, _ in ev})}")
        missing = [c for c in CLASSES if c not in classes_seeded]
        if missing:
            log(f"[track] WARNING no prompt for {missing} in chunk {ci}")
        t0 = time.time()
        sid = sam.start(list(range(cs, ce)), f"c{ci}")
        for (f, cls), pl in ev.items():
            sam.points(sid, f - cs, cls, [(x, y) for x, y, _ in pl], [l for *_, l in pl])
        prompted = sorted({f for f, _ in ev})
        start = min(max(prompted[len(prompted) // 2], cs + 1), ce - 2) - cs
        res = sam.propagate(sid, cs, start)
        raw[ci] = res
        raw_pk.write_bytes(pickle.dumps(raw))
        log(f"[track] chunk {ci} done in {time.time() - t0:.0f}s ({(time.time() - t0) / (ce - cs):.1f} s/frame)")
        for c in CLASSES + HELPERS:
            log(f"   {c:6s} frames {len(res[c]):3d}/{ce - cs}  areas " +
                " ".join(str(int(res[c][f].sum())) if f in res[c] else "-" for f in range(cs, ce, 4)))
    return 0


# --------------------------------------------------------- thin fin (2x) ---
def thin_pieces(Pf):
    """Parts of the silhouette thinner than ~9 px: blades seen edge on, fingers."""
    op = cv2.morphologyEx(Pf.astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    thin = Pf & ~dil(op, 1)
    n, lab, st, cen = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    ys = np.nonzero(Pf)[0]
    y0, y1 = ys.min(), ys.max()
    out = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < 150 or max(w, h) / max(min(w, h), 1) < 2.5:
            continue
        if cen[k][1] > y0 + 0.4 * (y1 - y0):            # blades are at the fin end (top)
            continue
        out.append((area, lab == k))
    return sorted(out, key=lambda t: -t[0])


def left_fin_piece(per, f):
    """The left fin = the FINS piece (>= 150 px) with the smallest x centroid."""
    m = per["FINS"].get(f)
    if m is None:
        return None
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    cs = [(cen[k][0], k) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= MIN_PART]
    if len(cs) < 2:
        return (lab == cs[0][1]) if cs else None
    return lab == min(cs)[1]


LEFT_RANGE = (284, 299)         # user anchors f286/289/292/295/299: one tracked left-fin object


def stage_thin(a):
    """Left fin on a tight 2x crop. THIN ranges: 5 positives along the blade on
    the middle frame (f223, f274). LEFT_RANGE: the user's own points are the prompts."""
    P = envelope()
    per, _ = load_raw()
    sam = Sam()
    res = {}
    pk = OUT / "thin.pkl"
    log = lambda s: print(s, flush=True)
    jobs = [(fa, fb, "blade") for fa, fb in THIN] + [(LEFT_RANGE[0], LEFT_RANGE[1], "user")]
    for (fa, fb, kind) in jobs:
        frames = list(range(fa, fb + 1))
        sel = {f: left_fin_piece(per, f) for f in frames}
        ms = [m for m in sel.values() if m is not None]
        pf = [(fa + fb) // 2] if kind == "blade" else [f for f in LEFT_FIN if fa <= f <= fb]
        if not ms or any(sel[f] is None for f in pf):
            log(f"[thin] f{fa}-{fb}: no left fin piece on the prompt frames, skipped")
            continue
        ys, xs = np.nonzero(np.any(np.stack(ms), 0))
        x0, x1 = max(0, xs.min() - 70), min(W, xs.max() + 70)
        y0, y1 = max(0, ys.min() - 70), min(H, ys.max() + 70)
        log(f"[thin] f{fa}-{fb} ({kind}): window x{x0}-{x1} y{y0}-{y1} ({x1 - x0}x{y1 - y0}) -> 2x bicubic crop")
        cdir = CACHE / f"cls_crop_{fa}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for f in frames:
            im = cv2.imread(str(FRAMES / f"{f:05d}.jpg"))[y0:y1, x0:x1]
            im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(cdir / f"{f:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 97])
        cw, ch = (x1 - x0) * 2, (y1 - y0) * 2
        sid = sam.start(frames, f"thin{fa}", src=cdir)
        for f in pf:
            piece = sel[f]
            pos = [(x, y) for x, y in fps_points(piece, 5)]            # 5 along the fin
            neg = [(x, y) for x, y in fps_points(dil(piece, 14) & ~dil(piece, 6), 4, min_dt=1.5)]
            if kind == "user":
                pos = [LEFT_FIN[f]] + pos[:4]                          # the user's anchor first
                sk = per["SKIN"].get(f)
                if sk is not None:
                    neg += fps_points(sk, 2)                           # SKIN negative
            pts = [((x - x0) * 2, (y - y0) * 2, 1) for x, y in pos] + [((x - x0) * 2, (y - y0) * 2, 0) for x, y in neg]
            log(f"[thin] f{f} prompt: {len(pos)} positives along the left fin"
                + (f" (first = user point {LEFT_FIN[f]})" if kind == 'user' else "") + f", {len(neg)} negatives")
            sam.pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": f - fa,
                                     "points": [[x / cw, y / ch] for x, y, _ in pts],
                                     "point_labels": [int(l) for *_, l in pts], "obj_id": OID["FINS"]})
        start = min(max(sorted(pf)[len(pf) // 2] - fa, 1), len(frames) - 2)
        got = sam.propagate(sid, fa, start)
        for f, m in got["FINS"].items():
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = cv2.resize(m.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
            full &= dil(P[f], 4)
            n, lb, st, _ = cv2.connectedComponentsWithStats(full.astype(np.uint8), 8)
            for k in range(1, n):                               # drop pieces < 150 px
                if st[k, 4] < MIN_PART:
                    full[lb == k] = False
            res[f] = full
        log(f"[thin] f{fa}-{fb} left-fin areas " + " ".join(str(int(res[f].sum())) if f in res else "-" for f in frames) +
            " | detection-based left piece " + " ".join(str(int(sel[f].sum())) if sel[f] is not None else "-" for f in frames))
    pk.write_bytes(pickle.dumps(res))
    return 0


# ----------------------------------------------------------------- finish ---
MIN_PART = 150
JUMP = 0.05


def clean_part(m):
    """Drop disconnected pieces < 150 px. No hole filling, ever."""
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < MIN_PART:
            m[lab == k] = 0
    return m > 0


def close3(m):
    """Closing of at most 3 px."""
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0


def load_raw():
    if (OUT / "per.pkl").exists():
        per = pickle.loads((OUT / "per.pkl").read_bytes())
        thin = pickle.loads((OUT / "thin.pkl").read_bytes()) if (OUT / "thin.pkl").exists() else {}
        thin = {f: m for f, m in thin.items() if abs(f - 223) <= 5 or abs(f - 274) <= 5 or LEFT_RANGE[0] <= f <= LEFT_RANGE[1]}
        return per, thin
    raw = pickle.loads((OUT / "raw.pkl").read_bytes())
    thin = pickle.loads((OUT / "thin.pkl").read_bytes()) if (OUT / "thin.pkl").exists() else {}
    thin = {f: m for f, m in thin.items() if abs(f - 223) <= 5 or abs(f - 274) <= 5 or LEFT_RANGE[0] <= f <= LEFT_RANGE[1]}
    per = {c: {} for c in CLASSES + HELPERS}
    for c in CLASSES + HELPERS:
        for f in range(F0, F1):
            own = next(ci for ci in range(len(CHUNKS)) if SPLITS[ci] <= f < SPLITS[ci + 1])
            for ci in [own] + [k for k in sorted(raw) if k != own]:
                m = raw.get(ci, {}).get(c, {}).get(f)
                if m is not None and m.sum() > 0:
                    per[c][f] = m
                    break
    return per, thin


def exclusive(f, per, thin, P):
    z = np.zeros((H, W), bool)
    Pc = dil(P[f], 3)
    _, l = luma01(f)
    m = {}
    for c in CLASSES:
        r = per[c].get(f, z)
        if c == "FINS":
            t = thin.get(f)
            if t is not None and not (LEFT_RANGE[0] <= f <= LEFT_RANGE[1]):   # 2x-crop blade: only next to the fin already found
                lp = left_fin_piece(per, f)
                t = t & dil(lp, 10) if lp is not None else None
            r = (r | (t if t is not None else z)) & dil(P[f], 8)   # fins are never refined by darkness: flat fins look light
        else:
            r = r & Pc
        if c == "HEAD":                                     # hair and mask are the darkest things on the head
            r = r & (l < 0.36)
        m[c] = r
    helpers = (per["WATCH"].get(f, z) | per["WAIST"].get(f, z))
    if helpers.any():                                       # watch and the white waist piece stay classless
        for c in ("SKIN", "SHORTS", "HEAD"):
            m[c] = m[c] & ~dil(helpers, 1)
    taken, out = z.copy(), {}
    for c in PRIORITY:
        out[c] = clean_part(close3(m[c])) & ~taken
        taken |= out[c]
    return {c: clean_part(out[c]) for c in PRIORITY}


def flow_warp(f_from, f_to, mask):
    """Predict `mask` (defined on f_to) on frame f_from with Farneback optical flow."""
    g0 = cv2.imread(str(FRAMES / f"{f_from:05d}.jpg"), 0)
    g1 = cv2.imread(str(FRAMES / f"{f_to:05d}.jpg"), 0)
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(mask.astype(np.float32), gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def flow_field(f_from, f_to):
    """Farneback flow on half-resolution frames, resized back (x2)."""
    g0 = cv2.resize(cv2.imread(str(FRAMES / f"{f_from:05d}.jpg"), 0), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    g1 = cv2.resize(cv2.imread(str(FRAMES / f"{f_to:05d}.jpg"), 0), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    return cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR) * 2.0


_GRID = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def remap_mask(fl, mask):
    return cv2.remap(mask.astype(np.float32), _GRID[0] + fl[..., 0], _GRID[1] + fl[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def flow_median(per, classes, r=2):
    """Symmetric (2r+1)-frame temporal median of the given classes, neighbours
    aligned with optical flow (no lag, single-frame jumps vanish)."""
    out = {c: {} for c in classes}
    for f in range(F0, F1):
        ks = [k for k in range(-r, r + 1) if F0 <= f + k < F1]
        fls = {k: flow_field(f, f + k) for k in ks if k != 0}
        need = (len(ks) + 1) // 2
        for c in classes:
            v = per[c][f].astype(np.float32)
            for k in ks:
                if k != 0:
                    v = v + remap_mask(fls[k], per[c][f + k])
            out[c][f] = v >= need - 1e-3 + (0 if len(ks) % 2 else 0.5)
    return out


def class_alphas(hard):
    """Soft, priority-composited class alphas. SKIN: 1 px in, 2-3 px feather.
    FINS / SHORTS / HEAD: 1 px out, 2 px feather. The feather never reaches more
    than 3 px outside the class."""
    out, remaining = {}, np.ones((H, W), np.float32)
    for c in PRIORITY:
        h = hard[c]
        if c == "SKIN":
            core, sig = ero(h, 1), 0.9
        else:
            core, sig = dil(h, 1), 0.6
        a = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sig) * dil(h, 3)
        a = np.clip(a, 0, 1) * remaining
        remaining = remaining - a
        out[c] = a
    return out


def stage_finish(a):
    P = envelope()
    per, thin = load_raw()
    log = lambda s: print(s, flush=True)
    for c in CLASSES:
        for f in range(F0, F1):
            per[c].setdefault(f, np.zeros((H, W), bool))
    pm = flow_median(per, ("SKIN", "FINS", "HEAD"))
    for f in range(F0, F1):                                  # SHORTS from the smoothed classes, then its own median
        per["SHORTS"][f] = derive_shorts_frame(f, pm, P)
    pm.update(flow_median(per, ("SHORTS",), r=3))
    pm.update({h: per[h] for h in HELPERS})
    hard = {f: exclusive(f, pm, thin, P) for f in range(F0, F1)}
    area = lambda f, c: int(hard[f][c].sum())
    # single-frame jumps (and a vanished class) are re-predicted from both neighbours by optical flow
    fixed = []
    for it in range(2):
        for f in range(F0 + 1, F1 - 1):
            for c in CLASSES:
                ap_, an_, ac = area(f - 1, c), area(f + 1, c), area(f, c)
                ref = 0.5 * (ap_ + an_)
                if ref < 300:
                    continue
                if abs(ac - ref) / ref > JUMP and abs(ap_ - an_) / ref < JUMP:
                    w = 0.5 * (flow_warp(f, f - 1, hard[f - 1][c]) + flow_warp(f, f + 1, hard[f + 1][c]))
                    new = w > 0.5
                    cand = dict(hard[f])
                    cand[c] = new
                    z = np.zeros((H, W), bool)
                    taken = z.copy()
                    for cc in PRIORITY:
                        cand[cc] = clean_part(cand[cc] & ~taken)
                        taken |= cand[cc]
                    log(f"  [flow] f{f} {c}: {ac} -> {int(cand[c].sum())} (neighbours {ap_}/{an_})")
                    hard[f] = cand
                    fixed.append((f, c))
    # numeric table
    lines = ["frame," + ",".join(CLASSES) + ",flags"]
    flags = []
    overlap = 0
    for f in range(F0, F1):
        fl = []
        for c in CLASSES:
            if area(f, c) == 0 and c == "FINS":
                fl.append("FINS=0")
            if f > F0:
                p_, q_ = area(f - 1, c), area(f, c)
                if max(p_, q_) > 0 and abs(q_ - p_) / max(p_, 1) > JUMP:
                    fl.append(f"{c} {100 * (q_ - p_) / max(p_, 1):+.0f}%")
        overlap += int((sum(hard[f][c].astype(np.uint8) for c in CLASSES) > 1).sum())
        lines.append(f"{f}," + ",".join(str(area(f, c)) for c in CLASSES) + "," + ";".join(fl))
        if fl:
            flags.append((f, fl))
    (OUT / "class_areas.csv").write_text("\n".join(lines) + "\n")
    log(f"[finish] overlap pixels {overlap}; flow fixes {fixed}; flagged frames {len(flags)}: " +
        "; ".join(f"f{f} {','.join(v)}" for f, v in flags[:40]))
    # alphas + hard masks to disk
    for c in CLASSES:
        (OUT / "alpha" / c).mkdir(parents=True, exist_ok=True)
        (OUT / "hard" / c).mkdir(parents=True, exist_ok=True)
    for f in range(F0, F1):
        al = class_alphas(hard[f])
        for c in CLASSES:
            cv2.imwrite(str(OUT / "alpha" / c / f"{f:05d}.png"), (al[c] * 255 + 0.5).astype(np.uint8))
            cv2.imwrite(str(OUT / "hard" / c / f"{f:05d}.png"), hard[f][c].astype(np.uint8) * 255)
    return 0


# ------------------------------------------------------------------ grade ---
SHEET_FRAMES = [210, 223, 248, 274, 289, 299]


def lab_stats(img_u8, sel):
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    px = lab[sel].astype(np.float32)
    return float(np.median(px[:, 0]) * 100 / 255), float(np.median(px[:, 2]) - 128)


def ring_delta(out_f, env, centre=None, radius=130, far=(30, 60)):
    """Median L*/b* of the 1-5 px ring outside the diver against the `far` px ring."""
    u8 = (np.clip(out_f, 0, 1) * 255 + 0.5).astype(np.uint8)
    d = cv2.distanceTransform((~env).astype(np.uint8), cv2.DIST_L2, 5)
    r1, r2 = (d >= 1) & (d <= 5), (d >= far[0]) & (d <= far[1])
    if centre is not None:
        yy, xx = np.ogrid[:H, :W]
        near = (yy - centre[1]) ** 2 + (xx - centre[0]) ** 2 <= radius ** 2
        r1, r2 = r1 & near, r2 & near
    if r1.sum() < 30 or r2.sum() < 30:
        return None
    (l1, b1), (l2, b2) = lab_stats(u8, r1), lab_stats(u8, r2)
    return round(l1 - l2, 2), round(b1 - b2, 2)


def anchors(hard, f):
    """Local check places: right fin tip, knee (leftmost skin), hands (lowest skin)."""
    out = {}
    sk, fn = hard["SKIN"], hard["FINS"]
    if sk.any():
        ys, xs = np.nonzero(sk)
        cx, cy = xs.mean(), ys.mean()
        i = np.argmin(xs)
        out["knee"] = (int(xs[i]), int(ys[i]))
        low = ys > np.quantile(ys, 0.97)
        out["hand"] = (int(xs[low].mean()), int(ys[low].mean()))
        if fn.any():
            fy, fx = np.nonzero(fn)
            right = fx > cx
            if right.any():
                d = np.hypot(fx[right] - cx, fy[right] - cy)
                k = np.argmax(d)
                out["right_fin_tip"] = (int(fx[right][k]), int(fy[right][k]))
    return out


def harmonic_fill(a, hole, scale=4, iters=700):
    """Fill `hole` with the smooth (Laplace) continuation of the values around it,
    computed at 1/scale resolution and upsampled: the water near the diver then
    joins the water 30 px away without a step."""
    import blocks_grade as B
    h, w = a.shape
    sm = cv2.resize(a, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    hm = cv2.resize(hole.astype(np.uint8), (w // scale, h // scale), interpolation=cv2.INTER_NEAREST) > 0
    cur = cv2.resize(B.plate_fill(a, hole), (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    cur[~hm] = sm[~hm]
    k = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], np.float32)
    for _ in range(iters):
        nb = cv2.filter2D(cur, -1, k, borderType=cv2.BORDER_REFLECT)
        cur[hm] = nb[hm]
    out = cv2.resize(cur, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.where(hole, out, a)


def grade_classes(task):
    import blocks_grade as B
    from reel_match import luma
    f, x, hard, al, y_hi, dist = task
    z = np.zeros((H, W), bool)
    env = np.zeros((H, W), bool)
    for c in CLASSES:
        env |= hard[c]
    support = dil(env, 6)
    # decontaminated water plate: the band around the diver is replaced by the surrounding water
    xb = x.copy()
    if support.any():
        # (1) the diver and the 6 px around him are rebuilt from the water around that band
        dd = cv2.distanceTransform((~env).astype(np.uint8), cv2.DIST_L2, 5)
        near = dil(env, 6)
        for ch in range(3):
            xb[..., ch] = np.where(near, harmonic_fill(x[..., ch], near), x[..., ch])
        # (2) radial levelling: the source has a faint halo (dark or bright) around him that
        # the 3x contrast expansion would turn into a band. Water brightness at 1..33 px is
        # brought to the level of the 30-60 px ring (1 at 34 px); per-frame, local, not exposure
        from reel_match import rgb2hsv
        far = (dd >= 30) & (dd <= 60)
        if far.sum() > 500:
            ramp = np.clip((34.0 - np.arange(36)) / 14.0, 0, 1)
            kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
            kern /= kern.sum()
            idx = np.clip(dd, 0, 35).astype(np.int32)
            for ch in range(3):                         # per channel: level and hue follow the far water
                vch = xb[..., ch]
                vfar = float(np.median(vch[far]))
                vk = np.array([np.median(vch[(dd >= k) & (dd < k + 1)]) if ((dd >= k) & (dd < k + 1)).sum() > 30 else np.nan
                               for k in range(36)], np.float32)
                ok = ~np.isnan(vk)
                if ok.sum() <= 8:
                    continue
                vk = np.interp(np.arange(36), np.arange(36)[ok], vk[ok])
                vk = np.convolve(np.pad(vk, 6, mode='edge'), kern, 'valid').astype(np.float32)
                gains = np.clip(1 + (vfar / np.maximum(vk, 1e-3) - 1) * ramp, 0.9, 1.1).astype(np.float32)
                g = gains[idx]
                g[env] = 1.0
                xb[..., ch] = np.clip(vch * g, 0, 1)
    hole = dil(env, 15)
    bg = B.grade_background(xb, hole=None, grain=False)           # xb is already diver-free: one consistent plate
    bg_raw = B.grade_background(x, hole=hole, grain=False)         # ambient grade of the diver's own pixels
    ring = dil(env, 40) & ~dil(env, 25)
    plate = np.median(bg[ring], 0) if ring.sum() > 50 else np.median(bg.reshape(-1, 3), 0)
    cols = {
        "SKIN": B.grade_diver(x, y_hi, dist, plate, mask=hard["SKIN"], force="skin"),
        "SHORTS": B.grade_diver(x, y_hi, dist, plate, mask=hard["SHORTS"], force="gear"),
        "HEAD": B.grade_diver(x, y_hi, dist, plate, mask=hard["HEAD"], force="gear"),
        "FINS": bg_raw,                                               # no warm skin correction on fins
    }
    out = bg.copy()
    tot = np.zeros((H, W), np.float32)
    for c in CLASSES:
        col = B.fill_rim(cols[c], hard[c], 2 if c != "SKIN" else 3) if hard[c].any() else cols[c]
        out = out * (1 - al[c][..., None]) + col * al[c][..., None]
        tot += al[c]
    out = B.add_grain(out, B.BG["grain"], np.random.default_rng(2000 + f))
    out = np.clip(out, 0, 1).astype(np.float32)
    # numeric checks on the graded frame
    fins = hard["FINS"]
    hot = float((out[fins][:, 0] > out[fins][:, 2]).mean()) if fins.any() else float("nan")
    rec = {"fins_hot": hot, "ring": ring_delta(out, env), "local": {}, "step": {}}
    for k, pt in anchors(hard, f).items():
        rec["local"][k] = ring_delta(out, env, pt)
        rec["step"][k] = ring_delta(out, env, pt, radius=70, far=(6, 12))     # fringe / band next to the edge
    return f, out, rec


def stage_grade(a):
    import multiprocessing as mp
    import dataclasses
    import blocks_grade as B
    from shades.config import Encode
    from shades.io import Encoder, iter_frames, probe
    from reel_match import luma
    info = probe(B.VIDEO)
    P = envelope()
    hard = {f: {c: cv2.imread(str(OUT / "hard" / c / f"{f:05d}.png"), 0) > 127 for c in CLASSES} for f in range(F0, F1)}
    alpha = lambda f: {c: cv2.imread(str(OUT / "alpha" / c / f"{f:05d}.png"), 0).astype(np.float32) / 255 for c in CLASSES}
    raws = {}
    for k, x in enumerate(iter_frames(B.VIDEO, info)):
        if k >= F1:
            break
        if k >= F0:
            raws[k] = x
    # his lit-skin level and size, smoothed over the clip like in the full render
    idx = list(range(F0, F1))
    yh = []
    for f in idx:
        m = ero(hard[f]["SKIN"], 2)
        yh.append(float(np.percentile(luma(np.clip(raws[f], 0, 1))[m], 90)) if m.sum() > 200 else np.nan)
    yh = np.array(yh); ok = ~np.isnan(yh)
    yh = B.smooth1d(np.interp(idx, np.array(idx)[ok], yh[ok]), 6.0, 18)
    size = np.array([B.diver_size(((P[f]) * 255).astype(np.uint8)) for f in idx])
    size = B.smooth1d(size, 8.0, 24)
    dist = np.clip((B.DV["size_near"] - size) / (B.DV["size_near"] - B.DV["size_far"]), 0, 1)
    tasks = [(f, raws[f], hard[f], alpha(f), float(yh[k]), float(dist[k])) for k, f in enumerate(idx)]
    out_mp4 = OUT / "classes_preview_210_299.mp4"
    enc = Encoder(B.VIDEO, out_mp4, dataclasses.replace(info, has_audio=False), Encode(crf=16, preset="slow"))
    pool = mp.get_context("fork").Pool(a.workers)
    metrics, keep = {}, {}
    for f, frame, rec in pool.imap(grade_classes, tasks, chunksize=1):
        enc.write(frame)
        metrics[f] = rec
        if f in SHEET_FRAMES:
            keep[f] = frame
    pool.close(); pool.join(); enc.close()
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    # ---- numeric report
    hot = {f: m["fins_hot"] for f, m in metrics.items()}
    bad_hot = {f: round(v * 100, 2) for f, v in hot.items() if v == v and v >= 0.01}
    bad_ring = {f: m["ring"] for f, m in metrics.items() if m["ring"] and (abs(m["ring"][0]) >= 2 or abs(m["ring"][1]) >= 2)}
    print(f"[grade] FINS warm (R>B) pixel share: max {100 * np.nanmax(list(hot.values())):.2f}% ; frames >= 1%: {bad_hot}")
    r = [m["ring"] for m in metrics.values() if m["ring"]]
    print(f"[grade] ring dL*/db* (1-5 px vs 30-60 px): max |dL*| {max(abs(v[0]) for v in r):.2f}, "
          f"max |db*| {max(abs(v[1]) for v in r):.2f}; frames out of tolerance: {bad_ring}")
    for place, rng in (("right_fin_tip", range(286, 296)), ("knee", range(210, 224)), ("hand", [210, 274])):
        v = [metrics[f]["local"].get(place) for f in rng if f in metrics and metrics[f]["local"].get(place)]
        w = [metrics[f]["step"].get(place) for f in rng if f in metrics and metrics[f]["step"].get(place)]
        if v:
            print(f"[grade] local {place}: vs 30-60 px ring (130 px circle, includes the real left-right water gradient): "
                  f"max |dL*| {max(abs(q[0]) for q in v):.2f}, max |db*| {max(abs(q[1]) for q in v):.2f}; "
                  f"vs 6-12 px ring (r=70, band/fringe test): max |dL*| {max(abs(q[0]) for q in w):.2f}, max |db*| {max(abs(q[1]) for q in w):.2f} over {len(v)} frames")
    # ---- the one contact sheet: class overlay + graded, 4 x 3 panels of 400 px
    panels = []
    for f in SHEET_FRAMES:
        x = raws[f]
        ov = (x * 255).astype(np.float32)
        for c in CLASSES:
            m = hard[f][c]
            col = np.array(COLOR[c], np.float32)
            ov[m] = 0.45 * ov[m] + 0.55 * col
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnt, -1, tuple(float(v) for v in col), 1)
        ys, xs = np.nonzero(P[f])
        cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) // 2 + 40
        x0, y0 = int(np.clip(cx - half, 0, W - 2 * half)), int(np.clip(cy - half, 0, H - 2 * half))
        for img, tag in ((ov, "classes"), (keep[f] * 255, "graded")):
            t = np.clip(img[y0:y0 + 2 * half, x0:x0 + 2 * half], 0, 255).astype(np.uint8)
            t = cv2.resize(t, (400, 400), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"f{f} {tag}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panels.append(t)
    rows = [np.concatenate(panels[i:i + 4], 1) for i in range(0, 12, 4)]
    sheet = np.concatenate(rows, 0)
    for k, c in enumerate(PRIORITY):
        cv2.rectangle(sheet, (1500 - 90, 10 + 22 * k), (1500 - 70, 28 + 22 * k), COLOR[c], -1)
        cv2.putText(sheet, c, (1500 - 62, 26 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / "contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[grade] contact sheet {OUT / 'contact_sheet.jpg'}; graded preview {out_mp4}")
    return 0


# --------------------------------------------------------------- assemble ---
def warp_soft(m, g, f):
    """Carry mask m from frame g to f; every step but the last is thresholded."""
    cur, step = m, (1 if f > g else -1)
    ks = list(range(g, f, step))
    for i, k in enumerate(ks):
        w = flow_warp(k + step, k, cur)
        cur = w if i == len(ks) - 1 else w > 0.5
    return cur.astype(np.float32)


def fill_user_points(per, P):
    """The left-fin anchors must be FINS and not SKIN; where the masks miss a
    point, the small colour-coherent piece of the silhouette around it is added
    to FINS (and taken out of SKIN)."""
    log = []
    pts = dict(LEFT_FIN)
    for f in RIGHT_POCKET_FRAMES:
        pt = right_pocket_point(f, P)
        if pt:
            pts.setdefault(f, pt)
    for f, (x, y) in sorted(pts.items()):
        fins = per["FINS"].get(f, np.zeros((H, W), bool))
        near = fins[max(0, y - 2):y + 3, max(0, x - 2):x + 3].any()
        if not near:
            im, _ = luma01(f)
            lab = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            c = np.median(lab[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3), 0)
            d = np.linalg.norm(lab - c, axis=2)
            cand = (d < 14) & P[f] & (np.hypot(*np.meshgrid(np.arange(W) - x, np.arange(H) - y)) < 45)
            n, lb, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
            k = lb[y, x]
            if k > 0:
                add = lb == k
                per["FINS"][f] = fins | add
                per["SKIN"][f] = per["SKIN"].get(f, np.zeros((H, W), bool)) & ~add
                log.append(f"f{f}: point {(x, y)} was outside FINS, added {int(add.sum())} px")
        sk = per["SKIN"].get(f, np.zeros((H, W), bool))
        if sk[y, x]:
            per["SKIN"][f] = sk & ~dil(per["FINS"].get(f, np.zeros((H, W), bool)), 0)
    return log


def derive_shorts_frame(f, per, P):
    _, l = luma01(f)
    used = np.zeros((H, W), bool)
    for c in ("SKIN", "FINS", "HEAD"):
        used |= per[c].get(f, np.zeros((H, W), bool))
    rem = P[f] & ~dil(used, 2) & (l < 0.52)
    rem = cv2.morphologyEx(rem.astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(rem.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((H, W), bool)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = lab == big
    near_big = dil(keep, 8)
    for k in range(1, n):                      # other dark pieces count only if big or touching it
        if k != big and (st[k, cv2.CC_STAT_AREA] >= 300 or (near_big & (lab == k)).any()):
            keep |= lab == k
    return clean_part(keep)


def stage_assemble(a):
    det = pickle.loads((OUT / "det.pkl").read_bytes())
    P = envelope()
    log = lambda s: print(s, flush=True)
    frames = list(range(F0, F1))
    per = {c: {} for c in CLASSES + HELPERS}
    src = {c: {} for c in CLASSES}
    for c in ("SKIN", "FINS", "HEAD"):
        for f in frames:
            for ph in PHRASES[c]:
                m = pick_class(c, get_det(det, f, ph), P[f])
                if m is not None:
                    per[c][f] = m
                    src[c][f] = "text"
                    break
    for c in ("SKIN", "FINS", "HEAD"):                    # fill the frames the detector missed from both sides
        have = sorted(per[c])
        for f in frames:
            if f in per[c]:
                continue
            before = [g for g in have if g < f and f - g <= 6]
            after = [g for g in have if g > f and g - f <= 6]
            parts = []
            if before:
                parts.append(warp_soft(per[c][before[-1]], before[-1], f))
            if after:
                parts.append(warp_soft(per[c][after[0]], after[0], f))
            if parts:
                per[c][f] = np.mean(parts, 0) > 0.5
                src[c][f] = "flow " + ("both" if len(parts) == 2 else "one side")
    miss = {c: [f for f in frames if f not in per[c]] for c in ("SKIN", "FINS", "HEAD")}
    log("[assemble] text detections: " + ", ".join(f"{c} {sum(1 for v in src[c].values() if v == 'text')}/90" for c in src) +
        "; still missing: " + str(miss))
    for f in frames:
        per["SHORTS"][f] = derive_shorts_frame(f, per, P)
    per["SKIN"] = {f: per["SKIN"].get(f, np.zeros((H, W), bool)) for f in frames}
    for line in fill_user_points(per, P):
        log("  [user point] " + line)
    (OUT / "per.pkl").write_bytes(pickle.dumps(per))
    return 0


# ------------------------------------------------------------------ hand ---
HAND_RANGE = (242, 272)           # session frames; the user's prompt frames are f248 and f265
HAND_PROMPTS = [248, 265]


def hand_tip_points(f, P, hard):
    """Positive points on the hand (bright, person pixels no class owns, next to the
    lowest extremity of the body) plus the skin right next to them."""
    import diver_classes as D
    avoid = hard["FINS"] | hard["SHORTS"] | hard["HEAD"]
    tips = D.hand_tips(P[f], avoid, n=1)
    x, y = tips[0]
    yy, xx = np.ogrid[:H, :W]
    disk = (yy - y) ** 2 + (xx - x) ** 2 <= 60 ** 2
    _, l = luma01(f)
    free = disk & P[f] & ~(hard["SKIN"] | avoid) & (l > 0.5)
    pts = fps_points(free, 4, min_dt=2.0) + fps_points(disk & hard["SKIN"] & dil(free, 20), 2, min_dt=3.0)
    neg = fps_points(hard["HEAD"], 1, min_dt=2.0) + fps_points(dil(disk & P[f], 12) & ~dil(P[f], 3), 3, min_dt=1.5)
    return (x, y), pts, neg


def stage_hand(a):
    P = envelope()
    sam = Sam()
    cs, ce = HAND_RANGE
    sid = sam.start(list(range(cs, ce)), "hand")
    log = lambda s: print(s, flush=True)
    for f in HAND_PROMPTS:
        hard = {c: cv2.imread(str(OUT / "hard" / c / f"{f:05d}.png"), 0) > 127 for c in CLASSES}
        tip, pos, neg = hand_tip_points(f, P, hard)
        log(f"[hand] f{f}: tip {tip}, {len(pos)} positives {pos}, {len(neg)} negatives")
        pts = [(x, y, 1) for x, y in pos] + [(x, y, 0) for x, y in neg]
        sam.points(sid, f - cs, "HAND", [(x, y) for x, y, _ in pts], [l for *_, l in pts])
    res = sam.propagate(sid, cs, (cs + ce) // 2 - cs)["HAND"]
    out = {}
    for f, m in res.items():
        out[f] = m & dil(P[f], 2)
    (OUT / "hand.pkl").write_bytes(pickle.dumps(out))
    log("[hand] areas " + " ".join(f"f{f}:{int(out[f].sum())}" for f in range(cs, ce) if f in out))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--frames", nargs="*", default=None)
    d.add_argument("--texts", nargs="*", default=None)
    d.set_defaults(fn=stage_detect)
    sub.add_parser("thin").set_defaults(fn=stage_thin)
    sub.add_parser("assemble").set_defaults(fn=stage_assemble)
    sub.add_parser("hand").set_defaults(fn=stage_hand)
    sub.add_parser("finish").set_defaults(fn=stage_finish)
    g = sub.add_parser("grade")
    g.add_argument("--workers", type=int, default=3)
    g.set_defaults(fn=stage_grade)
    t = sub.add_parser("track")
    t.add_argument("--chunks", nargs="*", default=None)
    t.set_defaults(fn=stage_track)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
