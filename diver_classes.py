#!/usr/bin/env python3
"""Four SAM 3.1 classes for the diver, frames 0-89: SKIN, SHORTS, FINS, MASK.

Each class is its own SAM 3.1 tracker object (own id, own text prompt):
    SKIN   "bare skin of the diver"      (lowest priority)
    SHORTS "white swim shorts"
    FINS   "swim fins"                   (both fins, one class)
    MASK   "diving mask and snorkel"     (highest priority)
Two helper objects (not output classes) keep things out of SKIN:
    HAIR   "hair"      WATCH  "wrist watch"

Priority where classes overlap: MASK > FINS > SHORTS > SKIN, so no pixel is in
two classes.

Pass 1 (per chunk): the text prompts detect each class on candidate frames,
the detections plus the user's point prompts (shorts/fins anchors, f0, f12,
f15, f20, f26, f30) seed one tracker session that is propagated both ways.
Pass 2 (per chunk): a fresh session replays every pass-1 prompt and adds the
corrections that need a mask to correct: negative points on foam / surface
reflections (f0-12, f43-50, f56-78), the far hand and fingertips (f66-74),
hair kept out of SKIN (f78-89), the mid-clip re-anchor at f26 and f30.
SAM 3.1 runs through the CPU port (shades/sam31_cpu.py).

    python diver_classes.py --input inputs/diver.mov
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
from shades.masking import _build_predictor, extract_jpeg_frames

REPO = Path(__file__).resolve().parent
CLASSES = ["SKIN", "SHORTS", "FINS", "MASK"]
PRIORITY = ["MASK", "FINS", "SHORTS", "SKIN"]          # first wins
HELPERS = ["HAIR", "WATCH"]
OID = {c: 101 + k for k, c in enumerate(CLASSES + HELPERS)}
TEXT = {"SKIN": "bare skin of the diver", "SHORTS": "white swim shorts",
        "FINS": "swim fins", "MASK": "diving mask and snorkel",
        "HAIR": "hair", "WATCH": "wrist watch"}
COLOR = {"SKIN": (255, 170, 60), "SHORTS": (60, 120, 255),
         "FINS": (255, 60, 255), "MASK": (60, 255, 60)}   # RGB
N = 90
CHUNKS = [(0, 48), (42, 90)]
SPLIT = 45                          # frames < SPLIT come from chunk 1

# frames on which the detector looks for each class, per chunk
DETECT = [
    {"SHORTS": [0, 15, 26, 30], "FINS": [0, 12, 20, 26, 30], "MASK": [0, 12, 26, 30],
     "SKIN": [0, 26, 30], "HAIR": [12, 30], "WATCH": [12, 26, 30]},
    {"SHORTS": [46, 66, 84], "FINS": [46, 84, 89], "MASK": [66, 78, 89],
     "SKIN": [46, 66, 84], "HAIR": [78, 89], "WATCH": [66]},
]

# user point prompts: frame -> class -> [(x, y, label)]  (label 1 pos, 0 neg)
USER = {
    0: {"SHORTS": [(660, 765, 1)],
        "SKIN": [(660, 765, 0), (815, 795, 0), (875, 765, 0)],
        "FINS": [(815, 795, 1), (875, 765, 1)]},
    12: {"FINS": [(890, 775, 1), (1010, 760, 1)],
         "SKIN": [(890, 775, 0), (1010, 760, 0)]},
    15: {"FINS": [(918, 732, 1), (1050, 725, 1)],
         "SKIN": [(918, 732, 0), (1050, 725, 0)]},
    20: {"FINS": [(964, 630, 1), (1108, 640, 1)],
         "SKIN": [(964, 630, 0), (1108, 640, 0)]},
    89: {"MASK": [(822, 470, 1)]},
}
# negative points for EVERY class: foam / reflection above the back (D) and
# the surface reflection on the top edge (C)
NEG_ALL = {
    0: [(520, 640), (620, 640), (720, 640), (820, 650)],
    12: [(700, 610), (800, 610), (900, 615)],
    43: [(650, 18), (780, 22), (900, 18), (1000, 20)],
    46: [(650, 18), (780, 22), (900, 18), (1000, 20)],
    50: [(650, 18), (780, 22), (900, 18), (1000, 20)],
}
FOAM_FRAMES = [6] + list(range(56, 79, 4))              # D (f6) and H
HAND_FRAMES = [66, 68, 70, 72, 74]                      # F
HAIR_FRAMES = [78, 82, 86, 89]                          # G
REFINE = sorted(set([0, 6, 12, 15, 20, 26, 30, 43, 46, 50] + FOAM_FRAMES
                    + HAND_FRAMES + HAIR_FRAMES))

POST = dict(min_part=150, max_hole=100, jump=0.08, erode=1, feather=2)


# ----------------------------------------------------------------- utils --
def read_mask(d: Path, i: int, shape=(1078, 1232)):
    m = cv2.imread(str(d / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
    return np.zeros(shape, bool) if m is None else m > 127


def dil(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def interior_point(m):
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return int(x), int(y)


def foam_points(img_bgr, person, n=3):
    """Bright foam / splash blobs just outside the diver, near the water line."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    ys, xs = np.nonzero(person)
    if len(ys) == 0:
        return []
    top = ys.min()
    band = np.zeros_like(person)
    band[max(0, top - 140):top + 90, :] = True
    cand = (g > 0.62) & band & ~dil(person, 10)
    num, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    order = sorted(range(1, num), key=lambda k: -st[k, cv2.CC_STAT_AREA])
    out = []
    for k in order:
        if st[k, cv2.CC_STAT_AREA] < 25 or len(out) >= n:
            break
        out.append((int(cen[k][0]), int(cen[k][1])))
    return out


def hand_tips(person, avoid, n=2):
    """Hull extremities of the diver below its centroid that are not fins or
    shorts: the hands. Returns points pulled 12 px inside the mask."""
    m = person & ~dil(avoid, 14)
    ys, xs = np.nonzero(m)
    if len(ys) < 200:
        return []
    cx, cy = xs.mean(), ys.mean()
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c[:, 0, :] for c in cnts if len(c) > 30]) if cnts else np.zeros((0, 2))
    pts = pts[pts[:, 1] > cy]
    if len(pts) == 0:
        return []
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    tips = []
    for k in np.argsort(-d):
        p = pts[k]
        if all(np.hypot(*(p - q)) > 150 for q in tips):
            tips.append(p)
        if len(tips) == n:
            break
    out = []
    for p in tips:
        v = np.array([cx, cy]) - p
        v = v / (np.linalg.norm(v) + 1e-6)
        q = p + v * 12
        out.append((int(q[0]), int(q[1])))
    return out


# ------------------------------------------------------------------ SAM ---
class Sam:
    def __init__(self, frames_dir: Path, cache: Path):
        self.frames, self.cache = frames_dir, cache
        self.pred, on_gpu = _build_predictor(Masking(parts={}), REPO, cache)

        def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
            o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
            o.update(refined_obj_id_to_mask or {})
            return o

        self.pred.model._build_sam2_output = _build_sam2_output
        print(f"[cls] SAM 3.1 on {'CUDA' if on_gpu else 'CPU (port)'}", flush=True)

    def start(self, cs, ce, tag):
        cdir = self.cache / f"sam_chunk_cls_{tag}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(self.frames / f"{i:05d}.jpg")
        return self.pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]

    def text(self, sid, cs, f, text):
        r = self.pred.handle_request({"type": "add_prompt", "session_id": sid,
                                      "frame_index": f - cs, "text": text})
        return [np.asarray(m) > 0 for m in r["outputs"]["out_binary_masks"]]

    def points(self, sid, cs, f, cls, pts, lab, w=1232, h=1078):
        norm = [[x / w, y / h] for x, y in pts]
        self.pred.handle_request({"type": "add_prompt", "session_id": sid,
                                  "frame_index": f - cs, "points": norm,
                                  "point_labels": [int(v) for v in lab], "obj_id": OID[cls]})

    def propagate(self, sid, cs, start):
        got = {c: {} for c in CLASSES + HELPERS}
        inv = {v: k for k, v in OID.items()}
        for resp in self.pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "both", "start_frame_index": start - cs}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                c = inv.get(int(oid))
                if c:
                    got[c][fi] = np.asarray(mm) > 0
        self.pred.handle_request({"type": "close_session", "session_id": sid})
        return got


def pick(cls, f, masks, person, user_pts):
    """Choose the detection that is the class (not the whole diver, not a
    reflection)."""
    pa = max(int(person.sum()), 1)
    lim = {"SKIN": (3000, 0.95 * pa), "SHORTS": (500, 0.45 * pa), "FINS": (250, 0.32 * pa),
           "MASK": (40, 0.07 * pa), "HAIR": (40, 0.08 * pa), "WATCH": (15, 0.012 * pa)}[cls]
    near = dil(person, 25 if cls == "FINS" else 4)
    cand = []
    for m in masks:
        a = int(m.sum())
        if a < lim[0] or a > lim[1] or (m & near).sum() / max(a, 1) < 0.8:
            continue
        cand.append(m)
    if not cand:
        return None
    pos = [(x, y) for x, y, l in user_pts if l == 1]
    if cls == "FINS":                       # both fins are one class
        cand = sorted(cand, key=lambda m: -m.sum())[:2]
        return np.any(np.stack(cand), 0)
    if pos:
        hit = [m for m in cand if any(m[y, x] for x, y in pos)]
        if hit:
            return max(hit, key=lambda m: m.sum())
    return max(cand, key=lambda m: m.sum())


# ----------------------------------------------------------------- chunk --
CHUNKS = [(0, 30), (24, 54), (48, 78), (72, 90)]
ORDER = [2, 1, 0, 3]            # start where the detector sees every class, carry outward
SPLITS = [0, 27, 51, 75, 90]    # frame ranges owned by each chunk (middle of the overlaps)
# detector attempts per chunk: (class, frame, [phrases tried in order]); the
# user's phrase first, then the one-word phrase that worked on this clip before
ATTEMPT = {
    2: [("SHORTS", 52, ["white swim shorts", "shorts"]), ("SHORTS", 66, ["white swim shorts", "shorts"]),
        ("FINS", 50, ["swim fins", "flipper"]), ("HAIR", 66, ["hair"]), ("HAIR", 56, ["hair"]),
        ("MASK", 66, ["diving mask and snorkel", "snorkel"]), ("MASK", 72, ["diving mask and snorkel", "snorkel"]),
        ("WATCH", 66, ["wrist watch", "watch"])],
    3: [("SHORTS", 84, ["white swim shorts", "shorts"]), ("HAIR", 84, ["hair"]),
        ("MASK", 84, ["diving mask and snorkel", "snorkel"]), ("MASK", 89, ["diving mask and snorkel", "snorkel", "diving mask"]),
        ("FINS", 86, ["swim fins", "flipper"]), ("WATCH", 78, ["wrist watch", "watch"])],
    1: [("SHORTS", 36, ["white swim shorts", "shorts"]), ("SHORTS", 46, ["white swim shorts", "shorts"]),
        ("FINS", 30, ["swim fins", "flipper"]), ("FINS", 40, ["swim fins", "flipper"]),
        ("HAIR", 36, ["hair"]), ("MASK", 46, ["diving mask and snorkel", "snorkel"]),
        ("WATCH", 26, ["wrist watch", "watch"])],
    0: [("FINS", 0, ["swim fins", "flipper"]), ("HAIR", 16, ["hair"]), ("WATCH", 12, ["wrist watch", "watch"]),
        ("SHORTS", 8, ["white swim shorts", "shorts"])],
}
LIM = {"SKIN": (3000, 0.97), "SHORTS": (300, 0.45), "FINS": (200, 0.32), "MASK": (40, 0.06),
       "HAIR": (40, 0.08), "WATCH": (10, 0.012)}
DARK_ONLY = {"FINS": 0.28, "MASK": 0.30, "HAIR": 0.30}      # median raw luma ceiling


def luma01(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255


def valid_mask(cls, m, person, img):
    pa = max(int(person.sum()), 1)
    a = int(m.sum())
    lo, hi = LIM[cls]
    if a < lo or a > hi * pa:
        return False
    if (m & dil(person, 25 if cls == "FINS" else 4)).sum() / a < 0.8:
        return False
    if cls in DARK_ONLY and float(np.median(luma01(img)[m])) > DARK_ONLY[cls]:
        return False
    return True


DARK_THR = {"FINS": 0.32, "MASK": 0.34, "HAIR": 0.34}


def refine_dark(cls, m, img):
    """Fins, mask and hair are the darkest things on the diver: a SAM mask that
    swelled over a bright leg is cut back to its dark pixels."""
    if cls not in DARK_THR:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    r = m & (luma01(img) < DARK_THR[cls])
    return cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def grow_dark(img_bgr, person, pts, thr=0.30):
    """Connected dark pixels (the black fins) containing each user point."""
    g = luma01(img_bgr)
    dark = (g < thr) & dil(person, 25)
    num, cc, _, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    out = np.zeros_like(dark)
    for x, y in pts:
        win = cc[max(0, y - 10):y + 11, max(0, x - 10):x + 11]
        ids = win[win > 0]
        if len(ids):
            out |= cc == np.bincount(ids).argmax()
    return out


def seed_events(m, user=()):
    pts, lab = prompts_from_mask(m, n_pos=5, n_neg=5)
    ev = [(int(p[0] * 1232), int(p[1] * 1078), l) for p, l in zip(pts, lab)]
    return ev + list(user)


def grow_white(img_bgr, person, x, y):
    """Colour region-grow around a user point (shorts on f0): pixels inside the
    diver mask whose Lab colour is close to the point's colour."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    c = np.median(lab[y - 4:y + 5, x - 4:x + 5].reshape(-1, 3), 0)
    d = np.linalg.norm(lab - c, axis=2)
    m = (d < 16) & dil(person, 2)
    num, cc, _, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    k = cc[y, x]
    return (cc == k) if k > 0 else np.zeros_like(m)


def run_chunk(sam: Sam, ci: int, person: dict, imgs: dict, done: dict, log, attempts=None):
    """Detect/seed the classes of chunk ci and track them both ways. `done`
    holds masks already produced on neighbouring chunks: [cls][frame]."""
    cs, ce = CHUNKS[ci]
    sid = sam.start(cs, ce, f"c{ci}")
    seeds: dict[tuple, np.ndarray] = {}
    got_n = {c: 0 for c in CLASSES + HELPERS}
    tried = set()
    t0 = time.time()
    for cls, f, phrases in (ATTEMPT if attempts is None else attempts).get(ci, []):
        if got_n[cls] >= 2 or (cls, f) in tried:
            continue
        tried.add((cls, f))
        for ph in phrases:
            ms = sam.text(sid, cs, f, ph)
            m = pick(cls, f, ms, person[f], USER.get(f, {}).get(cls, []))
            ok = m is not None and valid_mask(cls, m, person[f], imgs[f])
            log(f"  [detect] c{ci} f{f:02d} {cls:6s} '{ph}': {len(ms)} inst -> "
                f"{'none' if m is None else int(m.sum())}{'' if ok or m is None else ' (invalid)'}")
            if ok:
                seeds[(f, cls)] = m
                got_n[cls] += 1
                break
    # carried seeds from neighbouring chunks (their masks on frames of this chunk)
    for cls in CLASSES + HELPERS:
        if got_n[cls]:
            continue
        have = sorted(f for f in done.get(cls, {}) if cs <= f < ce and done[cls][f].sum() > 0)
        for f in (have[:1] + have[-1:]):
            seeds[(f, cls)] = done[cls][f]
            log(f"  [carry ] c{ci} f{f:02d} {cls:6s}: {int(done[cls][f].sum())}")
    # chunk 0: SHORTS from the user's point on f0 when nothing else exists
    if ci == 0 and not any(c == "SHORTS" for _, c in seeds):
        g = grow_white(imgs[0], person[0], 660, 765)
        if g.sum() > 400:
            seeds[(0, "SHORTS")] = g
            log(f"  [user  ] c0 f00 SHORTS: region-grown from (660,765): {int(g.sum())}")
    ev: dict[tuple, list] = {}
    for (f, cls), m in seeds.items():
        ev[(f, cls)] = seed_events(m)
    for f, per in USER.items():                  # the user's fin anchors, grown to the dark blob
        if f in (12, 15, 20) and cs <= f < ce:
            g = grow_dark(imgs[f], person[f], [(x, y) for x, y, _ in per["FINS"]])
            if g.sum() > 800:
                ev[(f, "FINS")] = seed_events(g)
                log(f"  [user  ] c{ci} f{f:02d} FINS: two fins grown from the anchors: {int(g.sum())}")
    if ci == 3 and not any(c == "MASK" for _, c in seeds):
        ev[(89, "MASK")] = [(822, 470, 1)]       # the user's reference spot, f89
        log("  [user  ] c3 f89 MASK: point (822,470)")
    # negative points on foam / surface reflections, only where a class has a positive
    for f in list(range(cs, ce)):
        negs = [(x, y, 0) for x, y in NEG_ALL.get(f, [])]
        if f in FOAM_FRAMES:
            negs += [(x, y, 0) for x, y in foam_points(imgs[f], person[f])]
        if negs:
            for (ff, cls), pl in ev.items():
                if ff == f and cls in CLASSES and any(l == 1 for *_, l in pl):
                    pl.extend(negs)
    prompted = sorted({f for f, _ in ev})
    if not prompted:
        sam.pred.handle_request({"type": "close_session", "session_id": sid})
        log(f"[cls] chunk {ci}: nothing to seed")
        return {c: {} for c in CLASSES + HELPERS}
    start = min(max(prompted[len(prompted) // 2], cs + 1), ce - 2)
    for (f, cls), pl in ev.items():
        sam.points(sid, cs, f, cls, [(x, y) for x, y, _ in pl], [l for *_, l in pl])
    res = sam.propagate(sid, cs, start)
    log(f"[cls] chunk {ci} ({cs}-{ce - 1}) done in {time.time() - t0:.0f}s; start {start}")
    out = {c: {} for c in CLASSES + HELPERS}
    for c in CLASSES + HELPERS:
        for f, m in res[c].items():
            m = refine_dark(c, m, imgs[f])
            if m.sum() > 0 and valid_mask(c, m, person[f], imgs[f]):
                out[c][f] = m
        log(f"   {c:6s} valid {len(out[c]):3d}/{ce - cs}  areas " +
            " ".join(f"{int(res[c][f].sum()) if f in res[c] else -1}" for f in range(cs, ce, 4)))
    return out


def find_watch(img_bgr, person, avoid, tips):
    """Fallback when the detector finds no wrist watch: the strap is a small,
    very dark band next to the hand. Dark connected component (40-900 px)
    inside the diver, outside the other classes, closest to a hand tip."""
    g = luma01(img_bgr)
    cand = person & (g < 0.20) & ~dil(avoid, 6)
    num, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    best, bd = None, 1e9
    for k in range(1, num):
        a = st[k, cv2.CC_STAT_AREA]
        if not 40 <= a <= 900:
            continue
        for t in tips:
            d = float(np.hypot(cen[k][0] - t[0], cen[k][1] - t[1]))
            if d < 170 and d < bd:
                best, bd = k, d
    return (lab == best) if best is not None else np.zeros_like(person)


# ------------------------------------------------------------ postprocess --
def clean(m, p=POST):
    m = m.astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    for k in range(1, num):
        if st[k, cv2.CC_STAT_AREA] < p["min_part"]:
            m[lab == k] = 0
    inv = (1 - m).astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = m.shape
    for k in range(1, num):
        x, y, ww, hh, a = st[k]
        touches = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not touches and a < p["max_hole"]:
            m[lab == k] = 1
    return m > 0


def finish(raw: dict, person: dict, hair: dict, watch: dict, frames):
    """raw[cls][f] bool -> exclusive, cleaned, eroded, feathered masks."""
    out = {c: {} for c in CLASSES}
    for f in frames:
        P = dil(person[f], 3)                   # water line / reflections: nothing above it
        cl = {}
        for c in CLASSES:
            if c == "SKIN":
                oth = np.zeros_like(P)
                for o in ("MASK", "FINS", "SHORTS"):
                    oth |= raw[o].get(f, np.zeros_like(P))
                m = person[f] & ~dil(oth, 1)
            else:
                m = raw[c].get(f, np.zeros_like(P)) & P
            if c == "SKIN":
                hw = hair.get(f, np.zeros_like(P))
                wt = watch.get(f, np.zeros_like(P))
                m = m & ~dil(hw, 1) & ~dil(wt, 3)
            cl[c] = clean(m)
        # fingertips stay in SKIN: hands are extremities of the diver mask
        avoid = cl["FINS"] | cl["SHORTS"] | cl["MASK"] | hair.get(f, np.zeros_like(P))
        for x, y in hand_tips(person[f], avoid):
            disk = np.zeros(P.shape, np.uint8)
            cv2.circle(disk, (x, y), 24, 1, -1)
            add = (disk > 0) & person[f] & ~avoid & ~dil(watch.get(f, np.zeros_like(P)), 3)
            cl["SKIN"] = cl["SKIN"] | add
        # priority: MASK > FINS > SHORTS > SKIN, so no pixel is in two classes;
        # then drop specks / fill holes per class and resolve once more
        for _ in range(2):
            taken = np.zeros_like(P)
            ex = {}
            for c in PRIORITY:
                ex[c] = cl[c] & ~taken
                taken |= ex[c]
            cl = {c: clean(ex[c]) for c in CLASSES}
        taken = np.zeros_like(P)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for c in PRIORITY:
            m = cl[c] & ~taken
            taken |= m
            out[c][f] = cv2.erode(m.astype(np.uint8), k3, iterations=POST["erode"]) > 0
    return out


def soft(binary: dict, f):
    """Feathered alphas that sum to <= 1 per pixel (2 px)."""
    a = {c: cv2.GaussianBlur(binary[c][f].astype(np.float32), (0, 0), 1.0) for c in CLASSES}
    s = sum(a.values())
    k = np.maximum(s, 1.0)
    return {c: np.clip(a[c] / k, 0, 1) for c in CLASSES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--reuse", action="store_true", help="reuse cached SAM output")
    ap.add_argument("--rerun", default="", help="chunks to re-run on top of the cache, e.g. 2,1,0")
    a = ap.parse_args()
    stem = a.input.stem
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    if not (frames / "00000.jpg").exists():
        extract_jpeg_frames(a.input, frames, 2)
    pdir = REPO / "masks" / f"{stem}__fwd" / "subject"
    hdir = REPO / "masks" / f"{stem}__hair" / "subject"
    out_dir = REPO / "masks" / f"{stem}__cls"
    qc = REPO / "output" / "qc_classes"
    qc.mkdir(parents=True, exist_ok=True)
    person = {f: read_mask(pdir, f) for f in range(N)}
    imgs = {f: cv2.imread(str(frames / f"{f:05d}.jpg")) for f in range(N)}
    pkl = out_dir / "sam_raw.pkl"
    log = lambda s: print(s, flush=True)

    if a.rerun and pkl.exists():
        raw, helper = pickle.loads(pkl.read_bytes())
        sam = Sam(frames, cache)
        done = {c: dict(raw[c]) for c in CLASSES}
        done.update({c: dict(helper[c]) for c in HELPERS})
        quick = {2: [("SHORTS", 52, ["shorts"]), ("SHORTS", 66, ["shorts"])], 1: [],
                 0: [("FINS", 0, ["swim fins"]), ("HAIR", 16, ["hair"])]}
        for ci in [int(v) for v in a.rerun.split(",")]:
            res = run_chunk(sam, ci, person, imgs, done, log, attempts=quick)
            lo, hi = SPLITS[ci], SPLITS[ci + 1]
            for c in CLASSES + HELPERS:
                dst = raw if c in CLASSES else helper
                for f in range(lo, hi):
                    dst[c].pop(f, None)
                    done[c].pop(f, None)
                for f, m in res[c].items():
                    if lo <= f < hi or f not in dst[c]:
                        dst[c][f] = m
                        done[c][f] = m
            pkl.write_bytes(pickle.dumps((raw, helper)))
        del sam
    elif a.reuse and pkl.exists():
        raw, helper = pickle.loads(pkl.read_bytes())
    else:
        sam = Sam(frames, cache)
        raw = {c: {} for c in CLASSES}
        helper = {c: {} for c in HELPERS}
        done = {c: {} for c in CLASSES + HELPERS}      # neighbours seed from these
        per_chunk = {}
        for ci in ORDER:
            res = run_chunk(sam, ci, person, imgs, done, log)
            per_chunk[ci] = res
            for c in CLASSES + HELPERS:
                for f, m in res[c].items():
                    done[c].setdefault(f, m)
        out_dir.mkdir(parents=True, exist_ok=True)
        # each chunk owns the middle of its range; fall back to any chunk that
        # produced a valid mask on that frame
        for c in CLASSES + HELPERS:
            dst = raw if c in CLASSES else helper
            dst[c] = {}
            for f in range(N):
                own = next(ci for ci in range(len(CHUNKS)) if SPLITS[ci] <= f < SPLITS[ci + 1])
                for ci in [own] + [k for k in range(len(CHUNKS)) if k != own]:
                    m = per_chunk[ci][c].get(f)
                    if m is not None and m.sum() > 0:
                        dst[c][f] = m
                        break
        pkl.write_bytes(pickle.dumps((raw, helper)))
        del sam

    # helpers: SAM hair/watch plus the earlier SAM hair masks
    hair = {f: helper["HAIR"].get(f, np.zeros_like(person[f])) | read_mask(hdir, f) for f in range(N)}
    watch = {f: helper["WATCH"].get(f, np.zeros_like(person[f])) for f in range(N)}
    wfb = 0
    for f in range(N):
        if watch[f].sum() >= 10:
            continue
        avoid = hair[f].copy()
        for c in ("FINS", "SHORTS", "MASK"):
            if raw[c].get(f) is not None:
                avoid |= raw[c][f]
        w = find_watch(imgs[f], person[f], avoid, hand_tips(person[f], avoid))
        if w.sum() >= 10:
            watch[f] = w
            wfb += 1
    print(f"[cls] wrist watch: SAM on {sum(1 for f in helper['WATCH'] if helper['WATCH'][f].sum() >= 10)} frames, "
          f"dark-band fallback on {wfb}", flush=True)
    fin = finish(raw, person, hair, watch, range(N))

    # ---- write masks (soft, exclusive) -----------------------------------
    for c in CLASSES:
        (out_dir / c).mkdir(parents=True, exist_ok=True)
    areas = {c: [] for c in CLASSES}
    for f in range(N):
        al = soft(fin, f)
        for c in CLASSES:
            cv2.imwrite(str(out_dir / c / f"{f:05d}.png"), (al[c] * 255 + 0.5).astype(np.uint8))
            areas[c].append(int(fin[c][f].sum()))
    # overlap check: no pixel in two classes
    bad = 0
    for f in range(N):
        s = sum(fin[c][f].astype(np.uint8) for c in CLASSES)
        bad += int((s > 1).sum())
    # ---- flagged frames ----------------------------------------------------
    flags = []
    for c in CLASSES:
        for f in range(1, N):
            p, q = areas[c][f - 1], areas[c][f]
            if p == 0 and q == 0:
                continue
            ch = abs(q - p) / max(p, 1)
            if ch > POST["jump"]:
                flags.append((f, c, p, q, ch))
    lines = ["frame," + ",".join(CLASSES) + ",flags"]
    fl_by = {}
    for f, c, p, q, ch in flags:
        fl_by.setdefault(f, []).append(f"{c} {ch * 100:+.0f}%".replace("+", "+") if q >= p else f"{c} -{ch * 100:.0f}%")
    for f in range(N):
        lines.append(f"{f}," + ",".join(str(areas[c][f]) for c in CLASSES) + "," + ";".join(fl_by.get(f, [])))
    (qc / "class_areas.csv").write_text("\n".join(lines) + "\n")
    (qc / "summary.json").write_text(json.dumps({"overlap_pixels": bad, "flags": len(flags)}, indent=1))
    print(f"[cls] masks -> {out_dir}; overlap pixels {bad}; flagged {len(flags)}", flush=True)

    # ---- the one picture: contact sheet -----------------------------------
    sheet_frames = [0, 12, 28, 46, 60, 70, 89]
    th = 520
    tiles = []
    for f in sheet_frames:
        im = cv2.cvtColor(imgs[f], cv2.COLOR_BGR2RGB).astype(np.float32)
        ov = im.copy()
        for c in CLASSES:
            m = fin[c][f]
            col = np.array(COLOR[c], np.float32)
            ov[m] = 0.45 * ov[m] + 0.55 * col
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnt, -1, tuple(float(v) for v in col), 1)
        ys, xs = np.nonzero(person[f])
        x0, x1 = max(0, xs.min() - 50), min(1232, xs.max() + 50)
        y0, y1 = max(0, ys.min() - 50), min(1078, ys.max() + 50)
        t = ov[y0:y1, x0:x1].astype(np.uint8)
        t = cv2.resize(t, (int(t.shape[1] * th / t.shape[0]), th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"f{f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(t)
    leg = np.zeros((th, 330, 3), np.uint8)
    for k, c in enumerate(PRIORITY):
        cv2.rectangle(leg, (18, 40 + 52 * k), (58, 76 + 52 * k), COLOR[c], -1)
        cv2.putText(leg, c, (72, 68 + 52 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(leg, "MASK > FINS > SHORTS > SKIN", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    tiles.append(leg)
    rows, cur, wsum = [], [], 0
    W = 2400
    for t in tiles:
        if wsum + t.shape[1] > W and cur:
            rows.append(cur); cur, wsum = [], 0
        cur.append(t); wsum += t.shape[1]
    rows.append(cur)
    rimg = []
    for r in rows:
        row = np.concatenate(r, 1)
        rimg.append(cv2.copyMakeBorder(row, 0, 0, 0, max(0, W - row.shape[1]), cv2.BORDER_CONSTANT))
    sheet = np.concatenate(rimg, 0)
    cv2.imwrite(str(qc / "contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[cls] contact sheet -> {qc / 'contact_sheet.jpg'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
