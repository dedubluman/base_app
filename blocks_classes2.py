#!/usr/bin/env python3
"""Round 3 for f210-299: classes WATCH > HEAD > FINS > SHORTS > SKIN, no unclassed
pixel inside the body, skin colour rebuilt per pixel.

    python blocks_classes2.py finish2     # masks, alphas, tables
    python blocks_classes2.py grade2      # skin v2, composite, sheet, mp4, tables

The seeds come from blocks_classes.py (SAM 3.1 text detections of "bare skin",
"black swim fins", "head", flow fill, the SAM hand tracker, the 2x-crop left
fin). WATCH has no text answer in SAM 3.1 ("wrist watch", "watch", "wristband"
return nothing, also on 3x crops), so it is derived from the dark pieces of the
person mask that no other class owns, next to the hand.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

import blocks_classes as K
from reel_match import rgb2hsv

H, W, F0, F1 = K.H, K.W, K.F0, K.F1
OUT2 = K.REPO / "work" / "classes2"
CLS2 = ["SKIN", "FINS", "SHORTS", "HEAD", "WATCH"]
PRIORITY2 = ["WATCH", "HEAD", "FINS", "SHORTS", "SKIN"]
FIN_CLOSE5 = range(279, 295)
JITTER = {"SKIN": 0.75, "FINS": 1.0}
HAND_R = 45


def ero(m, r):
    return cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def ell(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def close_px(m, k):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ell(k // 2 if k > 1 else 1)) > 0


def open_px(m, k):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, ell(k // 2 if k > 1 else 1)) > 0


def fill_small_holes(m, max_area):
    inv = (~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = m.copy()
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < max_area and x > 0 and y > 0 and x + w < W and y + h < H:
            out[lab == k] = True
    return out


def keep_components(m, n_keep, min_area, attach=None):
    cnt, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    cand = []
    for k in range(1, cnt):
        a = st[k, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        piece = lab == k
        if attach is not None and not (piece & attach).any():
            continue
        cand.append((a, k))
    out = np.zeros_like(m)
    for a, k in sorted(cand, reverse=True)[:n_keep]:
        out |= lab == k
    return out


_RAW = {}


def raw_frame(f):
    if f not in _RAW:
        if len(_RAW) > 6:
            _RAW.pop(next(iter(_RAW)))
        im, l = K.luma01(f)
        h, s, v = rgb2hsv(im)
        _RAW[f] = (im, l, h, s, v)
    return _RAW[f]


def water_like(f):
    """Open water: bright, saturated and bluer than skin or gear."""
    _, _, h, s, v = raw_frame(f)
    return (v > 0.60) & (s > 0.75) & (h >= 190) & (h <= 215)


def nearest_label(classes, region):
    """Index (in PRIORITY2 order) of the class nearest to every pixel of `region`."""
    dist = []
    for c in PRIORITY2:
        m = classes[c]
        dist.append(cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) if m.any()
                    else np.full((H, W), 1e6, np.float32))
    return np.argmin(np.stack(dist), 0)


def hand_tips(f, P, masks):
    import diver_classes as D
    avoid = masks["FINS"] | masks["SHORTS"] | masks["HEAD"]
    return D.hand_tips(P[f], avoid, n=2)


def derive_shorts(f, P, m, water):
    im, l, h, s, v = raw_frame(f)
    used = m["SKIN"] | m["FINS"] | m["HEAD"]
    rem = P[f] & ~K.dil(used, 2) & ~water
    dark = rem & (l < 0.44) & (v < 0.60)
    dark = open_px(close_px(dark, 3), 3)                    # pant leg closed (3 px), spikes removed (3 px)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((H, W), bool)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = lab == big
    near = K.dil(keep, 8)
    for k in range(1, n):
        if k != big and st[k, cv2.CC_STAT_AREA] >= 150 and (near & (lab == k)).any():
            keep |= lab == k
    return keep


def derive_watch(f, P, m, water):
    """Dark pieces of the person mask that no class owns, near a hand tip: the strap
    and dial. Joined with a 5 px closing so no unpainted gap is left between hand and arm."""
    im, l, h, s, v = raw_frame(f)
    tips = hand_tips(f, P, m)
    if not tips:
        return np.zeros((H, W), bool)
    owned = m["SKIN"] | m["FINS"] | m["HEAD"] | m["SHORTS"]
    free = close_px(P[f], 3) & ~owned & ~water
    dark = free & (l < 0.44)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    head_near = K.dil(m["HEAD"], 4)
    out = np.zeros((H, W), bool)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 15:
            continue
        cx, cy = cen[k]
        if min(np.hypot(cx - tx, cy - ty) for tx, ty in tips) > 170:
            continue
        piece = lab == k
        if (piece & head_near).any():
            continue
        out |= piece
    if out.any():
        out = close_px(K.dil(out, 1), 5) & (free | out | K.dil(m["SKIN"], 0))
        out &= ~(m["FINS"] | m["HEAD"])
    return out


def derive_dark(f, P, m, water):
    """SHORTS and WATCH from the dark pixels of the person mask (luma < 0.44, V < 0.60, no water):
    the biggest piece (+ pieces >= 150 px touching it) is SHORTS; the other small pieces within
    170 px of a hand tip are the strap/dial = WATCH. Independent of the SKIN seed (which
    flickers at the thigh), so the skin/shorts split follows the picture's own darkness."""
    im, l, h, s_, v = raw_frame(f)
    rem = P[f] & ~K.dil(m["FINS"] | m["HEAD"], 2) & ~water
    dark = open_px(close_px(rem & (l < 0.44) & (v < 0.60), 3), 3)            # pant leg closed, spikes opened (3 px)
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    shorts = np.zeros((H, W), bool)
    watch = np.zeros((H, W), bool)
    if n <= 1:
        return shorts, watch, np.zeros((H, W), bool)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    shorts = lab == big
    near = K.dil(shorts, 8)
    for k in range(1, n):
        if k != big and st[k, cv2.CC_STAT_AREA] >= 150 and (near & (lab == k)).any():
            shorts |= lab == k
    # watch candidates: dark pieces 6-60 px from a fingertip (the watch sits at the wrist, ~30 px from the
    # hand end), 25-700 px, looser darkness test than the shorts, never touching the hair or the shorts.
    # The choice among them is made over time (continuity), see choose_watch()
    body = P[f] & ~K.dil(m["FINS"] | m["HEAD"] | shorts, 3)
    by, bx = np.nonzero(body)
    tips = []
    if len(by) > 100:                                      # the hand end = the lowest end of the arm (fingertips)
        low = by >= np.percentile(by, 96)
        tips = [(float(bx[low].mean()), float(by[low].mean()))]
    head_near = K.dil(m["HEAD"], 4)
    darkw = close_px(rem & (l < 0.48) & (v < 0.62), 3) & ~K.dil(shorts, 3) & ~head_near
    cands = []
    if tips:
        nw, labw, stw, cenw = cv2.connectedComponentsWithStats(darkw.astype(np.uint8), 8)
        for k in range(1, nw):
            a_ = stw[k, cv2.CC_STAT_AREA]
            d_ = min(np.hypot(cenw[k][0] - tx, cenw[k][1] - ty) for tx, ty in tips)
            if 25 <= a_ <= 700 and 6 <= d_ <= 60:
                cands.append((labw == k, float(d_)))
    return shorts, cands, darkw


WATCH_AXES = (14, 8)                                      # half axes (px) of the watch ellipse: ~350 px, strap and dial


def choose_watch(frames, wcands, darkw, shorts, seeds, P):
    """One watch per frame. The piece nearest a fingertip is chosen (the wrist band; the older rule was continuity: the candidate that overlaps
    the flow-carried watch of the previous frame most; the biggest when none does). Its centre and
    main axis are kept, interpolated over frames without a candidate (<= 12 frames) and smoothed
    over 5 frames; the watch is then a constant-size ellipse there, so its size cannot jump."""
    pos = {}
    for f in frames:
        cands = wcands[f]
        if cands:                                          # the piece closest to a fingertip: the wrist band
            pick = min(cands, key=lambda cd: (cd[1], -cd[0].sum()))[0]
            ys, xs = np.nonzero(pick)
            c = np.cov(np.stack([xs, ys]))
            th = 0.5 * np.arctan2(2 * c[0, 1], c[0, 0] - c[1, 1])
            pos[f] = (float(xs.mean()), float(ys.mean()), th)
    have = sorted(pos)
    out = {f: np.zeros((H, W), bool) for f in frames}
    if not have:
        return out
    xs_, ys_ = np.array([pos[f][0] for f in have]), np.array([pos[f][1] for f in have])
    c2, s2 = np.array([np.cos(2 * pos[f][2]) for f in have]), np.array([np.sin(2 * pos[f][2]) for f in have])
    fr = np.array(frames, float)
    X, Y = np.interp(fr, have, xs_), np.interp(fr, have, ys_)
    C2, S2 = np.interp(fr, have, c2), np.interp(fr, have, s2)
    def med5(v):
        vp = np.pad(v, 2, mode="edge")
        return np.array([np.median(vp[i:i + 5]) for i in range(len(v))])
    X, Y, C2, S2 = (boxcar(med5(v), 5) for v in (X, Y, C2, S2))
    for i, f in enumerate(frames):
        gap = min(abs(f - g) for g in have)
        if gap > 12:
            continue
        m = np.zeros((H, W), np.uint8)
        ang = np.degrees(0.5 * np.arctan2(S2[i], C2[i]))
        cv2.ellipse(m, (int(round(X[i])), int(round(Y[i]))), WATCH_AXES, float(ang), 0, 360, 1, -1)
        out[f] = (m > 0) & K.dil(P[f], 2) & ~(seeds[f]["FINS"] | seeds[f]["HEAD"] | shorts[f])
    return out


def finalize(f, seeds, P, water):
    """seeds -> exclusive classes covering the whole closed silhouette."""
    z = np.zeros((H, W), bool)
    m = {c: seeds.get(c, z).copy() for c in CLS2}
    # priority
    taken, ex = z.copy(), {}
    for c in PRIORITY2:
        ex[c] = m[c] & ~taken
        taken |= ex[c]
    # piece rules: FINS = two pieces attached to the feet, HEAD = one piece
    ex["FINS"] = keep_components(fill_small_holes(ex["FINS"], 400), 2, K.MIN_PART, attach=K.dil(ex["SKIN"] | ex["SHORTS"], 10))
    ex["FINS"] = close_px(ex["FINS"], 5 if f in FIN_CLOSE5 else 3)
    ex["HEAD"] = keep_components(ex["HEAD"], 1, 40)
    ex["SHORTS"] = open_px(ex["SHORTS"], 3)
    # exclusivity again after the shape rules
    taken = z.copy()
    for c in PRIORITY2:
        ex[c] = ex[c] & ~taken
        taken |= ex[c]
    sil = close_px((P[f] & ~water) | taken, 3)
    lab = nearest_label(ex, sil)
    out = {c: sil & (lab == i) for i, c in enumerate(PRIORITY2)}
    # dark pieces < 150 px that belong to no big piece are not kept as their own class (FINS/HEAD)
    for c in ("FINS", "HEAD"):
        cnt, lb, st, _ = cv2.connectedComponentsWithStats(out[c].astype(np.uint8), 8)
        for k in range(1, cnt):
            if st[k, cv2.CC_STAT_AREA] < K.MIN_PART:
                piece = lb == k
                out[c][piece] = False
                rest = {cc: out[cc] for cc in PRIORITY2}
                rest[c] = np.zeros((H, W), bool)
                lab2 = nearest_label(rest, piece)
                for i, cc in enumerate(PRIORITY2):
                    if cc != c:
                        out[cc] |= piece & (lab2 == i)
    return out, sil


def sdf(m):
    d_in = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    return (d_in - d_out).astype(np.float32)


def smooth_seeds(seeds, update, r, classes=CLS2):
    """Flow-aligned average of signed distance fields over +-r frames (weights r+1-|k|),
    thresholded at 0: sub-pixel smooth edges, a one-frame step is spread over several frames."""
    frames = sorted(seeds)
    new = {}
    cache = {}
    gx, gy = K._GRID
    for f in update:
        ks = [k for k in range(-r, r + 1) if frames[0] <= f + k <= frames[-1]]
        fl = {k: K.flow_field(f, f + k) for k in ks if k != 0}
        for c in classes:
            acc, wsum = None, 0.0
            for k in ks:
                key = (f + k, c)
                if key not in cache:
                    if len(cache) > 120:
                        cache.pop(next(iter(cache)))
                    cache[key] = sdf(seeds[f + k][c])
                d = cache[key]
                if k != 0:
                    d = cv2.remap(d, gx + fl[k][..., 0], gy + fl[k][..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                w = r + 1 - abs(k)
                acc = d * w if acc is None else acc + d * w
                wsum += w
            new.setdefault(f, {})[c] = (acc / wsum) > 0
    return new


def edge_metric(a, b_prev_warped):
    xor = a ^ b_prev_warped
    per = int((a ^ (cv2.erode(a.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)).sum())
    return xor.sum() / max(per, 1)


def hand_path(P, hard, base=212):
    """One hand tip followed through the clip: found on `base`, then carried frame to
    frame by optical flow and snapped to the nearest extremity within 30 px, so the
    hand area is always measured on the same hand."""
    path = {base: tuple(hand_tips(base, P, hard[base])[0])}
    for step, rng in ((1, range(base + 1, F1)), (-1, range(base - 1, F0 - 1, -1))):
        for f in rng:
            x, y = path[f - step]
            fl = K.flow_field(f - step, f)
            px, py = int(np.clip(x + fl[y, x, 0], 0, W - 1)), int(np.clip(y + fl[y, x, 1], 0, H - 1))
            cand = hand_tips(f, P, hard[f])
            best = min(cand, key=lambda t: np.hypot(t[0] - px, t[1] - py)) if cand else None
            path[f] = tuple(best) if best is not None and np.hypot(best[0] - px, best[1] - py) <= 30 else (px, py)
    return path


def hand_area(f, P, hard, path):
    x, y = path[f]
    yy, xx = np.ogrid[:H, :W]
    return int((((yy - y) ** 2 + (xx - x) ** 2 <= HAND_R ** 2) & hard["SKIN"]).sum())


def stage_finish2(a):
    OUT2.mkdir(parents=True, exist_ok=True)
    P = K.envelope()
    log = lambda s: print(s, flush=True)
    per, thin = K.load_raw()
    pk = OUT2 / "pm.pkl"
    if pk.exists():
        pm = pickle.loads(pk.read_bytes())
    else:
        for c in K.CLASSES:
            for f in range(F0, F1):
                per[c].setdefault(f, np.zeros((H, W), bool))
        pm = K.flow_median(per, ("SKIN", "FINS", "HEAD"))
        pk.write_bytes(pickle.dumps(pm))
    hand = pickle.loads((K.OUT / "hand.pkl").read_bytes()) if (K.OUT / "hand.pkl").exists() else {}
    if hand:                                   # temporal flow median of the tracked hand object
        hz = {"HAND": {f: hand.get(f, np.zeros((H, W), bool)) for f in range(F0, F1)}}
        hand = {f: m for f, m in K.flow_median(hz, ("HAND",), r=2)["HAND"].items() if f in hand}
    fin_r = pickle.loads((OUT2 / "fin_right.pkl").read_bytes()) if (OUT2 / "fin_right.pkl").exists() else {}
    frames = list(range(F0, F1))
    water = {f: water_like(f) for f in frames}
    seeds, hard, sil = {}, {}, {}
    wcands, darkw = {}, {}
    for f in frames:
        t = thin.get(f)
        lp = K.left_fin_piece(pm, f) if t is not None else None
        if t is not None and not (K.LEFT_RANGE[0] <= f <= K.LEFT_RANGE[1]):
            t = (t & K.dil(lp, 10)) if lp is not None else None
        fins = (pm["FINS"][f] | (t if t is not None else False)) & K.dil(P[f], 8)
        if f in fin_r and f >= 289:               # right fin tracked by SAM where the text detector misses it
            fins = fins | fin_r[f]
        skin = pm["SKIN"][f]
        if f in hand and (247 <= f <= 249 or 263 <= f <= 270):       # the user's frames, next to the hand only
            z0 = np.zeros((H, W), bool)
            tips = hand_tips(f, P, {"FINS": pm["FINS"][f], "SHORTS": z0, "HEAD": pm["HEAD"][f]})
            yy, xx = np.ogrid[:H, :W]
            disk = np.zeros((H, W), bool)
            for x0_, y0_ in tips:
                disk |= (yy - y0_) ** 2 + (xx - x0_) ** 2 <= 80 ** 2
            skin = skin | (hand[f] & disk)
        s = {"SKIN": pm["SKIN"][f], "FINS": fins, "HEAD": pm["HEAD"][f] & (raw_frame(f)[1] < 0.40),
             "SHORTS": np.zeros((H, W), bool), "WATCH": np.zeros((H, W), bool)}
        s["SHORTS"], wcands[f], darkw[f] = derive_dark(f, P, s, water[f])
        if f in hand and (247 <= f <= 249 or 263 <= f <= 270):       # the user's frames: the tracked hand, bright pixels near the tip
            z0 = np.zeros((H, W), bool)
            tips = hand_tips(f, P, {"FINS": pm["FINS"][f], "SHORTS": s["SHORTS"], "HEAD": pm["HEAD"][f]})
            yy, xx = np.ogrid[:H, :W]
            disk = np.zeros((H, W), bool)
            for x0_, y0_ in tips:
                disk |= (yy - y0_) ** 2 + (xx - x0_) ** 2 <= 80 ** 2
            s["SKIN"] = s["SKIN"] | (hand[f] & disk & (raw_frame(f)[1] > 0.46) & ~s["WATCH"])
        seeds[f] = s
    chosen = choose_watch(frames, wcands, darkw, {f: seeds[f]["SHORTS"] for f in frames}, seeds, P)
    for f in frames:
        seeds[f]["WATCH"] = chosen[f]
        seeds[f]["SKIN"] = seeds[f]["SKIN"] | (darkw[f] & ~chosen[f])      # other dark pieces on the arm are skin shade
    # HEAD edge: guided filter on the original frame (r=4), single piece
    from matte import guided_filter
    for f in frames:
        g = raw_frame(f)[1].astype(np.float32)
        q = guided_filter(g, K.dil(seeds[f]["HEAD"], 0).astype(np.float32), 4, 1e-3)
        seeds[f]["HEAD"] = q > 0.5
    # WATCH in every frame: carry from the nearest frame that has one (flow), within 8 frames
    have = [f for f in frames if seeds[f]["WATCH"].sum() >= 25]
    carried = []
    for f in frames:
        if seeds[f]["WATCH"].sum() >= 25:
            continue
        near = [g for g in have if abs(g - f) <= 12]
        if near:
            g = min(near, key=lambda g: abs(g - f))
            seeds[f]["WATCH"] = seeds[f]["WATCH"] | (K.warp_chain(seeds[g]["WATCH"], g, f) & K.dil(P[f], 2))
            carried.append(f)
    log(f"[finish2] WATCH derived on {len(have)} frames, carried by flow on {len(carried)}: {carried[:12]}")
    # a watch under 90 px is grown (at most 3 px, never over FINS/HEAD) so smoothing and paint keep it
    small = []
    for f in frames:
        w_ = seeds[f]["WATCH"]
        n_ = 0
        while 0 < w_.sum() < 90 and n_ < 3:
            w_ = K.dil(w_, 1) & K.dil(P[f], 2) & ~(seeds[f]["FINS"] | seeds[f]["HEAD"])
            n_ += 1
        if n_:
            small.append(f)
        seeds[f]["WATCH"] = w_
    log(f"[finish2] small watches grown: {small}")
    raw_watch = {f: seeds[f]["WATCH"].copy() for f in frames}
    # now smooth every class in time (flow-aligned SDF average, +-2 frames)
    sm2 = smooth_seeds(seeds, frames, 2, classes=("HEAD",))
    sm3 = smooth_seeds(seeds, frames, 3, classes=("SKIN", "FINS", "SHORTS"))
    for f in frames:
        seeds[f]["HEAD"] = sm2[f]["HEAD"]
        for c in ("SKIN", "FINS", "SHORTS"):
            seeds[f][c] = sm3[f][c]
    for f in frames:
        hard[f], sil[f] = finalize(f, seeds[f], P, water[f])
    # ---- temporal checks
    def warped_prev(f, c):
        return K.remap_mask(K.flow_field(f, f - 1), hard[f - 1][c]) > 0.5

    def jitter_table():
        out = {c: {} for c in JITTER}
        for f in range(F0 + 1, F1):
            for c in JITTER:
                out[c][f] = edge_metric(hard[f][c], warped_prev(f, c))
        return out
    jt = jitter_table()
    log("[finish2] edge jitter (XOR / perimeter, px) after the SDF smoothing: " +
        ", ".join(f"{c} median {np.median(list(jt[c].values())):.2f}, > {JITTER[c]}: {[f for f, v in jt[c].items() if v > JITTER[c]]}" for c in JITTER))
    for it, rad in enumerate((4, 5)):                       # whole clip, never a window (a window leaves steps at its edges)
        bad = sorted({f for c in JITTER for f, v in jt[c].items() if v > JITTER[c]})
        if not bad:
            break
        sm = smooth_seeds({f: {c: hard[f][c] for c in CLS2} for f in frames}, frames, rad, classes=("SKIN", "FINS", "SHORTS"))
        for f in frames:
            for c in ("SKIN", "FINS", "SHORTS"):
                seeds[f][c] = sm[f][c]
            hard[f], sil[f] = finalize(f, seeds[f], P, water[f])
        jt = jitter_table()
        log(f"[finish2] whole-clip smoothing pass {it + 1} (+-{rad}): " + ", ".join(
            f"{c} still > {JITTER[c]}: {[f for f, v in jt[c].items() if v > JITTER[c]]}" for c in JITTER))
    # FINS edge on f288-293: 3-frame median of the flow-carried previous and next masks
    for f in range(288, 294):
        fp, fn = K.flow_field(f, f - 1), K.flow_field(f, f + 1)
        v = K.remap_mask(fp, hard[f - 1]["FINS"]) + hard[f]["FINS"].astype(np.float32) + K.remap_mask(fn, hard[f + 1]["FINS"])
        seeds[f]["FINS"] = v >= 1.5
        hard[f], sil[f] = finalize(f, seeds[f], P, water[f])
    # hand area: a >40 % drop against the previous frame is replaced from the nearest sound frame
    hpath = hand_path(P, hard)
    ha = {f: hand_area(f, P, hard[f], hpath) for f in frames}
    fixed = []
    for f in frames[1:]:
        if ha[f - 1] > 0 and ha[f] < 0.6 * ha[f - 1]:
            sound = [g for g in frames if g != f and ha[g] >= 0.8 * ha[f - 1] and abs(g - f) <= 8]
            if sound:
                g = min(sound, key=lambda g: abs(g - f))
                yy, xx = np.ogrid[:H, :W]
                disk = (yy - hpath[g][1]) ** 2 + (xx - hpath[g][0]) ** 2 <= (HAND_R + 15) ** 2
                carry = K.warp_chain(hard[g]["SKIN"] & disk, g, f) & K.dil(P[f], 2)
                seeds[f]["SKIN"] = seeds[f]["SKIN"] | carry
                hard[f], sil[f] = finalize(f, seeds[f], P, water[f])
                fixed.append((f, ha[f], hand_area(f, P, hard[f], hpath), g))
                ha[f] = hand_area(f, P, hard[f], hpath)
    log(f"[finish2] hand-area drops fixed by flow (frame, before, after, from): {fixed}")
    log(f"[finish2] hand area per frame: min {min(ha.values())} (frame {min(ha, key=ha.get)}); zero frames: {[f for f in frames if ha[f] == 0]}")
    # ---- numbers
    rows = ["frame,SKIN,FINS,SHORTS,HEAD,WATCH,hand_area,jitter_SKIN,jitter_FINS,flags"]
    flags = []
    jt = jitter_table()
    unclass = 0
    overlap = 0
    for f in frames:
        areas = {c: int(hard[f][c].sum()) for c in CLS2}
        s_union = np.zeros((H, W), bool)
        for c in CLS2:
            s_union |= hard[f][c]
        unclass += int((sil[f] & ~s_union).sum())
        overlap += int((sum(hard[f][c].astype(np.uint8) for c in CLS2) > 1).sum())
        fl = []
        if areas["WATCH"] == 0:
            fl.append("WATCH=0")
        if ha[f] == 0:
            fl.append("hand=0")
        for c in JITTER:
            if f in jt[c] and jt[c][f] > JITTER[c]:
                fl.append(f"{c} jitter {jt[c][f]:.2f}")
        if f > F0 and ha[f - 1] and ha[f] < 0.6 * ha[f - 1]:
            fl.append("hand -40%")
        rows.append(f"{f}," + ",".join(str(areas[c]) for c in ["SKIN", "FINS", "SHORTS", "HEAD", "WATCH"]) +
                    f",{ha[f]}," + ",".join(f"{jt[c].get(f, 0):.2f}" for c in JITTER) + "," + ";".join(fl))
        if fl:
            flags.append((f, fl))
    (OUT2 / "class_table.csv").write_text("\n".join(rows) + "\n")
    log(f"[finish2] unclassed pixels inside the silhouette: {unclass}; overlap pixels: {overlap}")
    log(f"[finish2] jitter medians SKIN {np.median(list(jt['SKIN'].values())):.2f} / FINS {np.median(list(jt['FINS'].values())):.2f}; flagged frames {len(flags)}: " +
        "; ".join(f"f{f} {','.join(v)}" for f, v in flags[:40]))
    for c in CLS2:
        (OUT2 / "hard" / c).mkdir(parents=True, exist_ok=True)
    for f in frames:
        for c in CLS2:
            cv2.imwrite(str(OUT2 / "hard" / c / f"{f:05d}.png"), hard[f][c].astype(np.uint8) * 255)
        cv2.imwrite(str(OUT2 / "hard" / f"sil_{f:05d}.png"), sil[f].astype(np.uint8) * 255)
    return 0


def stage_fin_right(a):
    """The right fin has no text detection on f290-297 (and merges with the left one on
    f298-299): track it from f288/f289, where it is a separate piece, with SAM point prompts."""
    import blocks_classes as K2
    P = K.envelope()
    sam = K.Sam()
    cs, ce = 286, 300
    log = lambda s: print(s, flush=True)
    sid = sam.start(list(range(cs, ce)), "finR")
    for f in (288, 289):
        fin = cv2.imread(str(K.OUT / "hard" / "FINS" / f"{f:05d}.png"), 0) > 127
        skin = cv2.imread(str(K.OUT / "hard" / "SKIN" / f"{f:05d}.png"), 0) > 127
        n, lab, st, cen = cv2.connectedComponentsWithStats(fin.astype(np.uint8), 8)
        pcs = sorted([(cen[k][0], k) for k in range(1, n) if st[k, 4] >= K.MIN_PART])
        right, left = lab == pcs[-1][1], lab == pcs[0][1]
        pos = K.fps_points(right, 5, min_dt=2.0)
        neg = K.fps_points(left, 3, min_dt=2.0) + K.fps_points(skin, 2, min_dt=3.0)
        log(f"[finR] f{f}: right fin piece {int(right.sum())} px; {len(pos)} positives {pos}; left fin piece {int(left.sum())}")
        pts = [(x, y, 1) for x, y in pos] + [(x, y, 0) for x, y in neg]
        sam.points(sid, f - cs, "FINS", [(x, y) for x, y, _ in pts], [l for *_, l in pts])
    res = sam.propagate(sid, cs, 289 - cs)["FINS"]
    out = {f: m & K.dil(P[f], 4) for f, m in res.items()}
    (OUT2 / "fin_right.pkl").write_bytes(pickle.dumps(out))
    log("[finR] right-fin object areas " + " ".join(f"f{f}:{int(out[f].sum())}" for f in range(cs, ce) if f in out))
    return 0


# ============================================================ part 2: skin ===
TGT = (60.0, 15.6, 20.4)                                   # mean L*, a*, b* of the skin
HUE_L, HUE_H = [30, 48, 50, 70, 75, 90], [44, 48, 50, 54, 56, 62]
RAT_L, RAT_R = [30, 48, 50, 70, 75, 90], [0.50, 0.42, 0.43, 0.38, 0.30, 0.22]
SMOOTH_N = 9
COLOR2 = {"SKIN": (255, 170, 60), "FINS": (255, 60, 255), "SHORTS": (60, 120, 255), "HEAD": (60, 255, 60),
          "WATCH": (255, 235, 0)}


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def rgb2lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)


def lab2rgb(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def masked_blur(a, m, sigma):
    mf = m.astype(np.float32)
    num = cv2.GaussianBlur(a * mf, (0, 0), sigma)
    den = cv2.GaussianBlur(mf, (0, 0), sigma)
    return np.where(den > 0.02, num / np.maximum(den, 1e-3), a)


def skin_metrics(L, a, b, core, m):
    """Measured the way the brief asks: Lab smoothed 2 px, over the skin core."""
    Ls, as_, bs = masked_blur(L, m, 2.0), masked_blur(a, m, 2.0), masked_blur(b, m, 2.0)
    l, aa, bb = Ls[core], as_[core], bs[core]
    hue = np.degrees(np.arctan2(bb, aa))
    chroma = np.hypot(aa, bb)
    order = np.argsort(l)
    n3 = max(len(l) // 3, 1)
    diff = float(abs(hue[order[-n3:]].mean() - hue[order[:n3]].mean()))
    return dict(L=float(l.mean()), a=float(aa.mean()), b=float(bb.mean()), hue_mean=float(hue.mean()),
                hue_std=float(hue.std()), third_diff=diff, corr=float(np.corrcoef(l, chroma)[0, 1]),
                chroma_mean=float(chroma.mean()))


def skin_crop(x, skin):
    ys, xs = np.nonzero(skin)
    y0, y1 = max(0, ys.min() - 12), min(H, ys.max() + 13)
    x0, x1 = max(0, xs.min() - 12), min(W, xs.max() + 13)
    return y0, y1, x0, x1


def clean_red(xc):
    """The raw red channel is nearly empty and blocky: guided filter, green as the guide (r=4)."""
    from matte import guided_filter
    xc = xc.copy()
    xc[..., 0] = np.clip(guided_filter(xc[..., 1].astype(np.float32), xc[..., 0].astype(np.float32), 4, 1e-3), 0, 1)
    return xc


def skin_gains(x, skin):
    """Per-channel linear gains that move the mean skin colour to the target mean."""
    y0, y1, x0, x1 = skin_crop(x, skin)
    xc = clean_red(x[y0:y1, x0:x1])
    m = skin[y0:y1, x0:x1]
    core = ero(m, 3)
    if core.sum() < 200:
        core = m
    lin = srgb2lin(xc)
    mean_lin = lin[core].mean(0)
    t_lin = srgb2lin(lab2rgb(np.array([[[TGT[0], TGT[1], TGT[2]]]], np.float32))[0, 0])
    return (t_lin / np.maximum(mean_lin, 1e-4)).astype(np.float64)


def skin_prepare(x, skin, g, other=None):
    """Steps 2, 1 and the smoothing part of 3: everything that does not depend on k and the mix."""
    from matte import guided_filter
    y0, y1, x0, x1 = skin_crop(x, skin)
    xc = clean_red(x[y0:y1, x0:x1])                                        # 2: red from green
    m = skin[y0:y1, x0:x1]
    core = ero(m, 3)
    if core.sum() < 200:
        core = m
    rgb = lin2srgb(srgb2lin(xc) * np.asarray(g, np.float32)[None, None, :])   # 1: gain per channel, linear light
    lab = rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    Ln = (L / 100.0).astype(np.float32)
    a_s = 50.0 * guided_filter(Ln, (a / 50.0).astype(np.float32), 2, 1e-3)    # 3: a*, b* smoothed, L* guided
    b_s = 50.0 * guided_filter(Ln, (b / 50.0).astype(np.float32), 2, 1e-3)
    h_r = np.degrees(np.arctan2(b_s, a_s))
    C_r = np.hypot(a_s, b_s)
    h0 = float(np.degrees(np.arctan2(np.sin(np.radians(h_r[core])).mean(), np.cos(np.radians(h_r[core])).mean())))
    dh = np.clip(((h_r - h0 + 180.0) % 360.0) - 180.0, -25.0, 25.0)          # hue deviation from the skin mean (wrap-safe, clipped)
    dC = np.clip(C_r - float(C_r[core].mean()), -12.0, 12.0)                  # chroma deviation from the skin mean
    if other is not None:                                                     # next to hair / shorts / fins / watch the raw colour is contaminated
        damp = np.where(other[y0:y1, x0:x1], 0.4, 1.0).astype(np.float32)
        dh, dC = dh * damp, dC * damp
    return dict(box=(y0, y1, x0, x1), m=m, core=core, L=L, dh=dh, dC=dC, Lb=masked_blur(L, m, 8.0),
                p98=float(np.percentile(L[core], 98)))


def skin_apply(pr, k, w, kc=1.0, tilt=0.0):
    """Widen a*, b* around the skin mean by k, optional L*-dependent hue/chroma curve (mix w),
    brightest 2 % neutral, mild local contrast, L* <= 88, means pinned to the targets."""
    L, m, core = pr["L"], pr["m"], pr["core"]
    # a*, b* deviations around the skin mean widened by k, in polar form (no hue flips where chroma is small)
    Ht = np.degrees(np.arctan2(TGT[2], TGT[1]))
    Ct = float(np.hypot(TGT[1], TGT[2]))
    h1 = Ht + k * pr["dh"]
    C1 = Ct + k * pr["dC"]
    if w > 0:
        hc = np.interp(L, HUE_L, HUE_H)
        Cc = np.interp(L, RAT_L, RAT_R) * L
        h1 = (1 - w) * h1 + w * hc
        C1 = (1 - w) * C1 + w * Cc
    if tilt:                                       # hue rises with L* (degrees per standard deviation of L*)
        Lc = L[core]
        h1 = h1 + tilt * (L - float(Lc.mean())) / max(float(Lc.std()), 1.0)
    if kc < 1.0:                                   # chroma deviation narrowed on its own: keeps hue variety, lowers L*-chroma correlation
        cm = float(C1[core].mean())
        C1 = cm + kc * (C1 - cm)
    h1 = h1 + (Ht - float(h1[core].mean()))              # the mean hue and mean chroma are pinned: frames cannot drift
    C1 = C1 + (Ct - float(C1[core].mean()))
    hi = L >= pr["p98"]
    C1 = np.where(hi, np.minimum(C1, 10.0 + 14.0 * np.clip((pr["p98"] + 4.0 - L) / 4.0, 0, 1)), C1)
    a2, b2 = C1 * np.cos(np.radians(h1)), C1 * np.sin(np.radians(h1))
    L2 = np.minimum(L + 0.15 * (L - pr["Lb"]), 88.0)
    L2 = np.minimum(L2 + (TGT[0] - L2[core].mean()), 88.0)
    return L2, a2, b2, skin_metrics(L2, a2, b2, core, m)


def skin_lab(x, skin, g, k, w, kc=1.0, other=None, tilt=0.0, want_metrics=True):
    pr = skin_prepare(x, skin, g, other)
    L2, a2, b2, met = skin_apply(pr, k, w, kc, tilt)
    return pr["box"], np.stack([L2, a2, b2], -1).astype(np.float32), pr["m"], pr["core"], met


HUE_STD_TARGET = 7.5


def solve_skin(x, skin, g, other=None):
    """Per frame: for each (chroma narrowing kc, curve mix w), k (<= 2.5) by bisection so the hue
    std lands on 7.5 deg; the first combination that also gives third difference >= 6.5 deg and
    L*-chroma correlation <= 0.78 wins (margins over the 6 / 6 / 0.80 targets)."""
    pr = skin_prepare(x, skin, g, other)
    best, best_viol, best_met = None, 1e9, None
    for kc in (1.0, 0.5, 0.25):
        for w in (0.0, 0.4, 0.6, 0.8, 1.0):
            lo, hi = 0.0, 2.5
            if skin_apply(pr, hi, w, kc)[3]["hue_std"] < HUE_STD_TARGET:
                k = hi
            elif skin_apply(pr, lo, w, kc)[3]["hue_std"] >= HUE_STD_TARGET:
                k = lo
            else:
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    if skin_apply(pr, mid, w, kc)[3]["hue_std"] < HUE_STD_TARGET:
                        lo = mid
                    else:
                        hi = mid
                k = 0.5 * (lo + hi)
            met = skin_apply(pr, k, w, kc)[3]
            viol = max(0, 6.5 - met["hue_std"]) + max(0, met["hue_std"] - 12.0) + max(0, 6.5 - met["third_diff"]) + 10 * max(0, met["corr"] - 0.78)
            if viol < best_viol:
                best, best_viol, best_met = (k, w, kc), viol, met
            if viol == 0:
                return k, w, kc, met
    return best[0], best[1], best[2], best_met


def boxcar(v, n=SMOOTH_N):
    v = np.asarray(v, np.float64)
    pad = n // 2
    vp = np.pad(v, [(pad, pad)] + [(0, 0)] * (v.ndim - 1), mode="edge")
    k = np.ones(n) / n
    if v.ndim == 1:
        return np.convolve(vp, k, "valid")
    return np.stack([np.convolve(vp[:, i], k, "valid") for i in range(v.shape[1])], 1)


def refine_skin(x, skin, g, k_s, other=None):
    """After the 9-frame smoothing of the gains and k: keep them (k may move by at most
    +-40 %) and re-solve the curve mix, chroma narrowing and the L*-dependent hue tilt on this
    frame against the targets (hue std 6.5-12 deg, third difference >= 6.5 deg, corr <= 0.78)."""
    pr = skin_prepare(x, skin, g, other)
    best, best_viol, best_met = None, 1e9, None
    for tilt in (0.0, 3.5, 5.5, 7.5):
        for km in (1.0, 1.2, 0.85, 1.4, 0.7):
            k = min(2.5, k_s * km)
            for kc in (1.0, 0.5, 0.25):
                for w in (0.0, 0.4, 0.6, 0.8, 1.0):
                    met = skin_apply(pr, k, w, kc, tilt)[3]
                    viol = (max(0, 6.5 - met["hue_std"]) + max(0, met["hue_std"] - 12.0) + max(0, 6.5 - met["third_diff"])
                            + 10 * max(0, met["corr"] - 0.78))
                    if viol < best_viol:
                        best, best_viol, best_met = (k, w, kc, tilt), viol, met
                    if viol == 0:
                        return k, w, kc, tilt, met
    return best[0], best[1], best[2], best[3], best_met


def refine_task(task):
    f, x, skin, other, g, k_s = task
    return (f,) + refine_skin(x, skin, g, k_s, other)


def solve_task(task):
    f, x, skin, other = task
    g = skin_gains(x, skin)
    k, w, kc, met = solve_skin(x, skin, g, other)
    return f, g, k, w, kc, met


def water_plates(x, env):
    """(background grade of the diver-free plate, ambient grade of his own pixels, ring colour)."""
    import blocks_grade as B
    xb = x.copy()
    dd = cv2.distanceTransform((~env).astype(np.uint8), cv2.DIST_L2, 5)
    near = K.dil(env, 6)
    for ch in range(3):
        xb[..., ch] = np.where(near, K.harmonic_fill(x[..., ch], near), x[..., ch])
    far = (dd >= 30) & (dd <= 60)
    if far.sum() > 500:
        ramp = np.clip((34.0 - np.arange(36)) / 14.0, 0, 1)
        kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        kern /= kern.sum()
        idx = np.clip(dd, 0, 35).astype(np.int32)
        for ch in range(3):
            vch = xb[..., ch]
            vfar = float(np.median(vch[far]))
            vk = np.array([np.median(vch[(dd >= q) & (dd < q + 1)]) if ((dd >= q) & (dd < q + 1)).sum() > 30 else np.nan
                           for q in range(36)], np.float32)
            ok = ~np.isnan(vk)
            if ok.sum() <= 8:
                continue
            vk = np.interp(np.arange(36), np.arange(36)[ok], vk[ok])
            vk = np.convolve(np.pad(vk, 6, mode='edge'), kern, 'valid').astype(np.float32)
            gains = np.clip(1 + (vfar / np.maximum(vk, 1e-3) - 1) * ramp, 0.9, 1.1).astype(np.float32)
            gg = gains[idx]
            gg[env] = 1.0
            xb[..., ch] = np.clip(vch * gg, 0, 1)
    bg = B.grade_background(xb, hole=None, grain=False)
    bg_raw = B.grade_background(x, hole=K.dil(env, 15), grain=False)
    ring = K.dil(env, 40) & ~K.dil(env, 25)
    plate = np.median(bg[ring], 0) if ring.sum() > 50 else np.median(bg.reshape(-1, 3), 0)
    return bg, bg_raw, plate


def fins_colour(bg_raw):
    """Ambient grade for the fins, but highlights neutral grey: L* <= 35, chroma rolled off above L* 8."""
    lab = rgb2lab(bg_raw)
    L = np.minimum(lab[..., 0], 33.0)
    f = 1.0 - 0.92 * np.clip((L - 8.0) / 14.0, 0, 1)
    return lab2rgb(np.stack([L, lab[..., 1] * f, lab[..., 2] * f], -1))


def watch_colour(x):
    """Dark neutral paint: L* 20-25, a* = b* = 0."""
    l = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    L = 20.0 + 3.0 * np.clip((l - 0.25) / 0.25, 0, 1)
    z = np.zeros_like(L)
    return lab2rgb(np.stack([L, z, z], -1))


def grade2_frame(task):
    import blocks_grade as B
    f, x, hard, y_hi, dist, g, k, w, kc, tilt = task
    S = np.zeros((H, W), bool)
    for c in CLS2:
        S |= hard[c]
    skin = hard["SKIN"]
    bg, bg_raw, plate = water_plates(x, S)
    cols = {"SHORTS": B.grade_diver(x, y_hi, dist, plate, mask=hard["SHORTS"], force="gear"),
            "HEAD": B.grade_diver(x, y_hi, dist, plate, mask=hard["HEAD"], force="gear"),
            "FINS": fins_colour(bg_raw), "WATCH": watch_colour(x)}
    other = K.dil(hard["HEAD"] | hard["SHORTS"] | hard["FINS"] | hard["WATCH"], 4)
    (y0, y1, x0, x1), lab, m, core, met = skin_lab(x, skin, g, k, w, kc, other, tilt)
    sk = np.zeros((H, W, 3), np.float32)
    sk[y0:y1, x0:x1] = lab2rgb(lab)
    cols["SKIN"] = B.fill_rim(sk, skin, 3)
    # light wrap: 15-20 % of the background colour over the outer 3 px of the silhouette (no blue outline)
    d = cv2.distanceTransform(S.astype(np.uint8), cv2.DIST_L2, 5)
    wrap = np.where((d > 0) & (d <= 3), 0.20 - 0.05 * (np.clip(d, 1, 3) - 1) / 2.0, 0.0).astype(np.float32)[..., None]
    cols["SKIN"] = cols["SKIN"] * (1 - wrap) + bg * wrap
    for c in ("SHORTS", "HEAD", "FINS", "WATCH"):
        if hard[c].any():
            cols[c] = B.fill_rim(cols[c], hard[c], 2)
    # composite: outer silhouette 1 px in + 2 px feather; classes cross-fade 1 px, no gaps
    A = np.clip(cv2.GaussianBlur(ero(S, 1).astype(np.float32), (0, 0), 0.85), 0, 1)[..., None]
    R4 = K.dil(S, 4)
    lab_idx = nearest_label(hard, R4)
    wsum = np.zeros((H, W), np.float32)
    col = np.zeros((H, W, 3), np.float32)
    for i, c in enumerate(PRIORITY2):
        wc = cv2.GaussianBlur(((lab_idx == i) & R4).astype(np.float32), (0, 0), 0.6)
        wsum += wc
        col += wc[..., None] * cols[c]
    col /= np.maximum(wsum, 1e-3)[..., None]
    out = bg * (1 - A) + col * A
    out = B.add_grain(out, B.BG["grain"], np.random.default_rng(3000 + f))
    out = np.clip(out, 0, 1).astype(np.float32)
    # measurement on the 8-bit result
    u8 = (out * 255 + 0.5).astype(np.uint8)
    lab8 = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    Lm, am, bm = lab8[..., 0] * 100 / 255, lab8[..., 1] - 128, lab8[..., 2] - 128
    cr = ero(skin, 3)
    mt = skin_metrics(Lm, am, bm, cr if cr.sum() > 200 else skin, skin)
    fins = hard["FINS"]
    Lf = Lm[fins]
    mt["fins_Lmax"] = float(np.percentile(Lf, 99.5)) if fins.any() else 0.0
    wc = next((e for e in (ero(hard["WATCH"], 2), ero(hard["WATCH"], 1)) if e.sum() > 20), hard["WATCH"])
    mt["watch_L"] = float(np.median(Lm[wc])) if wc.any() else 0.0
    mt["watch_ab"] = (float(np.median(am[wc])), float(np.median(bm[wc]))) if wc.any() else (0, 0)
    return f, out, mt


def stage_grade2(a):
    import multiprocessing as mp
    import dataclasses
    import blocks_grade as B
    from shades.config import Encode
    from shades.io import Encoder, iter_frames, probe
    from reel_match import luma
    info = probe(B.VIDEO)
    P = K.envelope()
    frames = [int(v) for v in a.only] if a.only else list(range(F0, F1))
    full = not a.only
    hard = {f: {c: cv2.imread(str(OUT2 / "hard" / c / f"{f:05d}.png"), 0) > 127 for c in CLS2} for f in range(F0, F1)}
    raws = {}
    for k_, x in enumerate(iter_frames(B.VIDEO, info)):
        if k_ >= F1:
            break
        if k_ >= F0:
            raws[k_] = x
    oth = {f: K.dil(hard[f]["HEAD"] | hard[f]["SHORTS"] | hard[f]["FINS"] | hard[f]["WATCH"], 4) for f in range(F0, F1)}
    pool = mp.get_context("fork").Pool(a.workers)
    # pass A: per-frame gains, k, curve mix (search against the targets)
    res = {}
    for f, g, k, w, kc, met in pool.imap(solve_task, [(f, raws[f], hard[f]["SKIN"], oth[f]) for f in range(F0, F1)], chunksize=1):
        res[f] = (g, k, w, kc)
    allf = list(range(F0, F1))
    lg = boxcar(np.log([res[f][0] for f in allf]))            # 7: gains and k smoothed over 9 frames
    kk = boxcar([res[f][1] for f in allf])
    ww = boxcar([res[f][2] for f in allf])
    kcs = boxcar([res[f][3] for f in allf])
    gs = {f: np.exp(lg[i]) for i, f in enumerate(allf)}
    par = {}
    for f, k, w, kc, tilt, met in pool.imap(refine_task, [(f, raws[f], hard[f]["SKIN"], oth[f], gs[f], float(kk[f - F0])) for f in allf], chunksize=1):
        par[f] = (gs[f], k, w, kc, tilt)
    # gear colours: his lit-skin level and size, as in the first round
    yh = []
    for f in allf:
        m = ero(hard[f]["SKIN"], 2)
        yh.append(float(np.percentile(luma(np.clip(raws[f], 0, 1))[m], 90)) if m.sum() > 200 else np.nan)
    yh = np.array(yh)
    ok = ~np.isnan(yh)
    yh = B.smooth1d(np.interp(allf, np.array(allf)[ok], yh[ok]), 6.0, 18)
    size = B.smooth1d(np.array([B.diver_size((P[f] * 255).astype(np.uint8)) for f in allf]), 8.0, 24)
    dist = np.clip((B.DV["size_near"] - size) / (B.DV["size_near"] - B.DV["size_far"]), 0, 1)
    tasks = [(f, raws[f], hard[f], float(yh[f - F0]), float(dist[f - F0]), *par[f]) for f in frames]
    out_frames, metrics = {}, {}
    enc = None
    if full:
        enc = Encoder(B.VIDEO, OUT2 / "blocks_dive_f210_299_graded.mp4", dataclasses.replace(info, has_audio=False), Encode(crf=16, preset="slow"))
    for f, frame, mt in pool.imap(grade2_frame, tasks, chunksize=1):
        metrics[f] = mt
        if enc is not None:
            enc.write(frame)
        if f in SHEET2 or not full:
            out_frames[f] = frame
    pool.close(); pool.join()
    if enc is not None:
        enc.close()
    (OUT2 / "skin_metrics.json").write_text(json.dumps({str(f): m for f, m in metrics.items()}, indent=1))
    pickle.dump(par, open(OUT2 / "skin_params.pkl", "wb"))
    # ---- tables
    def row(f):
        m = metrics[f]
        return (f"f{f}: L* {m['L']:.1f} | a* {m['a']:.1f} | b* {m['b']:.1f} | hue std {m['hue_std']:.1f} deg | "
                f"third diff {m['third_diff']:.1f} deg | L*-C* corr {m['corr']:.2f} | k {par[f][1]:.2f} mix {par[f][2]:.2f} kc {par[f][3]:.2f} tilt {par[f][4]:.1f}")
    print("[grade2] SKIN targets: L* 58-62, a* 14-16, b* 18-21, hue std >= 6, third diff >= 6, corr <= 0.80, frame-to-frame dL* <= 1.5 / dhue <= 1")
    for f in ([210, 235, 260, 290] if full else frames):
        if f in metrics:
            print("[grade2] " + row(f))
    fr = sorted(metrics)
    out_of = [f for f in fr if not (58 <= metrics[f]["L"] <= 62 and 14 <= metrics[f]["a"] <= 16 and 18 <= metrics[f]["b"] <= 21
                                    and metrics[f]["hue_std"] >= 6 and metrics[f]["third_diff"] >= 6 and metrics[f]["corr"] <= 0.80)]
    print(f"[grade2] frames outside any of the five targets: {len(out_of)} {out_of[:30]}")
    if len(fr) > 2 and fr == list(range(fr[0], fr[-1] + 1)):
        dL = [abs(metrics[f]["L"] - metrics[f - 1]["L"]) for f in fr[1:] if f - 1 in metrics]
        dh = [abs(metrics[f]["hue_mean"] - metrics[f - 1]["hue_mean"]) for f in fr[1:] if f - 1 in metrics]
        print(f"[grade2] frame-to-frame: max dL* {max(dL):.2f}, max dhue {max(dh):.2f} deg")
        print("[grade2] ranges over all frames: " + ", ".join(
            f"{k_} {min(metrics[f][k_] for f in fr):.2f}..{max(metrics[f][k_] for f in fr):.2f}" for k_ in ("L", "a", "b", "hue_std", "third_diff", "corr")))
        print(f"[grade2] FINS highlight L* max {max(metrics[f]['fins_Lmax'] for f in fr):.1f} (limit 35); "
              f"WATCH L* {min(metrics[f]['watch_L'] for f in fr):.1f}..{max(metrics[f]['watch_L'] for f in fr):.1f}, "
              f"a*/b* within +-{max(max(abs(metrics[f]['watch_ab'][0]), abs(metrics[f]['watch_ab'][1])) for f in fr):.1f}")
    if full:
        make_sheet(out_frames, hard, raws, P)
    return 0


SHEET2 = [210, 248, 265, 274, 290, 299]


def make_sheet(outs, hard, raws, P):
    panels_top, panels_bot = [], []
    for f in SHEET2:
        S = np.zeros((H, W), bool)
        for c in CLS2:
            S |= hard[f][c]
        ys, xs = np.nonzero(S)
        cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) // 2 + 30
        x0, y0 = int(np.clip(cx - half, 0, W - 2 * half)), int(np.clip(cy - half, 0, H - 2 * half))
        ov = (raws[f] * 255).astype(np.float32)
        for c in PRIORITY2[::-1]:
            m = hard[f][c]
            col = np.array(COLOR2[c], np.float32)
            ov[m] = 0.45 * ov[m] + 0.55 * col
        t0 = np.clip(ov[y0:y0 + 2 * half, x0:x0 + 2 * half], 0, 255).astype(np.uint8)
        t1 = np.clip(outs[f][y0:y0 + 2 * half, x0:x0 + 2 * half] * 255, 0, 255).astype(np.uint8)
        for t, tag, lst in ((t0, "classes", panels_top), (t1, "result", panels_bot)):
            t = cv2.resize(t, (266, 266), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"f{f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            lst.append(t)
    row1, row2 = np.concatenate(panels_top, 1), np.concatenate(panels_bot, 1)
    # hand regions 3x: f248 and f265, classes and result
    zooms = []
    for f in (248, 265):
        sk = hard[f]["SKIN"] | hard[f]["WATCH"]
        ys_, xs_ = np.nonzero(sk)
        low = ys_ >= np.percentile(ys_, 96)                       # the hand is the lowest end of the arm
        x, y = int(xs_[low].mean()), int(ys_[low].mean())
        x0, y0 = int(np.clip(x - 66, 0, W - 133)), int(np.clip(y - 90, 0, H - 133))
        ov = (raws[f] * 255).astype(np.float32)
        for c in PRIORITY2[::-1]:
            m = hard[f][c]
            ov[m] = 0.45 * ov[m] + 0.55 * np.array(COLOR2[c], np.float32)
        for img, tag in ((ov, "hand classes x3"), (outs[f] * 255, "hand result x3")):
            t = np.clip(img[y0:y0 + 133, x0:x0 + 133], 0, 255).astype(np.uint8)
            t = cv2.resize(t, (400, 400), interpolation=cv2.INTER_CUBIC)
            cv2.putText(t, f"f{f} {tag}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            zooms.append(t)
    row3 = np.concatenate(zooms, 1)
    sheet = np.concatenate([np.pad(row1, ((0, 0), (0, 1600 - row1.shape[1]), (0, 0))),
                            np.pad(row2, ((0, 0), (0, 1600 - row2.shape[1]), (0, 0))), row3], 0)
    for k_, c in enumerate(PRIORITY2):
        cv2.rectangle(sheet, (1500 - 90, 8 + 20 * k_), (1500 - 72, 24 + 20 * k_), COLOR2[c], -1)
        cv2.putText(sheet, c, (1500 - 64, 22 + 20 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(OUT2 / "contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[grade2] contact sheet {OUT2 / 'contact_sheet.jpg'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("finish2").set_defaults(fn=stage_finish2)
    sub.add_parser("finr").set_defaults(fn=stage_fin_right)
    g2 = sub.add_parser("grade2")
    g2.add_argument("--workers", type=int, default=4)
    g2.add_argument("--only", nargs="*", default=None)
    g2.set_defaults(fn=stage_grade2)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
