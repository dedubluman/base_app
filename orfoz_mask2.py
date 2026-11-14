#!/usr/bin/env python3
"""Second-generation grouper mask: one continuous SAM 3.1 track + pattern cleanup.

Why: the first masks came from independent text-prompted SAM chunks; each chunk
drew the black body / black crevice border differently, so the mask area
stepped by 15-40 % at the seams (frames 139, 207, 275, 309) and flat shadow
bands (left of the tail, above the back) were included.

Method
  evidence  patterned-body region from a top-hat spot detector (the fish is the
            only thing here with a cream spot pattern); flat shadow is not fish
  track     SAM 3.1 tracker-only propagation (~2 s/frame, as diver_masks_bwd.py)
            from the clearest frame, forward and backward, seeded by point
            prompts taken from the cleaned mask; chunks overlap by 12 frames and
            are joined by a linear cross-fade
  clean     largest component, small islands out, holes filled, open + close
  finish    erode, gaussian feather, +-2 frame smoothing, exit fade

    python orfoz_mask2.py --stage evid          # cleanup of the old masks only (no SAM)
    python orfoz_mask2.py --stage track         # SAM tracker pass + stitch
    python orfoz_mask2.py --stage finish        # alpha, exit fade, QC report
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
FR = REPO / ".cache" / "sam_frames_orfoz_sam"
OLD = REPO / "masks" / "orfoz_sam__cuttle" / "obj_1"
OUT = REPO / "masks" / "orfoz_v2"
W, H, N = 1232, 1078, 348
K = lambda d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def yl(i):
    """Boosted luma (the SAM proxy already is (4y)^0.6) 0..1, full res."""
    return cv2.imread(str(FR / f"{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255


def old_mask(i):
    p = OLD / f"{i:05d}.png"
    m = cv2.imread(str(p), 0) if p.exists() else None
    return (m > 127) if m is not None else np.zeros((H, W), bool)


def evidence(i, ref, thr_rel=0.30, th_px=41, margin=26, min_blob=25, inner_px=41):
    """Patterned-body region: convex hull of the cream spots and the pale tail
    edge (white top-hat blobs inside `ref`), grown by `margin` px. The hull keeps
    the dark body between the spots; flat shadow beyond the outermost pattern
    (left band, upper-right bulge, band left of the tail edge) falls outside."""
    y = yl(i)
    th = y - cv2.morphologyEx(y, cv2.MORPH_OPEN, K(th_px))
    near = cv2.erode(ref.astype(np.uint8), K(inner_px)) > 0   # ignore specks on the mask border
    p99 = np.percentile(th[ref], 99) if ref.any() else 0.2
    sp = ((th > max(0.04, thr_rel * p99)) & near).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sp, connectivity=8)
    keep = np.isin(lab, [c for c in range(1, n) if st[c, cv2.CC_STAT_AREA] >= min_blob])
    ys, xs = np.nonzero(keep)
    if len(xs) < 30:
        return np.ones((H, W), bool)
    hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
    e = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(e, hull, 1)
    return cv2.dilate(e, K(2 * margin + 1)) > 0


def clean(m, island=0.01, op=5, cl=9):
    """Largest component, islands < 1 % out, holes filled, open + close."""
    m8 = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    out = (lab == keep)
    for c in range(1, n):
        if c != keep and areas[c - 1] >= island * areas.max():
            pass  # only the largest component is kept by design
    ff = out.astype(np.uint8).copy()
    cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
    out = out | (ff == 0)
    o8 = out.astype(np.uint8)
    o8 = cv2.morphologyEx(o8, cv2.MORPH_OPEN, K(op))
    o8 = cv2.morphologyEx(o8, cv2.MORPH_CLOSE, K(cl))
    return o8 > 0


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def qc(masks, skip=()):
    """Consecutive-frame area change <= 10 %, IoU >= 0.85. Returns violations."""
    bad = []
    idx = sorted(masks)
    for a, b in zip(idx[:-1], idx[1:]):
        if b in skip:
            continue
        A, B = masks[a].sum(), masks[b].sum()
        da = abs(B / max(A, 1) - 1)
        u = iou(masks[a], masks[b])
        if da > 0.10 or u < 0.85:
            bad.append((b, round(da, 3), round(u, 3)))
    return bad


def stage_evid(a):
    t0 = time.time()
    res = {}
    prev = None
    for i in range(2, N):
        m = old_mask(i)
        e = evidence(i, m)
        c = clean(m & e)
        res[i] = c
        if i % 40 == 0:
            print(f"  frame {i}: old {m.sum()} -> trimmed {c.sum()}", flush=True)
    bad = qc(res)
    print(f"[evid] {len(res)} frames in {time.time() - t0:.0f}s; QC violations: {len(bad)}", flush=True)
    for b in bad[:40]:
        print("   frame %d area_change %.3f iou %.3f" % b)
    (OUT / "evid_dir").mkdir(exist_ok=True)
    for i, c in res.items():
        cv2.imwrite(str(OUT / "evid_dir" / f"{i:05d}.png"), c.astype(np.uint8) * 255)


def stage_track(a):
    """One continuous SAM 3.1 tracker pass from the anchor, forward and backward."""
    import inspect  # noqa: F401
    from diver_masks_bwd import prompts_from_mask
    from shades.config import Masking
    from shades.masking import _build_predictor

    A, L, OV = a.anchor, a.chunk, a.overlap
    cache = REPO / ".cache"
    pred, _ = _build_predictor(Masking(parts={}), REPO, cache)
    model = pred.model

    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        out = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        out.update(refined_obj_id_to_mask or {})
        return out

    model._build_sam2_output = _build_sam2_output   # keep tracker-only masks (see diver_masks_bwd.py)
    tdir = OUT / "track"
    shutil.rmtree(tdir, ignore_errors=True)
    tdir.mkdir(parents=True)
    final = {A: clean(old_mask(A)).astype(np.float32)}
    log = {}

    def run_chunk(cs, ce, seed, direction):
        t0 = time.time()
        seed_m = final[seed] > 0.5
        pts, lab = prompts_from_mask(seed_m)
        cdir = cache / "sam_chunk_orfoz_trk"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(FR / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": seed - cs, "text": "fish"})
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": seed - cs,
                             "points": pts, "point_labels": lab, "obj_id": 1})
        got = {}
        for resp in pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": direction, "start_frame_index": seed - cs}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            m = np.zeros((H, W), bool)
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                if int(oid) == 1:
                    m = np.asarray(mm) > 0
            got[fi] = m
        pred.handle_request({"type": "close_session", "session_id": sid})
        log[f"{direction}:{cs}-{ce - 1}"] = {"sec": round(time.time() - t0), "seed": seed,
                                            "areas": {str(k): int(v.sum()) for k, v in sorted(got.items())}}
        print(f"[track] {direction} chunk {cs}-{ce - 1} seed {seed}: {len(got)} frames, "
              f"seed area {int(seed_m.sum())}, last area {int(got[max(got) if direction == 'forward' else min(got)].sum())}, "
              f"{time.time() - t0:.0f}s", flush=True)
        return got

    # forward: A -> N-1
    seed = A
    while True:
        ce = min(N, seed + L)
        got = run_chunk(seed, ce, seed, "forward")
        for f in range(seed, ce):
            m = clean(got[f]).astype(np.float32) if got.get(f) is not None and got[f].any() else final.get(f, np.zeros((H, W), np.float32))
            if f in final and f > seed and f <= seed + OV:      # overlap with previous chunk: linear cross-fade
                w = (f - seed) / (OV + 1)
                final[f] = (1 - w) * final[f] + w * m
            elif f not in final:
                final[f] = m
        for f in range(seed, ce):
            cv2.imwrite(str(tdir / f"{f:05d}.png"), (final[f] * 255).astype(np.uint8))
        if ce >= N:
            break
        seed = ce - OV - 1
    # backward: A -> 0 (frames 0,1 are never detected in the old pass either; the tracker covers them)
    seed = A
    while True:
        cs = max(0, seed - L + 1)
        got = run_chunk(cs, seed + 1, seed, "backward")
        for f in range(seed, cs - 1, -1):
            m = clean(got[f]).astype(np.float32) if got.get(f) is not None and got[f].any() else final.get(f, np.zeros((H, W), np.float32))
            if f in final and f < seed and f >= seed - OV:
                w = (seed - f) / (OV + 1)
                final[f] = (1 - w) * final[f] + w * m
            elif f not in final:
                final[f] = m
        for f in range(cs, seed + 1):
            cv2.imwrite(str(tdir / f"{f:05d}.png"), (final[f] * 255).astype(np.uint8))
        if cs <= 0:
            break
        seed = cs + OV + 1
    (OUT / "track_log.json").write_text(json.dumps(log, indent=1))
    bad = qc({f: v > 0.5 for f, v in final.items()})
    print(f"[track] done. raw-track QC violations: {len(bad)}", flush=True)
    for b in bad[:30]:
        print("   frame %d area_change %.3f iou %.3f" % b)


def hull_trim(f, m, margin=26):
    """Leakage trim for the early frames: keep only the patterned body (+margin)."""
    return m & evidence(f, m, margin=margin)


def sdf_half(m):
    """Signed distance (px at half res), positive inside. Interpolating SDFs morphs a
    shape smoothly; blending binary masks makes the area jump when the weight crosses 0.5."""
    s = (cv2.resize(m.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    return (cv2.distanceTransform(s, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - s, cv2.DIST_L2, 5)).astype(np.float32)


def stage_finish(a):
    """tracked masks -> clean -> exit fade -> temporal SDF median/mean until QC is clean
    -> alpha (erode 0.2 % W, gaussian 0.7 % W, opacity fade)."""
    t0 = time.time()
    tdir, adir, bdir = OUT / "track", OUT / "alpha", OUT / "bin"
    for d in (adir, bdir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    cl = {}
    for f in range(N):
        m = cv2.imread(str(tdir / f"{f:05d}.png"), 0) > 127
        if a.hull_trim and f < 139 and m.any():
            m = hull_trim(f, m)
        cl[f] = clean(m) if m.any() else m
    area = np.array([cl[f].sum() for f in range(N)], float)
    peak = float(np.median(np.sort(area)[-20:]))          # peak area, robust to a one-frame spike
    f0 = next((f for f in range(a.exit_from, N) if area[f] < 0.25 * peak), None)
    fade = np.ones(N, np.float32)
    if f0 is not None:
        for f in range(f0, N):
            cl[f] = cl[f0 - 1]                             # freeze the shape, do not chase fragments
            fade[f] = max(0.0, 1.0 - (f - f0 + 1) / 10.0)
    print(f"[finish] cleaned in {time.time() - t0:.0f}s; peak area {peak:.0f}; exit fade from frame {f0}", flush=True)
    sd = np.stack([sdf_half(cl[f]) for f in range(N)])
    # one-frame spikes (e.g. the anchor frame) die in a +-2 temporal median of the SDF
    med = np.stack([np.median(sd[max(0, f - 2):f + 3], axis=0) for f in range(N)])
    win = a.smooth
    while True:
        wts = np.array([win + 1 - abs(k) for k in range(-win, win + 1)], np.float32)
        sm = {}
        for f in range(N):
            ks = [k for k in range(-win, win + 1) if 0 <= f + k < N]
            ww = np.array([win + 1 - abs(k) for k in ks], np.float32)
            acc = sum(w * med[f + k] for w, k in zip(ww, ks)) / ww.sum()
            m = cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR) > 0
            sm[f] = clean(m) if m.any() else m
        bad = qc({f: sm[f] for f in range(2, N) if sm[f].any() and fade[f] == 1.0})
        print(f"[qc] SDF smoothing +-{win} frames: violations {len(bad)}", flush=True)
        for b in bad[:30]:
            print("   frame %d area_change %.3f iou %.3f" % b, flush=True)
        if not bad or win >= 8:
            break
        win += 1
    er = max(1, int(round(0.002 * W)))
    sig = 0.007 * W
    for f in range(N):
        cv2.imwrite(str(bdir / f"{f:05d}.png"), sm[f].astype(np.uint8) * 255)
        m8 = cv2.erode(sm[f].astype(np.uint8), K(2 * er + 1)) if sm[f].any() else sm[f].astype(np.uint8)
        al = cv2.GaussianBlur(m8.astype(np.float32), (0, 0), sig) * fade[f]
        cv2.imwrite(str(adir / f"{f:05d}.png"), np.clip(al * 255 + 0.5, 0, 255).astype(np.uint8))
    ar2 = [int(sm[f].sum()) for f in range(N)]
    (OUT / "finish.json").write_text(json.dumps(dict(peak=peak, exit_frame=f0, smoothing_window=win,
                                                    violations=[list(b) for b in bad], area=ar2), indent=1))
    print(f"[finish] done in {time.time() - t0:.0f}s; smoothing window +-{win}; remaining violations {len(bad)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="evid")
    ap.add_argument("--anchor", type=int, default=212, help="clearest frame, inside 182-242")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--exit-from", dest="exit_from", type=int, default=309)
    ap.add_argument("--smooth", type=int, default=2, help="start half-window (frames) of the temporal SDF smoothing")
    ap.add_argument("--hull-trim", dest="hull_trim", action="store_true", help="old pattern-hull trim of frames <139 (off: tracker is tight)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"evid": stage_evid, "track": stage_track, "finish": stage_finish}[a.stage](a)
