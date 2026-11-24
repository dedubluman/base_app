#!/usr/bin/env python3
"""Targeted SAM 3.1 re-segmentation of the fish on selected keyframes (dense point prompts; the SAM 3.1 API
does not accept points together with a box).

For every selected keyframe the current fish keyframe mask (masks/<stem>__fish + merged fins) is turned into
  * positive clicks spread over the body incl. dark stripes and the fins,
  * negative clicks on thin structures that touch or cross the fish (stems, twigs: spurs outside the opened
    body and thin gaps inside its convex hull) and on a ring of water / sand / foliage around it,
and SAM 3.1 segments that single frame again. The result is accepted only if it agrees with the old mask
(IoU >= --min-iou), otherwise the old mask is kept (guards against SAM jumping onto a neighbouring stem).

Output: masks/<stem>__fishrefit/obj_1/<frame:05d>.png (binary 0/255) + meta.json (IoU, areas, clicks)

    python fish_resegment.py --input inputs/fish.mov --windows 0-30:3 150-225:2 345-428:3
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from fish_refine import load_objects, union_frames
from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent


def old_mask(stem, f, objs, fobjs):
    m = union_frames(objs, [1], [f]).get(f)
    for g in fobjs:                                   # merged fin parts (touching the fish)
        p = fobjs[g].get(f)
        if p is not None and m is not None:
            x = cv2.imread(str(p), 0) > 127
            d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)
            if x.sum() > 80 and d[x].min() <= 25:
                m = m | x
    return m


def make_prompts(m: np.ndarray, rng, n_pos=9):
    h, w = m.shape
    m8 = m.astype(np.uint8)
    ys, xs = np.nonzero(m8)
    box = [xs.min() / w, ys.min() / h, xs.max() / w, ys.max() / h]
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    r = max(3, int(0.045 * dist.max()))
    opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    body = cv2.dilate(opened, np.ones((5, 5), np.uint8))
    pos, neg = [], []
    # positives: farthest-point sampling among pixels well inside the body (covers stripes and fins alike)
    inner = np.argwhere(cv2.distanceTransform(opened, cv2.DIST_L2, 5) >= max(2.0, 0.35 * dist.max()))
    if len(inner) == 0:
        inner = np.argwhere(m8 > 0)
    picks = [inner[rng.integers(len(inner))]]
    for _ in range(n_pos - 1):
        d = np.min([np.hypot(*(inner - p).T) for p in picks], 0)
        picks.append(inner[int(d.argmax())])
    pos += [[p[1] / w, p[0] / h] for p in picks]
    # fins: thick parts that are in the mask but outside the opened body's core -> one positive each
    fin = (m8 > 0) & (cv2.dilate(opened, np.ones((9, 9), np.uint8)) == 0)
    n, lab, st, cen = cv2.connectedComponentsWithStats(fin.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, 4] > 300:
            cy, cx = np.argwhere(lab == i).mean(0)
            k = np.argwhere(lab == i)[np.argmin(np.hypot(*(np.argwhere(lab == i) - [cy, cx]).T))]
            pos.append([k[1] / w, k[0] / h])
    # negatives (1): spurs = thin parts of the mask that stick out of the opened body (stems touching the belly)
    spur = (m8 > 0) & (body == 0)
    n, lab, st, cen = cv2.connectedComponentsWithStats(spur.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if st[i, 4] > 20:
            pts = np.argwhere(lab == i)
            if cv2.distanceTransform((lab == i).astype(np.uint8), cv2.DIST_L2, 3).max() <= 6:   # thin: stem-like
                k = pts[len(pts) // 2]
                neg.append([k[1] / w, k[0] / h])
    # negatives (2): thin gaps inside the hull (stems crossing the fish are already excluded - reinforce)
    cnt, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(m8)
    cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cnt))], 1)
    gap = (hull > 0) & (m8 == 0)
    n, lab, st, cen = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        comp = (lab == i).astype(np.uint8)
        if 40 < st[i, 4] < 0.08 * m8.sum() and cv2.distanceTransform(comp, cv2.DIST_L2, 3).max() <= 9:
            k = np.argwhere(comp)[st[i, 4] // 2]
            neg.append([k[1] / w, k[0] / h])
    # negatives (3): ring of water / sand / foliage 18-34 px outside
    outer = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5)
    ring = np.argwhere((outer > 18) & (outer < 34))
    if len(ring):
        c = ring.mean(0)
        ang = np.arctan2(ring[:, 0] - c[0], ring[:, 1] - c[1])
        for a in np.linspace(-np.pi, np.pi, 10, endpoint=False):
            k = ring[np.argmin(np.abs(np.angle(np.exp(1j * (ang - a)))))]
            neg.append([k[1] / w, k[0] / h])
    return box, pos, neg[:24]


def parse_windows(ws):
    fr = set()
    for w in ws:
        rng, _, st = w.partition(":")
        a, b = [int(x) for x in rng.split("-")]
        fr |= set(range(a, b + 1, int(st or 3)))
    return sorted(fr)


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--windows", nargs="*", default=[], help="A-B:STEP in source frames")
    ap.add_argument("--min-iou", type=float, default=0.75)
    ap.add_argument("--frames", default=None, help="comma list (overrides --windows, for tests)")
    a = ap.parse_args()
    stem = a.input.stem
    frames = [int(x) for x in a.frames.split(",")] if a.frames else parse_windows(a.windows)
    cache = REPO / ".cache"
    fdir = cache / f"sam_frames_{stem}"
    objs, _ = load_objects(REPO / "masks" / f"{stem}__fish")
    fp = REPO / "masks" / f"{stem}__fins"
    fobjs = load_objects(fp)[0] if (fp / "meta.json").exists() else {}
    out = REPO / "masks" / f"{stem}__fishrefit"
    (out / "obj_1").mkdir(parents=True, exist_ok=True)
    meta_f = out / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    pred, _on = _build_predictor(Masking(max_num_objects=4, detector_batch=1), REPO, cache)
    rng = np.random.default_rng(0)
    h, w = cv2.imread(str(fdir / "00000.jpg")).shape[:2]
    t_all = time.time()
    for n_, f in enumerate(frames):
        if str(f) in meta and (out / "obj_1" / f"{f:05d}.png").exists():
            continue
        t0 = time.time()
        om = old_mask(stem, f, objs, fobjs)
        if om is None or om.sum() < 200:
            continue
        box, pos, neg = make_prompts(om, rng)
        cdir = cache / f"sam_chunk_{stem}_refit"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True)
        (cdir / "00000.jpg").symlink_to(fdir / f"{f:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        req = {"type": "add_prompt", "session_id": sid, "frame_index": 0, "obj_id": 1,
               "points": torch.tensor(pos + neg, dtype=torch.float32),
               "point_labels": torch.tensor([1] * len(pos) + [0] * len(neg), dtype=torch.int32)}
        pred.handle_request(req)
        cand = None
        for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                "propagation_direction": "forward", "start_frame_index": 0}):
            if int(resp["frame_index"]) != 0:
                continue
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                x = np.asarray(mm) > 0
                if x.shape != (h, w):
                    x = cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if cand is None or x.sum() > cand.sum():
                    cand = x
        pred.handle_request({"type": "close_session", "session_id": sid})
        iou = float((cand & om).sum() / max((cand | om).sum(), 1)) if cand is not None else 0.0
        ok = cand is not None and iou >= a.min_iou
        final = cand if ok else om
        cv2.imwrite(str(out / "obj_1" / f"{f:05d}.png"), final.astype(np.uint8) * 255)
        meta[str(f)] = {"iou_vs_old": round(iou, 3), "accepted": bool(ok), "area_old": int(om.sum()),
                        "area_new": int(cand.sum()) if cand is not None else 0, "n_pos": len(pos), "n_neg": len(neg)}
        meta_f.write_text(json.dumps(meta, indent=1))
        print(f"[refit] f{f}: IoU {iou:.2f} {'ACCEPT' if ok else 'keep old'}  area {om.sum()} -> "
              f"{cand.sum() if cand is not None else 0}  clicks +{len(pos)}/-{len(neg)}  {time.time() - t0:.0f}s "
              f"({n_ + 1}/{len(frames)})", flush=True)
    print(f"[refit] done in {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
