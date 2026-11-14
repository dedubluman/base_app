#!/usr/bin/env python3
"""SAM 3.1 masks for inputs/blocks_dive.mov (one free diver, 850 frames).

Sub-commands (all CPU, via shades/sam31_cpu.py):

  smoke    torch + CPU port + checkpoint + the "person" detector on frame 240
  timing   16 frames, plan (a) detector+tracker on every frame against
           plan (b) detector on the seed frame only, then tracker-only
           propagation both ways (diver_masks_bwd.py / diver_parts.py style)
  run      plan (b) over the whole clip, resumable chunk by chunk

Plan (b), per chunk of --chunk frames (default 32, step chunk-overlap):
  * seed frame = the chunk's middle frame, so every frame is at most half a
    chunk away from a detector mask and no chunk starts at an edge
  * the text prompt "person" finds the diver on the seed frame; the text
    prompt "flipper" finds the fins ("fin" finds nothing on earlier clips)
  * the diver and each fin component become tracker objects (ids 101, 102...)
    seeded with points taken from those masks, propagated both ways
  * frames that two chunks cover keep the mask tracked over the shorter
    distance from its seed

Masks land in masks/<stem>__diver/subject (person) and masks/<stem>__fins/subject
(fins, only where the tracker followed a fin), plus meta.json with the seed
frame, tracking distance and area of every frame.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from diver_masks_bwd import prompts_from_mask
from shades.config import Masking
from shades.masking import _build_predictor, extract_jpeg_frames

REPO = Path(__file__).resolve().parent
CACHE = REPO / ".cache"
VIDEO = REPO / "inputs" / "blocks_dive.mov"
STEM = VIDEO.stem
FRAMES = CACHE / f"sam_frames_{STEM}_bwd"
DIVER_ID, FIN_ID = 101, 110          # fins use 110, 111
HEAD_TEXTS = ["head", "hair"]      # first one that finds a head on the seed wins


def load_predictor():
    import torch
    t0 = time.time()
    pred, on_gpu = _build_predictor(Masking(parts={}), REPO, CACHE)
    print(f"[sam] torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"[sam] SAM 3.1 checkpoint checkpoints/sam3.1/sam3.1_multiplex.pt loaded on "
          f"{'CUDA' if on_gpu else 'CPU (patched port)'} in {time.time() - t0:.0f}s", flush=True)
    model = pred.model
    full_out = model._build_sam2_output

    def tracker_only(inference_state, frame_idx, refined_obj_id_to_mask=None):
        # tracker-only frames have no cached detector output: start from empty
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        o.update(refined_obj_id_to_mask or {})
        return o

    def set_mode(tracker: bool):
        model._build_sam2_output = tracker_only if tracker else full_out

    return pred, on_gpu, set_mode


def chunk_dir(cs: int, ce: int, tag: str) -> Path:
    d = CACHE / f"sam_chunk_{STEM}_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i in range(cs, ce):
        (d / f"{i - cs:05d}.jpg").symlink_to(FRAMES / f"{i:05d}.jpg")
    return d


def detect(pred, sid, local_f, text):
    r = pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": local_f, "text": text})
    o = r["outputs"]
    masks = [np.asarray(m) > 0 for m in o["out_binary_masks"]]
    probs = list(o.get("out_probs", [])) or [None] * len(masks)
    return masks, probs


def pick_diver(masks):
    """The diver is the only person: take the biggest detection and absorb any
    other detection that overlaps it (a split torso/legs)."""
    masks = [m for m in masks if m.sum() >= 200]
    if not masks:
        return None
    big = max(masks, key=lambda m: m.sum())
    return big


def fin_components(masks, diver, grow=201, min_area=80):
    """Fin masks near the diver, split into connected components (one tracker
    object per fin); the biggest two are kept."""
    near = cv2.dilate(diver.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
    comps = []
    for m in masks:
        if m.sum() < min_area or (m & near).sum() / max(m.sum(), 1) < 0.8:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] >= min_area:
                comps.append(lab == k)
    comps.sort(key=lambda c: -c.sum())
    return comps[:2]


def head_object(pred, sid, local_f, diver, texts=None, log=None):
    """Union of the detections of the head/hair prompts that sit on the diver:
    the tracker drops this small dark blob while it follows the body."""
    texts = texts or HEAD_TEXTS
    near = cv2.dilate(diver.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    head = np.zeros(diver.shape, bool)
    found = {}
    for t in texts:
        masks, _ = detect(pred, sid, local_f, t)
        ok = [m for m in masks
              if 150 <= m.sum() <= 0.25 * diver.sum() and (m & near).sum() >= 0.9 * m.sum()]
        found[t] = [int(m.sum()) for m in masks]
        for m in ok:
            head |= m
        if head.any():
            break
    if log is not None:
        log.update({"head_detections": found, "head_area": int(head.sum())})
    return head


def track_chunk(pred, cs, ce, seed, log_prefix="", want_fins=True, tag="b", want_head=True):
    """Plan (b) on frames [cs, ce) with the detector on `seed` only. Returns
    ({frame: diver_mask}, {frame: fin_mask}, info) or None if no diver found."""
    cdir = chunk_dir(cs, ce, tag)
    sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
    info = {"cs": cs, "ce": ce, "seed": seed}
    cands = [seed] + [f for f in (cs + (ce - cs) // 4, cs + 3 * (ce - cs) // 4, cs, ce - 1) if f != seed]
    # (the extra mask = fin objects that left the person mask + the head object)
    diver, dseed = None, None
    for f in cands:
        masks, probs = detect(pred, sid, f - cs, "person")
        diver = pick_diver(masks)
        if diver is not None:
            dseed = f
            break
    if diver is None:
        pred.handle_request({"type": "close_session", "session_id": sid})
        return None
    info["seed"] = dseed
    fins = []
    if want_fins:
        fm, _ = detect(pred, sid, dseed - cs, "flipper")
        fins = fin_components(fm, diver)
    inside = [float((c & diver).sum() / c.sum()) for c in fins]
    info["fins_found"] = len(fins)
    info["fin_inside_diver"] = [round(v, 2) for v in inside]
    # rule 4: fins the "person" mask already holds ride along with the diver
    # object; a fin that falls outside it becomes its own tracker object
    fins = [c for c, v in zip(fins, inside) if v < 0.9]
    info["n_fin_obj"] = len(fins)
    head = head_object(pred, sid, dseed - cs, diver, log=info) if want_head else np.zeros(diver.shape, bool)
    pts, lab = prompts_from_mask(diver, n_pos=6, n_neg=6)
    if head.any():
        # the diver object is told to keep the head too (one call: a second
        # add_prompt for the same object may clear the first one's points)
        hp, hl = prompts_from_mask(head, n_pos=3, n_neg=0)
        pts, lab = pts + hp, lab + hl
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": dseed - cs,
                         "points": pts, "point_labels": lab, "obj_id": DIVER_ID})
    for k, c in enumerate(fins):
        p, l = prompts_from_mask(c, n_pos=4, n_neg=4)
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": dseed - cs,
                             "points": p, "point_labels": l, "obj_id": FIN_ID + k})
    got_d = {dseed: diver}
    got_f = {dseed: np.any(fins, 0)} if fins else {}
    direction = "forward" if dseed == cs else "backward" if dseed == ce - 1 else "both"
    for resp in pred.handle_stream_request({
            "type": "propagate_in_video", "session_id": sid,
            "propagation_direction": direction, "start_frame_index": dseed - cs}):
        fi = cs + int(resp["frame_index"])
        o = resp["outputs"]
        fm = np.zeros(diver.shape, bool)
        dm = None
        for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
            oid = int(oid)
            if oid == DIVER_ID:
                dm = np.asarray(mm) > 0
            elif FIN_ID <= oid < FIN_ID + 4:
                fm |= np.asarray(mm) > 0
        if fi != dseed:
            got_d[fi] = dm if dm is not None else np.zeros(diver.shape, bool)
            if fins:
                got_f[fi] = fm
    pred.handle_request({"type": "close_session", "session_id": sid})
    # the "extra" mask = fin objects that left the person mask
    extra = {}
    for fi in got_d:
        e = np.zeros(diver.shape, bool)
        if fi in got_f:
            e |= got_f[fi]
        extra[fi] = e
    return got_d, extra, info


# --------------------------------------------------------------------------
def cmd_smoke(a):
    if not (FRAMES / "00000.jpg").exists():
        extract_jpeg_frames(VIDEO, FRAMES, 2)
    pred, on_gpu, set_mode = load_predictor()
    set_mode(True)
    f = 240
    cdir = chunk_dir(f, f + 4, "smoke")
    sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
    t0 = time.time()
    masks, probs = detect(pred, sid, 0, "person")
    dt = time.time() - t0
    print(f"[smoke] frame {f}: detector 'person' -> {len(masks)} mask(s), areas "
          f"{[int(m.sum()) for m in masks]}, probs {[None if p is None else round(float(p), 2) for p in probs]}, "
          f"{dt:.1f}s", flush=True)
    d = pick_diver(masks)
    assert d is not None, "SAM 3.1 returned no diver mask on frame 240"
    out = REPO / "work" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(FRAMES / f"{f:05d}.jpg"))
    ov = img.copy()
    ov[d] = (0.5 * ov[d] + 0.5 * np.array([40, 40, 255])).astype(np.uint8)
    cv2.imwrite(str(out / "smoke_240.jpg"), np.concatenate([img, ov], 1)[:, :, :])
    ys, xs = np.nonzero(d)
    print(f"[smoke] OK diver mask {int(d.sum())} px, bbox x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}", flush=True)
    pred.handle_request({"type": "close_session", "session_id": sid})
    return 0


def cmd_timing(a):
    if not (FRAMES / "00000.jpg").exists():
        extract_jpeg_frames(VIDEO, FRAMES, 2)
    pred, on_gpu, set_mode = load_predictor()
    cs, ce, seed = 232, 248, 240
    res = {}
    out = REPO / "work" / "timing"
    out.mkdir(parents=True, exist_ok=True)

    # (b) first: detector on the seed only, tracker-only both ways
    set_mode(True)
    t0 = time.time()
    r = track_chunk(pred, cs, ce, seed, tag="tb", want_fins=not a.no_fins)
    tb = time.time() - t0
    got_d, got_f, info = r
    res["b"] = {"seconds_total": round(tb, 1), "s_per_frame": round(tb / (ce - cs), 2), "info": info,
                "diver_area": {k: int(v.sum()) for k, v in sorted(got_d.items())},
                "fin_area": {k: int(v.sum()) for k, v in sorted(got_f.items())}}
    print(f"[timing] (b) seed+tracker: {tb:.0f}s for {ce - cs} frames = {tb / (ce - cs):.1f} s/frame, {info}", flush=True)
    for k, v in got_d.items():
        cv2.imwrite(str(out / f"b_{k:05d}.png"), v.astype(np.uint8) * 255)

    if a.only == "b":
        return 0
    # (a) detector + tracker on every frame (what diver_masks.py does per chunk)
    set_mode(False)
    t0 = time.time()
    cdir = chunk_dir(cs, ce, "ta")
    sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": "person"})
    areas, ids = {}, set()
    for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
        fi = cs + int(resp["frame_index"])
        o = resp["outputs"]
        ms = [np.asarray(m) > 0 for m in o["out_binary_masks"]]
        ids |= {int(i) for i in o["out_obj_ids"]}
        if ms:
            m = max(ms, key=lambda x: x.sum())
            areas[fi] = int(m.sum())
            cv2.imwrite(str(out / f"a_{fi:05d}.png"), m.astype(np.uint8) * 255)
    pred.handle_request({"type": "close_session", "session_id": sid})
    ta = time.time() - t0
    res["a"] = {"seconds_total": round(ta, 1), "s_per_frame": round(ta / (ce - cs), 2), "areas": areas}
    print(f"[timing] (a) detector+tracker: {ta:.0f}s for {ce - cs} frames = {ta / (ce - cs):.1f} s/frame", flush=True)

    ious = []
    for k in range(cs, ce):
        pa = cv2.imread(str(out / f"a_{k:05d}.png"), 0)
        pb = cv2.imread(str(out / f"b_{k:05d}.png"), 0)
        if pa is None or pb is None:
            continue
        pa, pb = pa > 127, pb > 127
        ious.append(round(float((pa & pb).sum() / max((pa | pb).sum(), 1)), 3))
    res["iou_a_vs_b"] = ious
    print(f"[timing] IoU (a vs b) per frame: {ious}", flush=True)
    frames_a, frames_b = ta / (ce - cs), tb / (ce - cs)
    res["projection_850_frames_h"] = {"a": round(850 * frames_a / 3600, 2), "b": round(850 * frames_b / 3600, 2)}
    print(f"[timing] projection for 850 frames: (a) {850 * frames_a / 3600:.1f} h, (b) {850 * frames_b / 3600:.1f} h", flush=True)
    (out / "timing.json").write_text(json.dumps(res, indent=1))
    return 0


def dark_outside(i, mask, reach=25, thr=0.34):
    """Dark pixels (hair, mask, fins) just outside a diver mask: ~0 when the head is held."""
    x = cv2.cvtColor(cv2.imread(str(FRAMES / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    near = cv2.dilate(mask.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8)) > 0
    return int(((y < thr) & near & ~mask).sum())


def cmd_headtest(a):
    pred, on_gpu, set_mode = load_predictor()
    set_mode(True)
    cs, ce = 196, 228
    t0 = time.time()
    got_d, extra, info = track_chunk(pred, cs, ce, 212, tag="ht")
    print(f"[headtest] {time.time() - t0:.0f}s, {info}", flush=True)
    out = REPO / "work" / "headtest"
    out.mkdir(parents=True, exist_ok=True)
    old = REPO / "masks" / f"{STEM}__diver" / "subject"
    for i in sorted(got_d):
        new = got_d[i] | extra[i]
        om = cv2.imread(str(old / f"{i:05d}.png"), 0)
        before = dark_outside(i, om > 127) if om is not None else None
        print(f"[headtest] f{i}: person-only dark px outside {before}, person+extra {dark_outside(i, new)}, "
              f"area {int(got_d[i].sum())} extra {int(extra[i].sum())}", flush=True)
        cv2.imwrite(str(out / f"{i:05d}.png"), new.astype(np.uint8) * 255)
    return 0


def cmd_run(a):
    if not (FRAMES / "00000.jpg").exists():
        extract_jpeg_frames(VIDEO, FRAMES, 2)
    n = len(list(FRAMES.glob("*.jpg")))
    s0, s1 = (0, n) if not a.frames else map(int, a.frames.split(":"))
    ddir = REPO / "masks" / f"{STEM}__diver" / "subject"
    fdir = REPO / "masks" / f"{STEM}__fins" / "subject"
    ddir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    adir = REPO / "masks" / f"{STEM}__diver_alt" / "subject"
    adir.mkdir(parents=True, exist_ok=True)
    mf = REPO / "masks" / f"{STEM}__diver" / "meta.json"
    meta = json.loads(mf.read_text()) if mf.exists() else {"chunks": {}, "frames": {}}
    step = a.chunk - a.overlap
    chunks, s = [], 0
    while True:
        e = min(n, s + a.chunk)
        chunks.append((s, e))
        if e == n:
            break
        s += step
    chunks = [(cs, ce) for cs, ce in chunks if ce > s0 and cs < s1]
    pred, on_gpu, set_mode = load_predictor()
    set_mode(True)
    t_all = time.time()
    for ci, (cs, ce) in enumerate(chunks):
        key = f"{cs}-{ce - 1}"
        if key in meta["chunks"] and meta["chunks"][key].get("ok"):
            continue
        t0 = time.time()
        seed = cs + (ce - cs) // 2
        r = track_chunk(pred, cs, ce, seed, tag="run")
        if r is None:
            meta["chunks"][key] = {"ok": False, "reason": "no diver detected on any candidate seed"}
            print(f"[run] chunk {key}: NO DIVER DETECTED", flush=True)
            mf.write_text(json.dumps(meta, indent=1))
            continue
        got_d, got_f, info = r
        for fi, m in got_d.items():
            dist = abs(fi - info["seed"])
            old = meta["frames"].get(str(fi))
            if old is not None and dist >= old["dist"]:
                cv2.imwrite(str(adir / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
            if old is None or dist < old["dist"]:
                if old is not None:
                    shutil.copy(ddir / f"{fi:05d}.png", adir / f"{fi:05d}.png")
                cv2.imwrite(str(ddir / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
                fm = got_f.get(fi)
                if fm is not None and fm.any():
                    cv2.imwrite(str(fdir / f"{fi:05d}.png"), fm.astype(np.uint8) * 255)
                else:
                    (fdir / f"{fi:05d}.png").unlink(missing_ok=True)
                meta["frames"][str(fi)] = {"chunk": key, "seed": info["seed"], "dist": dist,
                                           "area": int(m.sum()),
                                           "fin_area": int(fm.sum()) if fm is not None else 0}
        info["seconds"] = round(time.time() - t0, 1)
        info["ok"] = True
        meta["chunks"][key] = info
        mf.write_text(json.dumps(meta, indent=1))
        print(f"[run] chunk {ci + 1}/{len(chunks)} {key}: seed {info['seed']}, extra fin objects {info['n_fin_obj']} "
              f"(fins found {info['fins_found']}, inside diver {info['fin_inside_diver']}), {info['seconds']:.0f}s "
              f"= {info['seconds'] / (ce - cs):.1f} s/frame", flush=True)
    meta["num_frames"] = n
    meta["seconds_this_run"] = round(time.time() - t_all, 1)
    mf.write_text(json.dumps(meta, indent=1))
    print(f"[run] done in {time.time() - t_all:.0f}s", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke").set_defaults(fn=cmd_smoke)
    t = sub.add_parser("timing")
    t.add_argument("--only", choices=["a", "b"], default=None)
    t.add_argument("--no-fins", action="store_true")
    t.set_defaults(fn=cmd_timing)
    sub.add_parser("headtest").set_defaults(fn=cmd_headtest)
    r = sub.add_parser("run")
    r.add_argument("--frames", default=None, help="start:end (end exclusive)")
    r.add_argument("--chunk", type=int, default=32)
    r.add_argument("--overlap", type=int, default=4)
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
