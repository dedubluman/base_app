#!/usr/bin/env python3
"""SAM 3.1 masks for a fish clip: one text prompt per scene region.

Each ``name:prompt`` pair is segmented with SAM 3.1's multiplex video
predictor (overlapping chunks, instances linked across chunk seams by mask
IoU, exactly like ``cuttle_masks.py``). Output per region:

    masks/<stem>__<name>/obj_<gid>/<frame:05d>.png   0/255, source resolution
    masks/<stem>__<name>/meta.json                   frames, area, prob per object

    python fish_masks.py --input inputs/fish.mov \
        --regions fish:fish sand:sand grass:seagrass
    python fish_masks.py --input inputs/fish.mov --regions fish:fish --start 0 --end 12   # timing probe

``--stride N`` runs SAM on every Nth frame only (the chunk is a strided
sub-sequence, so SAM's tracker sees a faster-moving video); the in-between
frames are filled later by ``fish_refine.py`` with soft-mask interpolation.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from shades.config import Masking
from shades.masking import _build_predictor, _chunks, extract_jpeg_frames

REPO = Path(__file__).resolve().parent


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def run_region(pred, on_gpu, frames_dir, idx, name, prompt, out, a, w, h, stride):
    """Segment one region over source frames `idx` (list of frame numbers)."""
    if out.exists() and a.start == 0 and a.end is None and not a.resume:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_f = out / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"objects": {}, "chunks": []}

    G: dict[int, dict[int, np.ndarray]] = {}   # gid -> {frame: bool mask}
    P: dict[int, dict[int, float]] = {}
    for d in out.glob("obj_*"):
        gid = int(d.name[4:])
        G[gid] = {int(p.stem): cv2.imread(str(p), 0) > 127 for p in d.glob("*.png")}
    for k, v in meta["objects"].items():
        P[int(k)] = {int(f): p for f, p in v["prob"].items()}

    chunks = _chunks(len(idx), a.chunk, a.overlap)
    done = {tuple(c["range"]) for c in meta["chunks"]} if a.resume else set()
    for ci, (cs, ce) in enumerate(chunks):
        if (idx[cs], idx[ce - 1]) in done:        # finished before a crash / OOM kill
            print(f"[sam:{name}] chunk {ci + 1}/{len(chunks)} already done, skipping", flush=True)
            continue
        t0 = time.time()
        cdir = REPO / ".cache" / f"sam_chunk_{a.input.stem}_{name}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True)
        for j in range(cs, ce):
            (cdir / f"{j - cs:05d}.jpg").symlink_to(frames_dir / f"{idx[j]:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
        local: dict[int, dict[int, np.ndarray]] = {}
        lprob: dict[int, dict[int, float]] = {}
        for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            fi = idx[cs + int(resp["frame_index"])]
            o = resp["outputs"]
            pr = list(o.get("out_probs", [])) or [None] * len(o["out_obj_ids"])
            for oid, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], pr):
                m = np.asarray(m) > 0
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                if m.any():
                    local.setdefault(int(oid), {})[fi] = m
                    lprob.setdefault(int(oid), {})[fi] = None if p is None else float(p)
        pred.handle_request({"type": "close_session", "session_id": sid})

        # link chunk-local ids to global ids via IoU on the overlapping frames
        lids, gids = sorted(local), sorted(G)
        mapping: dict[int, int] = {}
        if lids and gids:
            C = np.zeros((len(lids), len(gids)))
            for x, l in enumerate(lids):
                for y, g in enumerate(gids):
                    shared = [f for f in local[l] if f in G[g]]
                    C[x, y] = np.mean([iou(local[l][f], G[g][f]) for f in shared]) if shared else 0
            for x, y in zip(*linear_sum_assignment(-C)):
                if C[x, y] > 0.3:
                    mapping[lids[x]] = gids[y]
        for l in lids:
            if l not in mapping:
                mapping[l] = max(list(G) + list(mapping.values()) + [0]) + 1
        seam_end = idx[cs + a.overlap // 2] if cs else -1
        for l, g in mapping.items():
            G.setdefault(g, {})
            P.setdefault(g, {})
            for f, m in local[l].items():
                if f in G[g] and f < seam_end:      # first half of an overlap: keep the older chunk
                    continue
                G[g][f] = m
                P[g][f] = lprob[l][f]
                d = out / f"obj_{g}"
                d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"{f:05d}.png"), m.astype(np.uint8) * 255)
        line = ", ".join(f"{l}->{mapping[l]}({len(local[l])}f)" for l in lids)
        print(f"[sam:{name}] chunk {ci + 1}/{len(chunks)} frames {idx[cs]}-{idx[ce - 1]}: "
              f"{line or 'nothing'}  {time.time() - t0:.0f}s ({(time.time() - t0) / (ce - cs):.1f}s/frame)",
              flush=True)
        meta["chunks"].append({"range": [idx[cs], idx[ce - 1]], "map": {str(k): v for k, v in mapping.items()}})
        meta["objects"] = {str(g): {"frames": len(G[g]),
                                    "median_area": float(np.median([m.sum() for m in G[g].values()])),
                                    "prob": {str(f): p for f, p in sorted(P[g].items())}}
                           for g in G}
        meta.update(num_frames=len(list(frames_dir.glob("*.jpg"))), width=w, height=h,
                    prompt=prompt, stride=stride, device="cuda" if on_gpu else "cpu")
        meta_f.write_text(json.dumps(meta, indent=1))
        del local
        gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--regions", nargs="+", required=True, metavar="NAME:PROMPT[:STRIDE]")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-objects", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                    help="keep masks/ and skip chunks already recorded in meta.json")
    a = ap.parse_args()

    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    cfg = Masking(max_num_objects=a.max_objects, detector_batch=2)
    full_dir = cache / f"sam_frames_{a.input.stem}"
    if not full_dir.exists() or not any(full_dir.glob("*.jpg")):
        extract_jpeg_frames(a.input, full_dir, cfg.jpeg_quality)
    n_all = len(list(full_dir.glob("*.jpg")))
    h, w = cv2.imread(str(full_dir / "00000.jpg")).shape[:2]
    end = a.end or n_all
    print(f"[sam] {n_all} frames {w}x{h}", flush=True)

    t0 = time.time()
    pred, on_gpu = _build_predictor(cfg, REPO, cache)
    print(f"[sam] SAM 3.1 multiplex loaded on {'CUDA' if on_gpu else 'CPU'} in {time.time() - t0:.0f}s", flush=True)

    for reg in a.regions:                          # NAME:PROMPT[:STRIDE]
        parts = reg.split(":")
        name, prompt = parts[0], parts[1]
        stride = int(parts[2]) if len(parts) > 2 else a.stride
        idx = list(range(a.start, end, stride))
        if idx[-1] != end - 1:
            idx.append(end - 1)                   # always include the last frame
        print(f"[sam:{name}] '{prompt}' on {len(idx)} frames ({a.start}-{end - 1}, stride {stride})", flush=True)
        out = REPO / "masks" / f"{a.input.stem}__{name}"
        run_region(pred, on_gpu, full_dir, idx, name, prompt, out, a, w, h, stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
