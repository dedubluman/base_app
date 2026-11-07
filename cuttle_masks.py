#!/usr/bin/env python3
"""SAM 3.1 instance masks for every cuttlefish in a clip.

One text prompt ("cuttlefish") per chunk: SAM 3.1's detector finds every
instance and its multiplex tracker propagates all of them together through
the chunk, each with its own object id. Chunks overlap; instances are linked
across chunk seams by mask IoU on the shared frames (Hungarian matching), so
every animal keeps one global id for the whole clip.

Output: masks/<stem>__cuttle/obj_<gid>/<frame:05d>.png (0/255, source res)
        masks/<stem>__cuttle/meta.json  (per object: frames, area, prob)

    python cuttle_masks.py --input inputs/cuttle.mov [--start 0 --end 40]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--prompt", default="cuttlefish")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-objects", type=int, default=8)
    ap.add_argument("--crop-y", type=int, default=None,
                    help="feed SAM a square WxW crop starting at this row (no aspect squash, "
                         "small animals stay larger); masks are pasted back at full frame")
    ap.add_argument("--crop-x", type=int, default=0, help="left edge of the crop (with --crop-y)")
    ap.add_argument("--crop-size", type=int, default=None,
                    help="side of the square crop (default: frame width); a tighter crop "
                         "gives SAM more pixels on a small subject such as thin fin rays")
    a = ap.parse_args()

    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    out = REPO / "masks" / f"{a.input.stem}__cuttle"
    full_dir = cache / f"sam_frames_{a.input.stem}"
    cfg = Masking(max_num_objects=a.max_objects, detector_batch=2)
    if not full_dir.exists() or not any(full_dir.glob("*.jpg")):
        extract_jpeg_frames(a.input, full_dir, cfg.jpeg_quality)
    n_all = len(list(full_dir.glob("*.jpg")))
    h, w = cv2.imread(str(full_dir / "00000.jpg")).shape[:2]
    frames_dir, cy, cx = full_dir, 0, 0
    if a.crop_y is not None:
        cy, cx = a.crop_y, a.crop_x
        cs = a.crop_size or w
        frames_dir = cache / f"sam_frames_{a.input.stem}_crop{cx}_{cy}_{cs}"
        if not frames_dir.exists():
            import subprocess
            frames_dir.mkdir()
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(a.input), "-fps_mode", "passthrough",
                            "-vf", f"crop={cs}:{cs}:{cx}:{cy}", "-q:v", "2", "-start_number", "0",
                            str(frames_dir / "%05d.jpg")], check=True)
    end = a.end or n_all
    print(f"[sam] {n_all} frames {w}x{h}; processing {a.start}-{end - 1}", flush=True)

    t0 = time.time()
    pred, on_gpu = _build_predictor(cfg, REPO, cache)
    print(f"[sam] SAM 3.1 multiplex loaded on {'CUDA' if on_gpu else 'CPU'} "
          f"in {time.time() - t0:.0f}s", flush=True)

    if out.exists() and a.start == 0:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_f = out / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"objects": {}, "chunks": []}

    # global masks: gid -> {frame: bool mask}
    G: dict[int, dict[int, np.ndarray]] = {}
    for d in out.glob("obj_*"):
        gid = int(d.name[4:])
        G[gid] = {int(p.stem): cv2.imread(str(p), 0) > 127 for p in d.glob("*.png")}
    P: dict[int, dict[int, float]] = {int(k): {int(f): p for f, p in v["prob"].items()}
                                      for k, v in meta["objects"].items()}

    chunks = [(s + a.start, e + a.start) for s, e in _chunks(end - a.start, a.chunk, a.overlap)]
    for ci, (cs, ce) in enumerate(chunks):
        t0 = time.time()
        cdir = cache / f"sam_chunk_{a.input.stem}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames_dir / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": 0, "text": a.prompt})
        local: dict[int, dict[int, np.ndarray]] = {}
        lprob: dict[int, dict[int, float]] = {}
        for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            pr = list(o.get("out_probs", [])) or [None] * len(o["out_obj_ids"])
            for oid, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], pr):
                m = np.asarray(m) > 0
                if cy or cx or m.shape[0] != h:
                    full = np.zeros((h, w), bool)
                    full[cy:cy + m.shape[0], cx:cx + m.shape[1]] = m
                    m = full
                if m.any():
                    local.setdefault(int(oid), {})[fi] = m
                    lprob.setdefault(int(oid), {})[fi] = None if p is None else float(p)
        pred.handle_request({"type": "close_session", "session_id": sid})

        # link chunk-local ids to global ids via IoU on overlapping frames
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
        for l, g in mapping.items():
            G.setdefault(g, {})
            P.setdefault(g, {})
            for f, m in local[l].items():
                # seam frames: keep the mask from the chunk where f is further
                # from the edge (less propagation drift) -> prefer the new chunk
                # past the overlap midpoint
                if f in G[g] and f < cs + a.overlap // 2:
                    continue
                G[g][f] = m
                P[g][f] = lprob[l][f]
                d = out / f"obj_{g}"
                d.mkdir(exist_ok=True)
                cv2.imwrite(str(d / f"{f:05d}.png"), m.astype(np.uint8) * 255)
        line = ", ".join(f"{l}->{mapping[l]}({len(local[l])}f)" for l in lids)
        print(f"[sam] chunk {ci + 1}/{len(chunks)} {cs}-{ce - 1}: {line or 'nothing'} "
              f"{time.time() - t0:.0f}s", flush=True)
        meta["chunks"].append({"range": [cs, ce], "map": {str(k): v for k, v in mapping.items()}})
        meta["objects"] = {str(g): {"frames": len(G[g]),
                                    "median_area": float(np.median([m.sum() for m in G[g].values()])),
                                    "prob": {str(f): p for f, p in sorted(P[g].items())}}
                           for g in G}
        meta.update(crop_y=cy, crop_x=cx, num_frames=n_all, width=w, height=h, prompt=a.prompt,
                    device="cuda" if on_gpu else "cpu")
        meta_f.write_text(json.dumps(meta, indent=1))
        del local
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
