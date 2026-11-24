#!/usr/bin/env python3
"""SAM 3.1 keyframe masks for a single subject (the striped fish).

Why a crop + stride
  * SAM resizes every frame to 1008x1008. A 1080x1920 portrait frame would be
    squashed vertically (the fish is small), so SAM is fed a square WxW crop
    (no distortion, ~0.93x scale) that covers the fish's whole travel range.
  * On this CPU SAM 3.1 costs ~13 s/frame. Every `--stride`-th frame is
    segmented (default 2); `fish_grade.py` fills the frames in between by
    optical-flow warping of the two neighbouring SAM masks.

How it runs
  * keyframes are cut into overlapping chunks (SAM keeps ~100 MB of features
    per frame in RAM). Each chunk is its own SAM 3.1 video session prompted
    with the text prompt on its first frame; SAM propagates it forward.
  * across chunk seams the fish is re-identified by mask IoU on the shared
    frames; the seam uses the later chunk's mask past the overlap midpoint
    (less propagation drift).
  * every finished chunk is written to disk, so a crash resumes where it was.

Output
  masks/<name>/key/<source_frame:05d>.png   0/255, full source resolution
  masks/<name>/meta.json                    per-keyframe prob / area / chunk

    python fish_masks.py --input inputs/fish.mov --name fish --crop-y 600
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.masking import _build_predictor, _chunks

REPO = Path(__file__).resolve().parent


def probe(path: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_packets", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_read_packets"])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--name", default=None, help="output folder name (default: input stem)")
    ap.add_argument("--prompt", default="fish")
    ap.add_argument("--crop-y", type=int, required=True,
                    help="top row of the square WxW crop fed to SAM")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=40, help="keyframes per SAM session")
    ap.add_argument("--overlap", type=int, default=4, help="shared keyframes between chunks")
    ap.add_argument("--max-objects", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N keyframes (tests)")
    a = ap.parse_args()

    name = a.name or a.input.stem
    out = REPO / "masks" / name
    key_dir = out / "key"
    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    W, H, n_src = probe(a.input)
    cy = a.crop_y
    assert 0 <= cy and cy + W <= H, "crop must fit inside the frame"
    src_idx = list(range(0, n_src, a.stride))
    if a.limit:
        src_idx = src_idx[:a.limit]
    n = len(src_idx)

    frames_dir = cache / f"sam_key_{name}_s{a.stride}_y{cy}"
    if not (frames_dir.exists() and len(list(frames_dir.glob("*.jpg"))) >= n):
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(a.input),
             "-vf", f"select='not(mod(n\\,{a.stride}))',crop={W}:{W}:0:{cy}",
             "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0",
             str(frames_dir / "%05d.jpg")], check=True)
    print(f"[sam] source {W}x{H}, {n_src} frames -> {n} keyframes (stride {a.stride}), "
          f"square crop rows {cy}-{cy + W}", flush=True)

    chunks = _chunks(n, a.chunk, a.overlap)
    key = {"n": n, "stride": a.stride, "crop_y": cy, "prompt": a.prompt,
           "chunks": [list(c) for c in chunks]}
    prog_f = out / "progress.json"
    meta = {"frames": {}, "chunks": []}
    done: set[int] = set()
    if prog_f.exists() and json.loads(prog_f.read_text())["key"] == key:
        prog = json.loads(prog_f.read_text())
        done, meta = set(prog["done"]), prog["meta"]
        print(f"[sam] resuming: {len(done)}/{len(chunks)} chunks already done", flush=True)
    else:
        shutil.rmtree(out, ignore_errors=True)
        key_dir.mkdir(parents=True)

    def load_key(k: int):
        p = key_dir / f"{src_idx[k]:05d}.png"
        return cv2.imread(str(p), 0) > 127 if p.exists() else None

    cfg = Masking(max_num_objects=a.max_objects, detector_batch=2)
    t0 = time.time()
    pred, on_gpu = _build_predictor(cfg, REPO, cache)
    print(f"[sam] SAM 3.1 multiplex loaded on {'CUDA' if on_gpu else 'CPU'} "
          f"in {time.time() - t0:.0f}s", flush=True)

    t_all = time.time()
    for ci, (cs, ce) in enumerate(chunks):
        if ci in done:
            continue
        t0 = time.time()
        cdir = cache / f"sam_chunk_{name}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for k in range(cs, ce):
            (cdir / f"{k - cs:05d}.jpg").symlink_to(frames_dir / f"{k:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": 0, "text": a.prompt})
        per_obj: dict[int, dict[int, np.ndarray]] = {}
        probs: dict[int, dict[int, float | None]] = {}
        for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            k = cs + int(resp["frame_index"])
            o = resp["outputs"]
            pr = list(o.get("out_probs", [])) or [None] * len(o["out_obj_ids"])
            for oid, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], pr):
                m = np.asarray(m) > 0
                if not m.any():
                    continue
                full = np.zeros((H, W), bool)           # paste the crop back
                full[cy:cy + m.shape[0], :m.shape[1]] = m
                per_obj.setdefault(int(oid), {})[k] = full
                probs.setdefault(int(oid), {})[k] = None if p is None else float(p)
        pred.handle_request({"type": "close_session", "session_id": sid})

        # which SAM object is the fish?
        shared = [k for k in range(cs, ce) if load_key(k) is not None] if ci else []
        prev = {k: load_key(k) for k in shared}

        def score(oid):
            fr = per_obj[oid]
            if shared:      # continuity with what the previous chunk produced
                return float(np.mean([iou(fr[k], prev[k]) if k in fr else 0.0 for k in shared]))
            return float(sum(m.sum() * (probs[oid][k] or 0.5) for k, m in fr.items()))

        chosen = max(per_obj, key=score) if per_obj else None
        if chosen is None:
            raise RuntimeError(f"SAM found no '{a.prompt}' in chunk {ci} (keyframes {cs}-{ce - 1})")
        if shared and score(chosen) < 0.3:
            raise RuntimeError(f"chunk {ci}: no object continues the previous chunk "
                               f"(best IoU {score(chosen):.2f}); inspect and re-prompt")
        for k, m in per_obj[chosen].items():
            if prev.get(k) is not None and k < cs + a.overlap // 2:
                continue                                 # seam: keep the earlier chunk's mask
            cv2.imwrite(str(key_dir / f"{src_idx[k]:05d}.png"), m.astype(np.uint8) * 255)
            meta["frames"][str(src_idx[k])] = {"chunk": ci, "obj": chosen, "prob": probs[chosen][k],
                                               "area": int(m.sum()),
                                               "objects_in_chunk": sorted(per_obj)}
        meta["chunks"].append({"range": [cs, ce], "chosen": chosen, "objects": sorted(per_obj),
                               "score": score(chosen), "seconds": round(time.time() - t0, 1)})
        done.add(ci)
        prog_f.write_text(json.dumps({"key": key, "done": sorted(done), "meta": meta}))
        print(f"[sam] chunk {ci + 1}/{len(chunks)} keyframes {cs}-{ce - 1}: objects "
              f"{sorted(per_obj)} -> fish {chosen} (score {score(chosen):.2f}) "
              f"{time.time() - t0:.0f}s", flush=True)
        del per_obj, prev
        gc.collect()

    missing = [i for i in src_idx if not (key_dir / f"{i:05d}.png").exists()]
    final = {"width": W, "height": H, "num_frames": n_src, "stride": a.stride, "crop_y": cy,
             "prompt": a.prompt, "keyframes": src_idx, "missing_keyframes": missing,
             "device": "cuda" if on_gpu else "cpu", "seconds": round(time.time() - t_all, 1),
             **meta}
    (out / "meta.json").write_text(json.dumps(final, indent=1))
    print(f"[sam] done: {n - len(missing)}/{n} keyframes masked, missing={missing}, "
          f"{final['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
