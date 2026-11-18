#!/usr/bin/env python3
"""SAM 3.1 pufferfish masks for a whole clip (or one still image).

The fish is small (~60 x 300 px in a 1080 x 1920 frame) and camouflaged on
sand, so SAM 3.1's open-vocabulary detector only finds it on some frames
("spotted fish" works on frame 0; "pufferfish"/"fish" do not; mid-clip
frames find nothing). The workflow is therefore:

  1. seed   text prompt "spotted fish" on the first frame -> SAM 3.1 mask
  2. track  SAM 3.1's multiplex tracker carries the fish forward in chunks;
            each chunk is a 1080x1080 window centred on the fish (SAM sees
            the fish at ~0.93x instead of ~0.53x scale: thin fins and the
            tail survive) and is seeded with point prompts taken from the
            previous chunk's last mask (positives along the body axis,
            negatives in a ring on the sand) - the repo's part_track.py
            pattern.

Output: masks/<stem>__puffer/obj_1/<frame:05d>.png (0/255, full frame)
        masks/<stem>__puffer/meta.json

    python puffer_masks.py --input inputs/pufferfish.mov
    python puffer_masks.py --input inputs/pufferfish_reference.png   # still
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from diver_masks_bwd import prompts_from_mask
from shades.config import Masking
from shades.masking import _build_predictor, extract_jpeg_frames

REPO = Path(__file__).resolve().parent
OBJ = 101


def masks_from(outputs, oid=None):
    res = []
    pr = list(outputs.get("out_probs", [])) or [None] * len(outputs["out_obj_ids"])
    for o, m, p in zip(outputs["out_obj_ids"], outputs["out_binary_masks"], pr):
        if oid is None or int(o) == oid:
            res.append((int(o), np.asarray(m) > 0, None if p is None else float(p)))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--prompt", default="spotted fish")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--win", type=int, default=1080, help="square tracking window (px)")
    a = ap.parse_args()

    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    stem = a.input.stem
    out = REPO / "masks" / f"{stem}__puffer"
    shutil.rmtree(out, ignore_errors=True)
    (out / "obj_1").mkdir(parents=True)
    frames = cache / f"sam_frames_{stem}"
    if not frames.exists() or not any(frames.glob("*.jpg")):
        extract_jpeg_frames(a.input, frames, 2)
    n = len(list(frames.glob("*.jpg")))
    first = cv2.imread(str(frames / "00000.jpg"))
    h, w = first.shape[:2]

    pred, on_gpu = _build_predictor(Masking(max_num_objects=4, detector_batch=1), REPO, cache)

    # point-prompt sessions: report the tracked object for every frame
    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        o.update(refined_obj_id_to_mask or {})
        return o

    print(f"[puffer] {n} frames {w}x{h} on {'CUDA' if on_gpu else 'CPU'}", flush=True)
    meta = {"prompt": a.prompt, "num_frames": n, "width": w, "height": h,
            "device": "cuda" if on_gpu else "cpu", "chunks": []}

    # 1) seed: text prompt on frame 0 (full frame)
    sdir = cache / f"sam_seed_{stem}"
    shutil.rmtree(sdir, ignore_errors=True)
    sdir.mkdir()
    (sdir / "00000.jpg").symlink_to(frames / "00000.jpg")
    sid = pred.handle_request({"type": "start_session", "resource_path": str(sdir)})["session_id"]
    r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                             "text": a.prompt})
    pred.handle_request({"type": "close_session", "session_id": sid})
    cands = masks_from(r["outputs"])
    if not cands:
        raise SystemExit(f"SAM 3.1 found no '{a.prompt}' on frame 0")
    _, seed, prob = max(cands, key=lambda c: (c[2] or 0, c[1].sum()))
    meta["seed"] = {"frame": 0, "prob": prob, "area": int(seed.sum()),
                    "candidates": len(cands)}
    cv2.imwrite(str(out / "obj_1" / "00000.png"), seed.astype(np.uint8) * 255)
    print(f"[puffer] seed frame 0: prob {prob:.2f}, area {int(seed.sum())} px", flush=True)
    if n == 1:
        meta["objects"] = {"1": {"frames": 1}}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        return 0

    # 2) track forward in fish-centred square windows
    pred.model._build_sam2_output = _build_sam2_output
    sf, areas = 0, {0: int(seed.sum())}
    while sf < n - 1:
        t0 = time.time()
        cs, ce = sf, min(n, sf + a.chunk)
        ys, xs = np.nonzero(seed)
        win = min(a.win, h, w)
        cy = int(np.clip(ys.mean() - win / 2, 0, h - win))
        cx = int(np.clip(xs.mean() - win / 2, 0, w - win))
        cdir = cache / f"sam_chunk_{stem}_puffer"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            img = cv2.imread(str(frames / f"{i:05d}.jpg"))
            cv2.imwrite(str(cdir / f"{i - cs:05d}.jpg"), img[cy:cy + win, cx:cx + win],
                        [cv2.IMWRITE_JPEG_QUALITY, 97])
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pts, lab = prompts_from_mask(seed[cy:cy + win, cx:cx + win], n_pos=5, n_neg=6)
        r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                 "points": pts, "point_labels": lab, "obj_id": OBJ})
        got0 = masks_from(r["outputs"], OBJ)
        seed_iou = None
        if got0:
            m0 = np.zeros((h, w), bool)
            m0[cy:cy + win, cx:cx + win] = got0[0][1]
            u = (m0 | seed).sum()
            seed_iou = round(float((m0 & seed).sum() / u), 3) if u else None
        last, last_f = seed, cs
        for resp in pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "forward", "start_frame_index": 0}):
            fi = cs + int(resp["frame_index"])
            if fi == cs:
                continue   # the seed frame keeps the previous chunk's tracked mask
            for _, mm, _ in masks_from(resp["outputs"], OBJ):
                m = np.zeros((h, w), bool)
                m[cy:cy + win, cx:cx + win] = mm
                cv2.imwrite(str(out / "obj_1" / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
                areas[fi] = int(m.sum())
                if m.sum() > 300:
                    last, last_f = m, fi
        pred.handle_request({"type": "close_session", "session_id": sid})
        ch = {"range": [cs, ce], "window_xy": [cx, cy], "seed_point_iou": seed_iou,
              "area_end": areas.get(ce - 1, 0), "seconds": round(time.time() - t0)}
        meta["chunks"].append(ch)
        print(f"[puffer] chunk {cs}-{ce - 1} win@({cx},{cy}) seed-IoU {seed_iou} "
              f"area {areas.get(cs + 1, 0)}->{areas.get(ce - 1, 0)} {ch['seconds']}s", flush=True)
        sf, seed = ce - 1, last
        if last_f < ce - 1:
            print(f"[puffer] fish lost after frame {last_f}", flush=True)
        meta["areas"] = {str(k): v for k, v in sorted(areas.items())}
        meta["objects"] = {"1": {"frames": len(areas)}}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(cdir, ignore_errors=True)
    shutil.rmtree(sdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
