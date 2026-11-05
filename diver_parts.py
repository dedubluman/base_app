#!/usr/bin/env python3
"""SAM 3.1 mask of a part of the diver (e.g. the white shorts).

The part is graded apart from the skin: a luminance key cannot tell sunlit
skin from white fabric. Per chunk, SAM 3.1 detects the part from a text
prompt on one seed frame (inside the diver mask), then its tracker head
propagates that mask through the chunk in both directions (no detector on
the other frames: ~5 s/frame on CPU instead of ~16 s).

    python diver_parts.py --input inputs/diver.mov --part shorts --text shorts
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--part", default="shorts")
    ap.add_argument("--text", default="shorts")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=4)
    a = ap.parse_args()

    stem = a.input.stem
    diver = REPO / "masks" / f"{stem}__fwd" / "subject"
    out = REPO / "masks" / f"{stem}__{a.part}" / "subject"
    out.mkdir(parents=True, exist_ok=True)
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    if not (frames / "00000.jpg").exists():
        extract_jpeg_frames(a.input, frames, 2)
    n = len(list(frames.glob("*.jpg")))

    pred, on_gpu = _build_predictor(Masking(parts={}), REPO, cache)
    model = pred.model

    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        # see diver_masks_bwd.py: keep tracked masks on frames the detector
        # never processed
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        o.update(refined_obj_id_to_mask or {})
        return o

    model._build_sam2_output = _build_sam2_output
    print(f"[part] SAM 3.1 on {'CUDA' if on_gpu else 'CPU'}; '{a.text}' over {n} frames", flush=True)

    chunks, s = [], 0
    while True:
        e = min(n, s + a.chunk)
        chunks.append((s, e))
        if e == n:
            break
        s = e - a.overlap
    log = {}
    for cs, ce in chunks:
        t0 = time.time()
        cdir = cache / f"sam_chunk_{stem}_{a.part}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        # seed: the first frame (of a few candidates) where the detector finds
        # the part inside the diver
        seed, seed_mask = None, None
        for f in sorted({cs + (ce - cs) // 2, cs, ce - 1}):
            dm = cv2.imread(str(diver / f"{f:05d}.png"), 0)
            if dm is None or (dm > 127).sum() < 2000:
                continue
            dm = cv2.dilate((dm > 127).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            r = pred.handle_request({"type": "add_prompt", "session_id": sid,
                                     "frame_index": f - cs, "text": a.text})
            o = r["outputs"]
            best, best_sc = None, 0.0
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                m = np.asarray(mm) > 0
                if m.sum() < 200:
                    continue
                inside = (m & dm).sum() / m.sum()
                if inside > 0.8 and m.sum() > best_sc:
                    best, best_sc = m & dm, m.sum()
            if best is not None:
                seed, seed_mask = f, best
                break
        if seed is None:
            pred.handle_request({"type": "close_session", "session_id": sid})
            print(f"[part] chunk {cs}-{ce - 1}: '{a.text}' not found, skipped", flush=True)
            log[f"{cs}-{ce - 1}"] = {"seed": None}
            continue
        pts, lab = prompts_from_mask(seed_mask, n_pos=4, n_neg=4)
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": seed - cs, "points": pts,
                             "point_labels": lab, "obj_id": 1})
        got = {seed: seed_mask}
        for resp in pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "both", "start_frame_index": seed - cs}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                if int(oid) == 1:
                    got[fi] = np.asarray(mm) > 0
        pred.handle_request({"type": "close_session", "session_id": sid})
        for fi, m in got.items():
            dm = cv2.imread(str(diver / f"{fi:05d}.png"), 0)
            if dm is not None:  # the part never leaves the diver
                m = m & (cv2.dilate((dm > 127).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
            cv2.imwrite(str(out / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
        log[f"{cs}-{ce - 1}"] = {"seed": seed, "seconds": round(time.time() - t0, 1),
                                 "areas": {str(k): int(v.sum()) for k, v in sorted(got.items())}}
        print(f"[part] chunk {cs}-{ce - 1}: seed {seed}, {len(got)} frames, "
              f"{time.time() - t0:.0f}s", flush=True)
    (out.parent / "meta.json").write_text(json.dumps({"num_frames": n, "text": a.text,
                                                       "chunks": log}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
