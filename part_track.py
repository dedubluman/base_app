#!/usr/bin/env python3
"""Extend a SAM 3.1 part mask past the frames where its text prompt stops
detecting it, by tracking the last good mask forward (or backward).

On this clip "diving mask" is detected on frames 0-87 only; from 88 on the
detector finds nothing although the mask is plainly visible. The last good
mask is turned into point prompts (diver_masks_bwd.prompts_from_mask) and
SAM 3.1's tracker carries it on, chunk by chunk, each chunk seeded from the
previous chunk's last mask. Output stays inside the (grown) diver mask.

    python part_track.py --input inputs/diver.mov --part goggles --from 87
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from diver_masks_bwd import prompts_from_mask
from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--from", dest="start", type=int, required=True, help="last good frame")
    ap.add_argument("--to", type=int, default=None, help="stop at this frame")
    ap.add_argument("--free", action="store_true",
                    help="part may leave the diver mask (fins): only kept near it")
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    stem = a.input.stem
    sub = REPO / "masks" / f"{stem}__{a.part}" / "subject"
    diver = REPO / "masks" / f"{stem}__fwd" / "subject"
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    n = len(list(frames.glob("*.jpg")))
    pred, on_gpu = _build_predictor(Masking(parts={}), REPO, cache)

    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        o.update(refined_obj_id_to_mask or {})
        return o

    pred.model._build_sam2_output = _build_sam2_output
    print(f"[track] {a.part} from frame {a.start} on {'CUDA' if on_gpu else 'CPU'}", flush=True)
    seed_f = a.start
    seed = cv2.imread(str(sub / f"{seed_f:05d}.png"), 0) > 127
    stop = n - 1 if a.to is None else min(n - 1, a.to)
    grow = 201 if a.free else 5
    while seed_f < stop:
        cs, ce = seed_f, min(stop + 1, seed_f + a.chunk)
        cdir = cache / f"sam_chunk_{stem}_track"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pts, lab = prompts_from_mask(seed, n_pos=4, n_neg=4)
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                             "points": pts, "point_labels": lab, "obj_id": 101})
        last = seed
        for resp in pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "forward", "start_frame_index": 0}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                if int(oid) != 101 or fi == cs:
                    continue
                m = np.asarray(mm) > 0
                dm = cv2.imread(str(diver / f"{fi:05d}.png"), 0)
                if dm is not None:
                    m &= cv2.dilate((dm > 127).astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
                cv2.imwrite(str(sub / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
                if fi == ce - 1 and m.sum() > 60:
                    last = m
        pred.handle_request({"type": "close_session", "session_id": sid})
        print(f"[track] frames {cs}-{ce - 1} done", flush=True)
        seed_f, seed = ce - 1, last
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
