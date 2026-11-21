#!/usr/bin/env python3
"""Track a hand-seeded part mask with SAM 3.1's tracker in both directions.

The text prompt "flipper" finds only one of the two fins, so the second fin is
seeded on one frame (GrabCut inside the diver mask) and carried forward and
backward chunk by chunk, each chunk seeded from the previous chunk's last mask.

    python part_track_bidir.py --input inputs/diver.mov --part fin2 --seed 30
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import cv2, numpy as np
from diver_masks_bwd import prompts_from_mask
from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    stem = a.input.stem
    sub = REPO / "masks" / f"{stem}__{a.part}" / "subject"
    diver = REPO / "masks" / f"{stem}__fwd" / "subject"
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    n = len(list(frames.glob("*.jpg")))
    pred, _ = _build_predictor(Masking(parts={}), REPO, cache)

    def _out(inference_state, frame_idx, refined_obj_id_to_mask=None):
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        o.update(refined_obj_id_to_mask or {})
        return o
    pred.model._build_sam2_output = _out

    for step in (1, -1):
        seed_f = a.seed
        seed = cv2.imread(str(sub / f"{seed_f:05d}.png"), 0) > 127
        end = n - 1 if step == 1 else 0
        while seed_f != end:
            e = min(max(seed_f + step * (a.chunk - 1), 0), n - 1) if step == 1 else max(seed_f - (a.chunk - 1), 0)
            idx = list(range(seed_f, e + step, step))
            cdir = cache / f"sam_chunk_{stem}_track2"
            shutil.rmtree(cdir, ignore_errors=True); cdir.mkdir()
            for k, i in enumerate(idx):
                (cdir / f"{k:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
            sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
            pts, lab = prompts_from_mask(seed, n_pos=4, n_neg=4)
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                 "points": pts, "point_labels": lab, "obj_id": 101})
            last = seed
            for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                    "propagation_direction": "forward", "start_frame_index": 0}):
                k = int(resp["frame_index"]); fi = idx[k]
                o = resp["outputs"]
                for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                    if int(oid) != 101 or k == 0:
                        continue
                    m = np.asarray(mm) > 0
                    dm = cv2.imread(str(diver / f"{fi:05d}.png"), 0)
                    if dm is not None:
                        m &= cv2.dilate((dm > 127).astype(np.uint8), np.ones((201, 201), np.uint8)) > 0
                    cv2.imwrite(str(sub / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
                    if k == len(idx) - 1 and m.sum() > 60:
                        last = m
            pred.handle_request({"type": "close_session", "session_id": sid})
            print(f"[track2] dir {step:+d} frames {idx[0]}..{idx[-1]} done", flush=True)
            seed_f, seed = idx[-1], last
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
