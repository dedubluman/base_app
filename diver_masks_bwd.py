#!/usr/bin/env python3
"""Backward SAM 3.1 pass for the diver mask (tracker head only).

The forward pass (diver_masks.py --direction fwd) runs the full SAM 3.1
detector + tracker on every frame. This pass re-tracks the diver in the
opposite direction: each chunk is seeded on its LAST frame with point prompts
taken from the forward mask there, then propagated backwards to the chunk's
first frame with SAM 3.1's tracker. Running the detector a second time would
cost ~3 h on CPU; the tracker costs ~2 s/frame.

    python diver_masks_bwd.py --input inputs/diver.mov [--frames 0:508]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.masking import _build_predictor, extract_jpeg_frames

REPO = Path(__file__).resolve().parent


def prompts_from_mask(m: np.ndarray, n_pos: int = 6, n_neg: int = 6):
    """Positive clicks deep inside the diver, spread along the body; negative
    clicks in a ring ~25 px outside the mask (water, bubbles, surface)."""
    h, w = m.shape
    m8 = m.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    pts, labels = [], []
    ys, xs = np.nonzero(dist >= max(2.0, 0.45 * dist.max()))
    if len(ys) == 0:
        ys, xs = np.nonzero(m8)
    # spread positives along the body's main axis
    if len(ys):
        c = np.stack([xs, ys], 1).astype(np.float32)
        mu = c.mean(0)
        _, _, vt = np.linalg.svd(c - mu, full_matrices=False)
        proj = (c - mu) @ vt[0]
        order = np.argsort(proj)
        for k in np.linspace(0, len(order) - 1, n_pos).astype(int):
            x, y = c[order[k]]
            pts.append([x / w, y / h]); labels.append(1)
    outer = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5)
    ring = (outer > 18) & (outer < 30)
    ys, xs = np.nonzero(ring)
    if len(ys):
        ang = np.arctan2(ys - ys.mean(), xs - xs.mean())
        for a in np.linspace(-np.pi, np.pi, n_neg, endpoint=False):
            k = np.argmin(np.abs(np.angle(np.exp(1j * (ang - a)))))
            pts.append([xs[k] / w, ys[k] / h]); labels.append(0)
    return pts, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--frames", default=None, help="start:end (end exclusive)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    stem = a.input.stem
    fwd = REPO / "masks" / f"{stem}__fwd" / "subject"
    out = REPO / "masks" / f"{stem}__bwd" / "subject"
    out.mkdir(parents=True, exist_ok=True)
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    if not (frames / "00000.jpg").exists():
        extract_jpeg_frames(a.input, frames, 2)
    n = len(list(frames.glob("*.jpg")))
    s0, s1 = (0, n) if a.frames is None else map(int, a.frames.split(":"))

    pred, on_gpu = _build_predictor(Masking(parts={}), REPO, cache)
    # The tracker-only ("partial") propagation only *refines* frames that a
    # full detector pass already cached; on any other frame
    # _build_sam2_output() returns {} and the tracked mask is dropped. Here no
    # detector pass runs on those frames, so start from an empty cache entry.
    model = pred.model

    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        out = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        out.update(refined_obj_id_to_mask or {})
        return out

    model._build_sam2_output = _build_sam2_output
    print(f"[bwd] SAM 3.1 on {'CUDA' if on_gpu else 'CPU'}; frames {s0}-{s1 - 1}", flush=True)

    # chunks laid out from the END so every chunk is seeded on its last frame
    chunks, e = [], s1
    while True:
        s = max(s0, e - a.chunk)
        chunks.append((s, e))
        if s == s0:
            break
        e = s + a.overlap
    log = {}
    for cs, ce in chunks:
        t0 = time.time()
        seed = ce - 1
        m_seed = cv2.imread(str(fwd / f"{seed:05d}.png"), 0)
        if m_seed is None or (m_seed > 127).sum() < 30:
            print(f"[bwd] chunk {cs}-{ce - 1}: no forward mask to seed on, skipped", flush=True)
            continue
        pts, lab = prompts_from_mask(m_seed > 127)
        cdir = cache / f"sam_chunk_{stem}_bwd"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        # A text prompt on the seed frame makes the wrapper process and keep
        # that frame; point-only sessions return no objects (see split_sea.py).
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": seed - cs, "text": "person"})
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": seed - cs, "points": pts,
                             "point_labels": lab, "obj_id": 1})
        got = {}
        for resp in pred.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid,
                "propagation_direction": "backward",
                "start_frame_index": seed - cs}):
            fi = cs + int(resp["frame_index"])
            o = resp["outputs"]
            if a.debug:
                print("[dbg]", fi, list(o.keys()), list(o.get("out_obj_ids", [])),
                      [int(np.asarray(mm).sum()) for mm in o.get("out_binary_masks", [])], flush=True)
            m = np.zeros(m_seed.shape, bool)
            for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                if int(oid) == 1:
                    m = np.asarray(mm) > 0
            got[fi] = m
        pred.handle_request({"type": "close_session", "session_id": sid})
        for fi, m in got.items():
            p = out / f"{fi:05d}.png"
            # Frames in the overlap with the later chunk were already written
            # by that chunk, tracked a shorter distance from its seed: keep them.
            if not p.exists() or fi < ce - a.overlap:
                cv2.imwrite(str(p), m.astype(np.uint8) * 255)
        log[f"{cs}-{ce - 1}"] = {"seconds": round(time.time() - t0, 1),
                                 "areas": {str(k): int(v.sum()) for k, v in sorted(got.items())}}
        print(f"[bwd] chunk {cs}-{ce - 1}: {len(got)} frames, {time.time() - t0:.0f}s", flush=True)
    (out.parent / "meta.json").write_text(json.dumps({"num_frames": n, "chunks": log}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
