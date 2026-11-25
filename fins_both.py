#!/usr/bin/env python3
"""Track BOTH fins with SAM 3.1 (diver_parts.py keeps only the largest 'flipper' detection per chunk).

Per chunk: SAM 3.1's detector runs on candidate frames with the text prompt "flipper"; every instance that
lies near the diver becomes its own tracker object (the frame with the most instances wins); all objects are
tracked together (multiplex tracker) in both directions. A chunk where the detector finds nothing is seeded from
the previous chunk's masks (connected components -> one tracker object each). Output: union mask per frame in
masks/<stem>__fins2/subject/.

    python fins_both.py --input inputs/player_export.mov --start 40
"""
from __future__ import annotations
import argparse, shutil, time
from pathlib import Path
import cv2, numpy as np
from diver_masks_bwd import prompts_from_mask
from shades.config import Masking
from shades.masking import _build_predictor, extract_jpeg_frames

REPO = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--text", default="flipper")
    ap.add_argument("--start", type=int, default=40); ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--overlap", type=int, default=4)
    a = ap.parse_args()
    stem = a.input.stem
    diver = REPO / "masks" / f"{stem}__fwd" / "subject"
    out = REPO / "masks" / f"{stem}__fins2" / "subject"; out.mkdir(parents=True, exist_ok=True)
    cache = REPO / ".cache"; frames = cache / f"sam_frames_{stem}_bwd"
    if not (frames / "00000.jpg").exists():
        extract_jpeg_frames(a.input, frames, 2)
    n = len(list(frames.glob("*.jpg")))
    pred, _ = _build_predictor(Masking(parts={}), REPO, cache)

    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {})); o.update(refined_obj_id_to_mask or {}); return o
    pred.model._build_sam2_output = _build_sam2_output

    def dmask(f, grow):
        dm = cv2.imread(str(diver / f"{f:05d}.png"), 0)
        if dm is None or (dm > 127).sum() < 1500: return None
        return cv2.dilate((dm > 127).astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0

    stop = n if a.end is None else min(n, a.end)
    chunks, s = [], a.start
    while True:
        e = min(stop, s + a.chunk); chunks.append((s, e))
        if e == stop: break
        s = e - a.overlap
    for cs, ce in chunks:
        t0 = time.time()
        cdir = cache / f"sam_chunk_{stem}_fins2"; shutil.rmtree(cdir, ignore_errors=True); cdir.mkdir()
        for i in range(cs, ce): (cdir / f"{i - cs:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        cands = [f for f in sorted({cs + (ce - cs) // 2, cs, ce - 1, cs + (ce - cs) // 4, cs + 3 * (ce - cs) // 4}) if dmask(f, 5) is not None]
        best = (0, None, [])
        for f in cands:
            r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": f - cs, "text": a.text})
            near = dmask(f, 201); ms = []
            for mm in r["outputs"]["out_binary_masks"]:
                m = np.asarray(mm) > 0
                if m.sum() >= 60 and (m & near).sum() / m.sum() > 0.8: ms.append(m)
            if len(ms) > best[0]: best = (len(ms), f, ms)
            if len(ms) >= 2: break
        seeds = []
        if best[0]: seeds = [(best[1], m) for m in best[2]]
        else:                                           # nothing detected: continue from the previous chunk's masks
            for f in (cs, ce - 1):
                pm = cv2.imread(str(out / f"{f:05d}.png"), 0)
                if pm is not None and (pm > 127).sum() >= 60:
                    nl, lab = cv2.connectedComponents((pm > 127).astype(np.uint8))
                    seeds = [(f, lab == k) for k in range(1, nl) if (lab == k).sum() >= 60]; break
        got = {}
        for k, (f, m) in enumerate(seeds):
            pts, lab = prompts_from_mask(m, n_pos=4, n_neg=4)
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": f - cs, "points": pts, "point_labels": lab, "obj_id": 200 + k})
            got[200 + k] = {f: m}
        if seeds:
            start = min(f for f, _ in seeds)
            direction = "forward" if start == cs else "backward" if start == ce - 1 else "both"
            for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                                    "propagation_direction": direction, "start_frame_index": start - cs}):
                fi = cs + int(resp["frame_index"]); o = resp["outputs"]
                for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                    if int(oid) in got and fi not in got[int(oid)]: got[int(oid)][fi] = np.asarray(mm) > 0
        pred.handle_request({"type": "close_session", "session_id": sid})
        union = {}
        for oid, fr in got.items():
            for fi, m in fr.items(): union[fi] = union.get(fi, False) | m
        for fi, m in union.items():
            dm = dmask(fi, 201)
            if dm is not None: m = m & dm
            cv2.imwrite(str(out / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
        print(f"[fins2] chunk {cs}-{ce - 1}: instances {best[0]} (frame {best[1]}), objects {len(seeds)}, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
