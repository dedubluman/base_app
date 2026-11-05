#!/usr/bin/env python3
"""Multi-class SAM 3.1 masks of the diver's parts: hair, shorts, fins, mask.

Each class is graded on its own (diver_reels.py / reels_grade.py): the
skin is the diver mask minus every part, so the skin grade never reaches
hair, fabric or gear. Per chunk, each class is detected by SAM 3.1 from its
text prompt on a seed frame (inside the diver mask), then all classes are
tracked together through the chunk in both directions by SAM 3.1's
multiplex tracker head (one tracker pass for all classes; no detector on
the other frames).

Prompts tested on this clip: "flipper" finds the fins ("fin", "swim fin",
"diving fin" find nothing); no prompt finds the small dive mask ("diving
mask", "goggles", "swim goggles", "snorkel mask", "face mask", "mask",
"eyewear"), so it is left to the dark-gear luminance fallback.

    python diver_parts.py --input inputs/diver.mov \
        --parts shorts=shorts fins=flipper hair=hair goggles="diving mask"
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
    ap.add_argument("--parts", nargs="+", required=True, help="name=text prompt")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--free", nargs="*", default=[],
                    help="parts not clipped to the diver mask (e.g. fins: the "
                         "'person' mask leaves them out); kept if near the diver")
    a = ap.parse_args()
    parts = dict(p.split("=", 1) for p in a.parts)
    ids = {name: k + 1 for k, name in enumerate(parts)}

    stem = a.input.stem
    diver = REPO / "masks" / f"{stem}__fwd" / "subject"
    outs = {}
    for name in parts:
        outs[name] = REPO / "masks" / f"{stem}__{name}" / "subject"
        outs[name].mkdir(parents=True, exist_ok=True)
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{stem}_bwd"
    if not (frames / "00000.jpg").exists():
        extract_jpeg_frames(a.input, frames, 2)
    n = len(list(frames.glob("*.jpg")))

    pred, on_gpu = _build_predictor(Masking(parts={}), REPO, cache)

    def _build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        # see diver_masks_bwd.py: keep tracked masks on frames the detector
        # never processed
        o = dict(inference_state["cached_frame_outputs"].get(frame_idx, {}))
        o.update(refined_obj_id_to_mask or {})
        return o

    pred.model._build_sam2_output = _build_sam2_output
    print(f"[parts] SAM 3.1 on {'CUDA' if on_gpu else 'CPU'}; {parts} over {n} frames", flush=True)

    def diver_mask(f, grow=9):
        dm = cv2.imread(str(diver / f"{f:05d}.png"), 0)
        if dm is None or (dm > 127).sum() < 1500:
            return None
        return cv2.dilate((dm > 127).astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0

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
        cdir = cache / f"sam_chunk_{stem}_parts"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        cands = [f for f in sorted({cs + (ce - cs) // 2, cs, ce - 1, cs + (ce - cs) // 4,
                                    cs + 3 * (ce - cs) // 4}) if diver_mask(f) is not None]
        seeds = {}
        # 1) detect each class from its text prompt on a seed frame
        for name, text in parts.items():
            for f in cands:
                dm = diver_mask(f)
                r = pred.handle_request({"type": "add_prompt", "session_id": sid,
                                         "frame_index": f - cs, "text": text})
                best, best_sc = None, 0
                for mm in r["outputs"]["out_binary_masks"]:
                    m = np.asarray(mm) > 0
                    if m.sum() < 60:
                        continue
                    if name in a.free:
                        near = diver_mask(f, grow=201)
                        if (m & near).sum() / m.sum() > 0.8 and m.sum() > best_sc:
                            best, best_sc = m, m.sum()
                    elif (m & dm).sum() / m.sum() > 0.8 and m.sum() > best_sc:
                        best, best_sc = m & dm, m.sum()
                if best is not None:
                    seeds[name] = (f, best)
                    break
        # 2) the detected masks become tracker objects; propagate all together
        for name, (f, m) in seeds.items():
            pts, lab = prompts_from_mask(m, n_pos=4, n_neg=4)
            pred.handle_request({"type": "add_prompt", "session_id": sid,
                                 "frame_index": f - cs, "points": pts,
                                 "point_labels": lab, "obj_id": 100 + ids[name]})
        got = {name: {f: m} for name, (f, m) in seeds.items()}
        if seeds:
            start = min(f for f, _ in seeds.values())
            # From a chunk edge, propagate one way only: after a pass that
            # starts on the first/last frame SAM 3.1 switches the second
            # direction to "fetch cached outputs", which do not exist here.
            direction = ("forward" if start == cs else
                         "backward" if start == ce - 1 else "both")
            for resp in pred.handle_stream_request({
                    "type": "propagate_in_video", "session_id": sid,
                    "propagation_direction": direction, "start_frame_index": start - cs}):
                fi = cs + int(resp["frame_index"])
                o = resp["outputs"]
                for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
                    for name in seeds:
                        if int(oid) == 100 + ids[name] and fi != seeds[name][0]:
                            got[name][fi] = np.asarray(mm) > 0
        pred.handle_request({"type": "close_session", "session_id": sid})
        for name, fr in got.items():
            for fi, m in fr.items():
                dm = diver_mask(fi, grow=5 if name not in a.free else 201)
                if dm is not None:  # a part never leaves the diver
                    m = m & dm
                cv2.imwrite(str(outs[name] / f"{fi:05d}.png"), m.astype(np.uint8) * 255)
        log[f"{cs}-{ce - 1}"] = {"seconds": round(time.time() - t0, 1),
                                 "seeds": {k: v[0] for k, v in seeds.items()},
                                 "missing": [k for k in parts if k not in seeds]}
        print(f"[parts] chunk {cs}-{ce - 1}: seeds {log[f'{cs}-{ce - 1}']['seeds']}, "
              f"missing {log[f'{cs}-{ce - 1}']['missing']}, {time.time() - t0:.0f}s", flush=True)
    for name in parts:
        (outs[name].parent / "meta.json").write_text(json.dumps(
            {"num_frames": n, "text": parts[name], "chunks": log}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
