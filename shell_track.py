#!/usr/bin/env python3
"""Track one tiny dropped shell with SAM 3.1 point prompts (text prompts miss it).

Seed: one click on the shell at --seed-frame (full-res x,y). A "seashell" text
prompt on the seed frame makes the wrapper keep that frame (see
diver_masks_bwd.py); the click then defines the object. Forward chunks are
re-seeded from the previous chunk's last mask centroid; a mask is rejected if
its area leaves [amin, amax] or it jumps further than --max-jump px per frame.

Output: masks/<stem>__shell/obj_1/<frame:05d>.png

    python shell_track.py --input inputs/player_export.mov --seed-frame 44 --x 836 --y 741
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent


def prompts(x, y, w, h, r_neg=26):
    pts = [[x / w, y / h]]
    lab = [1]
    for dx, dy in ((r_neg, 0), (-r_neg, 0), (0, r_neg), (0, -r_neg)):
        pts.append([(x + dx) / w, (y + dy) / h])
        lab.append(0)
    return pts, lab


def run_chunk(pred, frames, cdir, cs, ce, seed, x, y, w, h, direction):
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir()
    for i in range(cs, ce):
        (cdir / f"{i - cs:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
    sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
    pred.handle_request({"type": "add_prompt", "session_id": sid,
                         "frame_index": seed - cs, "text": "seashell"})
    pts, lab = prompts(x, y, w, h)
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": seed - cs,
                         "points": pts, "point_labels": lab, "obj_id": 1})
    got = {}
    for resp in pred.handle_stream_request({
            "type": "propagate_in_video", "session_id": sid,
            "propagation_direction": direction, "start_frame_index": seed - cs}):
        fi = cs + int(resp["frame_index"])
        o = resp["outputs"]
        for oid, mm in zip(o["out_obj_ids"], o["out_binary_masks"]):
            if int(oid) == 1:
                got[fi] = np.asarray(mm) > 0
    pred.handle_request({"type": "close_session", "session_id": sid})
    return got


def accept(got, seq, prev_xy, a):
    """Keep frames in order from the seed outward until the track is lost."""
    kept = {}
    for fi in seq:
        m = got.get(fi)
        if m is None or not (a.amin <= m.sum() <= a.amax):
            break
        ys, xs = np.nonzero(m)
        c = (float(xs.mean()), float(ys.mean()))
        if prev_xy is not None and np.hypot(c[0] - prev_xy[0], c[1] - prev_xy[1]) > a.max_jump:
            break
        kept[fi] = m
        prev_xy = c
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--seed-frame", type=int, required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--back", type=int, default=14, help="frames to track backward")
    ap.add_argument("--fwd-max", type=int, default=140)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--amin", type=int, default=40)
    ap.add_argument("--amax", type=int, default=1500)
    ap.add_argument("--max-jump", type=float, default=30.0)
    a = ap.parse_args()

    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{a.input.stem}"
    n_all = len(list(frames.glob("*.jpg")))
    h, w = cv2.imread(str(frames / "00000.jpg")).shape[:2]
    out = REPO / "masks" / f"{a.input.stem}__shell" / "obj_1"
    shutil.rmtree(out.parent, ignore_errors=True)
    out.mkdir(parents=True)
    pred, _ = _build_predictor(Masking(max_num_objects=2, detector_batch=2), REPO, cache)
    cdir = cache / f"sam_chunk_{a.input.stem}_shelltrack"

    def save(kept):
        for fi, m in kept.items():
            cv2.imwrite(str(out / f"{fi:05d}.png"), m.astype(np.uint8) * 255)

    # backward from the seed
    t0 = time.time()
    b0 = max(0, a.seed_frame - a.back)
    got = run_chunk(pred, frames, cdir, b0, a.seed_frame + 1, a.seed_frame, a.x, a.y, w, h, "backward")
    kept = accept(got, range(a.seed_frame, b0 - 1, -1), None, a)
    save(kept)
    print(f"[shell] backward {a.seed_frame}->{min(kept) if kept else '-'}: {len(kept)}f {time.time()-t0:.0f}s", flush=True)

    # forward in re-seeded chunks
    seed, x, y = a.seed_frame, a.x, a.y
    while seed < min(n_all - 1, a.seed_frame + a.fwd_max):
        t0 = time.time()
        ce = min(n_all, seed + a.chunk)
        got = run_chunk(pred, frames, cdir, seed, ce, seed, x, y, w, h, "forward")
        kept = accept(got, range(seed, ce), (x, y), a)
        save(kept)
        print(f"[shell] forward {seed}-{ce - 1}: tracked {len(kept)}f {time.time()-t0:.0f}s", flush=True)
        if len(kept) < ce - seed:
            print("[shell] track ended (lost / out of bounds)", flush=True)
            break
        last = kept[max(kept)]
        ys, xs = np.nonzero(last)
        seed, x, y = max(kept), float(xs.mean()), float(ys.mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
