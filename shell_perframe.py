#!/usr/bin/env python3
"""Per-frame SAM 3.1 masks for a tiny object whose centre is known per frame.

SAM's video tracker loses a ~13 px shell after the seed frame, but a click on
a single frame gives a clean mask (see shell_track.py). Centres come from a
template track (JSON {frame: [x, y, score]}); every frame is a one-frame
session seeded with that click. Frames whose SAM mask is implausible fall
back to a small disk at the tracked centre (listed in meta).

    python shell_perframe.py --input inputs/player_export.mov --positions pos.json
"""
import argparse, json, shutil, time
from pathlib import Path
import cv2, numpy as np
from shades.config import Masking
from shades.masking import _build_predictor
from shell_track import run_chunk

REPO = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--positions", type=Path, required=True)
    ap.add_argument("--amin", type=int, default=40)
    ap.add_argument("--amax", type=int, default=600)
    a = ap.parse_args()
    pos = {int(k): v for k, v in json.loads(a.positions.read_text()).items()}
    cache = REPO / ".cache"
    frames = cache / f"sam_frames_{a.input.stem}"
    h, w = cv2.imread(str(frames / "00000.jpg")).shape[:2]
    out = REPO / "masks" / f"{a.input.stem}__shell" / "obj_1"
    shutil.rmtree(out.parent, ignore_errors=True); out.mkdir(parents=True)
    pred, _ = _build_predictor(Masking(max_num_objects=2, detector_batch=2), REPO, cache)
    cdir = cache / f"sam_chunk_{a.input.stem}_shellpf"
    meta = {"sam": [], "fallback": []}
    for i in sorted(pos):
        x, y, s = pos[i]
        t0 = time.time()
        got = run_chunk(pred, frames, cdir, i, i + 1, i, x, y, w, h, "forward")
        m = got.get(i)
        ok = m is not None and a.amin <= m.sum() <= a.amax
        if ok:
            ys, xs = np.nonzero(m)
            ok = np.hypot(xs.mean() - x, ys.mean() - y) < 8
        if not ok:
            m = np.zeros((h, w), bool); cv2.circle(m.view(np.uint8), (int(round(x)), int(round(y))), 7, 1, -1)
            m = m.astype(bool); meta["fallback"].append(i)
        else:
            meta["sam"].append(i)
        cv2.imwrite(str(out / f"{i:05d}.png"), m.astype(np.uint8) * 255)
        print(f"[shell] {i} {'sam' if ok else 'disk'} area={int(m.sum())} {time.time()-t0:.0f}s", flush=True)
    (out.parent / "meta.json").write_text(json.dumps(meta))

if __name__ == "__main__":
    main()
