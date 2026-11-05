#!/usr/bin/env python3
"""Re-detect a SAM 3.1 part mask on frames where its tracked mask drifted.

The multiplex tracker in diver_parts.py can let a part grow onto a
neighbouring body part (on this clip the shorts mask slid onto the thigh
right after the seed of chunk 252-283). A frame is suspect when the part's
area leaves 0.75x-1.3x of the median of its +-15 neighbours. On each suspect
frame SAM 3.1's detector runs on that single frame from the text prompt and
its mask (inside the diver) replaces the tracked one. Originals are kept in
masks/<stem>__<part>/tracked/.

    python diver_parts_fix.py --input inputs/diver.mov --part shorts --text shorts
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--frames", default=None, help="explicit list a,b,c instead of auto")
    a = ap.parse_args()
    stem = a.input.stem
    sub = REPO / "masks" / f"{stem}__{a.part}" / "subject"
    keep = REPO / "masks" / f"{stem}__{a.part}" / "tracked"
    diver = REPO / "masks" / f"{stem}__fwd" / "subject"
    frames = REPO / ".cache" / f"sam_frames_{stem}_bwd"
    n = len(list(sub.glob("*.png")))
    area = np.array([(cv2.imread(str(sub / f"{i:05d}.png"), 0) > 127).sum() for i in range(n)], float)
    if a.frames:
        sus = [int(v) for v in a.frames.split(",")]
    else:
        sus = []
        for i in range(n):
            nb = [area[j] for j in range(max(0, i - 15), min(n, i + 16)) if j != i and area[j] > 0]
            if nb and area[i] > 0 and not 0.75 * np.median(nb) <= area[i] <= 1.3 * np.median(nb):
                sus.append(i)
    print(f"[fix] {a.part}: {len(sus)} suspect frames {sus}", flush=True)
    if not sus:
        return 0
    keep.mkdir(exist_ok=True)
    pred, _ = _build_predictor(Masking(parts={}), REPO, REPO / ".cache")
    one = REPO / ".cache" / "fix_one"
    log = {}
    for i in sus:
        shutil.rmtree(one, ignore_errors=True)
        one.mkdir()
        (one / "00000.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(one)})["session_id"]
        r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": a.text})
        pred.handle_request({"type": "close_session", "session_id": sid})
        dm = cv2.dilate((cv2.imread(str(diver / f"{i:05d}.png"), 0) > 127).astype(np.uint8),
                        np.ones((5, 5), np.uint8)) > 0
        best = None
        for mm in r["outputs"]["out_binary_masks"]:
            m = np.asarray(mm) > 0
            if m.sum() > 60 and (m & dm).sum() / m.sum() > 0.8:
                best = m & dm if best is None else best | (m & dm)
        p = sub / f"{i:05d}.png"
        if not (keep / p.name).exists():
            shutil.copy(p, keep / p.name)
        if best is None:
            log[i] = {"tracked": int(area[i]), "detected": None}
            print(f"[fix] frame {i}: detector found no {a.text}; tracked mask kept", flush=True)
            continue
        cv2.imwrite(str(p), best.astype(np.uint8) * 255)
        log[i] = {"tracked": int(area[i]), "detected": int(best.sum())}
        print(f"[fix] frame {i}: tracked {int(area[i])} px -> detected {int(best.sum())} px", flush=True)
    (sub.parent / "fix.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
