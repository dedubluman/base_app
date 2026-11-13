#!/usr/bin/env python3
"""SAM 3.1 concept prompts on keyframes (still-frame prompting, no tracking).

Findings on this footage (see README_uwgrade.md): "person" and "shorts" are
detected with p >= 0.9. Amorphous concepts are only found at low confidence:
"rock" -> individual boulders (p ~0.3), "stone wall" -> the whole rock mass
(p ~0.3) in some frames, "water surface" -> the upper surface / water column
(p ~0.1). "water" returns the whole frame minus the diver and is not used.
Detection thresholds are therefore lowered to 0.05 for these prompts; every
(frame, prompt) result is stored with its probabilities so the later steps
can weigh the seeds.

    python -m uwgrade.sam_keyframes --input inputs/player_export.mov \
        --frames 0:451:30 --prompts "rock|stone wall|water surface|shorts"
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if ":" in part:
            a, b, c = (int(x) for x in part.split(":"))
            out += list(range(a, b + 1, c))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--frames", required=True, help="e.g. 0:450:30,75,225")
    ap.add_argument("--prompts", required=True, help="'|' separated text prompts")
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--max-objects", type=int, default=32)
    ap.add_argument("--tag", default="keyframes")
    a = ap.parse_args()
    frames, prompts = parse_frames(a.frames), a.prompts.split("|")

    out = REPO / "masks" / f"{a.input.stem}__{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [(n, p) for n in frames for p in prompts
            if not (out / f"{n:05d}__{p.replace(' ', '_')}.npz").exists()]
    print(f"[kf] {len(frames)} frames x {len(prompts)} prompts, {len(todo)} to do", flush=True)
    if not todo:
        return 0

    import cv2
    import torch  # noqa: F401
    from shades.config import Masking
    from shades.masking import _build_predictor

    fdir = REPO / ".cache" / f"kf_frames_{a.tag}"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    sel = "+".join(f"eq(n\\,{n})" for n in frames)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.input), "-vf", f"select='{sel}'",
                    "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0",
                    str(fdir / "%05d.jpg")], check=True)
    assert len(list(fdir.glob("*.jpg"))) == len(frames), "keyframe extraction mismatch"

    pred, _ = _build_predictor(Masking(max_num_objects=a.max_objects), REPO, REPO / ".cache")
    pred.model.score_threshold_detection = a.thresh
    pred.model.new_det_thresh = a.thresh
    sid = pred.handle_request({"type": "start_session", "resource_path": str(fdir)})["session_id"]
    t_all = time.time()
    for i, (n, p) in enumerate(todo):
        k = frames.index(n)
        t0 = time.time()
        r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": k, "text": p,
                                 "output_prob_thresh": a.thresh})
        o = r["outputs"]
        ms = [np.asarray(m) > 0 for m in o["out_binary_masks"]]
        pr = np.array([float(x) for x in o.get("out_probs", [])], np.float32)
        h, w = (ms[0].shape if ms else cv2.imread(str(fdir / f"{k:05d}.jpg")).shape[:2])
        packed = np.packbits(np.stack(ms), axis=-1) if ms else np.zeros((0, h, (w + 7) // 8), np.uint8)
        np.savez_compressed(out / f"{n:05d}__{p.replace(' ', '_')}.npz", packed=packed, probs=pr,
                            shape=np.array([h, w]))
        print(f"[kf] {i + 1}/{len(todo)} frame {n} '{p}': {len(ms)} obj, p={np.round(pr[:3], 2).tolist()} "
              f"{time.time() - t0:.0f}s (elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)
    pred.handle_request({"type": "close_session", "session_id": sid})
    shutil.rmtree(fdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
