"""SAM 3.1 video-tracker masks for small diver parts (shorts, watch, mask).

The text detector finds none of these parts in this clip, and per-frame
point re-segmentation (diver_dark_points.py) drifts onto the skin because
the parts move a lot between stride-4 keyframes. Instead this uses SAM 3.1's
memory tracker: each chunk of the stride-4 proxy is its own session, seeded
on its first frame with clicks (first chunk: hand-placed seeds; later chunks:
clicks derived from the previous chunk's mask on the shared overlap frame),
then propagated through the chunk.

    python diver_dark_track.py shorts
Writes masks/diver__<part>_s4/{subject/*.png, meta.json}.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from diver_dark_points import SEEDS
from shades.config import Masking
from shades.masking import _build_predictor, extract_jpeg_frames

REPO = Path(__file__).resolve().parent
CHUNK, OVERLAP = 24, 2


def clicks_from_mask(m: np.ndarray, n_pos: int = 5, n_neg: int = 8):
    """Positives at well-separated interior points (distance-transform
    maxima), negatives on a ring just outside the mask (skin / water next
    to the part), so the tracker is told where the part ends."""
    m8 = (m > 0).astype(np.uint8)
    pts = []
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    d = dist.copy()
    for _ in range(n_pos):
        if d.max() < 2:
            break
        y, x = np.unravel_index(np.argmax(d), d.shape)
        pts.append([float(x), float(y), 1])
        cv2.circle(d, (int(x), int(y)), max(8, int(dist[y, x] * 1.5)), 0, -1)
    out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5)
    ys, xs = np.nonzero((out > 10) & (out < 22))
    if len(ys):
        for k in np.linspace(0, len(ys) - 1, n_neg).astype(int):
            pts.append([float(xs[k]), float(ys[k]), 0])
    return pts


def main(part: str) -> None:
    import torch

    cache = REPO / ".cache"
    proxy = cache / "diver__stick_s4.mp4"
    frames = cache / f"sam_frames_track_{part}"
    n = extract_jpeg_frames(proxy, frames, 2)
    h, w = cv2.imread(str(frames / "00000.jpg")).shape[:2]
    mdir = REPO / "masks" / f"diver__{part}_s4"
    if mdir.is_symlink():
        mdir.unlink()
    elif mdir.exists():
        shutil.rmtree(mdir)
    (mdir / "subject").mkdir(parents=True)
    pred, _ = _build_predictor(Masking(), REPO, cache)
    print(f"[track] {part}: {n} keyframes, SAM 3.1 loaded", flush=True)

    masks = np.zeros((n, h, w), bool)
    seed = SEEDS[part]
    s = 0
    ci = 0
    while s < n - 1 or (s == 0 and n == 1):
        e = min(n, s + CHUNK)
        t0 = time.time()
        cdir = cache / f"sam_track_chunk_{part}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for i in range(s, e):
            (cdir / f"{i - s:05d}.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        # a text prompt makes the wrapper keep the point-prompted object
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": 0, "text": part})
        pred.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": 0, "obj_id": 1,
            "points": torch.tensor([[x / w, y / h] for x, y, _ in seed], dtype=torch.float32),
            "point_labels": torch.tensor([int(l) for *_, l in seed], dtype=torch.int32)})
        got = 0
        for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            fi = s + int(resp["frame_index"])
            o = resp["outputs"]
            ids = [int(v) for v in o["out_obj_ids"]]
            if 1 in ids:
                mm = np.asarray(o["out_binary_masks"][ids.index(1)]) > 0
                if fi >= s + (OVERLAP if ci else 0) or not masks[fi].any():
                    masks[fi] = mm
                got += int(mm.any())
        pred.handle_request({"type": "close_session", "session_id": sid})
        areas = [int(masks[i].sum()) for i in range(s, e)]
        print(f"[track] chunk {ci} frames {s}-{e - 1}: {got} frames with mask, "
              f"area min/med/max {min(areas)}/{int(np.median(areas))}/{max(areas)}, "
              f"{time.time() - t0:.0f}s", flush=True)
        for i in range(s, e):
            cv2.imwrite(str(mdir / "subject" / f"{i:05d}.png"), masks[i].astype(np.uint8) * 255)
        if e == n:
            break
        # next chunk starts on the overlap frame, seeded from this chunk's mask
        s = e - OVERLAP
        ref = next((masks[k] for k in range(s, -1, -1) if masks[k].sum() > 60), None)
        seed = clicks_from_mask(ref) if ref is not None else SEEDS[part]
        ci += 1
    (mdir / "meta.json").write_text(json.dumps(
        {"num_frames": n, "stride": 4, "model": "SAM 3.1 multiplex tracker (points)",
         "part": part, "areas": [int(m.sum()) for m in masks]}))
    shutil.rmtree(frames, ignore_errors=True)
    print(f"[track] done {part}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
