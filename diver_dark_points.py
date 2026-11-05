"""Point-prompted SAM 3.1 masks for small diver parts (shorts, watch, mask).

SAM 3.1's text detector finds none of "shorts", "swim trunks", "swimsuit",
"briefs", "watch", "wristwatch", "goggles", "mask" in this clip, and its
multiplex video wrapper drops click-only objects during propagation. So each
stride-4 keyframe is segmented on its own from clicks, and the clicks follow
the part: the previous keyframe's mask is first moved to this keyframe with
dense optical flow, positives go to well-separated interior points of the
moved mask and negatives to a ring just outside it (skin / water next to the
part). Each result is checked against the moved mask (IoU, area ratio); a
bad one is re-segmented with fewer clicks, and if that fails too the moved
mask itself is kept, so a single miss cannot drift onto the skin.

    python diver_dark_points.py shorts            # -> masks/diver__shorts_s4/
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent

# first-keyframe clicks (source px): [x, y, 1=on the part / 0=not]
SEEDS = {
    "shorts": [[680, 320, 1], [700, 335, 1], [660, 355, 1], [720, 345, 1],
               [770, 290, 0], [560, 320, 0], [740, 425, 0], [650, 470, 0]],
    "watch": [[740, 420, 1], [735, 440, 1], [725, 405, 1],
              [650, 480, 0], [800, 450, 0], [690, 340, 0], [560, 400, 0]],
    "mask": [[905, 60, 1], [890, 105, 1], [870, 140, 1],
             [780, 40, 0], [980, 80, 0], [760, 200, 0], [640, 120, 0]],
}


def clicks_from_mask(m: np.ndarray, n_pos: int = 5, n_neg: int = 8):
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


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def warp_mask(m, g_from, g_to):
    """Move mask m from frame g_from to g_to (half-res grey flow)."""
    f = cv2.calcOpticalFlowFarneback(g_to, g_from, None, 0.5, 4, 25, 3, 5, 1.1, 0)
    h, w = m.shape
    f = cv2.resize(f, (w, h)) * 2.0
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(m.astype(np.float32), gx + f[..., 0], gy + f[..., 1],
                     cv2.INTER_LINEAR, borderValue=0) > 0.5


def main(part: str) -> None:
    import torch

    from shades.config import Masking
    from shades.masking import _build_predictor, extract_jpeg_frames

    cache = REPO / ".cache"
    frames = cache / f"sam_frames_pts_{part}"
    n = extract_jpeg_frames(cache / "diver__stick_s4.mp4", frames, 2)
    n = min(n, int(os.environ.get("PTS_MAX", n)))
    imgs = [cv2.imread(str(frames / f"{i:05d}.jpg")) for i in range(n)]
    h, w = imgs[0].shape[:2]
    greys = [cv2.cvtColor(cv2.resize(im, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY) for im in imgs]
    mdir = REPO / "masks" / f"diver__{part}_s4"
    if mdir.is_symlink():
        mdir.unlink()
    elif mdir.exists():
        shutil.rmtree(mdir)
    (mdir / "subject").mkdir(parents=True)
    pred, _ = _build_predictor(Masking(), REPO, cache)
    one = cache / f"sam_one_{part}"

    def segment(i, pts):
        shutil.rmtree(one, ignore_errors=True)
        one.mkdir(parents=True)
        (one / "00000.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(one)})["session_id"]
        # a text prompt makes the wrapper process (and keep) this frame
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": part})
        r = pred.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": 0, "obj_id": 1,
            "points": torch.tensor([[x / w, y / h] for x, y, _ in pts], dtype=torch.float32),
            "point_labels": torch.tensor([int(l) for *_, l in pts], dtype=torch.int32)})
        pred.handle_request({"type": "close_session", "session_id": sid})
        o = r["outputs"]
        ids = [int(v) for v in o["out_obj_ids"]]
        return (np.asarray(o["out_binary_masks"][ids.index(1)]) > 0) if 1 in ids else np.zeros((h, w), bool)

    qc = {}
    prev = None
    for i in range(n):
        t0 = time.time()
        if prev is None or prev.sum() < 60:
            m = segment(i, SEEDS[part])
            rec = {"how": "seed"}
        else:
            moved = warp_mask(prev, greys[i - 1], greys[i])
            if moved.sum() < 60:
                moved = prev
            best, how, sc = None, None, -1.0
            for tag, pts in (("moved5", clicks_from_mask(moved, 5, 8)),
                             ("moved3", clicks_from_mask(moved, 3, 10))):
                c = segment(i, pts)
                ar = c.sum() / max(moved.sum(), 1)
                s_ = iou(c, moved) if 0.5 <= ar <= 2.0 else 0.0
                if s_ > sc:
                    best, how, sc = c, tag, s_
                if sc >= 0.55:
                    break
            if sc < 0.35:
                best, how = moved, "flow-fallback"
            m = best
            rec = {"how": how, "iou_moved": round(sc, 3)}
        rec["area"] = int(m.sum())
        qc[i] = rec
        cv2.imwrite(str(mdir / "subject" / f"{i:05d}.png"), m.astype(np.uint8) * 255)
        prev = m
        print(f"[pts] {part} {i + 1}/{n}: {rec} {time.time() - t0:.0f}s", flush=True)
    (mdir / "meta.json").write_text(json.dumps(
        {"num_frames": n, "stride": 4, "model": "SAM 3.1 multiplex (per-keyframe clicks, flow-guided)",
         "part": part, "qc": qc}, indent=1))
    shutil.rmtree(frames, ignore_errors=True)
    print(f"[pts] done {part}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
