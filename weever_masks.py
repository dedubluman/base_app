#!/usr/bin/env python3
"""SAM 3.1 masks for a single fish (weever / Trachinus) on sand, CPU.

Two SAM 3.1 passes, both with the text prompt (default "fish"):

  coarse  full portrait frame, every --stride-th frame, one session. Gives
          the fish trajectory (box per sampled frame).
  fine    a 1080x1080 window that follows the fish (smoothed trajectory),
          so SAM sees the animal at ~0.93x native scale instead of the
          0.52x vertical squash of a 1080x1920 frame into 1008x1008. This
          is what keeps fin rays and the tail edge. Overlapping chunks;
          the fish is linked across chunk seams by IoU, and inside each
          chunk the object that best overlaps the coarse track is kept
          (shadows / debris detections are rejected).

Optional corrective point prompts (frame, x, y, label) in full-frame pixels
can be passed with --points JSON; they are mapped into the window.

Output: masks/<stem>__fish/fine/<frame:05d>.png  (0/255, 1080x1920)
        masks/<stem>__fish/coarse.json, meta.json

    python weever_masks.py --input inputs/weever.mov
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.masking import _build_predictor, _chunks

REPO = Path(__file__).resolve().parent


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def extract(video: Path, d: Path) -> int:
    if not (d.exists() and any(d.glob("*.jpg"))):
        d.mkdir(parents=True, exist_ok=True)
        # ffmpeg applies the display-matrix rotation -> upright portrait frames
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-fps_mode", "passthrough",
                        "-q:v", "1", "-qmin", "1", "-start_number", "0", str(d / "%05d.jpg")],
                       check=True)
    return len(list(d.glob("*.jpg")))


def run_session(pred, frame_paths, prompt, points=()):
    """One SAM 3.1 session over the given frame files; returns
    {obj_id: {local_idx: bool mask}} and {obj_id: {local_idx: prob}}."""
    import torch

    cdir = REPO / ".cache" / "sam_chunk_fish"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True)
    for i, p in enumerate(frame_paths):
        (cdir / f"{i:05d}.jpg").symlink_to(p)
    sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
    r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
    ids = list(r.get("outputs", {}).get("out_obj_ids", []))
    if points:
        # interactive SAM 3.1 flow: full text propagation first ...
        for _ in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            pass
        # ... then refinement of the text-detected fish itself: all clicks of
        # one frame in a single call (positive on the body, negative on the
        # cast shadow), on the fish's own object id.
        fish_id = int(ids[0]) if len(ids) == 1 else 0
        by_frame: dict[int, list] = {}
        for (li, xn, yn, lab) in points:
            by_frame.setdefault(int(li), []).append((xn, yn, lab))
        for li, pl in sorted(by_frame.items()):
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": li,
                                 "points": torch.tensor([[x, y] for x, y, _ in pl], dtype=torch.float32),
                                 "point_labels": torch.tensor([lab for *_, lab in pl], dtype=torch.int32),
                                 "obj_id": fish_id})
        print(f"[sam]   refine: text ids {ids}, {sum(len(v) for v in by_frame.values())} clicks "
              f"on {len(by_frame)} frames -> obj {fish_id}", flush=True)
    masks, probs = {}, {}
    for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
        fi = int(resp["frame_index"])
        o = resp["outputs"]
        pr = list(o.get("out_probs", [])) or [None] * len(o["out_obj_ids"])
        for oid, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], pr):
            m = np.asarray(m) > 0
            if m.any():
                masks.setdefault(int(oid), {})[fi] = m
                probs.setdefault(int(oid), {})[fi] = None if p is None else float(p)
    pred.handle_request({"type": "close_session", "session_id": sid})
    return masks, probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--prompt", default="fish")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=36)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--win", type=int, default=1080)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--only", choices=["coarse", "fine", "both"], default="both")
    ap.add_argument("--points", type=Path, default=None,
                    help="JSON list of [frame, x, y, label] corrective clicks (full-frame px)")
    a = ap.parse_args()

    cache = REPO / ".cache"
    fdir = cache / f"frames_{a.input.stem}"
    n = extract(a.input, fdir)
    H, W = cv2.imread(str(fdir / "00000.jpg")).shape[:2]
    out = REPO / "masks" / f"{a.input.stem}__fish"
    (out / "fine").mkdir(parents=True, exist_ok=True)
    print(f"[sam] {n} frames {W}x{H}", flush=True)

    t0 = time.time()
    cfg = Masking(max_num_objects=4, detector_batch=1)
    pred, on_gpu = _build_predictor(cfg, REPO, cache)
    dev = "cuda" if on_gpu else "cpu"
    print(f"[sam] SAM 3.1 multiplex ({cfg.checkpoint}) loaded on device={dev} "
          f"in {time.time() - t0:.0f}s", flush=True)

    # ---------------- coarse pass ----------------
    cf = out / "coarse.json"
    if a.only in ("coarse", "both") or not cf.exists():
        t0 = time.time()
        idx = list(range(0, n, a.stride))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        masks, probs = run_session(pred, [fdir / f"{i:05d}.jpg" for i in idx], a.prompt)
        # the fish = the object present in most sampled frames with largest area
        best = max(masks, key=lambda o: len(masks[o]) * np.median([m.sum() for m in masks[o].values()]))
        track = {}
        for li, m in masks[best].items():
            ys, xs = np.nonzero(m)
            track[idx[li]] = dict(box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                  c=[float(xs.mean()), float(ys.mean())], area=int(m.sum()))
        cv2.imwrite(str(out / "coarse_sample.png"), masks[best][len(idx) // 2].astype(np.uint8) * 255)
        cf.write_text(json.dumps({"objects": {str(o): len(v) for o, v in masks.items()},
                                  "fish_obj": best, "idx": idx, "track": track}, indent=1))
        print(f"[sam] coarse: {len(idx)} frames, objects {{id: n_frames}} "
              f"{ {o: len(v) for o, v in masks.items()} } -> fish {best} "
              f"({len(track)} frames) {time.time() - t0:.0f}s", flush=True)
        del masks
        gc.collect()
    if a.only == "coarse":
        return 0

    # ---------------- window trajectory ----------------
    C = json.loads(cf.read_text())
    tk = {int(k): v for k, v in C["track"].items()}
    ks = sorted(tk)
    fr = np.arange(n)
    cy = np.interp(fr, ks, [(tk[k]["box"][1] + tk[k]["box"][3]) / 2 for k in ks])
    top = np.interp(fr, ks, [tk[k]["box"][1] for k in ks])
    bot = np.interp(fr, ks, [tk[k]["box"][3] for k in ks])
    # smooth (sigma 6 frames) so the window glides like a slow camera tilt
    g = cv2.getGaussianKernel(37, 6).ravel()
    cy_s = np.convolve(np.pad(cy, 18, mode="edge"), g, mode="valid")
    wy = np.clip(np.round(cy_s - a.win / 2), 0, H - a.win).astype(int)
    margin = np.minimum(top - wy, wy + a.win - bot)
    print(f"[sam] window y {wy.min()}..{wy.max()}, min fish margin {margin.min():.0f}px", flush=True)
    wx = 0
    win_dir = cache / f"frames_{a.input.stem}_win"
    win_dir.mkdir(exist_ok=True)
    for i in range(n):
        p = win_dir / f"{i:05d}.jpg"
        if not p.exists():
            img = cv2.imread(str(fdir / f"{i:05d}.jpg"))
            cv2.imwrite(str(p), img[wy[i]:wy[i] + a.win, wx:wx + a.win], [cv2.IMWRITE_JPEG_QUALITY, 100])

    pts = json.loads(a.points.read_text()) if a.points else []

    # ---------------- fine pass ----------------
    end = a.end or n
    mf = out / "meta.json"
    meta = json.loads(mf.read_text()) if mf.exists() else {"chunks": [], "frames": {}}
    have = {int(p.stem) for p in (out / "fine").glob("*.png")} if a.start > 0 else set()
    prev: dict[int, np.ndarray] = {}
    for f in range(max(0, a.start - a.overlap), a.start):
        if f in have:
            prev[f] = cv2.imread(str(out / "fine" / f"{f:05d}.png"), 0) > 127

    def coarse_full(f):  # nearest coarse box as a full-frame bool box
        k = min(ks, key=lambda k: abs(k - f))
        x0, y0, x1, y1 = tk[k]["box"]
        b = np.zeros((H, W), bool)
        b[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40] = True
        return b

    for ci, (cs, ce) in enumerate(_chunks(end - a.start, a.chunk, a.overlap)):
        cs, ce = cs + a.start, ce + a.start
        t0 = time.time()
        cpts = [(f - cs, (x - wx) / a.win, (y - wy[f]) / a.win, lab)
                for f, x, y, lab in pts if cs <= f < ce]
        masks, probs = run_session(pred, [win_dir / f"{i:05d}.jpg" for i in range(cs, ce)],
                                   a.prompt, cpts)

        def to_full(li, m):
            f = cs + li
            full = np.zeros((H, W), bool)
            full[wy[f]:wy[f] + a.win, wx:wx + a.win] = m
            return full

        def score(o):
            fs = masks[o]
            seam = [li for li in fs if cs + li in prev]
            if seam:
                return 10 + np.mean([iou(to_full(li, fs[li]), prev[cs + li]) for li in seam])
            # no history: overlap with the coarse track box, weighted by coverage
            s = [np.logical_and(to_full(li, m), coarse_full(cs + li)).sum() / max(1, m.sum())
                 for li, m in fs.items()]
            return float(np.mean(s)) * len(fs) / (ce - cs)

        chosen = max(masks, key=score) if masks else None
        nf = 0
        for li in range(ce - cs):
            f = cs + li
            if chosen is None or li not in masks[chosen]:
                continue
            m = to_full(li, masks[chosen][li])
            if f in prev and f < cs + a.overlap // 2:
                continue  # first half of the seam: keep the previous chunk's mask
            cv2.imwrite(str(out / "fine" / f"{f:05d}.png"), m.astype(np.uint8) * 255)
            meta["frames"][str(f)] = {"chunk": len(meta["chunks"]), "obj": chosen, "clicks": bool(cpts), "area": int(m.sum()),
                                      "prob": probs[chosen].get(li), "wy": int(wy[f])}
            prev[f] = m
            nf += 1
        prev = {f: m for f, m in prev.items() if f >= ce - a.overlap}
        meta["chunks"].append({"range": [cs, ce], "objects": sorted(masks), "fish": chosen,
                               "clicks": len(cpts),
                               "score": float(score(chosen)) if chosen is not None else 0})
        meta.update(num_frames=n, width=W, height=H, prompt=a.prompt, device=dev,
                    checkpoint=cfg.checkpoint, model="SAM 3.1 multiplex", win=a.win)
        mf.write_text(json.dumps(meta, indent=1))
        print(f"[sam] chunk {ci + 1} frames {cs}-{ce - 1}: objects {sorted(masks)} -> fish {chosen} "
              f"(score {meta['chunks'][-1]['score']:.2f}), wrote {nf}, {time.time() - t0:.0f}s",
              flush=True)
        del masks
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
