#!/usr/bin/env python3
"""SAM 3.1 keyframe masks for the deep-abyss grade (divers + fish).

SAM 3.1 on CPU costs ~12 s per frame, so the clip is segmented on every
``--stride``-th frame only (default 6 -> 5 fps keyframes); abyss_propagate.py
then carries the masks to the in-between frames with optical flow.

Each job is one text prompt run through SAM 3.1's multiplex tracker in
overlapping chunks of keyframes. All instances of the class are merged into
ONE mask per keyframe (the grade treats every diver / every fish alike, so
cross-chunk identity linking is not needed). Spurious one-off detections are
dropped (probability and minimum-lifetime filters).

SAM resizes its input to a 1008x1008 square, which squashes a 9:16 frame by
1.8x and costs thin fins and tails. Each job can therefore feed SAM a square
WxW crop of the frame (``crop_y``); masks are pasted back at full frame size.

Output (one folder per job):
    masks/<stem>__abyss/<job>/kf_<frame:05d>.png   uint8 0/255, full source res
    masks/<stem>__abyss/<job>/meta.json            per keyframe: instances, probs

    python abyss_masks.py --input inputs/dive.mov                 # both jobs
    python abyss_masks.py --input inputs/dive.mov --jobs diver    # one job
    python abyss_masks.py --input inputs/dive.mov --stride 4 --chunk 24
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

from abyss_common import probe
from shades.config import Masking
from shades.masking import _build_predictor, _chunks

REPO = Path(__file__).resolve().parent

# job name -> (SAM text prompt, crop_y or None). The diver crop sits at the top
# of the frame (divers hang in the upper half), the fish crop is bottom-aligned
# (fish swim over the seabed). A square crop keeps the animals' aspect ratio.
JOBS = {
    "diver": ("scuba diver", 0),
    "fish": ("fish", -1),          # -1 -> bottom-aligned square
}


def extract_keyframes(src: Path, out: Path, stride: int, w: int, h: int, crop_y: int | None,
                      jpeg_q: int = 2) -> int:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    vf = f"select=not(mod(n\\,{stride}))"
    if crop_y is not None:
        vf += f",crop={w}:{w}:0:{crop_y}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-fps_mode", "passthrough",
                    "-q:v", str(jpeg_q), "-start_number", "0", str(out / "%05d.jpg")], check=True)
    return len(list(out.glob("*.jpg")))


def run_job(pred, name: str, prompt: str, crop_y: int | None, src: Path, info: dict, a, out_root: Path,
            cache: Path) -> None:
    w, h, n_all = info["w"], info["h"], info["frames"]
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    kf_frames = list(range(0, n_all, a.stride))
    meta_f = out / "meta.json"
    key = {"stride": a.stride, "prompt": prompt, "crop_y": crop_y, "chunk": a.chunk, "overlap": a.overlap,
           "n": n_all}
    meta = {"key": key, "done_chunks": [], "kf": {}}
    if meta_f.exists() and not a.fresh:
        old = json.loads(meta_f.read_text())
        if old.get("key") == key:
            meta = old
    elif a.fresh:
        for p in out.glob("kf_*.png"):
            p.unlink()

    fdir = cache / f"sam_kf_{src.stem}_{name}"
    nk = extract_keyframes(src, fdir, a.stride, w, h, crop_y)
    assert nk == len(kf_frames), (nk, len(kf_frames))
    chunks = _chunks(nk, a.chunk, a.overlap)
    print(f"[{name}] prompt={prompt!r} crop_y={crop_y} {nk} keyframes, {len(chunks)} chunks", flush=True)

    acc: dict[int, np.ndarray] = {}   # keyframe ordinal -> union mask of finished chunks
    for p in out.glob("kf_*.png"):
        acc[int(p.stem[3:]) // a.stride] = cv2.imread(str(p), 0) > 127

    for ci, (cs, ce) in enumerate(chunks):
        if ci in meta["done_chunks"]:
            continue
        t0 = time.time()
        cdir = cache / f"sam_chunk_{src.stem}_{name}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir()
        for k in range(cs, ce):
            (cdir / f"{k - cs:05d}.jpg").symlink_to(fdir / f"{k:05d}.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
        local: dict[int, dict[int, np.ndarray]] = {}
        lprob: dict[int, dict[int, float]] = {}
        for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            k = cs + int(resp["frame_index"])
            o = resp["outputs"]
            pr = list(o.get("out_probs", [])) or [None] * len(o["out_obj_ids"])
            for oid, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], pr):
                m = np.asarray(m) > 0
                if m.any():
                    local.setdefault(int(oid), {})[k] = m
                    lprob.setdefault(int(oid), {})[k] = 1.0 if p is None else float(p)
        pred.handle_request({"type": "close_session", "session_id": sid})

        kept = []
        for oid, fr in local.items():
            ok = {k: m for k, m in fr.items() if lprob[oid][k] >= a.min_prob}
            if len(ok) < a.min_life:           # spurious one-off detection
                continue
            kept.append(oid)
            for k, m in ok.items():
                if crop_y is not None:         # paste the square crop back at full frame
                    full = np.zeros((h, w), bool)
                    full[crop_y:crop_y + m.shape[0], :m.shape[1]] = m
                    m = full
                elif m.shape[0] != h:
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                acc[k] = acc.get(k, np.zeros((h, w), bool)) | m
                meta["kf"].setdefault(str(kf_frames[k]), []).append(
                    {"obj": oid, "chunk": ci, "prob": round(lprob[oid][k], 3), "area": int(m.sum())})
        for k in range(cs, ce):                # empty keyframes are written too (= "no subject")
            m = acc.get(k, np.zeros((h, w), bool))
            cv2.imwrite(str(out / f"kf_{kf_frames[k]:05d}.png"), m.astype(np.uint8) * 255)
        meta["done_chunks"].append(ci)
        meta_f.write_text(json.dumps(meta))
        print(f"[{name}] chunk {ci + 1}/{len(chunks)} kf {cs}-{ce - 1}: instances {sorted(local)} kept {kept} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del local, lprob
        gc.collect()
    shutil.rmtree(cache / f"sam_chunk_{src.stem}_{name}", ignore_errors=True)
    shutil.rmtree(fdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--jobs", default="diver,fish", help="comma list of: " + ", ".join(JOBS))
    ap.add_argument("--stride", type=int, default=6, help="segment every Nth frame")
    ap.add_argument("--chunk", type=int, default=20, help="keyframes per SAM session (RAM ~100 MB each)")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--min-prob", type=float, default=0.45)
    ap.add_argument("--min-life", type=int, default=3, help="drop instances seen on fewer keyframes per chunk")
    ap.add_argument("--max-objects", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite cached masks")
    a = ap.parse_args()

    info = probe(a.input)
    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    out_root = REPO / "masks" / f"{a.input.stem}__abyss"
    cfg = Masking(max_num_objects=a.max_objects, detector_batch=2)
    t0 = time.time()
    pred, on_gpu = _build_predictor(cfg, REPO, cache)
    print(f"[sam] SAM 3.1 multiplex on {'CUDA' if on_gpu else 'CPU'} ({time.time() - t0:.0f}s); "
          f"{info['w']}x{info['h']}, {info['frames']} frames, stride {a.stride}", flush=True)
    for name in a.jobs.split(","):
        prompt, cy = JOBS[name]
        crop_y = None if cy is None else (info["h"] - info["w"] if cy < 0 else cy)
        run_job(pred, name, prompt, crop_y, a.input, info, a, out_root, cache)
    print(f"[sam] all done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
