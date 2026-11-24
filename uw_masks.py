"""SAM 3.1 concept masks for the underwater cinematic grade.

Runs SAM 3.1 video propagation (text prompts) for several concepts on a
temporally sub-sampled, down-scaled copy of the clip, and stores one compact
mask stack per concept:

    <out>/<concept>.npz   masks  uint8[T_sub, h, w]  (0/255)
                          src_idx int[T_sub]          source frame index of each mask
                          prob   float[T_sub]         SAM score of the chosen track (nan if none)
    <out>/meta.json       clip info, prompts, timings

The grade (uw_grade.py) interpolates these stacks back to every source frame,
so masks stay temporally smooth even though SAM only ran on every Nth frame.
SAM 3.1 runs through the CPU port in shades/sam31_cpu.py when no CUDA GPU is
available (the upstream sam3/ clone is never modified).

    python uw_masks.py --video clip.mov --out masks_uw/clip \
        --concept subject=person --concept water="sea water" --concept sky=sky
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
from shades.masking import _PeakRSS, _build_predictor, _chunks, _iou

ROOT = Path(__file__).resolve().parent


def extract_frames(video: Path, out: Path, stride: int, width: int, limit: int | None,
                   start: int = 0, end: int | None = None):
    """JPEG-extract every `stride`-th source frame in [start, end), scaled to
    `width` px wide. Returns the list of source frame indices extracted."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = f"gte(n\\,{start})*" + (f"lt(n\\,{end})*" if end else "")
    vf = f"select='{rng}not(mod(n-{start}\\,{stride}))',scale={width}:-2:flags=area"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf,
           "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0"]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd.append(str(out / "%05d.jpg"))
    subprocess.run(cmd, check=True)
    n = len(list(out.glob("*.jpg")))
    return [start + i * stride for i in range(n)]


def run_concept(pred, name: str, prompt: str, frames_dir: Path, n: int,
                h: int, w: int, chunk: int, overlap: int, union: bool,
                work: Path, log, prompt_idx: int = 0, box=None):
    import torch  # noqa: F401  (predictor already imported torch)

    masks = np.zeros((n, h, w), bool)
    prob = np.full(n, np.nan, np.float32)
    have = np.zeros(n, bool)
    rss = _PeakRSS()
    for ci, (cs, ce) in enumerate(_chunks(n, chunk, overlap)):
        t0 = time.time()
        cdir = work / f"chunk_{name}"
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir(parents=True)
        for i in range(cs, ce):
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames_dir / f"{i:05d}.jpg")
        sid = pred.handle_request(
            {"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pidx = prompt_idx - cs if cs <= prompt_idx < ce else 0
        req = {"type": "add_prompt", "session_id": sid, "frame_index": pidx}
        if box is not None:   # visual prompt: normalised [x, y, w, h], positive
            req.update({"bounding_boxes": [list(box)], "bounding_box_labels": [1]})
        else:
            req["text"] = prompt
        pred.handle_request(req)
        per_obj: dict[int, dict[int, np.ndarray]] = {}
        probs: dict[int, dict[int, float]] = {}
        for resp in pred.handle_stream_request(
                {"type": "propagate_in_video", "session_id": sid}):
            fi = cs + int(resp["frame_index"])
            out = resp["outputs"]
            pr = list(out.get("out_probs", [])) or [None] * len(out["out_obj_ids"])
            for oid, m, p in zip(out["out_obj_ids"], out["out_binary_masks"], pr):
                m = np.asarray(m) > 0
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST) > 0
                per_obj.setdefault(int(oid), {})[fi] = m
                probs.setdefault(int(oid), {})[fi] = np.nan if p is None else float(p)
        pred.handle_request({"type": "close_session", "session_id": sid})

        overlap_idx = [i for i in range(cs, ce) if have[i]]

        def score(oid):
            fr = per_obj[oid]
            if overlap_idx:
                return float(np.mean([_iou(fr[i], masks[i]) if i in fr else 0.0
                                      for i in overlap_idx]))
            return float(sum(m.sum() for m in fr.values()))

        chosen = None
        if per_obj:
            chosen = None if union else max(per_obj, key=score)
        for i in range(cs, ce):
            if union:
                ms = [per_obj[o][i] for o in per_obj if i in per_obj[o]]
                m = np.any(ms, axis=0) if ms else None
                p = np.nanmax([probs[o][i] for o in per_obj if i in probs[o]]) if ms else np.nan
            else:
                m = per_obj.get(chosen, {}).get(i) if chosen is not None else None
                p = probs.get(chosen, {}).get(i, np.nan) if chosen is not None else np.nan
            if m is None:
                continue
            if have[i]:
                masks[i] |= m
            else:
                masks[i], have[i] = m, True
            prob[i] = p
        log(f"[{name}] chunk {ci + 1} frames {cs}-{ce - 1}: objects {sorted(per_obj)}"
            f"{'' if union else f' -> {chosen}'}  {time.time() - t0:.0f}s  "
            f"peak RSS {rss.take():.1f} GB")
        del per_obj, probs
        gc.collect()
    rss._stop = True
    return masks, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--concept", action="append", required=True,
                    help="name=text prompt (repeatable). Prefix name with + for union "
                         "of all SAM objects (water/sky); without it the single best track "
                         "is kept (subject).")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--width", type=int, default=540)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="only first N sub-sampled frames (test)")
    ap.add_argument("--range", default=None, help="source frames a:b (b exclusive) to process")
    ap.add_argument("--box", default=None,
                    help="x:y:w:h normalised box prompt (positive) given on --prompt-frame instead of text")
    ap.add_argument("--prompt-frame", type=int, default=None,
                    help="source frame on which the text prompt is given (default: first frame); "
                         "SAM propagates both ways from it")
    a = ap.parse_args()

    log = lambda s: print(s, flush=True)  # noqa: E731
    a.out.mkdir(parents=True, exist_ok=True)
    work = ROOT / ".cache" / f"uw_{a.video.stem}"
    work.mkdir(parents=True, exist_ok=True)
    frames_dir = work / "frames"
    r0, r1 = (int(x) for x in a.range.split(":")) if a.range else (0, None)
    src_idx = extract_frames(a.video, frames_dir, a.stride, a.width, a.limit, r0, r1)
    n = len(src_idx)
    h, w = cv2.imread(str(frames_dir / "00000.jpg")).shape[:2]
    log(f"[sam] {n} sub-sampled frames (stride {a.stride}) at {w}x{h}")

    cfg = Masking(max_num_objects=8, detector_batch=2)
    t0 = time.time()
    pred, on_gpu = _build_predictor(cfg, ROOT, ROOT / ".cache")
    log(f"[sam] SAM 3.1 loaded on {'CUDA' if on_gpu else 'CPU (bf16 port)'} in {time.time() - t0:.0f}s")

    timings = {}
    for spec in a.concept:
        name, prompt = spec.split("=", 1)
        union = name.startswith("+")
        name = name.lstrip("+")
        t0 = time.time()
        pidx = 0 if a.prompt_frame is None else int(np.argmin(np.abs(np.array(src_idx) - a.prompt_frame)))
        masks, prob = run_concept(pred, name, prompt, frames_dir, n, h, w,
                                  a.chunk, a.overlap, union, work, log, prompt_idx=pidx,
                                  box=None if a.box is None else [float(x) for x in a.box.split(':')])
        np.savez_compressed(a.out / f"{name}.npz", masks=masks.astype(np.uint8) * 255,
                            src_idx=np.array(src_idx), prob=prob)
        timings[name] = round(time.time() - t0, 1)
        log(f"[{name}] saved {a.out / (name + '.npz')} coverage "
            f"{masks.mean() * 100:.1f}% in {timings[name]}s")
    (a.out / "meta.json").write_text(json.dumps({
        "video": str(a.video), "stride": a.stride, "width": w, "height": h,
        "n_sub": n, "src_idx": src_idx, "concepts": a.concept,
        "device": "cuda" if on_gpu else "cpu", "seconds": timings}, indent=1))
    log("[sam] done")


if __name__ == "__main__":
    main()
