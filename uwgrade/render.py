#!/usr/bin/env python3
"""Grade the clip.

    # quick stills (PNG) for QC
    python -m uwgrade.render --video inputs/player_export.mov --params uwgrade/params_player_export.json \
        --stills 0,150,300,450 --out-dir output/underwater_grade/stills
    # full master (ProRes 422 HQ, original AAC audio copied) + H.264 delivery copy
    python -m uwgrade.render --video inputs/player_export.mov --params uwgrade/params_player_export.json \
        --out-dir output/underwater_grade --h264
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from uwgrade import video  # noqa: E402
from uwgrade.build_masks import paths, read_a  # noqa: E402
from uwgrade.grade import Grader, load_params  # noqa: E402

_G: Grader | None = None
_BASE: Path | None = None


def _init(params_path, base):
    global _G, _BASE
    _G = Grader(load_params(params_path))
    _BASE = Path(base)
    cv2.setNumThreads(1)


def load_alphas(base: Path, t: int, hw) -> dict:
    H, W = hw
    D = read_a(base / "diver", t)
    R = read_a(base / "rock", t)
    if R.shape != (H, W):
        R = cv2.resize(R, (W, H), interpolation=cv2.INTER_CUBIC)
    al = {"diver": D, "rock": np.clip(R, 0, 1), "surface": np.zeros((H, W), np.float32)}
    for k in ("skin", "gear", "shorts"):
        f = base / k / f"{t:05d}.png"
        al[k] = read_a(base / k, t) if f.exists() else (D if k == "skin" else np.zeros((H, W), np.float32))
    return al


def _work(args):
    t, rgb16 = args
    rgb = rgb16.astype(np.float32) / 65535.0
    _G.seed = t
    out = _G(rgb, load_alphas(_BASE, t, rgb.shape[:2]))
    return t, (np.clip(out, 0, 1) * 65535 + 0.5).astype("<u2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stills", default=None, help="comma list of frame numbers -> PNG stills, no encode")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--h264", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    info = video.probe(a.video)
    stem = a.video.stem
    base, _, _ = paths(stem)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.stills:
        wanted = [int(x) for x in a.stills.split(",")]
        _init(a.params, base)
        for t in wanted:
            _, rgb16 = next(video.decode(a.video, 16, start=t, count=1))
            tt, out = _work((t, rgb16))
            bgr = cv2.cvtColor((out >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(a.out_dir / f"graded_{t:05d}.png"), bgr)
            cv2.imwrite(str(a.out_dir / f"source_{t:05d}.png"), cv2.cvtColor((rgb16 >> 8).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print("still", t)
        return 0

    name = a.name or f"{stem}_graded_prores422hq"
    master = a.out_dir / f"{name}.mov"
    w = video.ProResWriter(master, a.video, info["w"], info["h"], info["fps_num"], info["fps_den"])
    t0 = time.time()
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.params, base)) as pool:
        batch = []
        for t, rgb16 in video.decode(a.video, 16):
            batch.append((t, rgb16))
            if len(batch) == a.workers * 2:
                for tt, out in pool.map(_work, batch):
                    w.write(out)
                    done += 1
                batch = []
                print(f"[render] {done}/{info['frames']}  {time.time() - t0:.0f}s", flush=True)
        if batch:
            for tt, out in pool.map(_work, batch):
                w.write(out)
                done += 1
    w.close()
    print(f"[render] master written: {master} ({done} frames, {time.time() - t0:.0f}s)")
    if a.h264:
        h = a.out_dir / f"{name.replace('prores422hq', 'h264')}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(master), "-map", "0:v:0", "-map", "0:a:0?",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-profile:v", "high",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                        "-c:a", "copy", "-movflags", "+faststart", str(h)], check=True)
        print("[render] H.264 written:", h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
