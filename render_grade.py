#!/usr/bin/env python3
"""Render the reference-matched underwater grade.

    python render_grade.py --input inputs/player_export.mov --masks masks_refined/player_export \
        --params grade_data/params.json --output output/UNDERWATER_CINEMATIC_FINAL_1080P.mp4

Per frame: environment LUT everywhere, gentle diver LUT inside the (temporally refined) SAM 3.1
alpha. Frames are decoded/encoded as 16-bit RGB, TPDF-dithered, scaled to fit 1920x1080 with
pillarboxing (never stretched or cropped), H.264 High yuv420p BT.709, +faststart, audio copied."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np
from shades.io import probe, iter_frames
import underwater_grade as ug

REPO = Path(__file__).resolve().parent


def load_alpha(mdir: Path | None, i: int, shape) -> np.ndarray | None:
    if mdir is None:
        return None
    p = mdir / f"{i:05d}.png"
    if not p.exists():
        return None
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return a.astype(np.float32)[..., None] / 255.0


class Out:
    def __init__(self, src, dst, info, crf, preset, W=1920, H=1080):
        ow = int(round(info.width * H / info.height / 2)) * 2
        self.rng = np.random.default_rng(1234)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", f"{info.width}x{info.height}", "-r", str(info.fps), "-i", "-", "-i", str(src),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
               "-vf", f"scale={ow}:{H}:flags=lanczos+accurate_rnd+full_chroma_int,"
                      f"pad={W}:{H}:{(W - ow)//2}:0:black,"
                      "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
               "-c:v", "libx264", "-profile:v", "high", "-preset", preset, "-crf", str(crf),
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-color_range", "tv", "-movflags", "+faststart", str(dst)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, x):
        n = (self.rng.random(x.shape[:2] + (1,), np.float32) - self.rng.random(x.shape[:2] + (1,), np.float32)) / 255.0
        self.p.stdin.write((np.clip(x + n, 0, 1) * 65535 + .5).astype(np.uint16).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--params", type=Path, default=REPO / "grade_data/params.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--frames", help="a:b frame range (test render, no audio sync)")
    ap.add_argument("--stills", help="comma list of frame indices -> PNGs next to --output instead of video")
    a = ap.parse_args()
    P = json.loads(a.params.read_text())
    env = ug.build_env_lut(33, P["env"]); dvr = ug.build_diver_lut(33, P["diver"])
    info = probe(a.input)
    still_idx = {int(x) for x in a.stills.split(",")} if a.stills else None
    lo, hi = (map(int, a.frames.split(":")) if a.frames else (0, info.n_frames))
    out = None if still_idx else Out(a.input, a.output, info, a.crf, a.preset)
    for i, f in enumerate(iter_frames(a.input, info)):
        if i >= hi: break
        if i < lo or (still_idx is not None and i not in still_idx):
            continue
        g = ug.apply_lut(f, env)
        al = load_alpha(a.masks, i, f.shape)
        if al is not None:
            g = g * (1 - al) + ug.apply_lut(f, dvr) * al
        if still_idx:
            cv2.imwrite(str(a.output.with_name(f"{a.output.stem}_{i:05d}.png")),
                        cv2.cvtColor((np.clip(g, 0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            out.write(g)
            if i % 50 == 0: print("frame", i, flush=True)
    if out: out.close()


if __name__ == "__main__":
    main()
