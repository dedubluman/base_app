#!/usr/bin/env python3
"""SAM 3.1 video tracking of the diver on every ``--stride``-th frame (``--offset`` picks which).

SAM 3.1 costs ~13 s/frame on 4 CPU cores, so each pass is split into a stride-2 stream (even frames,
then odd frames). Chunked and resumable (shades.masking.run_sam31).

    python -m uwgrade.sam_diver --input inputs/player_export.mov --stride 2 --offset 0   # even frames
    python -m uwgrade.sam_diver --input inputs/player_export.mov --stride 2 --offset 1   # odd frames

Why the odd frames are SAM-segmented too (measured afterwards against these SAM odd frames):
flow interpolation of an odd frame from its two even neighbours would have scored mean IoU 0.967, mean boundary
displacement 0.9 px (p90 1.4 px) -- fine on average -- but with a worst case of 11 px (frame 87, IoU 0.70) on the
fast-moving head. Single-frame errors like that flicker at the subject's edge, so every frame is SAM-direct and
interpolation is kept only as a fallback for frames SAM loses.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import shades.masking as M  # noqa: E402
from shades.config import Masking  # noqa: E402


def stride_proxy(src: Path, stride: int, offset: int = 0) -> Path:
    dst = REPO / ".cache" / (f"{src.stem}_s{stride}.mp4" if offset == 0 else f"{src.stem}_s{stride}o{offset}.mp4")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf",
                        f"select='eq(mod(n,{stride}),{offset})'", "-fps_mode", "passthrough",
                        "-c:v", "libx264", "-crf", "2", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--offset", type=int, default=0, help="track source frames n with n %% stride == offset")
    ap.add_argument("--text", default="person")
    ap.add_argument("--det-thresh", type=float, default=0.3,
                    help="lower than SAM's 0.4/0.65 so the small, hazy diver at the end of the clip is still picked up")
    a = ap.parse_args()
    proxy = stride_proxy(a.input, a.stride, a.offset)

    orig = M._build_predictor

    def build(cfg, repo_root, cache_dir):
        pred, on_gpu = orig(cfg, repo_root, cache_dir)
        pred.model.score_threshold_detection = a.det_thresh
        pred.model.new_det_thresh = a.det_thresh
        return pred, on_gpu

    M._build_predictor = build
    cfg = Masking(text_prompt=a.text, parts={}, chunk_frames=32, chunk_overlap=4, max_num_objects=4)
    tag = f"s{a.stride}" if a.offset == 0 else f"s{a.stride}o{a.offset}"
    mdir = REPO / "masks" / f"{a.input.stem}__{a.text.replace(' ', '_')}_{tag}"
    M.run_sam31(proxy, mdir, cfg, REPO, REPO / ".cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
