#!/usr/bin/env python3
"""SAM 3.1 video tracking of the diver on every ``--stride``-th frame.

SAM 3.1 costs ~13 s/frame on 4 CPU cores, so the diver is tracked on every
2nd frame (15 fps) and interpolated to the other frames in ``masks.py`` (flow
warp + edge snap). Chunked and resumable (shades.masking.run_sam31).

    python -m uwgrade.sam_diver --input inputs/player_export.mov --stride 2
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


def stride_proxy(src: Path, stride: int) -> Path:
    dst = REPO / ".cache" / f"{src.stem}_s{stride}.mp4"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf",
                        f"select='not(mod(n,{stride}))'", "-fps_mode", "passthrough",
                        "-c:v", "libx264", "-crf", "2", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--text", default="person")
    ap.add_argument("--det-thresh", type=float, default=0.3,
                    help="lower than SAM's 0.4/0.65 so the small, hazy diver at the end of the clip is still picked up")
    a = ap.parse_args()
    proxy = stride_proxy(a.input, a.stride)

    orig = M._build_predictor

    def build(cfg, repo_root, cache_dir):
        pred, on_gpu = orig(cfg, repo_root, cache_dir)
        pred.model.score_threshold_detection = a.det_thresh
        pred.model.new_det_thresh = a.det_thresh
        return pred, on_gpu

    M._build_predictor = build
    cfg = Masking(text_prompt=a.text, parts={}, chunk_frames=32, chunk_overlap=4, max_num_objects=4)
    mdir = REPO / "masks" / f"{a.input.stem}__{a.text.replace(' ', '_')}_s{a.stride}"
    M.run_sam31(proxy, mdir, cfg, REPO, REPO / ".cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
