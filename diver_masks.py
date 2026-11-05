#!/usr/bin/env python3
"""SAM 3.1 diver masks, propagated in both directions.

Pass "fwd" runs the chunked SAM 3.1 tracker (shades.masking.run_sam31) on the
clip as is. Pass "bwd" runs it on the time-reversed clip, so every chunk is
seeded from its *last* frame and tracks backwards. The two mask sets are
merged later (diver_reels.py) per frame.

    python diver_masks.py --input inputs/diver.mov --direction fwd
    python diver_masks.py --input inputs/diver.mov --direction bwd
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from shades.config import Masking
from shades.masking import run_sam31

REPO = Path(__file__).resolve().parent


def reversed_proxy(src: Path) -> Path:
    dst = REPO / ".cache" / f"{src.stem}_rev.mp4"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
                        "-vf", "reverse", "-c:v", "libx264", "-crf", "4",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--direction", choices=["fwd", "bwd"], required=True)
    ap.add_argument("--text", default="person")
    a = ap.parse_args()
    cfg = Masking(text_prompt=a.text, parts={}, chunk_frames=32, chunk_overlap=6,
                  max_num_objects=4)
    video = a.input if a.direction == "fwd" else reversed_proxy(a.input)
    mdir = REPO / "masks" / f"{a.input.stem}__{a.direction}"
    run_sam31(video, mdir, cfg, REPO, REPO / ".cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
