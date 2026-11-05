"""SAM 3.1 masks for diver_dark.py.

Runs SAM 3.1 (multiplex video predictor, local checkpoint; CPU port when no
CUDA) on every `stride`-th frame of [0, end) with one text prompt per run,
reusing shades.masking.run_sam31 (chunked sessions, cross-chunk IoU matching,
on-disk cache + resume). Masks land in masks/<stem>__<name>_s<stride>/.

    python diver_dark_sam.py --input clip.mov --name person --prompt person
    python diver_dark_sam.py --input clip.mov --name stick --prompt "selfie stick"
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import replace
from pathlib import Path

from shades.config import Masking
from shades.masking import run_sam31

REPO = Path(__file__).resolve().parent


def make_proxy(src: Path, dst: Path, stride: int, end: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
         "-vf", f"select='lt(n\\,{end})*not(mod(n\\,{stride}))',setpts=N/TB",
         "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "6",
         "-pix_fmt", "yuv420p", str(dst)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--end", type=int, default=588)
    ap.add_argument("--max-objects", type=int, default=4)
    a = ap.parse_args()

    cache = REPO / ".cache"
    stem = f"{a.input.stem}__{a.name}_s{a.stride}"
    proxy = cache / f"{stem}.mp4"
    if not proxy.exists():
        make_proxy(a.input, proxy, a.stride, a.end)
    cfg = replace(Masking(), text_prompt=a.prompt, max_num_objects=a.max_objects)
    run_sam31(proxy, REPO / "masks" / stem, cfg, REPO, cache)


if __name__ == "__main__":
    main()
