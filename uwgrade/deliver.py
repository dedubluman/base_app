#!/usr/bin/env python3
"""H.264 / MP4 delivery copies of the graded ProRes master at a given height (1080p).

    python -m uwgrade.deliver --master output/underwater_grade/player_export_graded_prores422hq.mov \
        --out output/underwater_grade/player_export_graded_1080p.mp4                    # height 1080, aspect kept, no crop
    python -m uwgrade.deliver --master ... --out ..._1920x1080.mp4 --canvas 1920x1080   # 16:9 frame, black side bars

The ProRes master is scaled (lanczos) and encoded in one step, so there is no extra generation loss. No crop, no
re-timing (every frame, same timestamps), the original AAC stream is copied untouched. BT.709 / limited range tagged.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from uwgrade import video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--canvas", default=None, help="WxH: pad the scaled picture onto this canvas (black bars), e.g. 1920x1080")
    ap.add_argument("--crf", type=int, default=15)
    a = ap.parse_args()
    info = video.probe(a.master)
    w = round(info["w"] * a.height / info["h"] / 2) * 2          # keep the aspect ratio, even width
    vf = f"scale={w}:{a.height}:flags=lanczos+accurate_rnd+full_chroma_int"
    if a.canvas:
        cw, ch = (int(x) for x in a.canvas.lower().split("x"))
        assert w <= cw and a.height <= ch, f"scaled picture {w}x{a.height} does not fit the {cw}x{ch} canvas"
        # yuv420p needs an even pad offset (chroma subsampling), so the picture sits at most 1 px off centre
        vf += f",pad={cw}:{ch}:{(cw - w) // 2 // 2 * 2}:{(ch - a.height) // 2 // 2 * 2}:black"
    vf += ",format=yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.master), "-map", "0:v:0", "-map", "0:a:0?",
                    "-vf", vf, "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                    "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                    "-c:a", "copy", "-movflags", "+faststart", str(a.out)], check=True)
    print(f"[deliver] {a.out}  ({w}x{a.height}{' on ' + a.canvas if a.canvas else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
