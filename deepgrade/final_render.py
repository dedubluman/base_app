"""Encode the retimed frames to the final 15.000 s H.264 MP4 (+ optional 10-bit ProRes master).

    python -m deepgrade.final_render --out output/underwater_cinematic_final_15s.mp4

Video: 450 frames at exactly 30 fps = 15.000 s, H.264 High yuv420p, BT.709 tv-range tags, TPDF-dithered
16-bit -> 8-bit conversion (no banding in the dark water gradients), CRF 14.
Audio: the original AAC track, pitch-preserving time stretch (rubberband) to exactly the video duration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .interp import load16
from .render import MasterEncoder

REPO = Path(__file__).resolve().parent.parent


def audio_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=duration", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return float(json.loads(out)["streams"][0]["duration"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=REPO / "output/v2/retimed_frames")
    ap.add_argument("--src", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--out", type=Path, default=REPO / "output/underwater_cinematic_final_15s.mp4")
    ap.add_argument("--master", type=Path, default=REPO / "output/underwater_cinematic_final_15s_master_prores.mov")
    ap.add_argument("--no-master", action="store_true")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=14)
    a = ap.parse_args()
    files = sorted(a.frames.glob("*.png"))
    n = len(files)
    assert n == round(a.duration * a.fps), f"{n} frames, expected {round(a.duration * a.fps)}"
    first = load16(files[0])
    h, w = first.shape[:2]
    tempo = audio_duration(a.src) / a.duration
    af = (f"rubberband=tempo={tempo:.6f}:pitch=1:transients=mixed:pitchq=quality,"
          f"apad=whole_dur={a.duration},atrim=end={a.duration},asetpts=N/SR/TB")
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
           "-i", str(a.src), "-map", "0:v:0", "-map", "1:a:0",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
           "-af", af,
           "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(a.crf),
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
           "-c:a", "aac", "-b:a", "192k", "-t", f"{a.duration:.3f}", "-movflags", "+faststart", str(a.out)]
    print("audio tempo", f"{tempo:.6f}", "->", a.duration, "s")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    master = None
    if not a.no_master:
        class Info:  # minimal stand-in for shades.io.VideoInfo
            width, height, fps = w, h, a.fps
        master = MasterEncoder(a.master, Info)
    rng = np.random.default_rng(1234)
    for i, f in enumerate(files):
        rgb = load16(f)
        if master is not None:
            master.write(rgb)
        noise = (rng.random(rgb.shape[:2] + (1,), np.float32) - rng.random(rgb.shape[:2] + (1,), np.float32)) / 255.0
        proc.stdin.write((np.clip(rgb + noise, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())
        if i % 50 == 0:
            print(f"[encode] {i}/{n}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    if master is not None:
        master.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
