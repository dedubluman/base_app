"""Contact sheet of graded frames (optionally source | graded) from a PNG sequence or a video.
    python -m deepgrade.sheet2 --graded output/v2/graded_frames --frames 0,14,45 --out sheet.jpg [--no-source]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
UP = Path("/root/.claude/uploads/70afcdbd-9d3d-51d2-9790-df796ada4833")


def grab_video(path: Path, i: int, w: int) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{i}),scale={w}:-2",
                        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def grab(path: Path, i: int, w: int) -> np.ndarray:
    if path.is_dir():
        im = cv2.imread(str(path / f"{i:05d}.png"), -1)
        im = (im / 257.0).astype(np.uint8) if im.dtype == np.uint16 else im
        return cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
    return grab_video(path, i, w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-source", action="store_true")
    ap.add_argument("--labels", default=None, help="comma separated labels, one per frame")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--refs", action="store_true", help="add the two reference photographs on top")
    a = ap.parse_args()
    idx = [int(x) for x in a.frames.split(",")]
    labels = a.labels.split(",") if a.labels else [f"frame {i}" for i in idx]
    tiles = []
    for i, lab in zip(idx, labels):
        g = grab(a.graded, i, a.width)
        t = g if a.no_source else np.concatenate([grab_video(a.source, i, a.width), g], 1)
        cv2.putText(t, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + a.cols], 1) for i in range(0, len(tiles), a.cols)]
    body = np.concatenate(rows, 0)
    if a.refs:
        refs = []
        for p in ("dee618b2-image.png", "999f1ff4-image.png"):
            im = cv2.imread(str(UP / p))
            if im is not None:
                refs.append(cv2.resize(im, (a.width, int(a.width * im.shape[0] / im.shape[1]))))
        if refs:
            h = max(r.shape[0] for r in refs)
            row = np.concatenate([cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in refs]
                                 + [np.zeros((h, body.shape[1] - a.width * len(refs), 3), np.uint8)], 1)
            cv2.putText(row, "reference photographs (targets)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            body = np.concatenate([row, body], 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), body, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet ->", a.out, body.shape)


if __name__ == "__main__":
    main()
