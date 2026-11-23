"""Grade a handful of keyframes with the single-frame SAM 3.1 masks and build a
contact sheet (source | graded | region weights).

    python -m deepgrade.preview --frames 0,30,60,78,108,150,186,204,228 --out output/preview
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shades.io import probe, read_frames_at

from .color import rgb_to_oklab
from .grade import frame_stats, grade_frame
from .params import Params
from .regions import compose, refine_diver

REPO = Path(__file__).resolve().parent.parent


def load_png(path: Path, shape) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, np.float32)
    return cv2.imread(str(path), 0).astype(np.float32) / 255.0


def preview_weights(mdir: Path, f: int, rgb: np.ndarray, p: Params):
    H, W, _ = rgb.shape
    luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    raw = np.maximum(load_png(mdir / "diver_single" / f"{f:05d}.png", (H, W)),
                     load_png(mdir / "snorkel" / f"{f:05d}.png", (H, W)))
    D = refine_diver(raw, luma,
                     p.diver_feather_px, p.diver_guided_r, p.diver_guided_eps)
    S = load_png(mdir / "sky" / f"{f:05d}.png", (H, W))
    R = np.maximum(load_png(mdir / "rocks" / f"{f:05d}.png", (H, W)),
                   load_png(mdir / "mountain" / f"{f:05d}.png", (H, W)))
    Wt = load_png(mdir / "water" / f"{f:05d}.png", (H, W))
    wcov = float((Wt * (1 - D)).mean() / max((1 - D).mean(), 1e-3))
    gate = 1.0 if wcov < 0.05 else 0.0
    return compose(D, S, R, Wt, gate, p.region_feather_px), gate


def debug_weights(w: dict) -> np.ndarray:
    col = {"D": (1.0, 0.45, 0.35), "S": (0.95, 0.9, 0.5), "R": (0.3, 0.9, 0.35),
           "U": (0.85, 0.6, 0.25), "W": (0.2, 0.35, 0.95)}
    img = np.zeros(w["D"].shape + (3,), np.float32)
    for k, c in col.items():
        img += w[k][..., None] * np.array(c, np.float32)
    return np.clip(img, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--frames", default="0,30,60,78,108,150,186,204,228")
    ap.add_argument("--out", type=Path, default=REPO / "output/preview")
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--final", action="store_true", help="use the built per-frame weights (masks/<stem>/final)")
    a = ap.parse_args()
    p = Params.load(a.params) if a.params else Params()
    idx = [int(x) for x in a.frames.split(",")]
    info = probe(a.video)
    frames = read_frames_at(a.video, info, idx)
    mdir = REPO / "masks" / a.video.stem
    a.out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    for f in idx:
        rgb = frames[f]
        if a.final:
            from .render import load_weights
            w = load_weights(mdir / "final", f)
            gate = json.loads((mdir / "final/gate.json").read_text())["gate"][str(f)]
        else:
            w, gate = preview_weights(mdir, f, rgb, p)
        st = frame_stats(rgb, rgb_to_oklab(rgb), w, p)
        st["gate"] = gate
        out = grade_frame(rgb, w, st, p)
        cv2.imwrite(str(a.out / f"graded_{f:03d}.png"),
                    cv2.cvtColor((out * 65535 + .5).astype(np.uint16), cv2.COLOR_RGB2BGR))
        dbg = debug_weights(w)
        tile = np.concatenate([rgb, out, dbg], axis=1)
        tile = cv2.resize(tile, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"f{f} gate={gate:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (1, 1, 1), 2)
        rows.append(tile)
        stats[f] = {k: v for k, v in st.items()}
    sheet = np.concatenate(rows, axis=0) if len(rows) <= 3 else None
    # grid of ncols tiles-rows
    ncol = 1
    sheets = []
    per = 3
    for i in range(0, len(rows), per):
        chunk = rows[i:i + per]
        sheets.append(np.concatenate(chunk, axis=0))
    for i, s in enumerate(sheets):
        cv2.imwrite(str(a.out / f"sheet_{i}.jpg"),
                    cv2.cvtColor((np.clip(s, 0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
    p.dump(a.out / "params.json")
    print("done", len(rows), "tiles ->", a.out)


if __name__ == "__main__":
    main()
