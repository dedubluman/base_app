"""Per-frame background colour of a source and a graded video (OKLab medians) + region weights.
    python -m deepgrade.diagnose --graded output/player_export_graded.mp4 --weights final --out output/diag_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe

from .color import lch, rgb_to_oklab

REPO = Path(__file__).resolve().parent.parent


def _frames(video: Path, w: int):
    if video.is_dir():                                  # 16-bit PNG sequence
        for f in sorted(video.glob("*.png")):
            im = cv2.cvtColor(cv2.imread(str(f), -1), cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
            yield cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
    else:
        yield from iter_frames(video, probe(video), scale_w=w)


def series(video: Path, fin: Path, w: int = 308):
    rows = []
    for t, fr in enumerate(_frames(video, w)):
        lab = rgb_to_oklab(fr)
        h, ww = lab.shape[:2]
        def m(k, thr=200):
            return cv2.resize(cv2.imread(str(fin / k / f"{t:05d}.png"), 0), (ww, h)) > thr
        D, S = m("D", 20), m("S", 128)
        bg = (m("W") | m("U")) & ~D & ~S
        # exclude bright particles / outliers so the median is the water itself
        sel = bg & (lab[..., 0] < np.percentile(lab[..., 0][bg], 90)) if bg.sum() > 300 else bg
        med = np.median(lab[sel], axis=0) if sel.sum() > 300 else np.array([np.nan] * 3)
        L, C, H = lch(med[None, None])
        u_frac = float(m("U").mean()); w_frac = float(m("W").mean())
        rows.append({"t": t, "L": float(med[0]), "a": float(med[1]), "b": float(med[2]),
                     "C": float(C[0, 0]), "h": float(H[0, 0]), "U": u_frac, "W": w_frac,
                     "meanL": float(lab[..., 0].mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--graded", type=Path, default=REPO / "output/player_export_graded.mp4")
    ap.add_argument("--weights", default="final")
    ap.add_argument("--out", type=Path, default=REPO / "output/diag_v1.json")
    a = ap.parse_args()
    fin = REPO / "masks" / "player_export" / a.weights
    src, gr = series(a.video, fin), series(a.graded, fin)
    a.out.write_text(json.dumps({"source": src, "graded": gr}))
    print("  t | SOURCE bg  L    C    h  | GRADED bg  L    C    h  | sand% water%")
    for t in list(range(0, 112, 4)) + list(range(120, 230, 15)):
        s, g = src[t], gr[t]
        print(f"{t:4d} |   {s['L']:.3f} {s['C']:.3f} {s['h']:4.0f}   |    {g['L']:.3f} {g['C']:.3f} {g['h']:4.0f}   | {100*g['U']:4.0f} {100*g['W']:4.0f}")
    gl = np.array([r["L"] for r in gr]); gh = np.array([r["h"] for r in gr]); gc = np.array([r["C"] for r in gr])
    sl = np.array([r["L"] for r in src])
    seg = slice(0, 120)
    print(f"\nunderwater frames 0-119: graded bg L range {np.nanmin(gl[seg]):.3f}-{np.nanmax(gl[seg]):.3f} "
          f"(source {np.nanmin(sl[seg]):.3f}-{np.nanmax(sl[seg]):.3f}); hue range {np.nanmin(gh[seg]):.0f}-{np.nanmax(gh[seg]):.0f}; chroma {np.nanmin(gc[seg]):.3f}-{np.nanmax(gc[seg]):.3f}")


if __name__ == "__main__":
    main()
