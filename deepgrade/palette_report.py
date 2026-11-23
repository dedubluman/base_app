"""Measure the graded video against the colour anchors and the reference photos.

    python -m deepgrade.palette_report --graded output/player_export_graded.mp4

Regions come from the final SAM 3.1 weights. Distances are OKLab Euclidean x100
(roughly delta-E-ok; < 4-6 reads as the same colour family at this scale).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe

from .color import hex_to_oklab, lch, rgb_to_oklab
from .params import Params

REPO = Path(__file__).resolve().parent.parent


def de(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 100)


def describe(lab):
    L, C, H = lch(np.asarray(lab, np.float32)[None, None])
    return f"L={lab[0]:.3f} C={C[0,0]:.3f} h={H[0,0]:.0f}"


def wmask(fin: Path, k: str, t: int, thr: float) -> np.ndarray:
    return cv2.imread(str(fin / k / f"{t:05d}.png"), 0).astype(np.float32) / 255.0 > thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--graded", type=Path, default=REPO / "output/player_export_graded.mp4")
    ap.add_argument("--out", type=Path, default=REPO / "output/palette_report.json")
    ap.add_argument("--step", type=int, default=6)
    a = ap.parse_args()
    p = Params()
    fin = REPO / "masks" / a.video.stem / "final"
    info = probe(a.graded)
    stratos, mocha, silver = (hex_to_oklab(h) for h in (p.stratos, p.mocha, p.silver))
    acc = {"water": [], "sand": [], "sand_near": [], "rock": [], "skin": [], "dark": [], "light": [], "sky": []}
    per_frame = {}
    for t, rgb in enumerate(iter_frames(a.graded, info)):
        if t % a.step:
            continue
        lab = rgb_to_oklab(rgb)
        L = lab[..., 0]
        D = wmask(fin, "D", t, 0.97)
        row = {}
        for key, m in (("water", wmask(fin, "W", t, 0.9)), ("sand", wmask(fin, "U", t, 0.9)),
                       ("rock", wmask(fin, "R", t, 0.9)), ("sky", wmask(fin, "S", t, 0.9))):
            if m.sum() > 2000:
                med = np.median(lab[m], axis=0)
                acc[key].append(med)
                row[key] = med.tolist()
            if key == "sand":
                near = m.copy()
                near[: int(0.5 * near.shape[0])] = False   # lower half = near seabed
                if near.sum() > 2000:
                    acc["sand_near"].append(np.median(lab[near], axis=0))
        # lit skin: upper body of the diver, mid/high luminance, not gear
        ys, xs = np.nonzero(D)
        if ys.size > 3000:
            y0, y1 = ys.min(), ys.max()
            band = np.zeros_like(D)
            band[y0:int(y0 + 0.6 * (y1 - y0 + 1))] = True
            C = np.hypot(lab[..., 1], lab[..., 2])
            sel = D & band & (L > 0.5) & (C > 0.015)
            if sel.sum() > 1500:
                acc["skin"].append(np.median(lab[sel], axis=0))
        nd = ~D
        under = wmask(fin, "S", t, 0.5)
        pool = nd & ~under
        if pool.sum() > 5000:
            lv = L[pool]
            acc["dark"].append(lab[pool & (L <= np.percentile(lv, 2))].mean(0))
            acc["light"].append(lab[pool & (L >= np.percentile(lv, 99.7))].mean(0))
        per_frame[t] = row
    print(f"{'region':8s} {'measured (median over frames)':34s} target                          dE")
    rep = {}
    refs = {"water": ("reference water", np.array([0.265, -0.024, -0.063])),
            "dark": ("Stratos", stratos), "sand": ("Mocha", mocha), "sand_near": ("Mocha", mocha), "rock": ("Mocha", mocha),
            "light": ("Silver Sand", silver), "skin": ("reference skin", np.array([0.569, 0.002, 0.026]))}
    for k, (tn, tv) in refs.items():
        if not acc[k]:
            continue
        m = np.median(np.array(acc[k]), axis=0)
        # distance in hue/chroma only for L-offset anchors (Stratos/Mocha are mid-tone anchors, not exposure targets)
        dE = de(m, tv)
        dE_ab = float(np.linalg.norm(m[1:] - np.asarray(tv)[1:]) * 100)
        print(f"{k:8s} {describe(m):34s} {tn:16s} {describe(tv):24s} dE={dE:5.1f}  dE(ab only)={dE_ab:5.1f}")
        rep[k] = {"measured": m.tolist(), "target": np.asarray(tv).tolist(), "dE": dE, "dE_ab": dE_ab}
    a.out.write_text(json.dumps({"summary": rep, "per_frame": per_frame}, indent=1))


if __name__ == "__main__":
    main()
