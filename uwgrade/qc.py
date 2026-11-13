#!/usr/bin/env python3
"""Quality control: side-by-side sheets, mask overlays, palette and stability numbers.

    python -m uwgrade.qc sheet   --video inputs/player_export.mov --graded <master.mov> --frames 5,225,445 --out sheet.jpg
    python -m uwgrade.qc masks   --video inputs/player_export.mov --frames 5,225,445 --out masks.jpg
    python -m uwgrade.qc metrics --video inputs/player_export.mov --graded <master.mov> --out qc_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from uwgrade import video  # noqa: E402
from uwgrade.build_masks import paths, read_a  # noqa: E402
from uwgrade.color import rgb2lab  # noqa: E402

REFS = [REPO / "references" / "ref1_rocks_above.jpg", REPO / "references" / "ref2_reflection.jpg"]


def _label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _fit(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def read_frame(path: Path, n: int) -> np.ndarray:
    _, f = next(video.decode(path, 8, start=n, count=1))
    return f


def sheet(src: Path, graded: Path, frames: list[int], out: Path, w: int = 460):
    rows = []
    for n in frames:
        a = _label(_fit(read_frame(src, n), w), f"source f{n}")
        b = _label(_fit(read_frame(graded, n), w), f"graded f{n}")
        rows.append(np.hstack([a, b]))
    left = np.vstack(rows)
    refs = []
    for i, r in enumerate(REFS):
        im = cv2.cvtColor(cv2.imread(str(r)), cv2.COLOR_BGR2RGB)
        refs.append(_label(_fit(im, w), f"REFERENCE {i + 1}"))
    H = left.shape[0]
    col = np.vstack(refs)
    if col.shape[0] < H:
        col = np.vstack([col, np.zeros((H - col.shape[0], col.shape[1], 3), np.uint8)])
    else:
        col = col[:H]
    img = np.hstack([left, col])
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def stills_sheet(stills: Path, frames: list[int], out: Path, w: int = 430, cols: int = 1):
    """source | graded pairs from render --stills PNGs, plus the two references."""
    rows = []
    for n in frames:
        a = cv2.cvtColor(cv2.imread(str(stills / f"source_{n:05d}.png")), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(cv2.imread(str(stills / f"graded_{n:05d}.png")), cv2.COLOR_BGR2RGB)
        rows.append(np.hstack([_label(_fit(a, w), f"source f{n}"), _label(_fit(b, w), f"graded f{n}")]))
    left = np.vstack(rows)
    refs = [_label(_fit(cv2.cvtColor(cv2.imread(str(r)), cv2.COLOR_BGR2RGB), w), f"REFERENCE {i + 1}") for i, r in enumerate(REFS)]
    col = np.vstack(refs)
    H = left.shape[0]
    col = np.vstack([col, np.zeros((max(H - col.shape[0], 0), col.shape[1], 3), np.uint8)])[:H]
    cv2.imwrite(str(out), cv2.cvtColor(np.hstack([left, col]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def masks_sheet(src: Path, frames: list[int], out: Path, w: int = 460):
    base, _, _ = paths(src.stem)
    tiles = []
    for n in frames:
        f = read_frame(src, n).astype(np.float32) / 255
        D = read_a(base / "diver", n)
        R = cv2.resize(read_a(base / "rock", n), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_CUBIC)
        sk, ge, sh = (read_a(base / k, n) for k in ("skin", "gear", "shorts"))
        env = f.copy()
        tint = np.stack([R, np.zeros_like(R), 1 - R], -1) * (1 - D)[..., None]       # red = rock, blue = water
        env = 0.55 * f + 0.45 * tint * 1.0
        parts = f.copy()
        parts = np.where(D[..., None] > 0.02, 0.45 * f + 0.55 * np.stack([sk, sh, ge], -1) * 1.0, 0.25 * f)  # R skin, G shorts, B gear
        edge = cv2.morphologyEx((D > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        diver = f.copy()
        diver[edge] = (1, 1, 0)
        row = [(_label(_fit((x * 255).clip(0, 255).astype(np.uint8), w), t))
               for x, t in ((diver, f"f{n} diver edge"), (env, "rock(red) / water(blue)"), (parts, "skin R shorts G gear B"))]
        tiles.append(np.hstack(row))
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def de76(a, b):
    return float(np.linalg.norm(a - b))


def metrics(src: Path, graded: Path, out: Path):
    base, _, _ = paths(src.stem)
    T = video.probe(src)["frames"]
    stats = []
    prev_alpha = {}
    flick = {"diver": [], "rock": []}
    prevL = None
    for i, g in video.decode(graded, 8):
        D = read_a(base / "diver", i)
        R = cv2.resize(read_a(base / "rock", i), (g.shape[1], g.shape[0]))
        lab = rgb2lab(g.astype(np.float32) / 255)[::2, ::2]
        d, r = D[::2, ::2], R[::2, ::2]
        row = {"t": i}
        for name, m in (("water", (d < 0.05) & (r < 0.15)), ("rock", (d < 0.05) & (r > 0.85)), ("diver", d > 0.9)):
            if m.sum() > 400:
                px = lab[m]
                row[name] = [float(x) for x in px.mean(0)] + [float(np.percentile(px[:, 0], 50))]
        stats.append(row)
        for k, A in (("diver", D), ("rock", R)):
            if k in prev_alpha:
                flick[k].append(float(np.abs(A - prev_alpha[k]).mean()))
            prev_alpha[k] = A
    res = {"frames": len(stats)}
    for name in ("water", "rock", "diver"):
        series = np.array([s[name][:3] for s in stats if name in s])
        if len(series) > 2:
            dd = np.linalg.norm(np.diff(series, axis=0), axis=1)
            res[name] = {"mean_Lab": series.mean(0).round(1).tolist(), "frame_to_frame_dE_mean": float(dd.mean()),
                         "frame_to_frame_dE_p99": float(np.percentile(dd, 99)), "frame_to_frame_dE_max": float(dd.max()),
                         "worst_frame": int(np.argmax(dd) + 1)}
    res["alpha_flicker_mean_abs_diff"] = {k: float(np.mean(v)) for k, v in flick.items() if v}
    res["alpha_flicker_max_abs_diff"] = {k: float(np.max(v)) for k, v in flick.items() if v}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def palette(src: Path, graded: Path, out: Path, step: int = 15):
    """Region-wise Lab statistics of the rendered master (every `step`-th frame) next to the reference targets."""
    base, _, _ = paths(src.stem)
    acc = {k: [] for k in ("water", "rock_lit", "rock_all", "skin", "shorts", "gear")}
    for i, g in video.decode(graded, 8):
        if i % step:
            continue
        lab = rgb2lab(g.astype(np.float32) / 255)[::2, ::2]
        D = read_a(base / "diver", i)[::2, ::2]
        R = cv2.resize(read_a(base / "rock", i), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_CUBIC)
        sk, ge, sh = (read_a(base / k, i)[::2, ::2] for k in ("skin", "gear", "shorts"))
        tot = np.maximum(sk + ge + sh, 1e-3)
        env = D < 0.05
        sets = {"water": env & (R < 0.15), "rock_all": env & (R > 0.85), "rock_lit": env & (R > 0.85) & (lab[..., 0] > 32),
                "skin": (D > 0.9) & (sk / tot > 0.8), "shorts": (D > 0.9) & (sh / tot > 0.8), "gear": (D > 0.9) & (ge / tot > 0.8)}
        for k, m in sets.items():
            if m.sum() > 300:
                acc[k].append(lab[m])
    ref = {"water": "REF water L p5/p50/p95 3.1/18.6/38.0, ab (-4.4..-5.2, -15.7..-17.7), hue 253-259",
           "rock_lit": "REF warm rock (L 28-50) hue 47-81, chroma 8-16",
           "skin": "REF1 (63.7, 5.2, 11.6)  REF2 (58.9, 9.8, 9.4)",
           "shorts": "REF1 (79.4, -6.0, -12.3) lit; REF2 (59.6, -3.5, -12.3) shade"}
    rows, res = [], {}
    for k, v in acc.items():
        px = np.concatenate(v)
        a, b = float(px[:, 1].mean()), float(px[:, 2].mean())
        hue = float(np.degrees(np.arctan2(b, a)) % 360)
        res[k] = {"n_px": int(len(px)), "L_p5_p50_p95": [round(float(x), 1) for x in np.percentile(px[:, 0], [5, 50, 95])],
                  "mean_ab": [round(a, 1), round(b, 1)], "hue": round(hue), "chroma": round(float(np.hypot(a, b)), 1), "reference": ref.get(k, "")}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheet", "masks", "metrics", "stills", "palette"])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--graded", type=Path)
    ap.add_argument("--frames", default="5,225,445")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fr = [int(x) for x in a.frames.split(",")]
    if a.cmd == "sheet":
        sheet(a.video, a.graded, fr, a.out)
    elif a.cmd == "palette":
        palette(a.video, a.graded, a.out)
    elif a.cmd == "stills":
        stills_sheet(a.graded, fr, a.out)
    elif a.cmd == "masks":
        masks_sheet(a.video, fr, a.out)
    else:
        metrics(a.video, a.graded, a.out)


if __name__ == "__main__":
    main()
