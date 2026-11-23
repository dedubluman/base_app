"""Grade v2 renderer: temporally smoothed anchors -> graded 16-bit frame sequence.

    python -m deepgrade.render2 --stage anchors                      # measure + smooth + cache anchors
    python -m deepgrade.render2 --stage grade --frames 0,24,48,72    # preview a few frames
    python -m deepgrade.render2 --stage grade                        # all frames -> output/v2/graded_frames
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe

from .color import rgb_to_oklab
from .grade2 import anchors, grade_frame2
from .params2 import Params2
from .render import load_weights, smooth_series

REPO = Path(__file__).resolve().parent.parent


def gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def compute_anchors(video: Path, fin: Path, p: Params2, log=print) -> list[dict]:
    info = probe(video)
    rows, t0 = [], time.time()
    for t, rgb in enumerate(iter_frames(video, info)):
        w = load_weights(fin, t)
        rows.append(anchors(rgb, rgb_to_oklab(rgb), w, p))
        if t % 40 == 0:
            log(f"[anchors] {t}/{info.n_frames} {time.time()-t0:.0f}s")
    return rows


def smooth_anchors(rows: list[dict], gates: dict, p: Params2) -> list[dict]:
    n = len(rows)
    num = lambda k: [None if (r.get(k) is None or np.isnan(r.get(k))) else r[k] for r in rows]
    bg = smooth_series(num("bg_med"), p.anchor_sigma_frames)
    sk = smooth_series(num("skin_med"), p.skin_sigma_frames)
    sm = smooth_series(num("sky_med"), 6.0)
    sh = smooth_series(num("skin_hue_src"), p.skin_sigma_frames)
    sdh = smooth_series(num("skin_dh"), p.skin_sigma_frames)
    skc = smooth_series(num("skin_kc"), p.skin_sigma_frames, log=True)
    g = [smooth_series([r["skin_gains"][c] if "skin_gains" in r else None for r in rows],
                       p.skin_sigma_frames, log=True) for c in range(3)]
    air_f = gauss(np.array([r["air_frac"] for r in rows]), 1.5)
    t_ = np.clip((air_f - p.air_lo) / (p.air_hi - p.air_lo), 0, 1)
    air = t_ * t_ * (3 - 2 * t_)
    out = []
    for i in range(n):
        d = {"bg_med": bg[i], "skin_med": sk[i], "sky_med": sm[i], "skin_hue_src": sh[i], "skin_dh": sdh[i], "skin_kc": skc[i], "air": float(air[i]),
             "gate": gates[str(i)]}
        if g[0][i] is not None:
            d["skin_gains"] = [g[c][i] for c in range(3)]
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--weights", default="final_v2")
    ap.add_argument("--stage", choices=["anchors", "grade"], default="grade")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--out", type=Path, default=REPO / "output/v2")
    ap.add_argument("--params", type=Path, default=None)
    a = ap.parse_args()
    p = Params2.load(a.params) if a.params else Params2()
    fin = REPO / "masks" / a.video.stem / a.weights
    a.out.mkdir(parents=True, exist_ok=True)
    anc_f = a.out / "anchors.json"
    gates = json.loads((fin / "gate.json").read_text())["gate"]

    if a.stage == "anchors" or not anc_f.exists():
        rows = compute_anchors(a.video, fin, p)
        sm = smooth_anchors(rows, gates, p)
        anc_f.write_text(json.dumps({"raw": rows, "smoothed": sm}, default=float))
        print("anchors ->", anc_f)
        if a.stage == "anchors":
            return
    sm = json.loads(anc_f.read_text())["smoothed"]
    info = probe(a.video)
    want = None if a.frames is None else {int(x) for x in a.frames.split(",")}
    gdir = a.out / ("graded_frames" if want is None else "preview_frames")
    gdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for t, rgb in enumerate(iter_frames(a.video, info)):
        if want is not None and t not in want:
            continue
        w = load_weights(fin, t)
        out = grade_frame2(rgb, w, sm[t], p)
        cv2.imwrite(str(gdir / f"{t:05d}.png"),
                    cv2.cvtColor((out * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR))
        if t % 20 == 0:
            print(f"[grade2] {t}/{info.n_frames} {time.time()-t0:.0f}s", flush=True)
    p.dump(a.out / "params2.json")
    print("done ->", gdir)


if __name__ == "__main__":
    main()
