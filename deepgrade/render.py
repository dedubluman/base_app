"""Grade every frame with the final SAM 3.1 region weights and encode.

    python -m deepgrade.render --video inputs/player_export.mov --out output/player_export_graded.mp4

Pass 1 measures per-frame anchors (water / sand / sky brightness, skin
white-balance gains) and smooths them over time (no exposure flicker).
Pass 2 grades and encodes: H.264 High yuv420p BT.709 (source audio copied) and,
unless --no-master, a 10-bit ProRes 422 HQ intermediate master.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

from .color import rgb_to_oklab
from .grade import frame_stats, grade_frame
from .params import Params

REPO = Path(__file__).resolve().parent.parent


def load_weights(fin: Path, t: int) -> dict:
    w = {}
    for k in "DSRMWU":
        if not (fin / k).exists():
            w[k] = np.zeros_like(w["D"]) if "D" in w else 0
            continue
        w[k] = cv2.imread(str(fin / k / f"{t:05d}.png"), 0).astype(np.float32) / 255.0
    tot = sum(w.values())
    out = {k: v / np.maximum(tot, 1e-3) for k, v in w.items()}
    pf = fin / "P" / f"{t:05d}.png"                       # swimwear inside the diver (auxiliary mask)
    out["P"] = (cv2.imread(str(pf), 0).astype(np.float32) / 255.0) if pf.exists() else np.zeros_like(out["D"])
    return out


def smooth_series(vals: list, sigma: float, log: bool = False):
    """NaN-aware: fill gaps by edge-extended interpolation, then Gaussian smooth."""
    x = np.array([np.nan if v is None else v for v in vals], np.float64)
    if log:
        x = np.log(x)
    ok = ~np.isnan(x)
    if not ok.any():
        return [None] * len(vals)
    idx = np.arange(len(x))
    x = np.interp(idx, idx[ok], x[ok])
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    y = np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")
    return list(np.exp(y) if log else y)


class MasterEncoder:
    """10-bit ProRes 422 HQ intermediate (BT.709 tagged), video only."""

    def __init__(self, dst: Path, info):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", f"{info.width}x{info.height}", "-r", str(info.fps), "-i", "-",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv422p10le",
               "-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               str(dst)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write((np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("master encode failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--out", type=Path, default=REPO / "output/player_export_graded.mp4")
    ap.add_argument("--master", type=Path, default=REPO / "output/player_export_master_prores.mov")
    ap.add_argument("--no-master", action="store_true")
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--range", nargs=2, type=int, default=None)
    ap.add_argument("--crf", type=int, default=14)
    a = ap.parse_args()
    p = Params.load(a.params) if a.params else Params()
    info = probe(a.video)
    n = info.n_frames
    fin = REPO / "masks" / a.video.stem / "final"
    lo, hi = a.range if a.range else (0, n - 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gates = json.loads((fin / "gate.json").read_text())["gate"]

    # ---- pass 1: anchors
    rows = {}
    for t, rgb in enumerate(iter_frames(a.video, info)):
        if t < lo or t > hi:
            continue
        w = load_weights(fin, t)
        rows[t] = frame_stats(rgb, rgb_to_oklab(rgb), w, p)
        if t % 25 == 0:
            print(f"[render] stats {t}/{n} {time.time()-t0:.0f}s", flush=True)
    ts = sorted(rows)
    sm = {}
    for key in ("water_med", "sand_med", "sky_med", "skin_med"):
        sm[key] = smooth_series([rows[t].get(key) if not (rows[t].get(key) is None or
                                 np.isnan(rows[t].get(key))) else None for t in ts], 4.0)
    gains = []
    for c in range(3):
        gains.append(smooth_series([rows[t]["skin_gains"][c] if "skin_gains" in rows[t] else None
                                    for t in ts], 5.0, log=True))
    smoothed = {}
    for i, t in enumerate(ts):
        d = {k: sm[k][i] for k in sm}
        d["gate"] = gates[str(t)]
        if gains[0][i] is not None:
            d["skin_gains"] = [gains[c][i] for c in range(3)]
        smoothed[t] = d
    a.out.with_suffix(".stats.json").write_text(json.dumps(
        {"raw": {str(t): rows[t] for t in ts}, "smoothed": {str(t): smoothed[t] for t in ts}},
        indent=1, default=float))
    print(f"[render] anchors done {time.time()-t0:.0f}s", flush=True)

    # ---- pass 2: grade + encode
    enc = Encoder(a.video, a.out, info, Encode(crf=a.crf, preset="slow", profile="high", dither=True))
    master = None if a.no_master else MasterEncoder(a.master, info)
    for t, rgb in enumerate(iter_frames(a.video, info)):
        if t < lo or t > hi:
            continue
        w = load_weights(fin, t)
        out = grade_frame(rgb, w, smoothed[t], p)
        enc.write(out)
        if master is not None:
            master.write(out)
        if t % 10 == 0:
            print(f"[render] graded {t}/{n} {time.time()-t0:.0f}s", flush=True)
    enc.close()
    if master is not None:
        master.close()
    p.dump(a.out.with_suffix(".params.json"))
    print(f"[render] done in {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
