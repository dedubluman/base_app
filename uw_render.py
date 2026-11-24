"""Render the region-aware cinematic underwater grade.

Decodes the source at native resolution/fps (10-bit-safe float pipeline from
shades/io.py), grades every frame with uw_look (SAM 3.1 masks interpolated by
uw_interp.py), and encodes H.264 / yuv420p / BT.709 with the original audio
track copied untouched.

    python uw_render.py --video clip.mov --masks masks_uw/clip --look look.json \
        --out out/clip_graded.mp4
    # quick preview of a few frames as PNG instead of a full render:
    python uw_render.py ... --stills 0,60,141,200,255,300 --still-dir out/stills
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Encode
from shades.io import Encoder, iter_frames, probe
from uw_look import Look, finish, mono, protect, refine_mask, region_weights, scene_weights


class Grader:
    def __init__(self, masks_dir: Path, look: Look):
        self.look = look
        self.full = {c: np.load(masks_dir / f"{c}_full.npy", mmap_mode="r")
                     for c in ("subject", "water", "sky") if (masks_dir / f"{c}_full.npy").exists()}

    def masks(self, t: int, rgb: np.ndarray) -> dict:
        out = {}
        for c in ("subject", "water", "sky"):
            if c in self.full:
                out[c] = refine_mask(np.asarray(self.full[c][t]), rgb)
        return out

    def grade(self, t: int, rgb: np.ndarray, return_parts: bool = False):
        sw = scene_weights(t)
        m = self.masks(t, rgb)
        f = mono(rgb)
        out = np.zeros_like(rgb)
        parts = {}
        for scene, w in sw.items():
            if w < 1e-3:
                continue
            reg = region_weights(scene, rgb, m.get("subject"), m.get("water"), m.get("sky"))
            sc = self.look.regional(f, scene, reg, rgb)
            if scene == "under":
                sc = protect(sc, reg["protect"], self.look.post.get("protect_desat", 0.30))
            out += w * sc
            parts[scene] = reg
        out = finish(out, t, self.look.post)
        return (out, {"scene_w": sw, "regions": parts, "masks": m}) if return_parts else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--look", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--stills", help="comma-separated frame indices to dump as PNG instead of rendering")
    ap.add_argument("--still-dir", type=Path)
    a = ap.parse_args()

    info = probe(a.video)
    g = Grader(a.masks, Look.load(a.look))
    print(f"[render] {info.width}x{info.height} {float(info.fps):.3f} fps, {info.n_frames} frames, "
          f"audio={info.has_audio}", flush=True)

    if a.stills:
        want = {int(x) for x in a.stills.split(",")}
        a.still_dir.mkdir(parents=True, exist_ok=True)
        for t, rgb in enumerate(iter_frames(a.video, info)):
            if t in want:
                out = g.grade(t, rgb)
                for tag, im in (("orig", rgb), ("graded", out)):
                    cv2.imwrite(str(a.still_dir / f"{t:04d}_{tag}.png"),
                                cv2.cvtColor((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8),
                                             cv2.COLOR_RGB2BGR))
                print(f"[render] still {t}", flush=True)
            if t >= max(want):
                break
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.video, a.out, info, Encode(crf=a.crf, preset=a.preset, profile="high", dither=True))
    t0 = time.time()
    n = 0
    for t, rgb in enumerate(iter_frames(a.video, info)):
        enc.write(g.grade(t, rgb))
        n += 1
        if t % 20 == 0:
            el = time.time() - t0
            print(f"[render] frame {t}/{info.n_frames}  {el / (t + 1):.2f}s/frame  "
                  f"eta {(info.n_frames - t - 1) * el / (t + 1) / 60:.1f} min", flush=True)
    enc.close()
    print(f"[render] wrote {n} frames to {a.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
