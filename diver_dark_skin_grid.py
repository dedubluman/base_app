"""Wide-range skin-tone grid (12 options) through the real diver_dark pipeline.

Each option is a three-point skin ramp (shadow / mid / highlight anchors)
plus how much water coolness it keeps. The ramp is luma-normalised, so the
lightness of a skin type is set via fg_stops relative to the medium tone
(T4), whose brightness matches the current choice: lighter skin types render
lighter, darker ones darker.

    python diver_dark_skin_grid.py                    # grids in output/skin12/
    python diver_dark_skin_grid.py --resolve T6       # -> output/skin12/choice.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import diver_dark as D
from diver_dark_variants import render_frame
from shades.io import probe, read_frames_at

REPO = Path(__file__).resolve().parent
# name, shadow, mid, high, fg_sat, fg_cool
TONES = [
    ("Porselen",      [0.72, 0.56, 0.52], [0.92, 0.80, 0.76], [0.97, 0.90, 0.87], 0.70, 0.05),
    ("Acik pembe",    [0.74, 0.52, 0.48], [0.90, 0.74, 0.68], [0.96, 0.86, 0.80], 0.75, 0.05),
    ("Seftali",       [0.74, 0.55, 0.44], [0.90, 0.76, 0.62], [0.96, 0.88, 0.76], 0.75, 0.05),
    ("Orta bej",      [0.68, 0.48, 0.38], [0.84, 0.66, 0.54], [0.92, 0.80, 0.70], 0.70, 0.08),
    ("Zeytin",        [0.56, 0.46, 0.32], [0.74, 0.64, 0.48], [0.86, 0.78, 0.64], 0.70, 0.08),
    ("Bugday",        [0.66, 0.46, 0.32], [0.82, 0.64, 0.46], [0.90, 0.78, 0.62], 0.75, 0.05),
    ("Bronz",         [0.60, 0.36, 0.24], [0.76, 0.52, 0.36], [0.88, 0.68, 0.52], 0.80, 0.03),
    ("Karamel",       [0.50, 0.30, 0.20], [0.66, 0.44, 0.30], [0.80, 0.62, 0.48], 0.80, 0.03),
    ("Koyu esmer",    [0.36, 0.22, 0.16], [0.50, 0.34, 0.25], [0.66, 0.50, 0.40], 0.80, 0.03),
    ("Notr gri",      [0.55, 0.50, 0.48], [0.72, 0.68, 0.65], [0.86, 0.83, 0.80], 0.60, 0.05),
    ("Su alti dogal", [0.66, 0.48, 0.42], [0.80, 0.66, 0.58], [0.90, 0.80, 0.74], 0.55, 0.22),
    ("Su alti soguk", [0.60, 0.50, 0.48], [0.74, 0.66, 0.62], [0.86, 0.80, 0.78], 0.50, 0.38),
]
REF = 3  # "Orta bej" keeps the current brightness


def options(base_stops: float) -> dict[str, dict]:
    ref_y = np.array(TONES[REF][2]) @ D.LUMA
    out = {}
    for k, (name, sh, md, hi, sat, cool) in enumerate(TONES, 1):
        rel = (np.array(md) @ D.LUMA) / ref_y
        stops = float(np.clip(base_stops + 2.2 * math.log2(rel) * 0.8, -0.8, 1.4))
        out[f"T{k}"] = {"name": name, "params": {
            "skin_shadow": sh, "skin_target": md, "skin_high": hi,
            "fg_sat": sat, "fg_cool": cool, "fg_stops": round(stops, 3)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "inputs/diver.mov")
    ap.add_argument("--frames", default="270,60")
    ap.add_argument("--out", type=Path, default=REPO / "output/skin12")
    ap.add_argument("--resolve", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cur = json.loads((REPO / "output/variants/choice.json").read_text())
    opts = options(cur.get("fg_stops", D.CFG["fg_stops"]))
    if a.resolve:
        p = {**cur, **opts[a.resolve.strip().upper()]["params"]}
        (a.out / "choice.json").write_text(json.dumps(p))
        print(json.dumps(p))
        return
    (a.out / "options.json").write_text(json.dumps(opts, indent=1))

    info = probe(a.input)
    h, w = info.height, info.width
    ids = [int(f) for f in a.frames.split(",")]
    frames = read_frames_at(a.input, info, ids)
    base = {**D.CFG, **cur}
    vig = D.vignette(h, w, base["vig_strength"], base["vig_inner"], base["vig_outer"])
    cw, ch, pad, top = 400, 350, 12, 36
    cols, rows = 4, 3
    for i in ids:
        sheet = np.full((rows * (ch + top + pad) + pad, pad + cols * (cw + pad), 3), 18, np.uint8)
        for k, (cid, o) in enumerate(opts.items()):
            cfg = {**base, **o["params"]}
            out, m = render_frame(frames[i], i, h, w, vig, cfg)
            ys, xs = np.where(m > 127)
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            bw = max(xs.max() - xs.min(), (ys.max() - ys.min()) * cw / ch) * 1.2
            bh = bw * ch / cw
            x1 = int(np.clip(cx - bw / 2, 0, w - bw)); y1 = int(np.clip(cy - bh / 2, 0, h - bh))
            crop = out[y1:y1 + int(min(bh, h)), x1:x1 + int(min(bw, w))]
            img = cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
            r, c = divmod(k, cols)
            x0 = pad + c * (cw + pad); y0 = top + r * (ch + top + pad)
            sheet[y0:y0 + ch, x0:x0 + cw] = img
            cv2.putText(sheet, f"{cid}  {o['name']}", (x0 + 4, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
            print(cid, i, flush=True)
        cv2.imwrite(str(a.out / f"grid_f{i:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
