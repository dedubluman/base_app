"""Colour-option grids for the diver's accessories (hair, shorts, watch,
diving mask + snorkel), rendered through the real diver_dark pipeline.

Each part gets a 3x3 grid of labelled colours (H1..H9, S1..S9, W1..W9,
M1..M9) on a frame where that part is clearly visible, cropped around it.
The skin stays at the current choice (output/variants/choice.json).

    python diver_dark_parts_grid.py
    python diver_dark_parts_grid.py --resolve "H2 S3 W5 M1"   # -> output/parts/choice.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import diver_dark as D
from shades.io import probe, read_frames_at

REPO = Path(__file__).resolve().parent

# display-referred RGB; realistic first, a few bold ones last
PALETTES = {
    "hair": ("H", "Sac", [
        ("Siyah", [0.06, 0.06, 0.06]), ("Koyu kahve", [0.17, 0.11, 0.08]),
        ("Kahve", [0.30, 0.20, 0.13]), ("Kullu kahve", [0.30, 0.27, 0.24]),
        ("Bal sarisi", [0.55, 0.42, 0.24]), ("Platin", [0.80, 0.75, 0.62]),
        ("Gumus gri", [0.55, 0.56, 0.58]), ("Gece mavisi", [0.06, 0.09, 0.18]),
        ("Petrol", [0.08, 0.30, 0.33])]),
    "shorts": ("S", "Sort", [
        ("Siyah", [0.05, 0.05, 0.05]), ("Antrasit", [0.18, 0.18, 0.19]),
        ("Lacivert", [0.06, 0.10, 0.26]), ("Haki", [0.36, 0.34, 0.22]),
        ("Beyaz", [0.86, 0.86, 0.84]), ("Petrol", [0.05, 0.26, 0.29]),
        ("Orman yesili", [0.08, 0.22, 0.12]), ("Hardal", [0.58, 0.44, 0.10]),
        ("Turuncu", [0.88, 0.40, 0.08])]),
    "watch": ("W", "Saat", [
        ("Siyah", [0.05, 0.05, 0.05]), ("Antrasit", [0.20, 0.20, 0.21]),
        ("Gumus", [0.66, 0.67, 0.68]), ("Altin", [0.72, 0.56, 0.26]),
        ("Lacivert", [0.06, 0.10, 0.28]), ("Beyaz", [0.88, 0.88, 0.86]),
        ("Haki", [0.36, 0.34, 0.22]), ("Turuncu", [0.90, 0.42, 0.08]),
        ("Neon sari", [0.86, 0.90, 0.10])]),
    "mask": ("M", "Maske+snorkel", [
        ("Siyah", [0.05, 0.05, 0.05]), ("Koyu gri", [0.20, 0.20, 0.21]),
        ("Duman gri", [0.42, 0.43, 0.45]), ("Lacivert", [0.06, 0.10, 0.28]),
        ("Beyaz", [0.88, 0.88, 0.86]), ("Petrol", [0.05, 0.28, 0.31]),
        ("Sari", [0.88, 0.72, 0.10]), ("Turuncu", [0.90, 0.42, 0.08]),
        ("Neon yesil", [0.40, 0.88, 0.20])]),
}
FRAMES = {"hair": 60, "shorts": 400, "watch": 60, "mask": 60}


def codes():
    out = {}
    for part, (pre, _, pal) in PALETTES.items():
        for k, (name, rgb) in enumerate(pal, 1):
            out[f"{pre}{k}"] = (part, name, rgb)
    return out


def load_mask(part, i):
    if part == "person":
        return cv2.imread(str(REPO / f"masks/diver__person_s2/subject/{i // 2:05d}.png"), 0)
    p = REPO / f"masks/diver__{part}_s4/subject/{i // 4:05d}.png"
    return cv2.imread(str(p), 0) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "inputs/diver.mov")
    ap.add_argument("--out", type=Path, default=REPO / "output/parts")
    ap.add_argument("--resolve", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cs = codes()
    if a.resolve:
        sel = {}
        for c in a.resolve.replace(",", " ").split():
            part, name, rgb = cs[c.upper()]
            sel[f"color_{part}"] = rgb
        skin = json.loads((REPO / "output/variants/choice.json").read_text())
        (a.out / "choice.json").write_text(json.dumps({**skin, **sel}))
        print(json.dumps({**skin, **sel}))
        return

    skin = json.loads((REPO / "output/variants/choice.json").read_text())
    info = probe(a.input)
    h, w = info.height, info.width
    frames = read_frames_at(a.input, info, sorted(set(FRAMES.values())))
    base = {**D.CFG, **skin}
    vig = D.vignette(h, w, base["vig_strength"], base["vig_inner"], base["vig_outer"])
    cw, ch, pad, top = 420, 420, 12, 36
    for part, (pre, title, pal) in PALETTES.items():
        i = FRAMES[part]
        # use the nearest stride-4 keyframe so the part mask is a real SAM 3.1 key
        i = (i // 4) * 4
        x = read_frames_at(a.input, info, [i])[i]
        pm = load_mask("person", i)
        hd = REPO / f"masks/diver__head_s4/subject/{i // 4:05d}.png"
        if hd.exists():
            pm = np.maximum(pm, cv2.imread(str(hd), 0))
        m = load_mask(part, i)
        if m is None or m.max() == 0:
            print("no mask for", part, "at", i)
            continue
        pm = np.maximum(pm, m)
        y = x @ D.LUMA
        a_p = D.refine(pm, y, base["feather"])
        a_n = np.clip(D.refine(m, y, base["feather"]), 0, 1)
        core = cv2.erode(((pm > 127) & (m < 60)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        tgt = D.to_lin(np.array(base["skin_target"], np.float32))
        lin = D.to_lin(x[core]); yy = lin @ D.LUMA
        g = D.skin_gains(lin[yy > np.percentile(yy, 35)].mean(0)[None], tgt, base["max_gain"], 1.0)[0]
        skin_rgb = D.decontaminate(D.grade_diver(x, g, vig, base, alpha=a_p), a_p)
        bg = D.grade_bg(x, vig, base)
        pmean = float((x[cv2.erode((m > 127).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] @ D.LUMA).mean())
        ys, xs = np.where(m > 127)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = min(max(xs.max() - xs.min(), ys.max() - ys.min()) * 2.2 + 120, h, w)
        x1 = int(np.clip(cx - side / 2, 0, w - side)); y1 = int(np.clip(cy - side / 2, 0, h - side))
        sheet = np.full((top + 3 * (ch + top + pad), pad + 3 * (cw + pad), 3), 18, np.uint8)
        for k, (name, rgb) in enumerate(pal):
            pc = D.decontaminate(D.grade_part(x, rgb, pmean, vig, base), a_n)
            fg = skin_rgb * (1 - a_n[..., None]) + pc * a_n[..., None]
            out = bg * (1 - a_p[..., None]) + fg * a_p[..., None]
            crop = out[y1:y1 + int(side), x1:x1 + int(side)]
            img = cv2.cvtColor((np.clip(crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
            r, c = divmod(k, 3)
            x0 = pad + c * (cw + pad); y0 = top + r * (ch + top + pad)
            sheet[y0:y0 + ch, x0:x0 + cw] = img
            cv2.putText(sheet, f"{pre}{k + 1}  {name}", (x0 + 4, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        # (title omitted: it would overlap the first label)
        cv2.imwrite(str(a.out / f"grid_{part}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", part, "frame", i, flush=True)


if __name__ == "__main__":
    main()
