"""Skin-tone variant grid for diver_dark.py.

Renders a 3x3 grid of real pipeline outputs (same masks, same background
grade, same skin code path as the final render) on chosen frames, so a pick
made by eye is exactly what the final video will look like.

    rows    A soft-cool / B natural / C warm      (temperature)
    columns 1 soft / 2 medium / 3 strong          (saturation + brightness)

Writes output/variants/grid_f<N>.jpg and output/variants/variants.json
(cell id -> CFG overrides). Render a pick with:

    python diver_dark.py --input SRC --params output/variants/B2.json
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

ROWS = {
    "A": dict(label="Soguk", fg_cool=0.40,
              skin_shadow=[0.58, 0.46, 0.44], skin_target=[0.70, 0.61, 0.57],
              skin_high=[0.82, 0.77, 0.74]),
    "B": dict(label="Dogal", fg_cool=0.12,
              skin_shadow=[0.68, 0.45, 0.38], skin_target=[0.78, 0.62, 0.53],
              skin_high=[0.88, 0.76, 0.67]),
    "C": dict(label="Sicak", fg_cool=0.00,
              skin_shadow=[0.76, 0.40, 0.28], skin_target=[0.86, 0.58, 0.42],
              skin_high=[0.95, 0.75, 0.58]),
}
COLS = {
    "1": dict(label="sonuk", fg_sat=0.50, fg_stops=0.20, patch_keep=0.80),
    "2": dict(label="orta", fg_sat=0.78, fg_stops=0.80, patch_keep=0.65),
    "3": dict(label="parlak", fg_sat=1.00, fg_stops=1.60, patch_keep=0.50, fg_lift=0.70),
}


def variants() -> dict[str, dict]:
    out = {}
    for r, rv in ROWS.items():
        for c, cv in COLS.items():
            p = {k: v for k, v in {**rv, **cv}.items() if k != "label"}
            out[f"{r}{c}"] = {"name": f"{rv['label']} / {cv['label']}", "params": p}
    return out


def render_frame(x, i, h, w, vig, cfg):
    m = cv2.imread(str(REPO / f"masks/diver__person_s2/subject/{i // 2:05d}.png"), 0)
    hd = REPO / f"masks/diver__head_s4/subject/{i // 4:05d}.png"
    if hd.exists() and i % 4 == 0:
        m = np.maximum(m, cv2.imread(str(hd), 0))
    core = cv2.erode((m > 127).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    tgt = D.to_lin(np.array(cfg["skin_target"], np.float32))
    lin = D.to_lin(x[core])
    yy = lin @ D.LUMA
    g = D.skin_gains(lin[yy > np.percentile(yy, 35)].mean(0)[None], tgt, cfg["max_gain"], 1.0)[0]
    a = D.refine(m, x @ D.LUMA, cfg["feather"])
    dv = D.decontaminate(D.grade_diver(x, g, vig, cfg, alpha=a), a)
    return D.grade_bg(x, vig, cfg) * (1 - a[..., None]) + dv * a[..., None], m


def resolve(choice: dict, vs: dict) -> dict:
    """Gemini's answer -> CFG overrides. Accepts {"pick": "B2"} or
    {"blend": {"B2": 0.6, "C2": 0.4}} (weights are normalised); numeric
    params and RGB anchors are interpolated linearly."""
    if "pick" in choice:
        return dict(vs[choice["pick"]]["params"])
    wts = {k: float(v) for k, v in choice["blend"].items()}
    tot = sum(wts.values())
    out: dict = {}
    for cid, wgt in wts.items():
        for k, v in vs[cid]["params"].items():
            v = np.asarray(v, float) * (wgt / tot)
            out[k] = out[k] + v if k in out else v
    return {k: (np.round(v, 3).tolist() if np.ndim(v) else round(float(v), 3)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "inputs/diver.mov")
    ap.add_argument("--frames", default="270,60")
    ap.add_argument("--out", type=Path, default=REPO / "output/variants")
    ap.add_argument("--resolve", default=None,
                    help='Gemini JSON, e.g. \'{"pick":"B2"}\' -> writes <out>/choice.json and exits')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.resolve:
        params = resolve(json.loads(a.resolve), variants())
        (a.out / "choice.json").write_text(json.dumps(params))
        print(json.dumps(params))
        return

    info = probe(a.input)
    ids = [int(f) for f in a.frames.split(",")]
    frames = read_frames_at(a.input, info, ids)
    h, w = info.height, info.width
    base = dict(D.CFG)
    vig = D.vignette(h, w, base["vig_strength"], base["vig_inner"], base["vig_outer"])
    vs = variants()
    (a.out / "variants.json").write_text(json.dumps(vs, indent=1))
    for cid, v in vs.items():
        (a.out / f"{cid}.json").write_text(json.dumps(v["params"]))

    cw, ch, pad, top = 520, 455, 14, 40
    for i in ids:
        sheet = np.full((top + 3 * (ch + pad + top) , pad + 3 * (cw + pad), 3), 18, np.uint8)
        for ri, r in enumerate(ROWS):
            for ci, c in enumerate(COLS):
                cid = f"{r}{c}"
                cfg = {**base, **vs[cid]["params"]}
                out, m = render_frame(frames[i], i, h, w, vig, cfg)
                ys, xs = np.where(m > 127)
                # crop around the diver (same aspect as the cell) so skin is large
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                bw = max(xs.max() - xs.min(), (ys.max() - ys.min()) * cw / ch) * 1.25
                bh = bw * ch / cw
                x1 = int(np.clip(cx - bw / 2, 0, w - bw)); y1 = int(np.clip(cy - bh / 2, 0, h - bh))
                out = out[y1:y1 + int(min(bh, h)), x1:x1 + int(min(bw, w))]
                img = cv2.cvtColor((np.clip(out, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
                x0 = pad + ci * (cw + pad)
                y0 = top + ri * (ch + pad + top)
                sheet[y0 + top - 6:y0 + top - 6 + ch, x0:x0 + cw] = img
                cv2.putText(sheet, f"{cid}  {vs[cid]['name']}", (x0 + 4, y0 + top - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
                print(cid, i, "done", flush=True)
        cv2.imwrite(str(a.out / f"grid_f{i:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
