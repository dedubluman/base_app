#!/usr/bin/env python3
"""Diver recolour + vivid aquarium background, all inside SAM 3.1 masks.

Masks (made beforehand on this clip):
    masks/<stem>__fwd/subject/*.png      diver_masks.py --direction fwd  ("person")
    masks/<stem>__<part>/subject/*.png   diver_parts.py  hair / goggles / fins / shorts

Per frame: the diver matte and part mattes come from diver_reels.build_mattes
(colour-checked parts, edge snapped to the body). Then
  background (outside the diver)  -> aquarium_grade.grade_aquarium
  skin  (diver minus every part)  -> reels_grade.grade_diver, skin colour = --skin
  hair / dive mask / fins         -> each recoloured to its own --scheme colour
                                     (luminance shading kept, glints neutral)

    python diver_recolor.py --input inputs/diver.mov --options 100   # choice sheets
    python diver_recolor.py --input inputs/diver.mov --skin B --scheme 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

import aquarium_grade as A
import diver_reels as D
import reels_grade as R
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

REPO = Path(__file__).resolve().parent
# parts inside the diver ("person") matte; the fins are outside it on this
# clip (SAM 3.1 "person" leaves them out), so they get their own matte
D.PARTS = ("hair", "shorts", "goggles")


def fin_mattes(stem, n, shape):
    """Fins: SAM 3.1 "flipper" masks (diver_parts.py --free fins), small
    specks dropped, 1 px grow + tight feather, 3-frame temporal median."""
    d = REPO / "masks" / f"{stem}__fins" / "subject"
    if not d.exists():
        return None
    k3 = np.ones((3, 3), np.uint8)
    out = []
    for i in range(n):
        m = D.drop_specks(D.read_mask(d, i, shape))
        a = cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8), k3).astype(np.float32), (0, 0), 1.0)
        out.append(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
    return D.median3(out)


def wb_and_depth(src, info, al, part_al):
    """Per-frame diver white balance + depth, as in diver_reels (skin core
    only, Gaussian-smoothed so the tone never flickers)."""
    n = info.n_frames
    means = np.full((n, 3), np.nan, np.float32)
    for i, x in enumerate(iter_frames(src, info)):
        if i >= n:
            break
        core = cv2.erode((al[i] > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for v in part_al.values():
            core &= v[i] < 128
        if core.sum() > 300:
            means[i] = np.median(x[core], 0)
    ok = ~np.isnan(means[:, 0])
    gains = np.full((n, 3), np.nan, np.float32)
    for i in np.nonzero(ok)[0]:
        gains[i] = R.diver_gains(means[i])
    depth = np.array([R.diver_depth(m) if o else np.nan for m, o in zip(means, ok)], np.float32)
    t = np.arange(n)

    def smooth(v, fallback, sigma):
        v = np.interp(t, t[ok], v[ok]) if ok.any() else np.full(n, fallback, np.float32)
        r = int(3 * sigma)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        return np.convolve(np.pad(v, r, mode="edge"), k / k.sum(), "valid").astype(np.float32)

    gains = np.stack([smooth(gains[:, c], R.DV["gains"][c], 10.0) for c in range(3)], 1)
    return gains, smooth(depth, 0.0, 15.0)


def set_skin(rgb):
    rgb = np.asarray(rgb, np.float32)
    R.DV["skin_rgb"] = rgb
    R.DV["skin_hue"] = A.skin_hue_of(rgb)


def grade_frame(x, alpha, parts, gains, depth, veil, scheme, fins=None):
    """x float RGB, alpha HxW 0..1 diver, parts {name: HxW 0..1}."""
    _, hair_hex, mask_hex, fins_hex = A.PART_SCHEMES[scheme]
    colours = {"hair": hair_hex, "goggles": mask_hex}
    bg = A.grade_aquarium(x, veil)
    # no navy gradient in the aquarium look: the diver follows a flat scene
    grad = np.ones((x.shape[0], x.shape[1], 1), np.float32)
    g = 1 + (gains - 1) * (1 - R.DV["depth_wb_keep"] * depth)
    xw = x * g
    yb = R.luma(xw)
    body = alpha > 0.9
    for m in parts.values():
        body &= m < 0.5
    body &= yb > R.DV["dark_lo"]
    skin_luma = float(np.percentile(yb[body], 70)) if body.sum() > 300 else None
    dv = R.grade_diver(x, grad, gains, depth, parts, skin_luma)
    for name, hx in colours.items():
        if name in parts:
            a = parts[name][..., None]
            dv = dv * (1 - a) + A.colorize(xw, a, hx) * a
    # a touch of the aquarium's clarity/contrast on the diver, so it sits in it
    s = dv * dv * (3 - 2 * dv)
    dv = np.clip(dv + (s - dv) * 0.25, 0, 1)
    a = alpha[..., None]
    comp = bg * (1 - a) + dv * a
    if alpha.max() > 0:   # light wrap, luminance only (no water colour on skin)
        ab = cv2.GaussianBlur(alpha, (0, 0), 3.0)
        ring = np.clip((alpha - ab) * 2.5, 0, 1)
        lb = cv2.GaussianBlur(R.luma(bg), (0, 0), 6.0)
        comp = comp + ((lb - R.luma(comp)) * 0.25 * ring)[..., None]
    if fins is not None and fins.max() > 0:
        af = fins[..., None]
        comp = comp * (1 - af) + A.colorize(xw, af, fins_hex) * af
    return np.clip(comp, 0, 1)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=REPO / "output" / "diver_aquarium.mp4")
    ap.add_argument("--skin", default="B", choices=sorted(A.SKIN_OPTIONS))
    ap.add_argument("--scheme", default="1", choices=sorted(A.PART_SCHEMES))
    ap.add_argument("--options", type=int, nargs="*", help="frame(s): write choice sheets only")
    a = ap.parse_args()
    info = probe(a.input)
    # count the frames actually decoded (this clip: 190 packets, 172 frames);
    # the SAM 3.1 masks are indexed by decoded frame
    n_dec = sum(1 for _ in iter_frames(a.input, info, scale_w=64))
    info = dataclasses.replace(info, n_frames=n_dec)
    w, h, n = info.width, info.height, info.n_frames
    stem = a.input.stem
    qc = REPO / "output" / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    raw, report = D.merge_masks(stem, n, (h, w))
    al, part_al, _ = D.build_mattes(a.input, info, stem, raw, log=print)
    del raw
    fins = fin_mattes(stem, n, (h, w))
    if fins is not None:
        part_al["fins"] = fins
    print(f"[mask] parts with SAM 3.1 masks: {sorted(part_al)}", flush=True)
    cover = {k: int(sum(f.any() for f in v)) for k, v in part_al.items()}
    print(f"[mask] frames covered per part (of {n}): {cover}", flush=True)
    gains, depth = wb_and_depth(a.input, info, al, part_al)
    sample = [x for i, x in enumerate(iter_frames(a.input, info)) if i % 10 == 0]
    veil = A.clip_veil(sample)
    del sample

    def parts_at(i):
        return {k: v[i].astype(np.float32) / 255 for k, v in part_al.items()
                if v[i].any() and k != "fins"}

    def fins_at(i):
        return None if fins is None else fins[i].astype(np.float32) / 255

    if a.options:
        for fi in a.options:
            x = next(xx for i, xx in enumerate(iter_frames(a.input, info)) if i == fi)
            m = al[fi] > 127
            mf = m | (fins[fi] > 127 if fins is not None else False)
            ys, xs = np.nonzero(mf)
            pad = 60
            y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)

            def tile(sk, sc, text):
                set_skin(A.SKIN_OPTIONS[sk][1])
                y = grade_frame(x, al[fi] / 255.0, parts_at(fi), gains[fi], float(depth[fi]), veil, sc, fins_at(fi))
                t = np.clip(y[y0:y1, x0:x1] * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1].copy()
                t = cv2.resize(t, (520, int(520 * t.shape[0] / t.shape[1])))
                return label(t, text)

            src = np.clip(x[y0:y1, x0:x1] * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1].copy()
            src = label(cv2.resize(src, (520, int(520 * src.shape[0] / src.shape[1]))), "Kaynak")
            skins = [src] + [tile(k, "1", f"{k} - {v[0]}") for k, v in A.SKIN_OPTIONS.items()]
            schemes = [src] + [tile("B", k, f"{k} - {v[0]}") for k, v in A.PART_SCHEMES.items()]
            cv2.imwrite(str(qc / f"{stem}_ten_secenekleri_{fi}.jpg"), np.hstack(skins),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(qc / f"{stem}_parca_renkleri_{fi}.jpg"), np.hstack(schemes),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            # mask overlay of the parts on this frame
            ov = np.clip(x * 255, 0, 255).astype(np.uint8)[..., ::-1].copy()
            col = {"hair": (0, 200, 255), "goggles": (255, 0, 255), "fins": (0, 255, 0),
                   "shorts": (255, 255, 0), "bubbles": (255, 255, 255)}
            ov[m] = (0.6 * ov[m] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
            pa = parts_at(fi)
            if fins is not None:
                pa["fins"] = fins_at(fi)
            for k, v in pa.items():
                pm = v > 0.5
                ov[pm] = (0.3 * ov[pm] + 0.7 * np.array(col.get(k, (128, 128, 128)))).astype(np.uint8)
            cv2.imwrite(str(qc / f"{stem}_parts_overlay_{fi}.jpg"), ov[y0:y1, x0:x1])
        print(f"[options] sheets in {qc}", flush=True)
        return 0

    set_skin(A.SKIN_OPTIONS[a.skin][1])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.input, a.output, info, Encode(crf=16, preset="slow", profile="high", dither=True))
    sheet_idx = sorted(set([0, n - 1] + list(range(0, n, int(round(1.5 * float(info.fps)))))))
    tiles, lum = [], []
    for i, x in enumerate(iter_frames(a.input, info)):
        if i >= n:
            break
        y = grade_frame(x, al[i] / 255.0, parts_at(i), gains[i], float(depth[i]), veil, a.scheme, fins_at(i))
        enc.write(y)
        lum.append(float(R.luma(y[::4, ::4]).mean()))
        if i in sheet_idx:
            b = np.clip(np.hstack([x, y]) * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1]
            tiles.append(cv2.resize(b, (w, h // 2)))
        if i % 30 == 0:
            print(f"[render] frame {i}/{n}", flush=True)
    enc.close()
    cv2.imwrite(str(qc / f"{stem}_aquarium_before_after.jpg"),
                np.vstack(tiles[:6]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    jumps = np.abs(np.diff(lum))
    o = probe(a.output)
    meta = {"skin": a.skin, "skin_rgb": list(A.SKIN_OPTIONS[a.skin][1]),
            "scheme": a.scheme, "scheme_colours": A.PART_SCHEMES[a.scheme],
            "parts_frames": cover, "veil": veil.round(4).tolist(),
            "max_luma_jump": round(float(jumps.max()), 4),
            "out": {"w": o.width, "h": o.height, "fps": str(o.fps), "frames": o.n_frames,
                    "audio": o.has_audio, "duration": o.duration}}
    (qc / f"{stem}_aquarium_report.json").write_text(json.dumps(meta, indent=1))
    print(f"[out] {a.output}: {o.width}x{o.height} {o.fps} fps, {o.n_frames} frames "
          f"(source {n}), audio={o.has_audio}; max luma jump {jumps.max():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
