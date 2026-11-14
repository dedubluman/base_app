#!/usr/bin/env python3
"""Reel-matched underwater grade: dark navy top, blue water, warm natural skin.

Targets come from measuring the reference Reel (prompts/sam31-reel-grade-v2.md):
  top water  V 0.09-0.21 H207-228   mid water V 0.12 H210 S0.45
  bright floor V 0.82 H191 S0.26    shaded floor V 0.56 H194 S0.48
  skin lit (232,188,164) H21 S0.29  skin shade (172,138,122) H19
The user's clip starts as blue-cyan (H183-197, S0.6-0.85, V0.4-0.5), so the
background step is a darken + small hue move + desaturate, matched to the
reference's end state rather than its hue delta.

All functions take float32 RGB 0..1 (BT.709 gamma) and are pure per-pixel
maps plus a fixed screen-space vertical weighting (no per-frame auto
exposure), so the grade is temporally stable by construction.
"""
from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(x):
    return x @ LUMA


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def rgb2hsv(x):
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h * 60.0, s, mx


def hsv2rgb(h, s, v):
    h = (h % 360.0) / 60.0
    c = v * s
    xx = c * (1 - np.abs(h % 2 - 1))
    z = np.zeros_like(h)
    sel = np.floor(h).astype(int) % 6
    r = np.choose(sel, [c, xx, z, z, xx, c])
    g = np.choose(sel, [xx, c, c, xx, z, z])
    b = np.choose(sel, [z, z, xx, c, c, xx])
    m = v - c
    return np.stack([r + m, g + m, b + m], -1)


# ---------------------------------------------------------- background --
BG = dict(
    # Step 1  Darken: tone curve + fixed screen-space vertical weighting
    v_gamma_top=1.45,        # V' = V^gamma darkens shadows/mids, keeps highlights
    v_gamma_bottom=0.92,     # bottom of frame: slight lift, the reference's bright seabed
    g_top=0.40, g_bottom=1.00,
    g_y0=0.28, g_y1=0.95,    # smoothstep window (fraction of frame height)
    floor=0.040,             # luminance floor so surface light never clips to flat black
    # Step 3  Blue/white: hue to the reference's graded 195-210, less saturation
    hue_target=205.0, hue_pull=0.75,
    sat_water=0.64,          # S x this (0.8 -> ~0.5)
    sat_hi=0.55,             # extra desaturation of bright areas toward blue-white
    hi_lo=0.42, hi_hi=0.85,
    # light ripples / surface light pop
    clarity=0.45, clarity_sigma=7.0, clarity_lo=0.20, clarity_hi=0.55,
)


def vertical_ramp(h, p=BG):
    t = smoothstep(p["g_y0"], p["g_y1"], np.linspace(0, 1, h, dtype=np.float32))
    return t[:, None].astype(np.float32)


def vertical_weight(h, w, p=BG):
    t = vertical_ramp(h, p)
    return p["g_top"] + (p["g_bottom"] - p["g_top"]) * t


def grade_background(x, p=BG):
    H, W = x.shape[:2]
    hue, s, v = rgb2hsv(np.clip(x, 0, 1))
    k_hi = smoothstep(p["hi_lo"], p["hi_hi"], v)

    # Step 3: water hues move toward the reference's blue-cyan (only hues that
    # are already water, so a warm object is not dragged blue)
    water = smoothstep(120, 150, hue) * (1 - smoothstep(235, 260, hue))
    hue = hue + (p["hue_target"] - hue) * p["hue_pull"] * water

    # Step 1: darken
    t = vertical_ramp(H, p)
    gam = p["v_gamma_top"] + (p["v_gamma_bottom"] - p["v_gamma_top"]) * t
    v1 = v ** gam
    g = vertical_weight(H, W, p)
    # bright light keeps more of itself (the reference's sand holds V 0.89 -> 0.82)
    g = g + (1 - g) * 0.60 * k_hi
    v1 = np.clip(v1 * g, p["floor"], 1)

    # Step 3: less saturated, bright areas go toward blue-white
    s1 = s * p["sat_water"] * (1 - p["sat_hi"] * k_hi)
    s1 = np.clip(s1 + 0.04 * (1 - k_hi), 0, 1)

    out = hsv2rgb(hue, s1, v1).astype(np.float32)

    # ripples / caustics: local contrast, only where there is light to show
    if p["clarity"] > 0:
        yl = luma(out)
        blur = cv2.GaussianBlur(yl, (0, 0), p["clarity_sigma"])
        k = smoothstep(p["clarity_lo"], p["clarity_hi"], yl)
        out = out + ((yl - blur) * p["clarity"] * k)[..., None]
    return np.clip(out, 0, 1)


# --------------------------------------------------------------- diver --
# Step 2  Restore skin tones: the diver is graded only inside the SAM 3.1
# mask, so none of the water's hue move reaches the skin.
#   * white balance undoes the red the water removed (gains measured from the
#     clip: shorts neutral)
#   * skin is pulled to the reference's flesh tone (H19-21, S0.29) while its
#     own luminance (shading, texture) is kept
#   * light shorts stay a clean off-white; fins/snorkel/hair stay dark
DV = dict(
    gains=np.array([1.90, 1.00, 1.02], np.float32),
    # flesh ratio R:G:B for H20 S~0.34 (a little over the target 0.29: the
    # ambient mix and the depth falloff take some back), normalised to luma 1
    skin_rgb=np.array([1.24, 0.95, 0.80], np.float32),
    skin_pull=0.92,          # 0 = white balance only, 1 = flat flesh colour
    lift=1.45,               # the reference diver is the brightest thing in frame
    dark_rel=(0.42, 0.62),   # luma relative to the diver's median: dark parts
    shorts_rel=(1.12, 1.35), # ... and light shorts
    shorts_chroma=0.16,      # shorts must also be near-neutral after balance
    shorts_rgb=np.array([0.99, 1.00, 1.03], np.float32),
    fins_rgb=np.array([0.90, 1.00, 1.10], np.float32),    # neutral, faintly blue black
    hair_rgb=np.array([1.12, 0.96, 0.85], np.float32),    # natural dark brown (1 : 0.86 : 0.76), luma-normalised
    hair_lift=0.85,          # hair stays darker than the skin
    depth_mix=0.20,          # share of the scene's vertical darkening
    ambient=0.05,            # share of ambient blue-cyan light
    ambient_rgb=np.array([0.35, 0.62, 0.78], np.float32),
)


def grade_diver(x, y_ref, p=DV, depth=None, parts=None):
    """x: raw float RGB frame; y_ref: median balanced luma of the diver in
    this frame (scalar, temporally smoothed by the caller); depth: the scene's
    vertical weighting; parts: optional {"hair"|"shorts"|"fins": float alpha
    HxW from its own SAM 3.1 mask}. Returns the graded diver colours for every
    pixel (use through the diver alpha).

    Each class gets its own colour and the class weights sum to 1 per pixel,
    so no class's colour reaches a neighbour:
      skin   -> flesh tone (H~20, S~0.3), own shading kept
      shorts -> clean off-white, fold contrast kept
      fins   -> neutral, faintly blue black
      hair   -> natural dark brown, contrast kept, none of the skin's warmth
    A class whose SAM 3.1 mask is empty on a frame falls back to a colour key
    for that frame (shorts: cool + bright, fins/hair: dark)."""
    xb = np.clip(x * p["gains"], 0, 1.5)
    yb = luma(xb)
    rel = yb / max(y_ref, 1e-3)
    chroma = (xb.max(-1) - xb.min(-1)) / np.maximum(xb.max(-1), 1e-3)
    parts = parts or {}

    # -- keys, used only where a class has no SAM 3.1 mask on this frame --
    k_dark = 1 - smoothstep(*p["dark_rel"], rel)
    k_dark = np.maximum(k_dark, 1 - smoothstep(0.17, 0.27, luma(np.clip(x, 0, 1))))
    cool = (xb[..., 2] - xb[..., 0]) / np.maximum(yb, 1e-3)
    k_cool = smoothstep(0.05, 0.15, cool) * smoothstep(0.55, 0.80, rel)
    k_bright = smoothstep(*p["shorts_rel"], rel) * (1 - smoothstep(p["shorts_chroma"] * 0.6, p["shorts_chroma"], chroma))
    k_short = np.clip(np.maximum(k_cool, k_bright) * (1 - k_dark), 0, 1)

    def has(name):
        a = parts.get(name)
        return a is not None and float(a.sum()) > 40

    # -- class weights (priority fins > hair > shorts > skin; sum to 1) --
    w_fins = parts["fins"] if has("fins") else (k_dark if not has("hair") else 0 * k_dark)
    w_hair = np.clip(parts["hair"] if has("hair") else 0 * k_dark, 0, 1) * (1 - w_fins)
    w_short = np.clip(parts["shorts"] if has("shorts") else k_short, 0, 1) * (1 - w_fins) * (1 - w_hair)
    w_skin = np.clip(1 - w_fins - w_hair - w_short, 0, 1)

    yd = np.clip(yb * p["lift"], 0, 1)[..., None]
    skin = yd * p["skin_rgb"]
    skin = xb * (1 - p["skin_pull"]) + skin * p["skin_pull"]
    shorts = yd * p["shorts_rgb"]
    fins = yb[..., None] * 0.80 * p["fins_rgb"]
    hair_y = (yb * p["hair_lift"])[..., None]
    hair = hair_y * p["hair_rgb"]
    out = (skin * w_skin[..., None] + shorts * w_short[..., None]
           + fins * w_fins[..., None] + hair * w_hair[..., None])
    if depth is not None:
        out = out * (1 + (depth - 1) * p["depth_mix"])
    # a little of the water's light so he sits in the scene
    out = out * (1 - p["ambient"]) + luma(out)[..., None] * p["ambient_rgb"] * p["ambient"] * 2.0
    return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------- pipeline --
def _main():
    import argparse
    import colorsys
    import dataclasses
    import json
    from pathlib import Path

    import matte as M
    from diver_reels import drop_specks, read_mask
    from shades.config import Encode
    from shades.io import Encoder, iter_frames, probe

    repo = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames", default=None, help="START:END (END exclusive)")
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--qc", type=Path, default=repo / "output" / "qc_reel")
    a = ap.parse_args()

    info = probe(a.input)
    n_all, w, h = info.n_frames, info.width, info.height
    f0, f1 = (0, n_all) if not a.frames else map(int, a.frames.split(":"))
    f1 = min(f1, n_all)
    idx = list(range(f0, f1))
    stem = a.input.stem
    fd = repo / "masks" / f"{stem}__fwd" / "subject"
    bd = repo / "masks" / f"{stem}__bwd" / "subject"
    a.qc.mkdir(parents=True, exist_ok=True)

    # pass 1: SAM 3.1 diver mask -> snapped alpha, plus the diver's luma
    raw = {}
    for i in idx:
        m = drop_specks(read_mask(fd, i, (h, w)))
        if bd.exists():
            b = drop_specks(read_mask(bd, i, (h, w)))
            if (m.sum() == 0 or m.sum() < 0.6 * b.sum()) and b.sum() > m.sum():
                m = m | b
        raw[i] = m
    missing = [i for i in idx if not raw[i].any()]
    print(f"[mask] frames {f0}-{f1 - 1}: {len(idx) - len(missing)} with a SAM 3.1 mask, "
          f"{len(missing)} without {missing[:10]}", flush=True)
    # separate SAM 3.1 masks for the diver's parts (diver_parts.py): fins may
    # leave the "person" mask, hair and shorts are clipped to it
    part_raw = {}
    for name in ("fins", "hair", "shorts"):
        d = repo / "masks" / f"{stem}__{name}" / "subject"
        if not d.exists():
            continue
        part_raw[name] = {}
        for i in idx:
            m = read_mask(d, i, (h, w))
            g = 201 if name == "fins" else 5
            near = cv2.dilate(raw[i].astype(np.uint8), np.ones((g, g), np.uint8)) > 0
            part_raw[name][i] = m & near
    for name, fr in part_raw.items():
        have = [i for i in idx if fr[i].any()]
        miss = [i for i in idx if not fr[i].any()]
        print(f"[parts] {name}: SAM 3.1 mask on {len(have)}/{len(idx)} frames"
              + (f", fallback key on {miss[:12]}{'...' if len(miss) > 12 else ''}" if miss else ""), flush=True)
    pmed = {}
    for name, fr in part_raw.items():
        u8 = {i: fr[i].astype(np.uint8) * 255 for i in idx}
        pmed[name] = {i: np.median(np.stack([u8[max(f0, i - 1)], u8[i], u8[min(f1 - 1, i + 1)]]), 0).astype(np.uint8)
                      for i in idx}
    alpha, yd = {}, {}
    for k, x in enumerate(iter_frames(a.input, info)):
        i = k
        if i < f0:
            continue
        if i >= f1:
            break
        al = M.snap_diver_alpha(raw[i], x)
        alpha[i] = np.clip(al * 255 + 0.5, 0, 255).astype(np.uint8)
        core = cv2.erode((alpha[i] > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        yd[i] = float(np.median(luma(np.clip(x * DV["gains"], 0, 1.5))[core])) if core.sum() > 200 else np.nan
    # symmetric 3-frame temporal median (no lag), then smoothed diver luma
    med = {i: np.median(np.stack([alpha[max(f0, i - 1)], alpha[i], alpha[min(f1 - 1, i + 1)]]), 0).astype(np.uint8)
           for i in idx}
    t = np.array(idx, np.float64)
    y = np.array([yd[i] for i in idx], np.float64)
    ok = ~np.isnan(y)
    y = np.interp(t, t[ok], y[ok]) if ok.any() else np.full(len(idx), 0.4)
    r = 18
    kk = np.exp(-0.5 * (np.arange(-r, r + 1) / 6.0) ** 2)
    ys = np.convolve(np.pad(y, r, mode="edge"), kk / kk.sum(), "valid")
    yref = dict(zip(idx, ys))

    # pass 2: grade + composite + encode
    out_info = dataclasses.replace(info, has_audio=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.input, a.output, out_info, Encode(crf=16, preset=a.preset))
    qc_every = int(round(2 * float(info.fps)))
    qc_ids = sorted({f0, f1 - 1} | {i for i in idx if (i - f0) % qc_every == 0})
    keep, keep_parts = {}, {}
    g_col = vertical_weight(h, w)
    for k, x in enumerate(iter_frames(a.input, info)):
        i = k
        if i < f0:
            continue
        if i >= f1:
            break
        bg = grade_background(x)
        parts = {nm: cv2.GaussianBlur(pm[i].astype(np.float32) / 255, (0, 0), 0.8) for nm, pm in pmed.items()}
        dv = grade_diver(x, yref[i], depth=g_col[..., None], parts=parts)
        al = cv2.GaussianBlur(med[i].astype(np.float32) / 255, (0, 0), 0.8)
        if "fins" in parts:                       # fins outside the person mask are still the diver
            al = np.maximum(al, parts["fins"])
        al = al[..., None]
        out = bg * (1 - al) + dv * al
        enc.write(out)
        if i in qc_ids:
            keep[i] = (x.copy(), out.copy(), med[i].copy())
            keep_parts[i] = {nm: pm[i].copy() for nm, pm in pmed.items()}
    enc.close()

    # QC sheets: mask overlay and before/after, scaled
    def thumb(img, tw=380):
        th = int(round(img.shape[0] * tw / img.shape[1]))
        return cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)

    ids = sorted(keep)
    cols = 5
    def sheet(fn, name):
        tiles = [fn(i) for i in ids]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[j:j + cols], 1) for j in range(0, len(tiles), cols)]
        cv2.imwrite(str(a.qc / name), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    def overlay(i):
        x, _, m = keep[i]
        t = thumb(x).copy()
        mm = cv2.resize(m, (t.shape[1], t.shape[0])) > 127
        edge = mm ^ (cv2.erode(mm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        t[mm] = (0.6 * t[mm] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
        t[edge] = (255, 255, 0)
        for nm, col in (("hair", (60, 255, 60)), ("shorts", (60, 120, 255)), ("fins", (255, 60, 255))):
            pmk = keep_parts[i].get(nm)
            if pmk is not None:
                q = cv2.resize(pmk, (t.shape[1], t.shape[0])) > 127
                t[q] = (0.35 * t[q] + 0.65 * np.array(col)).astype(np.uint8)
        cv2.putText(t, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return t
    def ba(i):
        x, o, _ = keep[i]
        return np.concatenate([thumb(x, 190), thumb(o, 190)], 0)
    sheet(overlay, "mask_overlay.jpg")
    sheet(ba, "before_after.jpg")

    # measurement table (graded output, diver-free water by role + diver skin)
    def stat(img, box, sel=None):
        H, W = img.shape[:2]
        reg = img[int(box[2] * H):int(box[3] * H), int(box[0] * W):int(box[1] * W)]
        s = sel[int(box[2] * H):int(box[3] * H), int(box[0] * W):int(box[1] * W)] if sel is not None else None
        px = reg[s] if s is not None else reg.reshape(-1, 3)
        if len(px) < 50:
            return None
        m = np.median(px, 0)
        hh, ss, vv = colorsys.rgb_to_hsv(*m)
        return dict(rgb=[int(round(c * 255)) for c in m], hue=int(round(hh * 360)), sat=round(float(ss), 2), val=round(float(vv), 2))
    roles = {"top_water": (0.02, 0.40, 0.03, 0.17), "mid_water": (0.02, 0.40, 0.31, 0.46),
             "bottom_water": (0.02, 0.40, 0.75, 0.95)}
    table = {}
    for i in ids:
        x, o, m = keep[i]
        free = cv2.dilate(m, np.ones((41, 41), np.uint8)) < 10
        rec = {k: stat(o, b, free) for k, b in roles.items()}
        core = cv2.erode((m > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        yb = luma(np.clip(x * DV["gains"], 0, 1.5))
        rel = yb / max(yref[i], 1e-3)
        skin = core & (rel > 0.7) & (rel < 1.1)
        rec["diver_skin"] = stat(o, (0, 1, 0, 1), skin)
        rec["diver_dark"] = stat(o, (0, 1, 0, 1), core & (rel < 0.45))
        table[i] = rec
    (a.qc / "measure.json").write_text(json.dumps(table, indent=1))
    print(f"[done] {a.output}  frames {f0}-{f1 - 1}  qc: {a.qc}", flush=True)


if __name__ == "__main__":
    _main()
