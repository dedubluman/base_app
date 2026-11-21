#!/usr/bin/env python3
"""Deep-navy cinematic underwater grade with a SAM 3.1 diver mask.

Needs masks from `diver_masks.py` (masks/<stem>__fwd/subject/*.png).
    python underwater_cinematic_grade.py --input inputs/diver.mov \
        --output output/diver_cinematic.mp4 [--preview 0,118,235]

Pure colour processing in float32 (no generative content): the water is
re-mapped to a navy gradient driven by the footage's own luminance (so ripples,
seabed and particles stay), a vignette darkens corners, and the SAM mask
drives a separate warm skin/subject grade. Settings live in SETTINGS and are
dumped next to the output as JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent

SETTINGS = dict(
    # --- environment -------------------------------------------------
    water_black=0.030,        # input luminance mapped to 0 (after WB)
    water_white=0.80,
    water_gamma=2.15,         # >1 sinks the water into deep navy
    water_detail_keep=0.45,   # share of original chroma kept (0..1)
    ramp=[(0.00, (1, 4, 12)), (0.15, (3, 16, 38)), (0.40, (8, 44, 82)),
          (0.70, (40, 100, 150)), (1.00, (140, 195, 225))],  # navy gradient map (RGB 0-255)
    vignette_strength=0.80,   # corner darkening
    vignette_power=2.1,
    bottom_dark=0.45,         # extra darkening toward the lower frame
    surface_boost=0.22,       # cool highlight lift on the bright surface
    # --- subject -----------------------------------------------------
    skin_match_w=[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0],  # per percentile (1,5,25,50,75,95,99): pull toward IMAGE 2 (highlights only; the clip's lit share differs)
    skin_toe=1.0,            # deepen skin shadows/lower mids (L* <= median) by this factor
    skin_low_gain=3.2,        # body-scale light falloff on L* (sigma 10..35 px band): sculpts the flat torso
    skin_shadow_stretch=1.5,  # stretch of the lower half of the tone curve around the median (deeper undersides)
    skin_mid_gain=2.3,       # local contrast on L* (sigma 3..10 px band)
    skin_fine_gain=1.30,      # texture on L* (< 3 px)
    skin_a_scale=0.98,        # keep red/orange (a*) controlled; warmth lives in b*
    skin_b_scale=1.14,
    skin_mid_chroma=0.85,     # muted bronze midtones (chroma factor at L* 30..65)
    skin_shadow_chroma=0.85,  # extra desaturation of skin shadows (L* < 25) -> cool-neutral
    skin_shadow_lift=7.0,     # L* added to the smooth base in the shadows (floor near IMAGE 2's p5 = 29)
    skin_ref_hi_scale=0.70,   # IMAGE 2's warm-pixel stats come from its lit shoulders only; scale its L* above the median
    skin_top_light=11.0,      # L* gain for sun-facing top of the body vs. undersides (shoulder tops bright, torso underside deep)
    skin_hi_knee=62.0,        # highlight roll-off: L* above this is compressed (IMAGE 2 p95 = 83.5)
    skin_hi_range=15.0,
    skin_vig_mix=0.30,        # share of the vignette applied to skin
    skin_wb=(1.85, 0.94, 0.72),   # recover red lost to water absorption
    skin_exposure=0.95,
    skin_contrast=1.12,
    skin_warm=0.42,              # push toward golden bronze
    skin_sat=1.0,
    skin_hi_gold=0.55,
    subject_rim=0.10,            # slight lift for separation
    mask_feather=1.2,
    # --- part split (skin / hair / gear / shorts) --------------------
    gear_lum=0.17,            # raw luminance below which a diver pixel is gear (lens, frame, snorkel, watch, fins)
    hair_lum=0.30,            # raw luminance below which pixels next to the SAM hair mask count as hair
    hair_grow=9,              # px the SAM hair mask may grow into dark pixels
    hair_max_area=9500,
    hair_top_frac=0.42,
    pole_reach=140,           # px: pole section in front of the body = hole in the diver mask with diver pixels both sides
    shorts_max_area=12000,
    fin2_lum_max=0.47,
    fin2_max_area=12000,
    fin2_min_area=300,
    fin_grow=22,
    fin_lum_tol=0.10,
    shorts_grow=14,           # px the SAM shorts mask may grow into similarly bright pixels
    shorts_lum_tol=0.14,
    part_feather=1.3,
    hair_tint=(1.00, 0.98, 1.04),   # cool-neutral dark hair
    hair_gain=0.58,
    hair_sat=0.55,
    gear_sat=0.30,
    gear_gain=0.55,
    lens_lum=0.28,
    snorkel_w=27,             # px: thinner than this inside the head area = snorkel tube
    tube_lum=0.42,            # dive-mask lens/frame: dark pixels in the band below the hair
    lens_band=0.85,           # band height in hair heights
    shorts_tint=(0.95, 1.00, 1.05),
    shorts_gain=1.45,
    # --- temporal ----------------------------------------------------
    exposure_ema=0.85,
)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def gradient_map(l, ramp):
    xs = np.array([r[0] for r in ramp], np.float32)
    out = np.empty(l.shape + (3,), np.float32)
    for c in range(3):
        ys = np.array([r[1][c] for r in ramp], np.float32) / 255.0
        out[..., c] = np.interp(l, xs, ys)
    return out


def vignette(h, w, s):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # slightly off-centre (subject sits centre-right/upper)
    cx, cy = 0.52 * w, 0.46 * h
    d = np.sqrt(((xx - cx) / (0.62 * w)) ** 2 + ((yy - cy) / (0.64 * h)) ** 2)
    v = 1.0 - s * np.clip(d, 0, 1.6) ** s_pow
    v = np.clip(v, 0.04, 1.0)
    bot = 1.0 - s_bot * smoothstep(0.35, 1.0, yy / h)
    return (v * bot).astype(np.float32)


s_pow = SETTINGS["vignette_power"]
s_bot = SETTINGS["bottom_dark"]


def grade_water(img, S, gain, vig):
    # neutralise the turquoise cast mildly so luminance reflects light, not tint
    l = lum(img)
    l = np.clip((l * gain - S["water_black"]) / (S["water_white"] - S["water_black"]), 0, 1)
    # local contrast: unsharp on luminance keeps ripples crisp
    blur = cv2.GaussianBlur(l, (0, 0), 14)
    l = np.clip(l + 0.35 * (l - blur), 0, 1)
    l = l ** S["water_gamma"]
    base = gradient_map(l, S["ramp"])
    # keep a little of the original colour variation (blue-ish only)
    orig = img / np.maximum(lum(img)[..., None], 1e-3)
    orig = np.clip(orig, 0.2, 3.0)
    tint = np.array([0.55, 0.95, 1.55], np.float32)  # cool target chroma
    chroma = (1 - S["water_detail_keep"]) * tint + S["water_detail_keep"] * orig * np.array([0.5, 0.9, 1.3], np.float32)
    # re-colour: scale base so luminance stays, bias chroma subtly
    out = base * (0.6 + 0.4 * chroma / chroma.mean(-1, keepdims=True))
    # cool surface highlights
    hi = smoothstep(0.55, 0.95, l)[..., None]
    out = out + S["surface_boost"] * hi * np.array([0.10, 0.22, 0.30], np.float32)
    return np.clip(out * vig[..., None], 0, 1)


def grade_subject(img, S, gain, vig, top=None):
    wb = np.array(S["skin_wb"], np.float32)
    keep = 1 - 0.85 * smoothstep(0.55, 0.9, lum(img))[..., None]   # bright specks/bubbles stay neutral, not orange
    x = np.clip(img * (1 + (wb - 1) * keep) * gain * S["skin_exposure"], 0, 1.5)
    # base tone: contrast around mid-grey, soft highlight roll-off
    bl = cv2.GaussianBlur(x, (0, 0), 6)
    x = x + 0.45 * (x - bl)
    x = 0.45 + (x - 0.45) * S["skin_contrast"]
    x = np.clip(x, 0, 1)
    x = x / (1 + 0.35 * x)  # gentle shoulder
    l = lum(x)
    # skin weight: warm, mid-bright, not near white (shorts) or black (fins)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    warm = smoothstep(-0.03, 0.06, r - b)
    midw = smoothstep(0.03, 0.14, l) * (1 - smoothstep(0.78, 0.97, l))
    skin = warm * midw
    # golden-bronze target: shift hue toward amber while holding luminance
    gold = np.stack([l * 1.18, l * 0.93, l * 0.58], -1)
    gold = gold / np.maximum(lum(gold)[..., None], 1e-3) * l[..., None]
    x = x * (1 - S["skin_warm"] * skin[..., None]) + gold * S["skin_warm"] * skin[..., None]
    # golden highlights
    hi = smoothstep(0.30, 0.8, l) * skin
    if top is not None:
        hi = hi * top
        x = x * (1 + 0.18 * top[..., None] * skin[..., None])
    x = x + S["skin_hi_gold"] * hi[..., None] * np.array([0.10, 0.045, -0.02], np.float32)
    # saturation
    l2 = lum(x)[..., None]
    x = l2 + (x - l2) * S["skin_sat"]
    # vignette shared but weaker on the subject
    return np.clip(x * (0.55 + 0.45 * vig[..., None]) + S["subject_rim"] * 0.0, 0, 1)


def grade_neutral(img, S, vig, tint, gain, sat, top=None, rim=0.0):
    """Hair / gear / shorts: no skin warmth, no white-balance push. Keeps original
    luminance structure, pulls chroma toward a cool-neutral tint."""
    l0 = lum(img)[..., None]
    x = l0 + (img - l0) * sat
    x = x * np.array(tint, np.float32) / np.mean(tint) * gain
    if top is not None and rim:
        x = x + rim * top[..., None] * np.array([0.03, 0.06, 0.10], np.float32)
    return np.clip(x * (0.55 + 0.45 * vig[..., None]), 0, 1)


def _grow(m, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def split_parts(img, full, hair_s, shorts_s, fins_s, S, fin2_s=None):
    """Return soft masks skin / hair / gear / shorts (SAM 3.1 part masks refined with
    colour cues inside the diver mask). skin = diver - hair - gear - shorts."""
    L = lum(img)
    if fin2_s is not None and fin2_s.any():   # second (raised) fin tracked from a seed: only fin-toned pixels, never leg skin
        fins_s = fins_s | (fin2_s & (L < S["fin2_lum_max"]))
    reach = _grow(full | fins_s, 3)
    # gear: SAM fins + very dark diver pixels (mask lens/frame, snorkel, watch, fins)
    dark = (L < S["gear_lum"]) & reach
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    # hair geometry from the main hair component only (SAM leaves stray blobs on the neck)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(hair_s.astype(np.uint8), 8)
    if nl > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        hair_s = lab == k
        ys, xs = np.nonzero(hair_s)
        hh = ys.max() - ys.min() + 1
        zone = np.zeros_like(full)
        y0, y1 = ys.max() - 6, int(ys.max() + S["lens_band"] * hh)
        zone[max(y0, 0):y1, max(xs.min() - 14, 0):xs.max() + 24] = True
        lens = ((L < S["lens_lum"]) & zone & reach).astype(np.uint8)
        lens = cv2.morphologyEx(lens, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        lens = cv2.morphologyEx(lens, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(lens, 8)
        if n2 > 1:
            k = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
            cs, _ = cv2.findContours((lab2 == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = np.zeros(L.shape, np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(cs)), 1)
            dark = dark | ((hull > 0) & zone & reach)
        # snorkel tube: thin structure beside the head (skin parts are thicker than the tube)
        head = np.zeros_like(full)
        head[max(ys.min() - 40, 0):int(ys.max() + 1.6 * hh), max(xs.min() - 60, 0):xs.max() + 70] = True
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (S["snorkel_w"], S["snorkel_w"]))
        thick = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_OPEN, k_open) > 0
        tube = full & ~thick & head & (L < S["tube_lum"])
        tube = cv2.morphologyEx(tube.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        if tube.any():  # grow along pixels of the tube's own tone (lower tube touches the cheek)
            ref = float(np.median(L[tube]))
            for _ in range(3):
                tube |= (np.abs(L - ref) < 0.07) & _grow(tube, 7) & full & head & ~hair_s
        dark = dark | tube
    fins_g = fins_s.copy()
    if fins_s.any():   # SAM may find only one fin: grow into pixels of the same tone (still inside the diver mask)
        refl = float(np.median(L[fins_s]))
        fins_g |= (np.abs(L - refl) < S["fin_lum_tol"]) & _grow(fins_s, S["fin_grow"]) & reach
    gear = fins_g | dark
    # hair: SAM hair + dark pixels hugging it
    hair = hair_s | ((L < S["hair_lum"]) & _grow(hair_s, S["hair_grow"]) & reach)
    hair &= ~dark | hair_s          # the dark mask lens never becomes hair
    # shorts: SAM shorts grown into similarly bright pixels (left leg/thigh skin is excluded by tolerance)
    shorts = shorts_s.copy()
    if shorts_s.any():
        ref = float(np.median(L[shorts_s]))
        shorts |= (np.abs(L - ref) < S["shorts_lum_tol"]) & _grow(shorts_s, S["shorts_grow"]) & full
    gear &= ~shorts
    hair &= ~gear
    f = S["part_feather"]
    sm = lambda m: cv2.GaussianBlur(m.astype(np.float32), (0, 0), f)
    fullf, hf, gf, sf = sm(full | fins_s), sm(hair), sm(gear), sm(shorts)
    skin = np.clip(fullf - np.maximum.reduce([hf, gf, sf]), 0, 1)
    return dict(full=fullf, skin=skin, hair=hf, gear=gf, shorts=sf)


def load_part(d: Path, n, shape):
    out = []
    for i in range(n):
        m = cv2.imread(str(d / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        out.append(np.zeros(shape, bool) if m is None else m > 127)
    return out


def vote3(ms):
    """Symmetric 3-frame majority vote: kills one-frame flicker without lag."""
    out = []
    for i in range(len(ms)):
        a, b, c = ms[max(i - 1, 0)], ms[i], ms[min(i + 1, len(ms) - 1)]
        out.append((a.astype(np.uint8) + b + c) >= 2)
    return out


REF = json.loads((REPO / "skin_reference.json").read_text())
_PCT = ["1", "5", "25", "50", "75", "95", "99"]


def sample_skin_L(img, P, rng):
    """L* values (cv2 Lab, 0..100) of this frame's exposed skin, subsampled."""
    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]
    v = L[P["skin"] > 0.9]
    if len(v) > 4000:
        v = rng.choice(v, 4000, replace=False)
    return v


def build_curve(src_L, S):
    """Monotone tone curve: clip skin L* percentiles -> IMAGE 2 skin percentiles."""
    src = np.percentile(src_L, [float(x) for x in _PCT])
    ref = np.array([REF["L_pct"][k] for k in _PCT])
    ref[3:] = ref[3] + (ref[3:] - ref[3]) * 1.0
    ref = np.where(ref > 50, 50 + (ref - 50) * S["skin_ref_hi_scale"], ref)
    w = np.array(S["skin_match_w"])
    src_t = src.copy()
    src_t[:4] = src[:4] * S["skin_toe"]            # shadow toe: real depth in the lower half
    piv = src_t[3]
    src_t[:3] = piv + (src_t[:3] - piv) * S["skin_shadow_stretch"]
    src_t = np.clip(src_t, 2, 98)
    tgt = w * ref + (1 - w) * src_t
    xs = np.concatenate([[0], src, [100]])
    ys = np.concatenate([[0], np.maximum.accumulate(tgt), [100]])
    ys = np.maximum.accumulate(ys)
    for i in range(1, len(xs)):          # strictly increasing x
        xs[i] = max(xs[i], xs[i - 1] + 1e-3)
    return xs.tolist(), ys.tolist(), src.tolist(), ref.tolist()


def nblur(x, w, sigma):
    """Mask-normalised Gaussian blur: skin pixels only, no bleed from water/hair/gear."""
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def grade_skin_lab(img, S, vig, top=None, w=None):
    """Skin only. L*: IMAGE 2 tone curve on the very smooth base + boosted body-scale,
    local and fine contrast. a*/b* from IMAGE 2's chroma-vs-L* table: shadows low chroma
    and cool, mids muted bronze, highlights champagne-gold (b* carries the warmth, a* held back)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    w = np.ones(L.shape, np.float32) if w is None else np.clip(w, 0, 1).astype(np.float32)
    B40, b10, b3 = nblur(L, w, 35), nblur(L, w, 10), nblur(L, w, 3)
    cx, cy = np.array(S["_curve"][0], np.float32), np.array(S["_curve"][1], np.float32)
    base = np.interp(B40, cx, cy).astype(np.float32)
    Lo = (base + S["skin_low_gain"] * (b10 - B40) + S["skin_mid_gain"] * (b3 - b10)
          + S["skin_fine_gain"] * (L - b3))
    if top is not None:
        Lo = Lo + S["skin_top_light"] * (top - 0.35) * np.clip(Lo / 55.0, 0.3, 1.2)
    Lo = Lo + S["skin_shadow_lift"] * (1 - smoothstep(30, 62, base))
    k_ = S["skin_hi_knee"]
    Lo = np.where(Lo > k_, k_ + S["skin_hi_range"] * np.tanh((Lo - k_) / S["skin_hi_range"]), Lo)
    Lo = np.clip(Lo, 0, 100)
    bl = np.array([b["L"] for b in REF["bands"]], np.float32)
    ba = np.array([b["a"] for b in REF["bands"]], np.float32)
    bb = np.array([b["b"] for b in REF["bands"]], np.float32)
    Ls = cv2.GaussianBlur(Lo, (0, 0), 2.0)
    a = np.interp(Ls, bl, ba) * S["skin_a_scale"]
    b = np.interp(Ls, bl, bb) * S["skin_b_scale"]
    shadow = 1 - smoothstep(12, 30, Ls)
    mid = smoothstep(30, 45, Ls) * (1 - smoothstep(55, 70, Ls))
    k = (1 - (1 - S["skin_shadow_chroma"]) * shadow) * (1 - (1 - S["skin_mid_chroma"]) * mid)
    a, b = a * k, b * k
    if top is not None:                   # sun-facing upper body leans champagne-gold (b*), not red
        hi = smoothstep(55, 85, Ls) * top
        b = b + 1.5 * hi
        a = a - 0.3 * hi
    a = cv2.GaussianBlur(a, (0, 0), 1.5)
    b = cv2.GaussianBlur(b, (0, 0), 1.5)
    out = cv2.cvtColor(np.stack([Lo, a, b], -1).astype(np.float32), cv2.COLOR_LAB2RGB)
    vm = S["skin_vig_mix"]
    return np.clip(out * ((1 - vm) + vm * vig[..., None]), 0, 1)


def merge_masks(mdir: Path, n, shape):
    ms = []
    for i in range(n):
        m = cv2.imread(str(mdir / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        ms.append(np.zeros(shape, np.uint8) if m is None else (m > 127).astype(np.uint8))
    return ms


def probe(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"], int(s["nb_frames"])


def read_frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    sz = w * h * 3
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def process(frames, masks, S, w, h):
    vig = vignette(h, w, S["vignette_strength"])
    ema, out = None, []
    for i, fr in enumerate(frames):
        img = fr.astype(np.float32) / 255.0
        m = masks['full'][i]
        # temporal exposure normalisation (EMA of median luminance)
        med = float(np.median(lum(img)))
        ema = med if ema is None else S["exposure_ema"] * ema + (1 - S["exposure_ema"]) * med
        gain = float(np.clip(0.30 / max(ema, 1e-3), 0.7, 1.4)) if False else 1.0
        bg = grade_water(img, S, gain, vig)
        ys = np.nonzero(m.any(1))[0]
        top = None
        if len(ys):
            yy = (np.arange(h, dtype=np.float32) - ys[0]) / max(ys[-1] - ys[0], 1)
            top = (1 - smoothstep(0.0, 0.6, yy))[:, None] * np.ones((1, w), np.float32)
        P = split_parts(img, m.astype(bool), masks['hair'][i], masks['shorts'][i], masks['fins'][i], S, masks['fin2'][i])
        sm = P['full'][..., None]
        x_skin = grade_skin_lab(img, S, vig, top, P['skin'])
        x_hair = grade_neutral(img, S, vig, S['hair_tint'], S['hair_gain'], S['hair_sat'], top, rim=0.5)
        x_gear = grade_neutral(img, S, vig, (1, 1, 1.03), S['gear_gain'], S['gear_sat'])
        x_sh = grade_neutral(img, S, vig, S['shorts_tint'], S['shorts_gain'], 0.25)
        wsum = np.maximum(P['skin'] + P['hair'] + P['gear'] + P['shorts'], 1e-3)[..., None]
        sub = (P['skin'][..., None] * x_skin + P['hair'][..., None] * x_hair
               + P['gear'][..., None] * x_gear + P['shorts'][..., None] * x_sh) / wsum
        sm = np.clip(np.maximum(sm, np.maximum.reduce([P['hair'], P['gear'], P['shorts']])[..., None]), 0, 1)
        # selfie stick: dark, neutral, outside the diver mask, touching the lower frame -> keep original look
        mx, mn = img.max(-1), img.min(-1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        dark = ((sat < 0.42) & (m == 0) & (np.arange(h)[:, None] > 0.30 * h)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        nl, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
        stick = np.zeros((h, w), np.float32)
        for k in range(1, nl):
            if st[k, cv2.CC_STAT_AREA] > 4000 and st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] >= h - 2:
                cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pts = np.vstack(cs)
                cv2.fillConvexPoly(stick, cv2.convexHull(pts), 1.0)
        if stick.any():  # reach the ring/collar at the top of the pole (dark or unsaturated only)
            near = cv2.dilate(stick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
            ring = ((near > 0) & (m == 0) & ((sat < 0.55) | (lum(img) < 0.16))).astype(np.float32)
            stick = np.maximum(stick, ring)
        if stick.any():  # upper pole: the part in front of the body is a hole in the diver mask (diver pixels on both sides)
            fu = (m > 0).astype(np.uint8)
            k = S["pole_reach"]
            left = cv2.dilate(fu, np.ones((1, k), np.uint8), anchor=(k - 1, 0))     # a diver pixel within k px to the left
            right = cv2.dilate(fu, np.ones((1, k), np.uint8), anchor=(0, 0))        # ... and to the right
            hole = ((fu == 0) & (left > 0) & (right > 0)).astype(np.uint8)
            hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            seed = cv2.dilate((stick > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
            nl2, lab2, _, _ = cv2.connectedComponentsWithStats(np.maximum(hole, (stick > 0).astype(np.uint8)), 8)
            ids = np.unique(lab2[(seed > 0) & (hole > 0)])
            ids = ids[ids > 0]
            if len(ids):
                upper = np.isin(lab2, ids) & (fu == 0)
                stick = np.maximum(stick, upper.astype(np.float32))
        stick = cv2.GaussianBlur(cv2.dilate(stick, np.ones((3, 3), np.uint8)), (0, 0), 1.0)[..., None]
        l0 = lum(img)[..., None]
        st_col = np.clip((l0 + (img - l0) * 0.25) * 0.85 * (0.5 + 0.5 * vig[..., None]), 0, 1)
        bg = bg * (1 - stick) + st_col * stick
        P['stick'] = stick[..., 0]
        yield np.clip((bg * (1 - sm) + sub * sm) * 255 + 0.5, 0, 255).astype(np.uint8), P


def save_diag(d: Path, i, img_u8, P):
    """Binary masks + one overlay per frame: diver_full, skin_only, hair_only, excluded_gear_and_pole."""
    d.mkdir(parents=True, exist_ok=True)
    gear_pole = np.maximum(P['gear'], np.maximum(P['stick'], P['shorts']))
    layers = {"diver_full": P['full'], "skin_only": P['skin'], "hair_only": P['hair'],
              "excluded_gear_and_pole": gear_pole}
    for k, v in layers.items():
        cv2.imwrite(str(d / f"{i:05d}_{k}.png"), (v > 0.5).astype(np.uint8) * 255)
    ov = img_u8.astype(np.float32)
    for v, c in ((P['skin'], (255, 190, 120)), (P['hair'], (255, 40, 40)),
                 (P['gear'], (40, 120, 255)), (P['shorts'], (60, 255, 90)), (P['stick'], (200, 60, 255))):
        a_ = (0.55 * (v > 0.5))[..., None]
        ov = ov * (1 - a_) + np.array(c, np.float32) * a_
    cv2.imwrite(str(d / f"{i:05d}_overlay.jpg"), cv2.cvtColor(ov.astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--preview", default="")
    ap.add_argument("--calib-max", type=int, default=10**9, help="only calibrate on frames below this index (partial masks)")
    ap.add_argument("--set", action="append", default=[], help="KEY=JSON value")
    a = ap.parse_args()
    S = dict(SETTINGS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        S[k] = json.loads(v)
    global s_pow, s_bot
    s_pow, s_bot = S["vignette_power"], S["bottom_dark"]
    w, h, fps, n = probe(a.input)
    mdir = a.masks or REPO / "masks" / f"{a.input.stem}__fwd" / "subject"
    root = REPO / "masks"
    stem = a.input.stem
    masks = {"full": merge_masks(mdir, n, (h, w))}
    for k in ("hair", "shorts", "fins"):
        pm = load_part(root / f"{stem}__{k}" / "subject", n, (h, w))
        if k == "shorts":   # tracker drift onto torso (area explosion): fall back to the last good mask
            last = None
            for i, m in enumerate(pm):
                if m.sum() > S["shorts_max_area"] and last is not None:
                    pm[i] = last.copy()
                elif m.sum() <= S["shorts_max_area"]:
                    last = m
        if k == "hair":     # tracker swelled over mask + face on some frames: keep the top (hair) part of the blob only
            for i, m in enumerate(pm):
                if m.sum() > S["hair_max_area"]:
                    ys, _ = np.nonzero(m)
                    cut = ys.min() + S["hair_top_frac"] * (ys.max() - ys.min() + 1)
                    keep = m.copy()
                    keep[int(cut):] = False
                    pm[i] = keep
        masks[k] = vote3(pm)
    f2 = load_part(root / f"{stem}__fin2" / "subject", n, (h, w))
    # the seeded tracker loses the fin once it leaves the frame (huge blobs / nothing): keep only plausible frames
    f2 = [m if S["fin2_min_area"] <= m.sum() <= S["fin2_max_area"] else np.zeros_like(m) for m in f2]
    masks["fin2"] = vote3(f2)
    frames = list(read_frames(a.input, w, h)) if a.preview else None
    # calibration: skin L* distribution of the clip (sampled frames), fixed for the whole render -> no flicker
    rng = np.random.default_rng(0)
    smp = []
    src_iter = frames if frames is not None else read_frames(a.input, w, h)
    for i, fr in enumerate(src_iter):
        if i % 6 == 0 and i < min(n, a.calib_max):
            one = {k: [v[i]] for k, v in masks.items()}
            P = split_parts(fr.astype(np.float32) / 255.0, one['full'][0].astype(bool), one['hair'][0],
                            one['shorts'][0], one['fins'][0], S, one['fin2'][0])
            if (P['skin'] > 0.9).any():
                smp.append(sample_skin_L(fr.astype(np.float32) / 255.0, P, rng))
    cx, cy, src_p, ref_p = build_curve(np.concatenate(smp), S)
    S["_curve"] = [cx, cy]
    S["_skin_L_src_pct"], S["_skin_L_ref_pct"] = src_p, ref_p
    print("skin L* src pct", np.round(src_p, 1), "-> ref", np.round(ref_p, 1))
    if frames is None:
        frames = read_frames(a.input, w, h)
    if a.preview:
        idx = [int(x) for x in a.preview.split(",")]
        sel = {i: frames[i] for i in idx}
        rows = []
        for i in idx:
            one = {k: [v[i]] for k, v in masks.items()}
            g, P = next(process([sel[i]], one, S, w, h))
            save_diag(REPO / "output" / "diagnostics", i, sel[i], P)
            rows.append(np.hstack([sel[i], g]))
        cv2.imwrite(str(REPO / "output" / "preview.jpg"),
                    cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", "12",
                            "-preset", "slow", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for g, _ in process(frames, masks, S, w, h):
        enc.stdin.write(g.tobytes())
    enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(a.input), "-map", "0:v",
                    "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", str(a.output)], check=True)
    print("tmp frames", subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(tmp)]).decode().strip())
    tmp.unlink()
    a.output.with_suffix(".settings.json").write_text(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
