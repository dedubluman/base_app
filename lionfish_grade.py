#!/usr/bin/env python3
"""Lionfish subject isolation + cinematic grade. Masks: SAM 3.1 (CPU port).

Pipeline
  1. SAM 3.1 ("fish" text prompt, two overlapping square crops so the small
     fish are not squashed) -> per-object tracked masks (cuttle_masks.py).
  2. build_union_masks: duplicate detections from the two crops are merged
     by IoU into one identity per fish, written per proxy frame as binary
     'all' / 'lion' masks plus an id map; an area-continuity QC is logged.
  3. Per source frame: the two neighbouring proxy masks are warped into the
     frame with dense optical flow (forward + backward neighbour, no
     ghosting), the edge is snapped to the image with a guided filter
     (matte.snap_diver_alpha, constrained to a 1 px band, no halo).
  4. Grade, three separate looks:
       lionfish   - relative-contrast gradient map (crimson -> fire orange ->
                    amber -> gold) so translucent fins glow and rays stay dark
       other fish - per-fish tone normalisation (temporally smoothed) onto a
                    natural blue-silver-olive-gold ramp
       background - dehaze-lite, desaturated, dark petrol/teal, vignette, grain
  5. Single H.264 MP4 (CRF 16, slow, yuv420p, BT.709), source audio copied.

    python lionfish_grade.py --input clip.mov --masks DIR_A DIR_B \
        --crop-ys 100 840 --workdir WORK --output out.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from matte import snap_diver_alpha
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

W3 = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(x):
    return x @ W3


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def ramp(t, pts):
    pts = np.asarray(pts, np.float32)
    return np.stack([np.interp(t, pts[:, 0], pts[:, 1 + c]) for c in range(3)], -1).astype(np.float32)


# ---------------------------------------------------------------- masks ----
def index_objects(mask_dirs, tags="AB"):
    objs = {}
    for tag, d in zip(tags, mask_dirs):
        for od in sorted(Path(d).glob("obj_*")):
            m = {int(p.stem): p for p in od.glob("*.png")}
            if m:
                objs[f"{tag}{od.name[4:]}"] = (tag, m)
    return objs


def load_mask(path, crop, h, w):
    """crop = (x, y) of the crop the SAM run saw (full frame -> (0, 0))."""
    a = cv2.imread(str(path), 0) > 127
    if a.shape[0] != h or a.shape[1] != w:
        full = np.zeros((h, w), bool)
        cx, cy = crop
        full[cy:cy + a.shape[0], cx:cx + a.shape[1]] = a
        a = full
    return a


def _small(path, crop, h, w, q=4):
    return cv2.resize(load_mask(path, crop, h, w).astype(np.uint8), (w // q, h // q),
                      interpolation=cv2.INTER_AREA) > 0


def build_union_masks(objs, crops, h, w, out_dir, min_frames=6, iou_t=0.3):
    """objs: {name: (tag, {frame: png})}; crops: {tag: (x, y)}.
    Tag 'L' = the tight-crop lionfish run: if present it IS the lionfish
    (finer fin rays); the coarse full-crop lionfish detections are only a
    per-frame fallback where the tight run lost it."""
    q = 4
    lion_names = [n for n, (t, fr) in objs.items() if t == "L" and len(fr) >= min_frames]
    sm = {n: {f: _small(p, crops[tag], h, w) for f, p in list(fr.items())[::8]}
          for n, (tag, fr) in objs.items() if len(fr) >= min_frames}
    area = {n: np.median([m.sum() for m in v.values()]) for n, v in sm.items()}

    def iou(a, b):
        sh = [f for f in sm[a] if f in sm[b]]
        return np.mean([(sm[a][f] & sm[b][f]).sum() / max((sm[a][f] | sm[b][f]).sum(), 1)
                        for f in sh]) if sh else 0.0

    names = sorted([n for n in sm if objs[n][0] != "L"], key=lambda k: -area[k])
    groups = []
    for n in names:
        for g in groups:
            if iou(n, g[0]) > iou_t:
                g.append(n)
                break
        else:
            groups.append([n])
    fallback = []
    if lion_names:
        lion = max(lion_names, key=lambda k: area[k])
        for g in list(groups):
            if iou(g[0], lion) > iou_t:
                fallback = g
                groups.remove(g)
        groups = [[lion]] + groups
    print(f"[masks] {len(groups)} fish identities; lionfish = {groups[0]} "
          f"(coarse fallback {fallback})", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted({f for g in groups for n in g for f in objs[n][1]})
    areas = {gi: {} for gi in range(len(groups))}
    for f in frames:
        allm = np.zeros((h, w), bool)
        lm = np.zeros((h, w), bool)
        ids = np.zeros((h, w), np.uint8)
        for gi, g in enumerate(groups):          # largest first, small fish overwrite
            gm = np.zeros((h, w), bool)
            for n in g:
                tag, fr = objs[n]
                if f in fr:
                    gm |= load_mask(fr[f], crops[tag], h, w)
            if gi == 0 and not gm.any():
                for n in fallback:
                    tag, fr = objs[n]
                    if f in fr:
                        gm |= load_mask(fr[f], crops[tag], h, w)
            if gm.any():
                areas[gi][f] = int(gm.sum())
                allm |= gm
                ids[gm] = gi + 1
                if gi == 0:
                    lm = gm
        cv2.imwrite(str(out_dir / f"all_{f:05d}.png"), allm.astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / f"lion_{f:05d}.png"), lm.astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / f"ids_{f:05d}.png"), ids)
    # QC: area continuity per identity (jumps flag a tracker error)
    qc = {}
    for gi, ar in areas.items():
        ks = sorted(ar)
        flags = [k for a, k in zip(ks, ks[1:]) if k - a == 1 and
                 not (0.62 < ar[k] / max(ar[a], 1) < 1.6)]
        qc[str(gi)] = {"frames": [ks[0], ks[-1]] if ks else None, "n": len(ks),
                       "median_area": float(np.median(list(ar.values()))) if ar else 0,
                       "area_jumps_at_proxy_frames": flags}
    (out_dir / "qc.json").write_text(json.dumps(qc, indent=1))
    print("[masks] QC", json.dumps(qc), flush=True)
    return len(frames)


def _gray_at(video, n):
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "fatal", "-i", str(video), "-vf",
                          f"select=eq(n\\,{n}),format=gray", "-frames:v", "1", "-vsync", "0",
                          "-f", "image2pipe", "-vcodec", "png", "-"],
                         check=True, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(raw, np.uint8), 0)


def repair_masks(mdir, video, stride, h, w, min_ratio=0.75):
    """Forward/backward validation of every non-lion identity: a frame whose
    area collapses (or balloons) relative to both neighbours is rebuilt from
    those neighbours, each warped into it with dense optical flow."""
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ids_f = {int(p.stem.split("_")[1]): p for p in mdir.glob("ids_*.png")}
    frames = sorted(ids_f)
    ids = {k: cv2.imread(str(ids_f[k]), 0) for k in frames}
    gids = sorted({int(g) for k in frames for g in np.unique(ids[k]) if g > 1})
    fixed = []
    for gid in gids:
        area = {k: int((ids[k] == gid).sum()) for k in frames}
        pres = [k for k in frames if area[k] > 0]
        # collapsed frames: well below the larger neighbour (collapses can span
        # a few frames, so also compare with the 2nd neighbour on each side)
        bad = set()
        for k in pres:
            nb = [area.get(k + d, 0) for d in (-2, -1, 1, 2)]
            if area[k] and max(nb) > 300 and area[k] < min_ratio * max(nb):
                bad.add(k)
        for k in sorted(bad):
            lo = next((q for q in range(k - 1, k - 6, -1) if q in area and area[q] > 300 and q not in bad), None)
            hi = next((q for q in range(k + 1, k + 6) if q in area and area[q] > 300 and q not in bad), None)
            if lo is None and hi is None:
                continue
            gk = _gray_at(video, k * stride)
            acc, wsum = np.zeros((h, w), np.float32), 0.0
            for kk in (lo, hi):
                if kk is None:
                    continue
                wt = 1.0 / abs(kk - k)
                fl = cv2.calcOpticalFlowFarneback(gk, _gray_at(video, kk * stride), None, 0.5, 3, 25, 3, 5, 1.1, 0)
                acc += wt * cv2.remap((ids[kk] == gid).astype(np.float32), gx + fl[..., 0],
                                      gy + fl[..., 1], cv2.INTER_LINEAR)
                wsum += wt
            rep = (acc / wsum) > 0.5
            rep = cv2.morphologyEx(rep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            before = area[k]
            ids[k][ids[k] == gid] = 0
            ids[k][rep & (ids[k] == 0)] = gid
            allm = cv2.imread(str(mdir / f"all_{k:05d}.png"), 0) > 127
            allm |= (ids[k] == gid)
            cv2.imwrite(str(mdir / f"ids_{k:05d}.png"), ids[k])
            cv2.imwrite(str(mdir / f"all_{k:05d}.png"), allm.astype(np.uint8) * 255)
            fixed.append((gid, k, before, int((ids[k] == gid).sum())))
    print(f"[masks] repaired (identity, proxy frame, area before, after): {fixed}", flush=True)
    (mdir / "repairs.json").write_text(json.dumps(fixed))
    return fixed


class MaskStore:
    def __init__(self, d):
        self.d, self.cache = d, {}

    def get(self, kind, k):
        key = (kind, k)
        if key not in self.cache:
            p = self.d / f"{kind}_{k:05d}.png"
            im = cv2.imread(str(p), 0) if p.exists() else None
            self.cache[key] = im if kind == "ids" else (None if im is None else im > 127)
            if len(self.cache) > 12:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def interp_frame(store, f, stride, gray_f, gray_proxy, h, w):
    """-> (all_float, lion_float, ids_uint8) at source frame f."""
    k0 = f // stride
    t = (f - k0 * stride) / stride
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    acc = {"all": np.zeros((h, w), np.float32), "lion": np.zeros((h, w), np.float32)}
    wsum, ids_src, best = 0.0, None, -1
    for k, wt in ((k0, 1 - t), (k0 + 1, t)):
        ma = store.get("all", k) if wt > 0 else None
        if ma is None or k not in gray_proxy:
            continue
        if k * stride == f:
            def mp(x, nearest=False):
                return x
        else:
            fl = cv2.calcOpticalFlowFarneback(gray_f, gray_proxy[k], None, 0.5, 3, 25, 3, 5, 1.1, 0)
            mx, my = gx + fl[..., 0], gy + fl[..., 1]

            def mp(x, nearest=False, mx=mx, my=my):
                return cv2.remap(x, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        for kind in ("all", "lion"):
            acc[kind] += wt * mp(store.get(kind, k).astype(np.float32))
        if wt > best:
            best, ids_src = wt, mp(store.get("ids", k), True)
        wsum += wt
    if not wsum:
        return acc["all"], acc["lion"], np.zeros((h, w), np.uint8)
    return acc["all"] / wsum, acc["lion"] / wsum, ids_src


def snap_regions(mask, rgb):
    """Guided-filter edge snap per connected component (works for tiny fish
    and the large lionfish alike), constrained to a 1 px band, no halo."""
    out = np.zeros(mask.shape, np.float32)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for c in range(1, n):
        x0, y0, bw, bh, ar = st[c]
        if ar < 12:
            continue
        sl = (slice(max(y0 - 10, 0), y0 + bh + 10), slice(max(x0 - 10, 0), x0 + bw + 10))
        a = snap_diver_alpha(lab[sl] == c, rgb[sl])
        out[sl] = np.maximum(out[sl], a)
    return out


# ---------------------------------------------------------------- grade ----
# lionfish colour ramp (t = tone-mapped, shadow-lifted fish luminance 0..1).
# Wide and smooth: crimson in the shade, scarlet/red-orange in the midtones,
# amber only in the lights. The floor is a lit crimson, never black.
LION_RAMP = [
    (0.00, 0.420, 0.075, 0.070),   # sienna-crimson (shadow, clearly not black)
    (0.18, 0.600, 0.085, 0.075),   # crimson
    (0.38, 0.780, 0.085, 0.070),   # rich scarlet
    (0.58, 0.900, 0.190, 0.065),   # fiery red-orange
    (0.78, 1.000, 0.380, 0.090),   # orange
    (0.92, 1.000, 0.520, 0.150),   # amber
    (1.00, 1.000, 0.600, 0.220),   # soft golden light
]


def estimate_airlight(frames):
    return np.median(np.stack([np.percentile(f.reshape(-1, 3), 97, axis=0) for f in frames]),
                     0).astype(np.float32)


def dehaze(rgb, air, veil, red_gain=1.45):
    j = np.clip((rgb - veil * air) / (1 - veil), 0, 1.5)
    return j * np.array([red_gain, 1.0, 1.0], np.float32)


def norm_blur(x, w, sigma, scale=4):
    """Normalised blur of x over weight map w (computed at 1/scale res)."""
    h0, w0 = w.shape
    sm = (w0 // scale, h0 // scale)
    xs = cv2.resize(x * (w[..., None] if x.ndim == 3 else w), sm, interpolation=cv2.INTER_AREA) \
        if scale > 1 else x * (w[..., None] if x.ndim == 3 else w)
    ws = cv2.resize(w, sm, interpolation=cv2.INTER_AREA) if scale > 1 else w
    xb = cv2.GaussianBlur(xs, (0, 0), sigma / scale)
    wb = cv2.GaussianBlur(ws, (0, 0), sigma / scale)
    if x.ndim == 3:
        wb = wb[..., None]
    out = xb / np.maximum(wb, 1e-4)
    return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_CUBIC) if scale > 1 else out


def decontaminate(col, region, iters=1):
    """Edge decontamination. Colour is only trusted deep inside `region`
    (mask eroded by `iters` px); the rim takes the colour of the nearest interior
    pixels, so water mixed into the edge pixels can never form a halo."""
    inner = cv2.erode(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                      iterations=iters) > 0
    if inner.sum() < 20:
        return col
    wgt = inner.astype(np.float32)
    ext = norm_blur(col, wgt, 2.5, scale=1)
    return np.where(inner[..., None], col, ext)


def soft_knee(c, k=0.78, r=0.45):
    return np.where(c > k, k + (c - k) * r, c)


def scene_look(x, air, vig):
    """The scene's own grade (what the water, rock and sand get). Fish are run
    through it too, so every fish shares the scene's light and atmosphere."""
    x = np.clip((x - 0.30 * air) / 0.70, 0, 1)                 # lift the worst of the haze
    l = luma(x)[..., None]
    x = l + (x - l) * 0.28                                     # desaturate
    x = np.clip(x + 0.55 * (x - cv2.GaussianBlur(x, (0, 0), 9.0)), 0, 1)   # texture recovery
    x = x ** 1.12
    x = x / (1 + 0.95 * x * x) * 1.22                          # shoulder: sand compressed, rock lifted
    x = x * np.array([0.57, 0.88, 1.00], np.float32)           # petrol / dark teal
    x = x * 0.80 + np.array([0.006, 0.018, 0.026], np.float32)
    return np.clip(x * vig[..., None], 0, 1)


def grade_bg(rgb, air, a_all, vig):
    excl = 1 - cv2.dilate(a_all, np.ones((9, 9), np.uint8))
    fill = norm_blur(rgb, excl.astype(np.float32), 8, scale=2)
    x = np.where(a_all[..., None] > 0.02, fill, rgb)
    return scene_look(cv2.GaussianBlur(x, (0, 0), 1.0), air, vig)


def grade_lion(rgb, air, veil, a_lion, bg, scene, vig):
    """Lionfish: its own photographic light structure is kept (so fins, rays and
    body marks stay organic), the shadows are opened, and the colour comes from
    a smooth ramp. Then it is integrated into the water: atmosphere, ambient
    colour in the shade, softness and edge wrap matching the scene."""
    m = a_lion > 0.5
    if m.sum() < 500:
        return scene
    # Natural lionfish. Everything is relative to the water around the fish, so the
    # translucent membrane stays at the water's own level (it simply shows the water),
    # the rays and bands go darker than the water (dark brown/black) and the accents go
    # lighter (near-white): organic transmission instead of paint.
    # raw luminance, NOT the dehazed one: subtracting the veil clips the dark fin zones to
    # zero and throws their ray structure away before any tone mapping can see it
    Y = cv2.GaussianBlur(luma(rgb), (0, 0), 0.6)
    excl = (1 - cv2.dilate(a_lion, np.ones((31, 31), np.uint8))).astype(np.float32)
    water = norm_blur(Y, excl, 40)
    r = (Y - water) / (water + 0.05)
    r = r + 0.5 * (r - cv2.GaussianBlur(r, (0, 0), 2.5))         # crisp rays, restrained
    Lbg = norm_blur(luma(bg), excl, 40)                          # graded water level here
    # base level (slow) + local contrast ratio (fast): the ray / band structure is shown
    # relative to the fin's own local brightness, so dark proximal fin zones keep their
    # cream-and-brown stripes instead of collapsing into one dark mass
    # equalise the LOCAL contrast of the fin zone (fixed window: no per-frame jitter) so the
    # faint rays of the dark proximal fin become readable; used for the detail signal only
    x0, y0, sz = 155, 850, 640
    mm = a_lion > 0.5
    ys = cv2.GaussianBlur(luma(rgb), (0, 0), 0.9)
    win = np.where(mm, ys, np.median(ys[mm]))[y0:y0 + sz, x0:x0 + sz]
    wlo, whi = np.percentile(win, [0.5, 99.5])
    w8 = (np.clip((win - wlo) / max(whi - wlo, 1e-3), 0, 1) * 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(w8).astype(np.float32) / 255
    Yc = np.full_like(Y, 0.5)
    Yc[y0:y0 + sz, x0:x0 + sz] = eq
    Yb = cv2.GaussianBlur(Y, (0, 0), 3.0)
    Ycb = cv2.GaussianBlur(Yc, (0, 0), 3.0)
    D = np.clip(np.log((Yc + 0.05) / (Ycb + 0.05)), -0.9, 0.9)
    rb = (Yb - water) / (water + 0.05)
    B = Lbg * (1.45 + np.where(rb < 0, 0.60 * rb, 2.3 * rb))      # lit presence: membrane a little lighter than the water
    L = B * np.exp(1.75 * D)
    L = np.where(L > 0.45, 0.45 + (L - 0.45) * 0.55, L)          # accents roll off softly
    L = np.clip(L, 0.055, 0.70)
    L = 0.70 * (L / 0.70) ** 1.22                                # deeper dark bands, crisp light ones
    L = np.clip(L, 0.045, 0.70)
    warm = np.array([1.20, 1.00, 0.70], np.float32); warm = warm / luma(warm)
    cool = np.array([1.00, 1.00, 0.95], np.float32); cool = cool / luma(cool)
    wt = smoothstep(0.17, 0.38, L)[..., None]
    col_t = np.clip((warm * (1 - wt) + cool * wt) * L[..., None], 0, 1)    # sepia shade -> near-white accents
    o = smoothstep(0.05, 0.28, np.maximum(np.abs(r), 0.7 * np.abs(D)))
    o = cv2.GaussianBlur((0.32 + 0.68 * o).astype(np.float32), (0, 0), 1.0)
    scene_nv = scene / np.maximum(vig, 0.3)[..., None]           # the vignette is applied once, below
    col = scene_nv * (1 - o[..., None]) + col_t * o[..., None]
    t = np.clip(L / 0.5, 0, 1)
    # --- submerge it: same water, same light -------------------------------------
    # 1) tame the pop: documentary-level saturation, luminance preserved
    ll = luma(col)[..., None]
    col = ll + (col - ll) * 1.0
    # 2) water absorbs red first and adds blue-green: transmission + veil in the
    #    colour of the surrounding water (stronger in the shade of the body)
    amb = norm_blur(bg, excl, 30)
    shade = (1 - t)[..., None]
    col = col * np.array([0.90, 0.96, 1.00], np.float32)
    v = 0.12 + 0.12 * shade
    col = col * (1 - v) + np.clip(amb * 1.25, 0, 1) * v
    # 3) the scene's own light falloff (vignette) and depth darkening
    yy = np.linspace(0, 1, col.shape[0], dtype=np.float32)[:, None, None]
    col = col * vig[..., None]                                  # exactly the scene's own falloff
    col = decontaminate(col, m, iters=3)
    # 4) edge wrap: the rim takes the water's colour and light, with a wide falloff
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    wrap = (np.exp(-d / 4.5) * 0.88)[..., None]
    col = col * (1 - wrap) + bg * wrap                         # the rim IS the water: no seam
    return np.clip(cv2.GaussianBlur(col, (0, 0), 0.9), 0, 1)    # match the footage's optical softness


def grade_fish(rgb, air, a_other, ids, state, bg, scene):
    """Other fish: graded with the scene's own look (same light/atmosphere), then
    opened up just enough to read as fish, with a little colour and definition.
    Brightness is tied to the water around each fish (smoothed per identity)."""
    out = np.zeros_like(rgb)
    l_s = luma(scene)
    for gid in np.unique(ids[a_other > 0.5]):
        if gid <= 1:
            continue
        m = (ids == gid) & (a_other > 0.3)
        if m.sum() < 30:
            continue
        mm = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        mm = mm if mm.sum() > 20 else m
        ring = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        ring &= ~(cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        ring_l = float(np.median(l_s[ring])) if ring.sum() > 50 else 0.2
        med = float(np.median(l_s[mm]))
        g = float(np.clip(0.55 * ring_l / max(med, 0.02), 0.9, 3.0))   # fish read as clear, natural silhouettes
        gp = state.get(("g", int(gid)))
        g = g if gp is None else 0.92 * gp + 0.08 * g
        state[("g", int(gid))] = g
        col = scene * g
        ll = luma(col)[..., None]
        col = ll + (col - ll) * 1.25                                 # keep the scene's colour, a little richer
        # species-neutral natural body tone: olive-silver, blended in so the fish
        # reads as a fish, not as a teal-tinted cut-out
        body = ll * np.array([1.05, 1.01, 0.90], np.float32)
        col = 0.50 * col + 0.50 * body
        det = l_s - cv2.GaussianBlur(l_s, (0, 0), 1.8)
        col = col * (1 + 0.8 * np.clip(det / (l_s + 0.04), -0.20, 0.20))[..., None]   # gentle definition
        col = np.clip(soft_knee(col, 0.60, 0.5), 0, 1)
        col = decontaminate(col, m, iters=2)
        out[m] = col[m]
    return out


def make_vignette(h, w, strength=0.50):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w * 0.52) / (w * 0.78)) ** 2 + ((yy - h * 0.60) / (h * 0.70)) ** 2)
    return (1 - strength * smoothstep(0.45, 1.20, d)).astype(np.float32)


def process(rgb, am, al, ids, air, veil, vig, noise, state, grain=0.010):
    lion_b = al > 0.5
    # SAM masks of small fish sit a pixel or two outside the body: pull them in
    oth_b = cv2.erode(((am > 0.5) & ~lion_b).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    a_all = snap_regions(lion_b | oth_b, rgb)
    a_lion = np.minimum(snap_regions(lion_b, rgb), a_all)
    a_oth = np.clip(a_all - a_lion, 0, 1)
    bg = grade_bg(rgb, air, a_all, vig)
    scene = scene_look(rgb, air, vig)                            # same look applied to the fish pixels
    lion = grade_lion(rgb, air, veil, a_lion, bg, scene, vig)
    fish = grade_fish(rgb, air, a_oth, ids, state, bg, scene)
    # foreground colour = weighted by the SAME soft masks that make the alpha, normalised:
    # never multiplied by a fractional alpha, so no dark rim at the boundary
    wl = cv2.GaussianBlur(a_lion, (0, 0), 1.3) + 1e-4
    wf = cv2.GaussianBlur(a_oth, (0, 0), 1.3) + 1e-4
    fg = (lion * wl[..., None] + fish * wf[..., None]) / (wl + wf)[..., None]
    # soft, sub-pixel edge: slight feather (colour is already decontaminated)
    af = np.clip(cv2.GaussianBlur(a_all, (0, 0), 1.3) * 1.10, 0, 1)
    # fin membrane is translucent: let some water show through toward the rim
    dl = cv2.distanceTransform((a_lion > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    edge_t = 0.55 + 0.45 * smoothstep(0.0, 7.0, dl)
    af = np.where(cv2.dilate(a_lion, np.ones((9, 9), np.uint8)) > 0.5, af * edge_t, af)
    out = bg * (1 - af[..., None]) + fg * af[..., None]
    out = out + (noise * grain)[..., None]                       # one grain over everything
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", nargs="+", type=Path, required=True)
    ap.add_argument("--crop-ys", nargs="+", type=int, required=True)
    ap.add_argument("--lion-masks", type=Path, default=None,
                    help="SAM 3.1 'lionfish' run on a tight crop (see --lion-crop)")
    ap.add_argument("--lion-crop", nargs=2, type=int, default=None, metavar=("X", "Y"))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--stills", default="")
    ap.add_argument("--stills-dir", type=Path, default=None)
    ap.add_argument("--veil", type=float, default=0.45)
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--rebuild-masks", action="store_true")
    a = ap.parse_args()

    info = probe(a.input)
    h, w = info.height, info.width
    mdir = a.workdir / "union_masks"
    if a.rebuild_masks or not (mdir / "done").exists():
        dirs, tags = list(a.masks), "AB"[:len(a.masks)]
        crops = {t: (0, cy) for t, cy in zip(tags, a.crop_ys)}
        if a.lion_masks:
            dirs.append(a.lion_masks)
            tags += "L"
            crops["L"] = tuple(a.lion_crop)
        build_union_masks(index_objects(dirs, tags), crops, h, w, mdir, a.min_frames)
        repair_masks(mdir, a.input, a.stride, h, w)
        (mdir / "done").write_text("1")
    store = MaskStore(mdir)

    samples = list(iter_frames(a.input, info, scale_w=270))
    air = estimate_airlight(samples[::40])
    print(f"[grade] airlight {air}", flush=True)
    vig = make_vignette(h, w)
    rng = np.random.default_rng(7)
    still_idx = {int(s) for s in a.stills.split(",") if s}
    enc = Encoder(a.input, a.output, info, Encode()) if a.output else None

    gray_proxy = {}
    for i, f in enumerate(iter_frames(a.input, info)):
        if i % a.stride == 0:
            gray_proxy[i // a.stride] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    state = {}
    for i, f in enumerate(iter_frames(a.input, info)):
        if not enc and i not in still_idx:
            continue
        g = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        am, al, ids = interp_frame(store, i, a.stride, g, gray_proxy, h, w)
        noise = cv2.resize(rng.standard_normal((h // 2, w // 2)).astype(np.float32), (w, h))
        out = process(f, am, al, ids, air, a.veil, vig, noise, state)
        if enc:
            enc.write(out)
        if i in still_idx:
            d = a.stills_dir or a.output.parent
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"still_{i:05d}.jpg"),
                        cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(d / f"src_{i:05d}.jpg"),
                        cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 20 == 0:
            print(f"[grade] frame {i}/{info.n_frames}", flush=True)
    if enc:
        enc.close()


if __name__ == "__main__":
    main()
