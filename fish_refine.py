#!/usr/bin/env python3
"""SAM 3.1 keyframe masks -> clean per-frame mattes for the grade.

Input  (from fish_masks.py):  masks/<stem>__fish/obj_*/  and  masks/<stem>__sand/obj_*/
Output (full res, 8 bit):     masks/<stem>__mattes/{fish,seabed,midground,background}/<frame:05d>.png
                              masks/<stem>__mattes/report.json   (areas, repairs, gaps)
                              masks/<stem>__mattes/overlay.mp4   (QC preview, 540x960)

What it does
  fish        SAM keyframes (stride 3) -> area-outlier repair -> optical-flow
              (DIS) warped soft interpolation of the in-between frames -> guided
              filter against the frame's own luma so the edge snaps to real
              fin/blade edges (a blade in front of the fish stays a hole).
  seabed      SAM "sandy bottom" keyframes (stride 6) -> flow interpolation ->
              hole filling (debris / blades lying on the sand count as seabed)
              -> temporal smoothing -> wide feather.
  midground   derived, soft: a contact band just above the sand edge (immediate
              seagrass blades, stones, debris) that fades out with distance.
  background  complement: 1 - fish - max(seabed, midground)  (upper rock, dense
              weed, open water). SAM 3.1 returns no detections for seagrass /
              rock / weed text prompts on this clip, so these two are derived
              from the SAM seabed + fish masks instead of segmented directly.

    python fish_refine.py --input inputs/fish.mov
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from fish_color import gblur, guided, luma

REPO = Path(__file__).resolve().parent


# ----------------------------------------------------------------- decoding
def probe(video: Path):
    r = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(video)])
    s = json.loads(r)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_read_frames"]), s["r_frame_rate"]


def decode_source(video: Path) -> np.memmap:
    """Whole clip as uint8 RGB in a memory-mapped cache file (BT.709, tv -> full range)."""
    w, h, n, _ = probe(video)
    f = REPO / ".cache" / f"{video.stem}_src_rgb24.u8"
    if not f.exists() or f.stat().st_size != n * h * w * 3:
        f.parent.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-fps_mode", "passthrough",
                        "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24",
                        "-f", "rawvideo", str(f)], check=True)
    return np.memmap(f, np.uint8, "r", shape=(n, h, w, 3))


# ------------------------------------------------------------- mask loading
def load_objects(mdir: Path):
    """{gid: {frame: Path}} + meta."""
    meta = json.loads((mdir / "meta.json").read_text())
    objs = {int(g): {int(p.stem): p for p in (mdir / f"obj_{g}").glob("*.png")} for g in meta["objects"]}
    return objs, meta


def union_frames(objs, gids, frames):
    """{frame: bool mask} = union of the chosen objects."""
    out = {}
    for f in frames:
        m = None
        for g in gids:
            p = objs[g].get(f)
            if p is not None:
                x = cv2.imread(str(p), 0) > 127
                m = x if m is None else (m | x)
        if m is not None:
            out[f] = m
    return out


# ------------------------------------------------------------ repair / flow
def shift(m, dx, dy):
    H, W = m.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)


def repair_outliers(masks: dict[int, np.ndarray], tol=0.30, log=print):
    """A keyframe whose area deviates from the median of its neighbours by more
    than `tol` is a SAM failure (mask bloating into the water or collapsing onto
    one stripe). Rebuild it from the nearest good keyframes, each shifted to the
    interpolated centroid, and report it."""
    idx = sorted(masks)
    area = {i: int(masks[i].sum()) for i in idx}
    bad = set()
    for k, i in enumerate(idx):
        nb = [area[j] for j in idx[max(k - 3, 0):k + 4] if j != i and area[j] > 0]
        if nb and abs(area[i] / np.median(nb) - 1) > tol:
            bad.add(i)
    fixed = {}
    for i in sorted(bad):
        prev = max((j for j in idx if j < i and j not in bad), default=None)
        nxt = min((j for j in idx if j > i and j not in bad), default=None)
        src = [j for j in (prev, nxt) if j is not None]
        if not src:
            continue
        cs = {j: np.argwhere(masks[j]).mean(0) for j in src if masks[j].any()}
        if not cs:
            continue
        if len(cs) == 2:
            w = (i - prev) / (nxt - prev)
            tgt = cs[prev] * (1 - w) + cs[nxt] * w
            acc = sum(shift(masks[j], *(tgt - cs[j])[::-1]) * (1 - abs(i - j) / (nxt - prev)) for j in cs)
        else:
            (j, c), = cs.items()
            acc = masks[j].astype(np.float32)
        new = acc > 0.5
        log(f"  repaired keyframe {i}: area {area[i]} -> {int(new.sum())}")
        fixed[i] = (area[i], int(new.sum()))
        masks[i] = new
    return fixed


_dis = None


def flow_between(g0, g1):
    """Dense flow with I0(x) ~= I1(x + flow(x))."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(np.ascontiguousarray(g0), np.ascontiguousarray(g1), None)


def warp_back(mask_f32, flow):
    """mask_t(x) = mask_k(x + flow_t->k(x)): pull a keyframe mask into frame t."""
    H, W = mask_f32.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(mask_f32, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def clean_keyframe(m: np.ndarray, min_frac=0.03):
    """Morphological clean-up of a fish keyframe mask: fill small holes, cut thin spurs (stems touching the
    body) and drop tiny detached fragments. Fins are thick, stems / twigs are thin, so an opening whose
    radius scales with the body keeps the former and removes the latter."""
    m8 = m.astype(np.uint8)
    if not m8.any():
        return m
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    r = max(3, int(0.045 * dist.max()))
    opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    keep = m8 & cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # small holes (particles, speckle) are not gaps between head and body: fill the ones under 150 px
    inv = (1 - keep).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = keep.shape
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 150 and not (x == 0 or y == 0 or x + w == W or y + h == H):
            keep[lab == i] = 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    if n > 2:
        big = st[1:, 4].max()
        for i in range(1, n):
            if st[i, 4] < min_frac * big:
                keep[lab == i] = 0
    return keep > 0


def temporal_median_keys(keys, src, pad=60):
    """Flow-aligned majority vote over each keyframe and its two neighbouring keyframes: a fin / stem bit that is
    present or missing in a single keyframe only (intermittent detection) is voted out; anything seen on two
    consecutive keyframes is kept. Returns the new dict and the number of changed pixels."""
    kf = sorted(keys)
    gray = lambda t: cv2.cvtColor(np.ascontiguousarray(src[t]), cv2.COLOR_RGB2GRAY)
    out = dict(keys)
    changed = 0
    H, W = keys[kf[0]].shape
    for i in range(1, len(kf) - 1):
        k, p_, n_ = kf[i], kf[i - 1], kf[i + 1]
        ys, xs = np.nonzero(keys[k] | keys[p_] | keys[n_])
        if not len(ys):
            continue
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
        gk = gray(k)[y0:y1, x0:x1]
        votes = keys[k][y0:y1, x0:x1].astype(np.float32)
        for nb in (p_, n_):
            fl = flow_between(gk, gray(nb)[y0:y1, x0:x1])
            votes += warp_back(keys[nb][y0:y1, x0:x1].astype(np.float32), fl)
        new = keys[k].copy()
        new[y0:y1, x0:x1] = votes >= 1.5
        changed += int((new != keys[k]).sum())
        out[k] = new
    return out, changed


class SoftInterp:
    """Soft per-frame masks (float32 0..1) from keyframe masks, computed on demand.

    scale   flow resolution (fraction of the output resolution)
    roi_pad if set, flow and warping run in the bounding box of the two keyframe masks (+pad) only
            (fish); otherwise on the whole frame (seabed)
    out     output scale: 1.0 = full resolution, 0.25 = quarter resolution (seabed; saves memory)"""

    def __init__(self, keys, src, scale, roi_pad, out=1.0):
        self.src, self.scale, self.roi_pad, self.out = src, scale, roi_pad, out
        self.H, self.W = src.shape[1:3]
        self.oh, self.ow = int(round(self.H * out)), int(round(self.W * out))
        rs = (lambda m: cv2.resize(m.astype(np.float32), (self.ow, self.oh), interpolation=cv2.INTER_AREA)) \
            if out != 1.0 else (lambda m: m.astype(np.float32))
        self.keys = {k: rs(v) for k, v in keys.items()}
        self.kf = sorted(self.keys)
        self._gray = {}
        self._cache = {}

    def gray(self, t):
        g = self._gray.get(t)
        if g is None:
            g = cv2.cvtColor(np.ascontiguousarray(self.src[t]), cv2.COLOR_RGB2GRAY)
            if self.out != 1.0:
                g = cv2.resize(g, (self.ow, self.oh), interpolation=cv2.INTER_AREA)
            if len(self._gray) > 12:
                self._gray.pop(next(iter(self._gray)))
            self._gray[t] = g
        return g

    def __call__(self, t):
        if t in self._cache:
            return self._cache[t]
        m = self._compute(t)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        self._cache[t] = m
        return m

    def _compute(self, t):
        if t in self.keys:
            return self.keys[t]
        a = max((k for k in self.kf if k < t), default=None)
        b = min((k for k in self.kf if k > t), default=None)
        if a is None or b is None:                       # before the first / after the last key: hold
            return self.keys[a if a is not None else b]
        wb = (t - a) / (b - a)
        H, W = self.oh, self.ow
        if self.roi_pad is not None:
            ys, xs = np.nonzero((self.keys[a] > 0.5) | (self.keys[b] > 0.5))
            if not len(ys):
                return np.zeros((H, W), np.float32)
            y0, y1 = max(ys.min() - self.roi_pad, 0), min(ys.max() + self.roi_pad + 1, H)
            x0, x1 = max(xs.min() - self.roi_pad, 0), min(xs.max() + self.roi_pad + 1, W)
        else:
            y0, y1, x0, x1 = 0, H, 0, W
        acc = np.zeros((y1 - y0, x1 - x0), np.float32)
        gt = self.gray(t)[y0:y1, x0:x1]
        for k, wk in ((a, 1 - wb), (b, wb)):
            gk = self.gray(k)[y0:y1, x0:x1]
            if self.scale != 1:
                sz = (max(int((x1 - x0) * self.scale), 8), max(int((y1 - y0) * self.scale), 8))
                fl = flow_between(cv2.resize(gt, sz, interpolation=cv2.INTER_AREA),
                                  cv2.resize(gk, sz, interpolation=cv2.INTER_AREA))
                fl = cv2.resize(fl, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) / self.scale
            else:
                fl = flow_between(gt, gk)
            acc += wk * warp_back(self.keys[k][y0:y1, x0:x1], fl)
        m = np.zeros((H, W), np.float32)
        m[y0:y1, x0:x1] = acc
        return m


def temporal_fish(S: SoftInterp, t: int, n: int, pad=40, wide=False):
    """Temporal filter of the soft fish mask with the neighbours warped onto frame t by optical flow (a moving
    silhouette is not smeared). 1-2-1 normally; 1-4-6-4-1 (+-2 frames) where `wide` (the narrow turning
    fish, frames ~140-235), which kills popping and drift there at the cost of a little lag."""
    cur = S(t)
    taps = {1: 0.25, 2: 0.0} if not wide else {1: 4 / 16, 2: 1 / 16}
    acc = (0.5 if not wide else 6 / 16) * cur
    for off, wgt in taps.items():
        if wgt == 0:
            continue
        for nb in (t - off, t + off):
            if not 0 <= nb < n:
                acc += wgt * cur
                continue
            m = S(nb)
            ys, xs = np.nonzero((cur > 0.05) | (m > 0.05))
            if not len(ys):
                continue
            H, W = cur.shape
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
            fl = flow_between(S.gray(t)[y0:y1, x0:x1], S.gray(nb)[y0:y1, x0:x1])
            acc[y0:y1, x0:x1] += wgt * warp_back(m[y0:y1, x0:x1], fl)
    return acc


# ------------------------------------------------------------ region logic
def fish_alpha(frame_rgb, soft, feather=0.8, rad=4, choke=0.0, edge_shift=0.0):
    """Soft interpolated mask -> edge-aware alpha aligned to the frame."""
    ys, xs = np.nonzero(soft > 0.05)
    if not len(ys):
        return np.zeros_like(soft)
    H, W = soft.shape
    y0, y1 = max(ys.min() - 24, 0), min(ys.max() + 25, H)
    x0, x1 = max(xs.min() - 24, 0), min(xs.max() + 25, W)
    L = luma(frame_rgb[y0:y1, x0:x1].astype(np.float32) / 255)
    sub = np.clip(soft[y0:y1, x0:x1], 0, 1)
    # sharpen the interpolated soft mask a little before snapping it to edges
    sub = np.clip((sub - (0.5 + edge_shift)) * 2.2 + 0.5, 0, 1)
    a = np.clip(guided(L, sub, rad, 1e-4), 0, 1)
    support = cv2.dilate((soft[y0:y1, x0:x1] > 0.15).astype(np.uint8), np.ones((3, 3), np.uint8))
    a = gblur(a * support, feather)
    if choke > 0:                                  # slight negative choke: no background / stem spill
        a = np.clip((a - choke) / (1 - choke), 0, 1)
    out = np.zeros_like(soft)
    out[y0:y1, x0:x1] = a
    return out


def fill_small_holes(m: np.ndarray, max_frac=0.05):
    """Fill holes (debris, blades, the fish itself) that do not touch the border."""
    inv = (~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = m.shape
    out = m.copy()
    for i in range(1, n):
        x, y, w, h, a = st[i]
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if not touches and a < max_frac * H * W:
            out[lab == i] = True
    return out


def temporal_smooth(seq: np.ndarray, passes=2):
    """[T,H,W] float: binomial 1-2-1 along time, edge frames repeated."""
    s = seq
    for _ in range(passes):
        p = np.concatenate([s[:1], s, s[-1:]], 0)
        s = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    return s


def write_png(path: Path, a: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--fish-gids", default=None, help="comma list of SAM object ids that are the fish (default: auto)")
    ap.add_argument("--sand-gids", default=None)
    ap.add_argument("--midground-px", type=float, default=0.075, help="contact band height / frame height")
    ap.add_argument("--feather-sand", type=float, default=7.0)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--touch-px", type=float, default=25.0, help="a part is merged if it comes this close to the fish mask")
    ap.add_argument("--refit", action="store_true", help="use masks/<stem>__fishrefit (fish_resegment.py)")
    ap.add_argument("--tmedian", action="store_true", help="flow-aligned temporal majority vote over keyframes")
    ap.add_argument("--clean", action="store_true", help="spur / hole / fragment clean-up of the fish keyframes")
    ap.add_argument("--choke", type=float, default=0.0, help="negative choke of the final alpha (0-0.4)")
    ap.add_argument("--edge-shift", type=float, default=0.0, help="moves the matte edge inward before edge snapping (0-0.3)")
    ap.add_argument("--wide", type=int, nargs=2, default=[-1, -1], metavar=("A", "B"),
                    help="frame range that gets the 5-tap temporal window")
    ap.add_argument("--parts", default="", help="comma list of extra SAM parts merged into the fish (e.g. fins)")
    ap.add_argument("--only-fish", action="store_true", help="skip seabed/midground/background + preview")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N frames (development)")
    a = ap.parse_args()
    stem = a.input.stem
    src = decode_source(a.input)
    n, H, W, _ = src.shape
    if a.limit:
        n = min(n, a.limit)
    out_root = REPO / "masks" / f"{stem}__mattes"
    report = {"frames": n, "size": [W, H]}

    # ------------------------------------------------------------- fish
    fobjs, fmeta = load_objects(REPO / "masks" / f"{stem}__fish")
    gids = [int(x) for x in a.fish_gids.split(",")] if a.fish_gids else \
        [g for g, v in fobjs.items() if len(v) >= a.min_frames]
    print(f"[refine] fish objects used: {gids} "
          f"({ {g: len(fobjs[g]) for g in fobjs} } frames each)")
    keys = union_frames(fobjs, gids, sorted({f for g in gids for f in fobjs[g]}))
    # extra SAM parts that belong to the fish (fins): kept only where they touch the fish silhouette
    for part in [x for x in a.parts.split(",") if x]:
        pdir = REPO / "masks" / f"{stem}__{part}"
        if not (pdir / "meta.json").exists():
            print(f"[refine] part '{part}' not found, skipped")
            continue
        pobjs, _ = load_objects(pdir)
        added = 0
        for f in sorted({f for g in pobjs for f in pobjs[g]}):
            if f not in keys:
                continue
            # a fin is attached to the body by a short edge, so the test is contact distance, not the share
            # of its area near the silhouette
            dist = cv2.distanceTransform((~keys[f]).astype(np.uint8), cv2.DIST_L2, 3)
            for g in pobjs:
                p = pobjs[g].get(f)
                if p is None:
                    continue
                m = cv2.imread(str(p), 0) > 127
                if m.sum() > 80 and dist[m].min() <= a.touch_px and m.sum() < 0.6 * keys[f].sum():
                    keys[f] = keys[f] | m
                    added += 1
            keys[f] = cv2.morphologyEx(keys[f].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        print(f"[refine] part '{part}': merged {added} instances into the fish keyframes")
    # targeted SAM 3.1 re-segmentation (fish_resegment.py) used as EVIDENCE, not as a replacement: point-prompted SAM
    # under-segments the belly on some frames, so only large parts of the keyframe mask that SAM's refit rejects by
    # more than 10 px (stem / foliage grabs) are removed; edges everywhere else stay as they are
    rf = REPO / "masks" / f"{stem}__fishrefit"
    if a.refit and (rf / "meta.json").exists():
        rmeta = json.loads((rf / "meta.json").read_text())
        removed = 0
        for f, v in rmeta.items():
            f = int(f)
            if not v["accepted"] or f not in keys:
                continue
            new = cv2.imread(str(rf / "obj_1" / f"{f:05d}.png"), 0) > 127
            near = cv2.dilate(new.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0
            spill = keys[f] & ~near
            n_, lab, st, _ = cv2.connectedComponentsWithStats(spill.astype(np.uint8), connectivity=8)
            for i in range(1, n_):
                if 30 < st[i, 4] < 0.10 * keys[f].sum():
                    keys[f][lab == i] = False
                    removed += int(st[i, 4])
        report["refit_spill_px_removed"] = removed
        print(f"[refine] SAM re-segmentation evidence: {removed} spill px removed over {len(rmeta)} keyframes")
    if a.clean:
        before = {f: int(m.sum()) for f, m in keys.items()}
        for f in list(keys):
            keys[f] = clean_keyframe(keys[f])
        report["clean_area_change_pct"] = round(float(np.mean([(int(keys[f].sum()) - before[f]) / max(before[f], 1) for f in keys]) * 100), 2)
    if a.tmedian:
        keys, ch = temporal_median_keys(keys, src)
        keys = {f: clean_keyframe(m) for f, m in keys.items()} if a.clean else keys
        report["tmedian_changed_px"] = ch
        print(f"[refine] temporal median over keyframes changed {ch} px")
    report["fish_keyframes"] = len(keys)
    report["fish_repairs"] = repair_outliers(keys)
    S = SoftInterp(keys, src, scale=1.0, roi_pad=60)
    for t in range(n):
        alpha = fish_alpha(src[t], temporal_fish(S, t, n, wide=(a.wide[0] <= t <= a.wide[1])), choke=a.choke, edge_shift=a.edge_shift)
        write_png(out_root / "fish" / f"{t:05d}.png", alpha)
        if t % 60 == 0:
            print(f"[refine] fish matte {t}/{n}", flush=True)
    area = np.array([float((cv2.imread(str(out_root / "fish" / f"{t:05d}.png"), 0) / 255.0).sum()) for t in range(n)])
    report["fish_area"] = [int(x) for x in area]
    print(f"[refine] fish mattes done; area px min/median/max = {area.min():.0f}/{np.median(area):.0f}/{area.max():.0f}")
    if a.only_fish:
        (out_root / "report.json").write_text(json.dumps(report, indent=1))
        return 0

    # ------------------------------------------------------------- seabed
    sobjs, smeta = load_objects(REPO / "masks" / f"{stem}__sand")
    sg = [int(x) for x in a.sand_gids.split(",")] if a.sand_gids else \
        [g for g, v in sobjs.items() if len(v) >= a.min_frames]
    print(f"[refine] seabed objects used: {sg}")
    skeys = union_frames(sobjs, sg, sorted({f for g in sg for f in sobjs[g]}))
    report["sand_repairs"] = repair_outliers(skeys, tol=0.35)
    SS = SoftInterp(skeys, src, scale=1.0, roi_pad=None, out=0.25)      # quarter-res flow + masks
    seq = temporal_smooth(np.stack([SS(t) for t in range(n)]), passes=2)
    band_px = a.midground_px * H
    report["sand_keyframes"] = len(skeys)

    for t in range(n):
        up = cv2.resize(seq[t], (W, H), interpolation=cv2.INTER_LINEAR)
        m = up > 0.5
        m = fill_small_holes(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                              np.ones((9, 9), np.uint8)) > 0)
        seabed = gblur(m.astype(np.float32), a.feather_sand)
        # contact band above the sand edge: distance (px) to the nearest seabed pixel
        dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
        band = np.clip(1 - dist / band_px, 0, 1) ** 1.6
        band = np.where(m, 0, band)
        midground = gblur(band.astype(np.float32), 10)
        fg = np.maximum(seabed, midground)
        alpha = cv2.imread(str(out_root / "fish" / f"{t:05d}.png"), 0).astype(np.float32) / 255
        bg = np.clip((1 - alpha) * (1 - fg), 0, 1)
        write_png(out_root / "seabed" / f"{t:05d}.png", seabed)
        write_png(out_root / "midground" / f"{t:05d}.png", midground)
        write_png(out_root / "background" / f"{t:05d}.png", bg)
        if t % 60 == 0:
            print(f"[refine] regions frame {t}", flush=True)

    # ------------------------------------------------------------- QC preview
    ov = out_root / "overlay.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "540x960",
                            "-framerate", "30000/1001", "-i", "-", "-c:v", "libx264", "-crf", "24",
                            "-pix_fmt", "yuv420p", str(ov)], stdin=subprocess.PIPE)
    for t in range(n):
        im = src[t][..., ::-1].astype(np.float32)
        r = lambda nm: cv2.imread(str(out_root / nm / f"{t:05d}.png"), 0).astype(np.float32)[..., None] / 255
        col = {"seabed": (255, 160, 0), "midground": (0, 255, 255), "background": (255, 0, 160), "fish": (0, 0, 255)}
        for nm in ("background", "seabed", "midground", "fish"):
            im = im * (1 - 0.45 * r(nm)) + 0.45 * r(nm) * np.array(col[nm], np.float32)
        enc.stdin.write(cv2.resize(np.clip(im, 0, 255).astype(np.uint8), (540, 960)).tobytes())
    enc.stdin.close()
    enc.wait()
    (out_root / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[refine] done -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
