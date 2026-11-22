"""From SAM 3.1 key-frame detections to per-frame soft region masks.

1. assemble(): SAM text-prompt masks -> disjoint grading regions (sky, rock, veg, water, seabed).
   The waterline is the boundary of SAM's 'water' mask (never a fixed horizontal line).
2. propagate(): masks of neighbouring key frames are carried to every frame with dense optical
   flow (DIS) frame-to-frame, forwards from the earlier key and backwards from the later one,
   then cross-faded. Disagreement between the two carries is the drift metric.
3. refine(): edge-aware (guided-filter) snapping to the actual image edges at full resolution."""
from __future__ import annotations
import cv2
import numpy as np

CH = ["sky", "rock", "veg", "water", "seabed"]
SCALE = 0.5


def _open_close(m, k_open=0, k_close=0):
    if k_close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    if k_open:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    return m


def _fill_holes(m, max_area):
    inv = (1 - m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m.copy()
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a <= max_area and x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0]:
            out[lab == i] = 1
    return out


def _drop_small(m, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, 4] >= min_area:
            keep[lab == i] = 1
    return keep


def assemble(frame_rgb: np.ndarray, sam: dict) -> np.ndarray:
    """-> (H,W,5) uint8 {0,1} in CH order at full resolution."""
    H, W = frame_rgb.shape[:2]
    L = frame_rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    g = lambda k: sam[k].astype(np.uint8)
    S = _fill_holes(_drop_small(g("sky"), 4000), 60000)
    # sun disc is not 'sky' for SAM: close it, but only over bright pixels
    Sc = _open_close(S, 0, 81)
    S = np.maximum(S, (Sc & (L > 0.78)).astype(np.uint8))
    sky_frac = S.mean()
    Wt = _fill_holes(_drop_small(g("water"), 6000), 20000)
    V = _drop_small(g("green vegetation"), 1500) & (1 - S)
    Rk = np.maximum(g("rock cliff"), g("rock"))
    if sky_frac < 0.03:                                  # fully underwater frame
        # water column everywhere; seabed = rock detections in the lower part of the frame
        lower = np.zeros_like(Wt); lower[int(0.55 * H):] = 1
        Sb = Rk & lower
        Wc = (1 - Sb).astype(np.uint8)
        out = np.stack([np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), Wc, Sb], -1)
        return out.astype(np.uint8)
    Wt = Wt & (1 - S)
    V = V & (1 - Wt)
    Sb = Rk & Wt
    # everything above the waterline that is neither sky nor vegetation is rock / cliff
    R = (1 - S) & (1 - V) & (1 - Wt)
    R = _drop_small(R, 3000)
    Wt = Wt & (1 - Sb)
    return np.stack([S, R, V, Wt, Sb], -1).astype(np.uint8)


def _small(frame_rgb):
    g = cv2.cvtColor((np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def _warp(m, flow):
    h, w = m.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(m, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def propagate(grays, keymasks: dict[int, np.ndarray]):
    """grays: list of half-res uint8 frames. keymasks: {frame: (h,w,5) float32 at half-res}.
    Returns (N,h,w,5) float16 and per-frame drift (1-IoU between fwd and bwd carries)."""
    n = len(grays)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    bf = [None] * n; ff = [None] * n
    for k in range(1, n):
        bf[k] = dis.calc(grays[k], grays[k - 1], None)       # k -> k-1 : carries masks forward
    for k in range(n - 1):
        ff[k] = dis.calc(grays[k], grays[k + 1], None)       # k -> k+1 : carries masks backward
    keys = sorted(keymasks)
    out = np.zeros((n,) + keymasks[keys[0]].shape, np.float16)
    drift = np.zeros(n, np.float32)
    for a, b in zip(keys[:-1], keys[1:]):
        fwd = {a: keymasks[a]}
        for j in range(a + 1, b):
            fwd[j] = _warp(fwd[j - 1], bf[j])
        bwd = {b: keymasks[b]}
        for j in range(b - 1, a, -1):
            bwd[j] = _warp(bwd[j + 1], ff[j])
        out[a] = keymasks[a]
        for j in range(a + 1, b):
            lam = (j - a) / (b - a)
            out[j] = (1 - lam) * fwd[j] + lam * bwd[j]
            x, y = fwd[j] > 0.5, bwd[j] > 0.5
            u = (x | y).sum()
            drift[j] = 1 - (x & y).sum() / u if u else 0
    out[keys[-1]] = keymasks[keys[-1]]
    for k in range(0, keys[0]):
        out[k] = keymasks[keys[0]]
    for k in range(keys[-1] + 1, n):
        out[k] = keymasks[keys[-1]]
    return out, drift


def _guided(I, p, r, eps):
    ks = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.boxFilter(I, -1, ks), cv2.boxFilter(p, -1, ks)
    cov = cv2.boxFilter(I * p, -1, ks) - mI * mp
    var = cv2.boxFilter(I * I, -1, ks) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return cv2.boxFilter(a, -1, ks) * I + cv2.boxFilter(b, -1, ks)


def refine(masks_small: np.ndarray, frame_rgb: np.ndarray, feather: float = 1.2):
    """(h,w,5) half-res soft masks + full-res frame -> dict of full-res soft masks, sum <= 1."""
    H, W = frame_rgb.shape[:2]
    guide = cv2.cvtColor((np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    out = {}
    for i, k in enumerate(CH):
        up = cv2.resize(masks_small[..., i].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)
        up = cv2.GaussianBlur(up, (0, 0), 2.0)
        r = np.clip(_guided(guide, up, 6, 2e-3), 0, 1)
        out[k] = cv2.GaussianBlur(r, (0, 0), feather)
    tot = sum(out.values())
    s = np.maximum(tot, 1.0)
    return {k: v / s for k, v in out.items()}
