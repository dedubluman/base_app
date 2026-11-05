"""Matte refinement on top of the SAM 3.1 masks.

snap_diver_alpha  - edge-aware diver alpha: a guided filter on the frame's
                    luminance snaps the transition to the real body edge, and
                    the result is choked into a band around the SAM 3.1
                    boundary (at most +1 px outside, solid from -4 px inside),
                    so no feather or halo reaches the water or bubbles.
clean_diver_edge  - drops pixels near the SAM 3.1 diver boundary that match
                    the surrounding water/bubbles much better than any of the
                    diver's own classes (skin, hair, shorts, fins) on this
                    frame: bubble clouds hugging the head or body near the
                    surface no longer get the skin/hair grade.
bubbles            (from refine_parts) - bubbles in front of the head or fins,
                    taken out of the hair/fins masks because they match the
                    frame's bubble colours, are graded neutral, not as skin.
                    A general bubble class over the whole body was tried and
                    dropped: thin caustic highlights on sunlit skin look the
                    same to colour and top-hat tests.
refine_parts      - checks every SAM 3.1 part mask (shorts, hair, fins)
                    against colour models built on this frame: pixels inside
                    a part mask that look like the frame's skin or the
                    surrounding water are given back to that class. Fixes
                    tracker drift (e.g. the shorts mask sliding onto a thigh)
                    and parts bleeding into bubbles.
"""
from __future__ import annotations

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _ellipse(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, grey guide."""
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    cov = cv2.boxFilter(I * p, -1, k) - mI * mp
    var = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def snap_diver_alpha(raw: np.ndarray, x: np.ndarray) -> np.ndarray:
    """raw: bool HxW SAM 3.1 diver mask; x: float RGB 0..1. Returns float
    alpha 0..1 whose edge follows the image and never leaves the band
    [SAM boundary - 4 px, SAM boundary + 1 px]."""
    m = raw.astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return np.zeros(raw.shape, np.float32)
    small = area < 4000
    inner = cv2.erode(m, _ellipse(2 if small else 4)).astype(np.float32)
    outer = cv2.dilate(m, _ellipse(1)).astype(np.float32)
    I = (x @ LUMA).astype(np.float32)
    p = cv2.erode(m, _ellipse(1)).astype(np.float32)
    g = np.clip(guided_filter(I, p, 2 if small else 4, 1e-3), 0, 1)
    a = np.minimum(np.maximum(g, inner), outer)
    # a last sub-pixel smoothing, kept inside the band
    a = np.minimum(cv2.GaussianBlur(a, (0, 0), 0.7), outer)
    return a.astype(np.float32)


def _gauss_model(feat: np.ndarray, n_max: int = 6000):
    if len(feat) > n_max:
        feat = feat[np.random.default_rng(0).choice(len(feat), n_max, replace=False)]
    mu = feat.mean(0)
    cov = np.cov(feat.T) + np.eye(feat.shape[1]) * 4.0   # regularised (Lab units)
    inv = np.linalg.inv(cov)
    logdet = float(np.linalg.slogdet(cov)[1])
    return mu, inv, logdet


def _loglik(feat: np.ndarray, model) -> np.ndarray:
    mu, inv, logdet = model
    d = feat - mu
    return -0.5 * (np.einsum("ij,jk,ik->i", d, inv, d) + logdet)


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """Fill enclosed holes of a bool mask."""
    m8 = m.astype(np.uint8)
    h, w = m8.shape
    flood = np.pad(m8, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return m | holes


def _surround_models(lab: np.ndarray, m8: np.ndarray) -> dict:
    """Colour models of what surrounds the diver on this frame: open water,
    and bubbles (the brightest 15% of the surrounding ring, where the bubble
    trail usually continues outside the mask)."""
    ring = (cv2.dilate(m8, _ellipse(35)) > 0) & ~(cv2.dilate(m8, _ellipse(3)) > 0)
    out = {}
    if ring.sum() < 300:
        return out
    L = lab[..., 0]
    hi = np.percentile(L[ring], 85)
    water = ring & (L < hi)
    bubbles = ring & (L >= hi)
    if water.sum() > 300:
        out["water"] = _gauss_model(lab[water])
    if bubbles.sum() > 200:
        out["bubbles"] = _gauss_model(lab[bubbles])
    return out


def refine_parts(x: np.ndarray, diver: np.ndarray, parts: dict) -> tuple[dict, dict]:
    """x: float RGB 0..1; diver: bool HxW; parts: {name: bool HxW}.
    Returns ({name: bool HxW refined}, {name: fraction of pixels removed}).
    Pixels taken out of the hair or fins mask because they match this
    frame's bubbles (bubbles in front of the head or the fins) come back as
    an extra "bubbles" mask, graded neutral instead of as skin."""
    if not parts:
        return parts, {}
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2Lab)
    feat_all = lab * np.array([0.6, 1.0, 1.0], np.float32)  # shading matters less
    d8 = diver.astype(np.uint8)
    any_part = np.zeros(diver.shape, np.uint8)
    for m in parts.values():
        any_part |= m.astype(np.uint8)
    skin_core = (cv2.erode(d8, _ellipse(4)) > 0) & ~(cv2.dilate(any_part, _ellipse(4)) > 0)
    models = {}
    if skin_core.sum() > 300:
        models["skin"] = _gauss_model(feat_all[skin_core])
    models.update(_surround_models(feat_all, d8))
    out, removed = {}, {}
    bubbles = np.zeros(diver.shape, bool)
    for name, m in parts.items():
        m8 = m.astype(np.uint8)
        area = int(m8.sum())
        core = cv2.erode(m8, _ellipse(3 if area > 3000 else 1)) > 0
        if area < 80 or core.sum() < 60 or not models:
            out[name], removed[name] = m, 0.0
            continue
        if name == "hair":   # hair masks can hold bubbles: learn its darker half
            core &= feat_all[..., 0] <= np.median(feat_all[..., 0][core])
        # white fabric looks like bubbles, so the shorts are only tested
        # against skin (that is the drift we fix); other parts against all
        rivals = {k: v for k, v in models.items() if name != "shorts" or k == "skin"}
        if not rivals:
            out[name], removed[name] = m, 0.0
            continue
        own = _gauss_model(feat_all[core])
        idx = np.nonzero(m)
        f = feat_all[idx]
        ll_own = _loglik(f, own)
        ll_other = np.max(np.stack([_loglik(f, md) for md in rivals.values()]), 0)
        keep = np.zeros_like(m)
        keep[idx] = ll_own >= ll_other - 1.0      # ties stay with the part
        if name in ("hair", "fins") and "bubbles" in rivals:
            names = list(rivals)
            win = np.argmax(np.stack([_loglik(f, rivals[k]) for k in names]), 0)
            bub = np.zeros_like(m)
            bub[idx] = (~keep[idx]) & (win == names.index("bubbles"))
            bubbles |= bub
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(1))
        # keep only the pieces that hold the part's confident core, then
        # close small holes inside the original SAM 3.1 mask
        n, lab_cc, _, _ = cv2.connectedComponentsWithStats(keep, 8)
        good = np.unique(lab_cc[core & (lab_cc > 0)])
        keep = np.isin(lab_cc, good[good > 0])
        keep = (cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(2)) > 0) & m
        keep = _fill_holes(keep) & m   # fabric folds/highlights stay fabric
        out[name] = keep
        removed[name] = round(1.0 - keep.sum() / max(1, area), 3)
    bubbles = cv2.morphologyEx(bubbles.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    if bubbles.any():
        out["bubbles"] = bubbles
    return out, removed


def clean_diver_edge(x: np.ndarray, raw: np.ndarray, parts: dict,
                     band_px: int = 30, margin: float = 3.0) -> tuple[np.ndarray, float]:
    """raw: bool HxW diver mask; parts: {name: bool}. Returns (cleaned mask,
    fraction of the mask removed). Only pixels within band_px of the
    boundary can go, and only when water beats every body class by `margin`
    in log-likelihood (conservative: lit skin is kept)."""
    m8 = raw.astype(np.uint8)
    area = int(m8.sum())
    if area < 1500:
        return raw, 0.0
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2Lab) * np.array([0.6, 1.0, 1.0], np.float32)
    inner = cv2.erode(m8, _ellipse(max(3, band_px // 3))) > 0
    any_part = np.zeros_like(m8)
    for pm in parts.values():
        any_part |= pm.astype(np.uint8)
    body = {}
    skin = inner & ~(cv2.dilate(any_part, _ellipse(3)) > 0)
    if skin.sum() > 300:
        body["skin"] = _gauss_model(lab[skin])
    for name, pm in parts.items():
        if name == "shorts":   # white fabric would vouch for bubbles
            continue
        core = cv2.erode(pm.astype(np.uint8), _ellipse(2)) > 0
        if name == "hair" and core.sum() > 150:
            # hair masks can hold bubbles: learn hair from its darker half
            L = lab[..., 0]
            core &= L <= np.median(L[core])
        if core.sum() > 150:
            body[name] = _gauss_model(lab[core])
    others = _surround_models(lab, m8)
    if not body or not others:
        return raw, 0.0
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    cand = raw & (dist <= band_px)
    idx = np.nonzero(cand)
    f = lab[idx]
    ll_body = np.max(np.stack([_loglik(f, md) for md in body.values()]), 0)
    ll_out = np.max(np.stack([_loglik(f, md) for md in others.values()]), 0)
    drop = np.zeros_like(raw)
    drop[idx] = ll_out > ll_body + margin
    drop = cv2.morphologyEx(drop.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(1)) > 0
    if "shorts" in parts:      # the shorts' own mask is never cut
        drop &= ~(cv2.dilate(parts["shorts"].astype(np.uint8), _ellipse(2)) > 0)
    out = raw & ~drop
    # keep the body in one piece: components that hold the inner core
    n, lab_cc, _, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    good = np.unique(lab_cc[inner & (lab_cc > 0)])
    out = np.isin(lab_cc, good[good > 0])
    out = (cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(2)) > 0) & raw
    return out, round(1.0 - out.sum() / area, 4)

