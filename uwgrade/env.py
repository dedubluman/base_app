"""Environment masks (rocks / seabed vs water column) completed from SAM 3.1 seeds.

SAM 3.1's text concepts only give *partial*, low-confidence cover of these amorphous
classes (see README_uwgrade.md): "rock" -> individual boulders, "stone wall" -> the
rock mass in some frames, "water surface" -> the upper surface / water column in the
frames where it is visible. Those masks are therefore used as **seeds**. A small
gradient-boosted classifier on colour / texture / haze / height features, trained on the
pooled seeds of all keyframes, completes them to a dense per-pixel probability for every
frame. The classifier is deterministic per pixel, so it does not flicker the way a
per-frame detector would; a short temporal filter on the (low-frequency) result finishes it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .color import blur, rgb2lab

P_ROCK = 0.12      # 'rock' instance seeds
P_WALL = 0.20      # 'stone wall' seeds (the rock mass); lower-prob walls were the water column
P_SURF = 0.06      # 'water surface' seeds
P_SHORTS = 0.30
DS = 4             # classifier runs at 1/4 resolution


def load_kf(kfdir: Path, frame: int, prompt: str):
    f = kfdir / f"{frame:05d}__{prompt.replace(' ', '_')}.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    h, w = (int(v) for v in z["shape"])
    pk = z["packed"]
    if pk.shape[0] == 0:
        return np.zeros((0, h, w), bool), z["probs"]
    return np.unpackbits(pk, axis=-1)[..., :w].astype(bool), z["probs"]


def seeds(kfdir: Path, frame: int, shape) -> dict:
    h, w = shape
    out = {"rock": np.zeros(shape, bool), "surf": np.zeros(shape, bool), "shorts": np.zeros(shape, bool)}
    for prompt, thr, key in (("rock", P_ROCK, "rock"), ("stone wall", P_WALL, "rock"),
                             ("water surface", P_SURF, "surf"), ("shorts", P_SHORTS, "shorts")):
        ms, ps = load_kf(kfdir, frame, prompt)
        if ms is None:
            continue
        for m, p in zip(ms, ps):
            if p >= thr:
                out[key] |= m
    return out


def features(rgb_u8: np.ndarray) -> np.ndarray:
    """(H/DS, W/DS, F) float32 features of one frame."""
    H, W = rgb_u8.shape[:2]
    h, w = H // DS, W // DS
    small = cv2.resize(rgb_u8, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lab = rgb2lab(small)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    def lstd(x, k):
        m = cv2.boxFilter(x, -1, (k, k))
        m2 = cv2.boxFilter(x * x, -1, (k, k))
        return np.sqrt(np.maximum(m2 - m * m, 0))

    hp = blur(np.abs(L - blur(L, 1.5)), 2.0)
    yy = np.broadcast_to(np.linspace(0, 1, h, dtype=np.float32)[:, None], (h, w))
    return np.stack([blur(L, 1.0), blur(a, 1.0), blur(b, 1.0), lstd(L, 5), lstd(L, 11), hp,
                     blur(L, 8), blur(a, 8), blur(b, 8), yy], -1).astype(np.float32)


def train(frames_u8, kfdir: Path, kf_frames: list[int], exclude=None, per_class: int = 4000, log=print):
    """Pooled seed training. exclude(t) -> bool HxW diver-ish mask removed from the seeds."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X, y, g = [], [], []
    for t in kf_frames:
        rgb = np.asarray(frames_u8[t])
        H, W = rgb.shape[:2]
        sd = seeds(kfdir, t, (H, W))
        ex = exclude(t) if exclude else None
        f = features(rgb)
        h, w = f.shape[:2]
        for cls, key in ((1, "rock"), (0, "surf")):
            m = sd[key]
            if ex is not None:
                m = m & ~ex
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0     # keep seeds off boundaries
            idx = np.flatnonzero(m.ravel())
            if len(idx) == 0:
                continue
            take = rng.choice(idx, min(per_class, len(idx)), replace=False)
            X.append(f.reshape(-1, f.shape[-1])[take])
            y.append(np.full(len(take), cls))
            g.append(np.full(len(take), t))
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    log(f"[env] training samples: rock {int((y == 1).sum())}, water/surface {int((y == 0).sum())} "
        f"from {len(set(g.tolist()))} keyframes")
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.08,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    clf.fit(X, y)
    return clf, (X, y, g)


def cv_auc(X, y, g, log=print):
    """Leave-one-keyframe-out AUC (only keyframes that have both classes)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    res = {}
    for t in sorted(set(g.tolist())):
        te = g == t
        if len(set(y[te].tolist())) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.08,
                                             l2_regularization=1.0, class_weight="balanced", random_state=0)
        clf.fit(X[~te], y[~te])
        res[t] = float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    log(f"[env] leave-one-keyframe-out AUC: {res}")
    return res


def p_rock(clf, rgb_u8: np.ndarray) -> np.ndarray:
    f = features(rgb_u8)
    h, w = f.shape[:2]
    p = clf.predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(h, w).astype(np.float32)
    return p
