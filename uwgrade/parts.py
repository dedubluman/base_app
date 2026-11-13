"""Diver parts (skin / white shorts / dark gear) inside the SAM 3.1 diver alpha.

SAM 3.1 detects "shorts" with p >= 0.9 on most keyframes but loses it on a few (very
small / foreshortened diver), and it has no usable text concept for "exposed skin" vs
"snorkel, hair, fins". So the parts are completed from SAM seeds:

    shorts  = SAM "shorts" masks on the keyframes where they were found
    gear    = dark pixels inside the diver (hair, snorkel, mask, fins, deep shadow)
    skin    = the rest of the diver

A small classifier on colour features, trained on the pooled keyframe seeds, labels every
diver pixel in every frame. The classifier never leaves the SAM diver alpha, so the
diver boundary stays SAM's.
"""
from __future__ import annotations

import cv2
import numpy as np

from .color import blur, rgb2lab

GEAR_L = 24.0   # source L* below which a diver pixel is "dark gear" (hair / snorkel / fins)


def part_features(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = rgb2lab(rgb_u8.astype(np.float32) / 255.0)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    f = np.stack([blur(L, 1.0), blur(a, 1.0), blur(b, 1.0), blur(L, 5), blur(a, 5), blur(b, 5),
                  np.abs(L - blur(L, 2.0))], -1).astype(np.float32)
    return f, lab


def train(frames_u8, diver_fn, shorts_seed_fn, kf_frames, min_shorts_frac=0.003, per_class=5000, log=print):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X, y, g = [], [], []
    for t in kf_frames:
        D = diver_fn(t) > 0.9
        S = shorts_seed_fn(t)
        if D.sum() < 2000 or S.sum() < min_shorts_frac * D.size:
            continue
        f, lab = part_features(np.asarray(frames_u8[t]))
        De = cv2.erode(D.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        Se = (cv2.erode(S.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & De
        near_s = cv2.dilate(S.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        dark = De & (lab[..., 0] < GEAR_L) & ~near_s
        skin = De & ~near_s & ~dark
        for cls, m in ((2, Se), (1, dark), (0, skin)):
            idx = np.flatnonzero(m.ravel())
            if len(idx) < 50:
                continue
            take = rng.choice(idx, min(per_class, len(idx)), replace=False)
            X.append(f.reshape(-1, f.shape[-1])[take])
            y.append(np.full(len(take), cls))
            g.append(np.full(len(take), t))
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    log(f"[parts] training samples skin {int((y == 0).sum())} gear {int((y == 1).sum())} shorts {int((y == 2).sum())} "
        f"from {len(set(g.tolist()))} keyframes")
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=120, learning_rate=0.1, l2_regularization=1.0,
                                         class_weight="balanced", random_state=0)
    clf.fit(X, y)
    return clf, (X, y, g)


def loo_accuracy(X, y, g, log=print):
    from sklearn.ensemble import HistGradientBoostingClassifier

    res = {}
    for t in sorted(set(g.tolist())):
        te = g == t
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=120, learning_rate=0.1, l2_regularization=1.0,
                                             class_weight="balanced", random_state=0)
        clf.fit(X[~te], y[~te])
        pred = clf.predict(X[te])
        res[t] = {"acc": float((pred == y[te]).mean()),
                  "shorts_recall": float((pred[y[te] == 2] == 2).mean()) if (y[te] == 2).any() else None}
    log(f"[parts] leave-one-keyframe-out: " + ", ".join(f"{t}: acc {v['acc']:.2f}" for t, v in res.items()))
    return res


def classify(clf, rgb_u8: np.ndarray, diver: np.ndarray) -> np.ndarray:
    """(3,H,W) float32 probabilities [skin, gear, shorts] (sum 1 inside the diver)."""
    H, W = diver.shape
    out = np.zeros((3, H, W), np.float32)
    out[0] = 1.0
    ys, xs = np.nonzero(diver > 0.02)
    if len(ys) == 0:
        return out
    y0, y1 = max(ys.min() - 6, 0), min(ys.max() + 7, H)
    x0, x1 = max(xs.min() - 6, 0), min(xs.max() + 7, W)
    f, lab = part_features(rgb_u8[y0:y1, x0:x1])
    p = clf.predict_proba(f.reshape(-1, f.shape[-1])).reshape(y1 - y0, x1 - x0, -1).astype(np.float32)
    for k in range(3):
        out[k, y0:y1, x0:x1] = blur(p[..., k], 0.8)
    s = out.sum(0, keepdims=True)
    return out / np.maximum(s, 1e-6)
