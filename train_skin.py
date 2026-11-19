#!/usr/bin/env python3
"""Learn the soft colour qualifier for SKIN on RAW frames from SAM 3.1 keyframe part masks.

skin     = diver mask (eroded) minus dilated {swim shorts, fins, hair}
non-skin = {swim shorts, fins, hair} inside the diver mask
Logistic regression on raw Lab features (luma + hue/chroma gates). Leave-one-keyframe-out check reports how much matte mass
leaks into shorts/fins/hair on frames the qualifier never saw. Writes skin.clf_w / skin.clf_b into grade_params.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import gradelib as G  # noqa: E402

KEYS = ["shorts", "fins", "hair"]


def load_kf(frames, mask_video):
    kfs = sorted((REPO / ".cache/parts/kf_tmp").glob("*.jpg"))
    cap = cv2.VideoCapture(str(mask_video))
    masks = {}
    i = 0
    while True:
        ok, m = cap.read()
        if not ok:
            break
        if i in frames:
            masks[i] = m[..., 0]
        i += 1
    data = []
    for f, jp in zip(sorted(frames), kfs):
        npz = np.load(REPO / ".cache/parts" / f"f{f:04d}.npz")
        rgb = cv2.imread(str(jp))[..., ::-1].astype(np.float32) / 255
        h, w = rgb.shape[:2]
        person = cv2.resize(masks[f], (w, h), interpolation=cv2.INTER_AREA) > 127
        pe = cv2.erode(person.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        non = np.zeros_like(person)
        for k in KEYS:
            non |= npz[k]
        non &= cv2.dilate(person.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        skin = pe & ~(cv2.dilate(non.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        data.append({"f": f, "rgb": rgb, "person": person, "skin": skin, "non": non & pe,
                     "feat": G.skin_features(G.to_lab(rgb))})
    return data


def fit_logreg(X, y, w, l2=1e-2, iters=400):
    wt = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):   # Newton / IRLS
        z = X @ wt + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        s = np.maximum(p * (1 - p), 1e-4) * w
        g = np.concatenate([X.T @ ((p - y) * w) + l2 * wt, [((p - y) * w).sum()]])
        Xb = np.hstack([X, np.ones((len(X), 1))])
        Hm = (Xb * s[:, None]).T @ Xb + l2 * np.eye(X.shape[1] + 1)
        step = np.linalg.solve(Hm, g)
        wt -= step[:-1]; b -= step[-1]
        if np.abs(step).max() < 1e-6:
            break
    return wt, float(b)


def sample(d, rng, n=6000):
    xs, ys = [], []
    for lab, m in ((1, d["skin"]), (0, d["non"])):
        idx = np.flatnonzero(m.ravel())
        if len(idx):
            idx = rng.choice(idx, min(n, len(idx)), replace=False)
            xs.append(d["feat"].reshape(-1, d["feat"].shape[-1])[idx]); ys.append(np.full(len(idx), lab))
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (np.zeros((0, 5)), np.zeros(0))


def train(ds, rng):
    X, y = zip(*[sample(d, rng) for d in ds])
    X, y = np.concatenate(X), np.concatenate(y)
    w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1)) * len(y)
    return fit_logreg(X, y, w)


def matte(d, wt, b):
    z = d["feat"] @ wt + b
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return cv2.GaussianBlur(p.astype(np.float32), (0, 0), 1.0) * d["person"]


def main():
    frames = sorted(int(p.stem[1:]) for p in (REPO / ".cache/parts").glob("f*.npz"))
    ds = load_kf(frames, REPO / "out/player_export_mask.mp4")
    rng = np.random.default_rng(0)
    report = []
    for i, d in enumerate(ds):   # leave-one-keyframe-out
        tr = [x for j, x in enumerate(ds) if j != i]
        wt, b = train(tr, rng)
        m = matte(d, wt, b)
        tot = m.sum() + 1e-6
        rep = {"frame": d["f"], "non_skin_px": int(d["non"].sum()), "skin_px": int(d["skin"].sum()),
               "leak_mass_into_nonskin": float(m[d["non"]].sum() / tot),
               "skin_recall": float(m[d["skin"]].sum() / max(d["skin"].sum(), 1)),
               "nonskin_mean_matte": float(m[d["non"]].mean()) if d["non"].any() else None}
        report.append(rep)
    wt, b = train(ds, rng)
    p = REPO / "out/grade_params.json"
    d = json.loads(p.read_text())
    d.setdefault("skin", dict(G.DEFAULT["skin"]))
    d["skin"]["clf_w"] = wt.tolist(); d["skin"]["clf_b"] = b
    d["skin_qualifier_cv"] = report
    p.write_text(json.dumps(d, indent=1))
    for r in report:
        print(r["frame"], "skin_px", r["skin_px"], "non_px", r["non_skin_px"], "leak %.3f" % r["leak_mass_into_nonskin"], "recall %.3f" % r["skin_recall"],
              "nonskin_mean %s" % (None if r["nonskin_mean_matte"] is None else round(r["nonskin_mean_matte"], 3)))


if __name__ == "__main__":
    main()
