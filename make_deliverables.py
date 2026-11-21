#!/usr/bin/env python3
"""Verification package from the rendered file: original / v1 / fixed / IMAGE 2 comparison, skin zooms,
per-layer diagnostic masks (start, middle, end), flicker + hair-colour stats, and the mask archive."""
import json, shutil, sys
from pathlib import Path
import cv2, numpy as np
sys.argv = ["x"]
import underwater_cinematic_grade as U

R = U.REPO
O = R / "output" / "underwater_cinematic"
D = O / "diagnostics"
D.mkdir(parents=True, exist_ok=True)
S = json.loads((O / "diver_cinematic_graded.settings.json").read_text())
w, h, fps, n = U.probe(R / "inputs" / "diver.mov")
root = R / "masks"
full = U.merge_masks(root / "diver__fwd" / "subject", n, (h, w))
parts = {}
for k in ("hair", "shorts", "fins"):
    pm = U.load_part(root / f"diver__{k}" / "subject", n, (h, w))
    if k == "shorts":
        last = None
        for i, m in enumerate(pm):
            if m.sum() > S["shorts_max_area"] and last is not None: pm[i] = last.copy()
            elif m.sum() <= S["shorts_max_area"]: last = m
    if k == "hair":
        for i, m in enumerate(pm):
            if m.sum() > S["hair_max_area"]:
                ys, _ = np.nonzero(m); keep = m.copy()
                keep[int(ys.min() + S["hair_top_frac"] * (ys.max() - ys.min() + 1)):] = False; pm[i] = keep
    parts[k] = U.vote3(pm)
f2 = U.load_part(root / "diver__fin2" / "subject", n, (h, w))
parts["fin2"] = U.vote3([m if S["fin2_min_area"] <= m.sum() <= S["fin2_max_area"] else np.zeros_like(m) for m in f2])
o = cv2.VideoCapture(str(R / "inputs" / "diver.mov")); g = cv2.VideoCapture(str(O / "diver_cinematic_graded.mp4"))
v1 = cv2.VideoCapture(str(O / "previous" / "diver_cinematic_graded_v1.mp4")); ref = cv2.imread(str(R / "reference" / "image2_target.png"))
keep, skinL, hair = {}, [], []
for i in range(n):
    _, fo = o.read(); _, fg = g.read(); _, f1 = v1.read()
    P = U.split_parts(cv2.cvtColor(fo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255, full[i].astype(bool),
                      parts["hair"][i], parts["shorts"][i], parts["fins"][i], S, parts["fin2"][i])
    lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)
    sk = P["skin"] > 0.9; hm = P["hair"] > 0.9
    skinL.append(lab[..., 0][sk].mean() * 100 / 255)
    if hm.any(): hair.append((lab[..., 0][hm].mean() * 100 / 255, lab[..., 1][hm].mean() - 128, lab[..., 2][hm].mean() - 128))
    if i in (0, 118, 235):
        keep[i] = (fo, f1, fg, P)
        gray = cv2.cvtColor(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)  # overlay on the graded frame
        one = {kk: [vv[i]] for kk, vv in {"full": full, **parts}.items()}
        _, P2 = next(U.process([cv2.cvtColor(fo, cv2.COLOR_BGR2RGB)], one, S, w, h))   # includes the pole ('stick') mask
        U.save_diag(D, i, cv2.cvtColor(fo, cv2.COLOR_BGR2RGB), P2)
d = np.abs(np.diff(np.array(skinL))); hd = np.array(hair)
stats = {"skin_mean_L_step_max": round(float(d.max()), 2), "skin_mean_L_step_median": round(float(np.median(d)), 3),
         "hair_mean_Lab_median": np.median(hd, 0).round(1).tolist(), "hair_b_max": round(float(hd[:, 2].max()), 1),
         "frames": n}
(O / "verification_stats.json").write_text(json.dumps(stats, indent=1)); print(stats)
def tag(t, im):
    im = im.copy(); cv2.putText(im, t, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA); return im
rr = cv2.resize(ref, (int(ref.shape[1] * h / ref.shape[0]), h))
rows = [np.hstack([tag(f"original f{i}", fo), tag("previous (v1)", f1), tag("fixed grade", fg), tag("IMAGE 2 target", rr)])
        for i, (fo, f1, fg, P) in keep.items()]
cv2.imwrite(str(O / "comparison_original_v1_fixed_vs_IMAGE2.jpg"), cv2.resize(np.vstack(rows), None, fx=.45, fy=.45), [cv2.IMWRITE_JPEG_QUALITY, 92])
zs = []
for i, (fo, f1, fg, P) in keep.items():
    ys, xs = np.nonzero(P["skin"] > .5); y0, y1 = int(np.percentile(ys, 10)), int(np.percentile(ys, 70)); x0, x1 = int(np.percentile(xs, 5)), int(np.percentile(xs, 95))
    c = lambda im: cv2.resize(im[y0:y1, x0:x1], (480, int(480 * (y1 - y0) / (x1 - x0))))
    zs.append(np.hstack([c(fo), c(f1), c(fg)]))
hm_ = max(z.shape[0] for z in zs)
cv2.imwrite(str(O / "skin_zoom_original_v1_fixed.jpg"), np.vstack([cv2.copyMakeBorder(z, 0, hm_ - z.shape[0], 0, 0, cv2.BORDER_CONSTANT) for z in zs]), [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(str(O / "skin_zoom_IMAGE2.jpg"), cv2.resize(ref[300:760, 420:1000], (480, int(480 * 460 / 580))))
for k in ("fwd", "hair", "shorts", "fins", "fin2"):
    dst = O / "masks" / ("diver_full" if k == "fwd" else k); dst.mkdir(parents=True, exist_ok=True)
    for f in (root / f"diver__{k}" / "subject").glob("*.png"): shutil.copy(f, dst / f.name)
