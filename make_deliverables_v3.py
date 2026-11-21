#!/usr/bin/env python3
"""Verification package for the refinement pass: original / previous (video_1) / new contact sheets with
head and body close-ups, per-layer diagnostic masks, warm-bleed + temporal + corner metrics."""
import json, shutil, sys
from pathlib import Path
import cv2, numpy as np
sys.argv = ["x"]
import underwater_cinematic_grade as U

R = U.REPO; O = R / "output" / "underwater_cinematic"; D = O / "diagnostics"; D.mkdir(parents=True, exist_ok=True)
NEW = O / "diver_cinematic_graded_v3.mp4"; OLD = O / "previous" / "diver_cinematic_graded_v2.mp4"
S = json.loads(NEW.with_suffix(".settings.json").read_text())
w, h, fps, n = U.probe(R / "inputs" / "diver.mov")
root = R / "masks"
class MS:
    """Lazy per-frame access to the cleaned part masks (same cleaning + 3-frame vote as the grade script)."""
    def __init__(self):
        rd = lambda k, i: cv2.imread(str(root / f"diver__{k}" / "subject" / f"{i:05d}.png"), 0)
        self.rd = rd
        self.sh_ok = []
        last = None
        for i in range(n):
            m = rd("shorts", i)
            a_ = 0 if m is None else int((m > 127).sum())
            if a_ <= S["shorts_max_area"]: last = i
            self.sh_ok.append(last if (a_ > S["shorts_max_area"]) else i)
    def raw(self, k, i):
        m = self.rd(k, i); return np.zeros((h, w), bool) if m is None else m > 127
    def part(self, k, i):
        if k == "shorts": return self.raw("shorts", self.sh_ok[i] if self.sh_ok[i] is not None else i)
        m = self.raw(k, i)
        if k == "hair" and m.sum() > S["hair_max_area"]:
            ys, _ = np.nonzero(m); keep = m.copy(); keep[int(ys.min() + S["hair_top_frac"] * (ys.max() - ys.min() + 1)):] = False; return keep
        if k == "fin2" and not (S["fin2_min_area"] <= m.sum() <= S["fin2_max_area"]): return np.zeros_like(m)
        return m
    def voted(self, k, i):
        a_, b_, c_ = self.part(k, max(i - 1, 0)), self.part(k, i), self.part(k, min(i + 1, n - 1))
        return (a_.astype(np.uint8) + b_ + c_) >= 2
    def one(self, i):
        return {"full": [self.raw("fwd", i)], **{k: [self.voted(k, i)] for k in ("hair", "shorts", "fins", "fin2")}}
mst = MS()
ref = cv2.imread(str(R / "reference" / "image2_target.png"))
cap = {k: cv2.VideoCapture(str(p)) for k, p in (("o", R / "inputs" / "diver.mov"), ("old", OLD), ("new", NEW))}
SEL = (0, 60, 118, 190, 235)
keep, warm = {}, {"old": [], "new": []}
tone = {"old": [], "new": []}
lab_of = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]
for i in range(n):
    fr = {k: c.read()[1] for k, c in cap.items()}
    if i in SEL:
        one = mst.one(i)
        _, P = next(U.process([cv2.cvtColor(fr["o"], cv2.COLOR_BGR2RGB)], one, S, w, h))
        keep[i] = (fr, P)
        U.save_diag(D, i, cv2.cvtColor(fr["o"], cv2.COLOR_BGR2RGB), P)
        ys, xs = np.nonzero(P["hair"] > 0.5)
        if len(ys):   # head gear region: hair + dark gear inside the head box
            y0, y1, x0, x1 = ys.min() - 60, ys.max() + 140, xs.min() - 70, xs.max() + 90
            reg = np.zeros((h, w), bool); reg[max(y0, 0):y1, max(x0, 0):x1] = True
            reg &= ((P["hair"] > 0.5) | (P["gear"] > 0.5)) & (P["skin"] < 0.3)
            for k in ("old", "new"):
                L, a, b = np.moveaxis(lab_of(fr[k]), -1, 0)
                tan = (a > 4) & (b > 10) & (L > 12)
                warm[k].append(float((tan & reg).sum() / max(reg.sum(), 1)))
    sh = mst.voted("shorts", i); fullm = mst.raw("fwd", i)
    for k in ("old", "new"):
        L = lab_of(fr[k])[..., 0]
        sk = (cv2.erode(sh.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        tone[k].append((float(L[sk].mean()) if sk.any() else np.nan, float(L[fullm & ~sh].mean())))
def steps(arr):
    a = np.array(arr, float); a = a[~np.isnan(a)]; d = np.abs(np.diff(a)); return round(float(d.max()), 2), round(float(np.median(d)), 3)
stats = {"tan_fraction_in_hair_gear_snorkel_region": {k: [round(x, 4) for x in v] for k, v in warm.items()},
         "frames_checked": list(SEL),
         "shorts_mean_L_step_max_median": {k: steps([t[0] for t in v]) for k, v in tone.items()},
         "diver_mean_L_step_max_median": {k: steps([t[1] for t in v]) for k, v in tone.items()}}
def corner(k):
    c = cv2.VideoCapture(str(NEW if k == "new" else OLD)); out = []
    for f in (0, 118, 235):
        c.set(cv2.CAP_PROP_POS_FRAMES, f); im = c.read()[1][..., ::-1].astype(float)
        out.append(im[int(.8 * h):, :int(.2 * w)].reshape(-1, 3).mean(0).round(1).tolist() + im[int(.8 * h):, int(.8 * w):].reshape(-1, 3).mean(0).round(1).tolist())
    return out
stats["lower_corners_RGB_LL_LR_per_frame"] = {"old": corner("old"), "new": corner("new"), "IMAGE2": [0.5, 15, 22.4, 1.0, 16.2, 23.6]}
(O / "verification_stats_v3.json").write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))
def tag(t, im):
    im = im.copy(); cv2.putText(im, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA); return im
rows = []
for i in (0, 118, 235):
    fr, P = keep[i]; rows.append(np.hstack([tag(f"original f{i}", fr["o"]), tag("previous (video_1)", fr["old"]), tag("new refined", fr["new"])]))
cv2.imwrite(str(O / "contact_sheet_full.jpg"), cv2.resize(np.vstack(rows), None, fx=.5, fy=.5), [cv2.IMWRITE_JPEG_QUALITY, 92])
crow = []
for i in (0, 118, 235):
    fr, P = keep[i]
    ys, xs = np.nonzero(P["hair"] > 0.5); cy, cx = int(np.median(ys)), int(np.median(xs))
    hy0, hx0 = max(cy - 150, 0), max(cx - 170, 0)
    ys, xs = np.nonzero(P["full"] > 0.5); by0, by1, bx0, bx1 = int(np.percentile(ys, 30)), int(ys.max()) + 10, int(xs.min()) - 10, int(xs.max()) + 10
    hc = lambda im: cv2.resize(im[hy0:hy0 + 330, hx0:hx0 + 340], (340, 330)); bc = lambda im: cv2.resize(im[by0:by1, bx0:bx1], (int(340 * (bx1 - bx0) / max(by1 - by0, 1) * 0.0 + 340), 330))
    crow.append(np.hstack([tag(f"(a) head f{i}", hc(fr["o"])), hc(fr["old"]), hc(fr["new"])]))
    crow.append(np.hstack([tag(f"(b) body f{i}", bc(fr["o"])), bc(fr["old"]), bc(fr["new"])]))
cv2.imwrite(str(O / "contact_sheet_crops.jpg"), np.vstack(crow), [cv2.IMWRITE_JPEG_QUALITY, 92])
for k in ("fwd", "hair", "shorts", "fins", "fin2"):
    dst = O / "masks" / ("diver_full" if k == "fwd" else k); dst.mkdir(parents=True, exist_ok=True)
    for f in (root / f"diver__{k}" / "subject").glob("*.png"): shutil.copy(f, dst / f.name)
