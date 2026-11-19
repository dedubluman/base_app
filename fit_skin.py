#!/usr/bin/env python3
"""Fit the small skin correction (target a/b, dL) so SKIN pixels (skin matte > .8) of the rendered output hit the reference
after-frame skin statistics (HSL hue, HSL saturation, Lab L*). Measured on the real render (matte, veil, vignette included)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
import gradelib as G

REF = {  # from SAM person - (shorts, fins, hair), warm gate a*>2 on the reference after frames
    "head": {"hue": 26.0, "sat": 0.47, "L": 68.9, "frames": [15, 30]},
    "fin": {"hue": 14.0, "sat": 0.47, "L": 68.8, "frames": [524, 599]},
}


def prep(params):
    B = params["bg"]; luts = G.make_luts(B)
    items = []
    for name, r in REF.items():
        for f in r["frames"]:
            raw = cv2.imread(f".cache/ref/raw_{f}.png")[..., ::-1].astype(np.float32) / 255
            m = cv2.imread(f".cache/ref/mask_{f}.png", 0).astype(np.float32) / 255
            H, W = m.shape
            A = G.global_grade(raw, B, luts)
            keep = cv2.dilate((m > .004).astype(np.uint8), np.ones((31, 31), np.uint8)) == 0
            dab = np.array(B["anchor_ab"], np.float32) - G.to_lab(A)[..., 1:][keep].mean(0)
            lab = G.to_lab(A); lab[..., 1] += dab[0]; lab[..., 2] += dab[1]; A = G.from_lab(lab)
            items.append((name, raw, m, A, G.vignette_map(H, W, params["finish"]["vignette"], params["finish"]["vignette_pow"])))
    return items


def measure(items, params):
    out = []
    for name, raw, m, A, vig in items:
        sk = G.skin_matte(raw, m, params["skin"])
        o = G.render(raw, m, sk, params, vig=vig, A=A)
        sel = sk > 0.8
        px = o[sel]
        h, s, l = G.hsl_stats(px)
        L = G.to_lab(o)[..., 0][sel].mean()
        out.append((name, h, s, float(L), int(sel.sum())))
    return out


if __name__ == "__main__":
    p = Path("out/grade_params.json"); params = json.loads(p.read_text())
    params["skin"]["ab_strength"] = 1.0
    items = prep(params)

    def resid(x):
        params["skin"].update(skin_ab=[float(x[0]), float(x[1])], L_restore=float(x[2]), dL=float(x[3]))
        r = []
        for (name, h, s, L, n) in measure(items, params):
            t = REF[name]
            r += [G.hue_diff(h, t["hue"]) / 5, (s - t["sat"]) / 0.08, (L - t["L"]) / 4]
        return np.array(r)

    params["skin"].setdefault("L_restore", 0.0)
    x0 = np.array([11.0, 17.0, 0.9, 0.0])
    sol = least_squares(resid, x0, bounds=([-5, -5, 0.0, -10], [30, 30, 1.2, 15]), diff_step=0.03, x_scale=[3, 3, 0.2, 3], max_nfev=30)
    params["skin"].update(skin_ab=[float(sol.x[0]), float(sol.x[1])], L_restore=float(sol.x[2]), dL=float(sol.x[3]))
    res = measure(items, params)
    params["fit_report_skin"] = [{"frame_pair": n, "hue": round(h, 1), "sat": round(s, 3), "L": round(L, 1), "skin_px": k} for n, h, s, L, k in res]
    p.write_text(json.dumps(params, indent=1))
    print(json.dumps({"skin_ab": params["skin"]["skin_ab"], "L_restore": params["skin"]["L_restore"], "dL": params["skin"]["dL"], "nfev": int(sol.nfev)}))
    for n, h, s, L, k in res:
        t = REF[n]; print(n, "hue %.1f (ref %.0f, d=%.1f)  sat %.3f (ref %.2f)  L* %.1f (ref %.1f)  px %d" % (h, t["hue"], G.hue_diff(h, t["hue"]), s, t["sat"], L, t["L"], k))
