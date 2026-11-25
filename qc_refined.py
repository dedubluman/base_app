#!/usr/bin/env python3
"""Every-frame QC of the refined render against the source and an environment-only reference grade.

    python qc_refined.py --video output/UNDERWATER_CINEMATIC_FINAL_REFINED.mp4 --out work/qc_refined.json

For all frames (decoded from the final H.264 file, not from the pipeline):
  outside   mean |refined - env-only grade| outside the dilated diver mask (the existing grade must be untouched; the only
            difference allowed is H.264 quantisation noise)
  halo      (mean luma 1-3 px outside the diver - mean luma 8-12 px outside) in refined minus the same in the env-only grade;
            > ~3 /255 would be a visible glow or dark fringe added around the subject
  skin      R/G and B/G of the output skin (bright and shaded bands), frame-to-frame jumps
  shorts    Lab a*, b* of the output shorts (neutral/cool white expected)
  other     luma difference of fin/hair/gear pixels vs the env-only grade (they must not be altered)
  masks     skin-weight area jumps, class coverage
  flicker   global luma/a/b frame-to-frame jumps relative to the source's own"""
import argparse, json
from pathlib import Path
import cv2, numpy as np
import underwater_grade as ug

L = np.array([.2126, .7152, .0722], np.float32)
REPO = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--src", default="inputs/player_export.mov")
    ap.add_argument("--classes", default="masks_classes/player_export"); ap.add_argument("--diver", default="masks_refined/player_export")
    ap.add_argument("--params", default="grade_data/params_refined.json"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.loads(Path(a.params).read_text()); env = ug.build_env_lut(33, P["env"])
    cv_ = cv2.VideoCapture(a.video); cs = cv2.VideoCapture(a.src); rows = []; prev = None
    k3 = np.ones((3, 3), np.uint8)
    i = 0
    while True:
        ok1, fo = cv_.read(); ok2, fs = cs.read()
        if not ok1 or not ok2: break
        out = cv2.cvtColor(fo, cv2.COLOR_BGR2RGB); src = cv2.cvtColor(fs, cv2.COLOR_BGR2RGB)
        ref = (np.clip(ug.apply_lut(src.astype(np.float32) / 255., env), 0, 1) * 255 + .5).astype(np.uint8)
        dv = cv2.imread(f"{a.diver}/{i:05d}.png", 0) > 7
        z = np.load(f"{a.classes}/classes_{i:05d}.npz"); ws = z["skin"].astype(np.float32); wh = z["shorts"].astype(np.float32); core = z["skin_core"]
        alpha = cv2.imread(f"{a.diver}/{i:05d}.png", 0).astype(np.float32) / 255
        r = {"i": i}
        d9 = cv2.dilate(dv.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
        diff = np.abs(out.astype(np.int16) - ref.astype(np.int16)).mean(2)
        r["outside_diff"] = float(diff[~d9].mean())
        # halo rings (outside the diver mask)
        dd = lambda n: cv2.dilate(dv.astype(np.uint8), np.ones((2 * n + 1, 2 * n + 1), np.uint8)) > 0
        ring_in = dd(3) & ~dd(1); ring_far = dd(12) & ~dd(8)
        yo = out.astype(np.float32) @ L; yr = ref.astype(np.float32) @ L
        r["halo"] = float((yo[ring_in].mean() - yo[ring_far].mean()) - (yr[ring_in].mean() - yr[ring_far].mean())) if ring_in.sum() > 50 else None
        # skin
        m = core & (ws > .9)
        if m.sum() > 200:
            px = out[m].astype(np.float32) / 255.; y = px @ L
            for nm, lo, hi in (("shade", .10, .25), ("lit", .25, .60)):
                s = (y >= lo) & (y < hi)
                r[f"skin_rg_{nm}"] = float(np.median(px[s, 0] / px[s, 1])) if s.sum() > 100 else None
                r[f"skin_bg_{nm}"] = float(np.median(px[s, 2] / px[s, 1])) if s.sum() > 100 else None
            r["skin_Y"] = float(np.median(y))
        r["skin_area"] = float(ws.sum())
        # shorts (Lab)
        msh = wh > .9
        if msh.sum() > 150:
            lab = cv2.cvtColor(out[msh][None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
            r["shorts_a"] = float(np.median(lab[:, 1]) - 128); r["shorts_b"] = float(np.median(lab[:, 2]) - 128); r["shorts_L"] = float(np.median(lab[:, 0]) * 100 / 255)
        # other (fins/hair/gear): must equal env-only grade
        other = (alpha > .9) & (ws < .02) & (wh < .02)
        if other.sum() > 150: r["other_diff"] = float(diff[other].mean())
        # global
        g = cv2.cvtColor(cv2.resize(out, (308, 270)), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3).mean(0); gs = cv2.cvtColor(cv2.resize(src, (308, 270)), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3).mean(0)
        r["glob"] = g.tolist(); r["glob_src"] = gs.tolist()
        r["edge_black"] = float(((yo[:, :6] <= 18).mean() + (yo[:, -6:] <= 18).mean() + (yo[:6] <= 18).mean() + (yo[-6:] <= 18).mean()) / 4)
        rows.append(r); i += 1
        if i % 100 == 0: print("qc", i, flush=True)
    json.dump(rows, open(a.out, "w"))
    n = len(rows); print("frames compared", n)
    f = lambda k: np.array([r[k] if r.get(k) is not None else np.nan for r in rows], float)
    od = f("outside_diff"); print(f"outside-diver |refined - env-only| mean {np.nanmean(od):.2f}/255  max {np.nanmax(od):.2f} (frame {int(np.nanargmax(od))})")
    h = f("halo"); print(f"halo (added ring contrast)    mean {np.nanmean(h):+.2f}  min {np.nanmin(h):+.2f}  max {np.nanmax(h):+.2f} (frame {int(np.nanargmax(h))})  frames |halo|>3: {[int(x) for x in np.flatnonzero(np.abs(np.nan_to_num(h))>3)][:20]}")
    od2 = f("other_diff"); print(f"fins/hair/gear |refined - env-only| mean {np.nanmean(od2):.2f}  max {np.nanmax(od2):.2f}")
    for nm in ("shade", "lit"):
        rg, bg = f(f"skin_rg_{nm}"), f(f"skin_bg_{nm}")
        dj = np.abs(np.diff(rg)); print(f"skin {nm:5s}: R/G mean {np.nanmean(rg):.2f} range {np.nanmin(rg):.2f}-{np.nanmax(rg):.2f} | B/G mean {np.nanmean(bg):.2f} range {np.nanmin(bg):.2f}-{np.nanmax(bg):.2f} | frame-to-frame R/G jump p95 {np.nanpercentile(dj,95):.3f} max {np.nanmax(dj):.3f}")
    sa, sb, sl = f("shorts_a"), f("shorts_b"), f("shorts_L"); print(f"shorts Lab: a* {np.nanmin(sa):+.1f}..{np.nanmax(sa):+.1f} (median {np.nanmedian(sa):+.1f}) b* {np.nanmin(sb):+.1f}..{np.nanmax(sb):+.1f} (median {np.nanmedian(sb):+.1f}) L {np.nanmin(sl):.0f}..{np.nanmax(sl):.0f}; frames measured {int(np.sum(~np.isnan(sa)))}")
    ar = f("skin_area"); rel = np.abs(np.diff(ar)) / np.maximum(ar[:-1], 1); print("skin weight area frame-to-frame change: p95 %.1f%% max %.1f%% at frame %d" % (np.percentile(rel, 95) * 100, rel.max() * 100, int(rel.argmax()) + 1))
    G = np.array([r["glob"] for r in rows]); S = np.array([r["glob_src"] for r in rows]); dG = np.abs(np.diff(G, axis=0)); dS = np.abs(np.diff(S, axis=0))
    print("global L/a/b max frame jump  out:", dG.max(0).round(2), " source:", dS.max(0).round(2), "| out a range", (G[:, 1] - 128).min().round(1), (G[:, 1] - 128).max().round(1), "b range", (G[:, 2] - 128).min().round(1), (G[:, 2] - 128).max().round(1))
    eb = f("edge_black"); print("frames with black edge band:", int((eb > .97).sum()), " max edge-black fraction %.2f" % eb.max())


if __name__ == "__main__":
    main()
