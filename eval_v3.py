#!/usr/bin/env python3
"""Numeric verification of a graded render (no images).

 1. container check (frames / duration / size / fps / audio) vs source
 2. region table on 3 SAM-labelled keyframes: skin / shorts / fins / hair  x  before(raw), after(output), global-only(skin matte off), reference-after
    hue / HSL saturation / Lab L,a,b; non-skin regions must move like the water does (|dHue(after, global-only)| < 5 deg, no extra saturation)
 3. skin vs reference after-frame skin: |dHue|<5, |dSat|<8%, |dL*|<4
 4. temporal: background mean L*/b* frame-to-frame jumps, skin / non-skin matte area jumps (>25 %)
usage: python eval_v3.py <source> <graded.mp4> [skin.mp4]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import gradelib as G

REPO = Path(__file__).resolve().parent
D = Path("/root/.claude/uploads/dc050da2-0edc-5bdb-a909-5d669a008d2e")
FRAMES = [15, 310, 524]


def probe(p):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(p)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    return {"w": v["width"], "h": v["height"], "fps": v["r_frame_rate"], "frames": int(v["nb_frames"]), "dur": float(v["duration"]),
            "pix": v["pix_fmt"], "codec": v["codec_name"], "audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def get_frame(path, i):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{i}),scale=in_color_matrix=bt709:in_range=tv,format=rgb24",
                        "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True).stdout
    info = probe(path)
    return np.frombuffer(p, np.uint8).reshape(info["h"], info["w"], 3).astype(np.float32) / 255


def stat(rgb, m):
    if m.sum() < 30:
        return None
    lab = G.to_lab(rgb)[m].mean(0)
    h, s, l = G.hsl_stats(rgb[m])
    return {"L": float(lab[0]), "a": float(lab[1]), "b": float(lab[2]), "hue": h, "sat": s, "n": int(m.sum())}


def fmt(x):
    return "   -   " if x is None else f"{x['L']:5.1f} {x['a']:5.1f} {x['b']:5.1f} h{x['hue']:4.0f} s{x['sat']:.2f}"


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    skin_v = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "out/player_export_skin.mp4"
    params = json.loads((REPO / "out/grade_params.json").read_text())
    ps, po = probe(src), probe(out)
    print("container: src", ps, "\n           out", po)
    print("frames equal:", ps["frames"] == po["frames"], "| duration diff %.3fs" % abs(ps["dur"] - po["dur"]), "| size/fps equal:", (ps["w"], ps["h"], ps["fps"]) == (po["w"], po["h"], po["fps"]), "| audio:", po["audio"])
    B = params["bg"]; luts = G.make_luts(B)
    rows = []
    worst = {"nonskin_dhue": 0.0, "nonskin_dsat": -9.0}
    for f in FRAMES:
        npz = np.load(REPO / ".cache/parts" / f"f{f:04d}.npz")
        raw = get_frame(src, f); res = get_frame(out, f)
        H, W = raw.shape[:2]
        m = get_frame(REPO / "out/player_export_mask.mp4", f)[..., 0] if False else (cv2.VideoCapture(str(REPO / "out/player_export_mask.mp4")))
        # mask / skin matte frames
        def gray(path, idx):
            r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
            return np.frombuffer(r, np.uint8).reshape(H, W).astype(np.float32) / 255
        mk = gray(REPO / "out/player_export_mask.mp4", f)
        sk = gray(skin_v, f)
        # global-only reference: same pipeline, skin matte off (static anchor)
        A = G.global_grade(raw, B, luts)
        keep = cv2.dilate((mk > .004).astype(np.uint8), np.ones((31, 31), np.uint8)) == 0
        lab = G.to_lab(A); dab = np.array(B["anchor_ab"], np.float32) - lab[..., 1:][keep].mean(0)
        lab[..., 1] += dab[0]; lab[..., 2] += dab[1]; A = G.from_lab(lab)
        vig = G.vignette_map(H, W, params["finish"]["vignette"], params["finish"]["vignette_pow"])
        glob = G.render(raw, mk, sk, params, vig=vig, A=A, skin_on=False)
        regs = {}
        pe = cv2.erode((mk > .5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for k in ("shorts", "fins", "hair"):
            r = cv2.resize(npz[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            regs[k] = r & pe
        regs["skin"] = sk > 0.8
        for k, r in regs.items():
            b0, a0, g0 = stat(raw, r), stat(res, r), stat(glob, r)
            rows.append((f, k, b0, a0, g0))
            if k != "skin" and a0 and g0:
                worst["nonskin_dhue"] = max(worst["nonskin_dhue"], abs(G.hue_diff(a0["hue"], g0["hue"])) if min(a0["sat"], g0["sat"]) > 0.08 else 0.0)
                worst["nonskin_dsat"] = max(worst["nonskin_dsat"], a0["sat"] - g0["sat"])
    # reference after regions
    refs = {}
    for name, a in (("head", "f9831d23-image"), ("fin", "3d75cb83-image")):
        im = cv2.imread(str(D / (a + ".jpg"))); s = 960 / im.shape[1]
        im = cv2.resize(im, (960, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        z = np.load(REPO / f".cache/parts/{a}.npz"); person = z["person"]
        pe = cv2.erode(person.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        non = np.zeros_like(person)
        for k in ("shorts", "fins", "hair"):
            non |= z[k]
        lab = G.to_lab(im)
        sk = pe & ~(cv2.dilate(non.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (lab[..., 1] > 2)
        refs[name] = {"skin": stat(im, sk), **{k: stat(im, z[k] & pe) for k in ("shorts", "fins", "hair")}}
    print("\nregion table  [L a b hue sat]   (frame, region): before(raw) | after(output) | global-only | reference-after(head / fin)")
    for f, k, b0, a0, g0 in rows:
        print(f"{f:4d} {k:6s} | {fmt(b0)} | {fmt(a0)} | {fmt(g0)} | {fmt(refs['head'].get(k))} / {fmt(refs['fin'].get(k))}")
    print("\nnon-skin vs global-only: worst |dHue| %.1f deg (regions with sat>0.08), worst sat increase %+.3f" % (worst["nonskin_dhue"], worst["nonskin_dsat"]))
    # skin vs reference
    for f, k, b0, a0, g0 in rows:
        if k == "skin" and a0:
            tgt = refs["head"]["skin"] if f == 15 else refs["fin"]["skin"]
            print(f"skin frame {f}: dHue {G.hue_diff(a0['hue'], tgt['hue']):+.1f} deg | dSat {a0['sat'] - tgt['sat']:+.3f} ({(a0['sat'] / tgt['sat'] - 1) * 100:+.0f}%) | dL* {a0['L'] - tgt['L']:+.1f}")
    # temporal numbers on the output (1/4 res, every frame)
    cap = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(out), "-vf", f"scale=308:269,format=rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    mcap = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(REPO / "out/player_export_mask.mp4"), "-vf", "scale=308:269,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    Ls, bs = [], []
    while True:
        a = cap.stdout.read(308 * 269 * 3); b = mcap.stdout.read(308 * 269)
        if len(a) < 308 * 269 * 3 or len(b) < 308 * 269:
            break
        im = np.frombuffer(a, np.uint8).reshape(269, 308, 3).astype(np.float32) / 255
        mm = cv2.dilate((np.frombuffer(b, np.uint8).reshape(269, 308) > 10).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
        lab = G.to_lab(im)[mm]
        Ls.append(lab[:, 0].mean()); bs.append(lab[:, 2].mean())
    dL, db = np.abs(np.diff(Ls)), np.abs(np.diff(bs))
    print(f"\nbackground frame-to-frame: |dL*| p95 {np.percentile(dL, 95):.3f} max {dL.max():.3f} | |db*| p95 {np.percentile(db, 95):.3f} max {db.max():.3f} | frames with jump>1.0: {int(((dL > 1) | (db > 1)).sum())}")
    rep = json.loads((REPO / "out/player_export_report_v3.json").read_text()) if (REPO / "out/player_export_report_v3.json").exists() else {}
    if rep:
        for k, v in rep["area_report"].items():
            print(f"matte area '{k}': median {v['area_px_median']:.0f}px, p95 rel jump {v['rel_jump_p95']:.3f}, frames >25%: {v['count']} {v['frames_jump_over_25pct']}")


if __name__ == "__main__":
    main()
