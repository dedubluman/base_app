#!/usr/bin/env python3
"""SAM 3.1-assisted, region-aware Mediterranean grade of the retimed frames.

    python slowmo_grade.py masks  --frames DIR --kf DIR --out masks.npz   # assemble + propagate
    python slowmo_grade.py stills --frames DIR --masks masks.npz --out DIR --idx 0,40,...
    python slowmo_grade.py render --frames DIR --masks masks.npz --video out.mp4 --audio a.wav
Frames are the optical-flow-retimed output frames (o0000.png ...), so masks and picture share
one timeline by construction."""
from __future__ import annotations
import argparse, glob, json, re, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
from slowmo import grade as G
from slowmo import segment as SG

COL = {"sky": (60, 140, 255), "rock": (60, 200, 255), "veg": (60, 200, 60), "water": (255, 160, 40), "seabed": (200, 60, 200)}  # BGR overlay


def load(frames: Path, k: int):
    return cv2.cvtColor(cv2.imread(str(frames / f"o{k:04d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def cmd_masks(a):
    n = len(glob.glob(str(a.frames / "o*.png")))
    keys = sorted(int(re.search(r"kf(\d+)", p).group(1)) for p in glob.glob(str(a.kf / "kf*.npz")))
    print(n, "frames,", len(keys), "key frames")
    grays, keym = [], {}
    for k in range(n):
        grays.append(SG._small(load(a.frames, k)))
    for k in keys:
        z = np.load(a.kf / f"kf{k:04d}.npz")
        sam = {p: z[p] for p in z.files if p != "probs"}
        m = SG.assemble(load(a.frames, k), sam).astype(np.float32)
        keym[k] = cv2.resize(m, None, fx=SG.SCALE, fy=SG.SCALE, interpolation=cv2.INTER_AREA)
    prop, drift = SG.propagate(grays, keym)
    np.savez_compressed(a.out, masks=prop, drift=drift, keys=np.array(keys))
    bad = np.argsort(-drift)[:8]
    print("temporal drift (1-IoU of forward vs backward carry): mean %.3f max %.3f; worst frames %s" % (drift.mean(), drift.max(), bad.tolist()))


def overlay(img, w):
    o = (img * 255).astype(np.uint8)[..., ::-1].astype(np.float32)
    for k, c in COL.items():
        o = o * (1 - 0.45 * w[k][..., None]) + 0.45 * w[k][..., None] * np.array(c, np.float32)
    return np.clip(o, 0, 255).astype(np.uint8)


def cmd_stills(a):
    z = np.load(a.masks)["masks"]
    a.out.mkdir(parents=True, exist_ok=True)
    for k in [int(x) for x in a.idx.split(",")]:
        img = load(a.frames, k)
        w = SG.refine(z[k].astype(np.float32), img)
        out = G.grade(img, w, a.strength)
        cmp = np.hstack([(img[..., ::-1] * 255).astype(np.uint8), (out[..., ::-1] * 255 + .5).astype(np.uint8)])
        cv2.imwrite(str(a.out / f"cmp_{k:04d}.png"), cmp)
        cv2.imwrite(str(a.out / f"mask_{k:04d}.png"), overlay(img, w))
    print("stills ->", a.out)


def cmd_render(a):
    z = np.load(a.masks)["masks"]
    n = len(z)
    h, w = load(a.frames, 0).shape[:2]
    raw = a.video.with_suffix(".video.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", "30000/1001", "-i", "-",
                          "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
                          "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", "-color_range", "tv", str(raw)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    for k in range(n):
        img = load(a.frames, k)
        out = G.grade(img, SG.refine(z[k].astype(np.float32), img), a.strength)
        out = out + (rng.random(out.shape, dtype=np.float32) - 0.5) / 255.0          # dither: no banding in blue gradients
        p.stdin.write((np.clip(out, 0, 1) * 65535 + .5).astype(np.uint16).tobytes())
        if k % 25 == 0:
            print("frame", k, flush=True)
    p.stdin.close(); p.wait()
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(raw)]
    if a.audio:
        cmd += ["-i", str(a.audio), "-c:a", "aac", "-b:a", "192k"]
    subprocess.run(cmd + ["-c:v", "copy", "-movflags", "+faststart", str(a.video)], check=True)
    raw.unlink()
    print("wrote", a.video)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks", "stills", "render"])
    ap.add_argument("--frames", type=Path); ap.add_argument("--kf", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--masks", type=Path); ap.add_argument("--idx", default="0"); ap.add_argument("--video", type=Path)
    ap.add_argument("--audio", type=Path); ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--crf", type=int, default=14)
    a = ap.parse_args()
    {"masks": cmd_masks, "stills": cmd_stills, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
