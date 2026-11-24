#!/usr/bin/env python3
"""Quality checks for the deep-abyss grade.

  compare   graded still vs the reference still: per-band / per-subject luma + Lab numbers
            and a side-by-side image (source | graded | reference)
  sheet     before/after contact sheet of a rendered video (optionally with SAM mask outlines)
  check     container / stream checks of the rendered MP4 against the source

    python abyss_qc.py compare --video inputs/dive.mov --frame 241 --graded output/stills/g00241.png \
        --reference ref.jpg --sheet output/compare_241.jpg
    python abyss_qc.py sheet --video inputs/dive.mov --graded output/dive_abyss_1080p.mp4 --out output/sheet.jpg
    python abyss_qc.py check --video inputs/dive.mov --graded output/dive_abyss_1080p.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
import sys
from pathlib import Path

import cv2
import numpy as np

from abyss_common import luma, probe, read_frames, rgb_to_lab

REPO = Path(__file__).resolve().parent


def load_rgb(p: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    im = im[..., :3][..., ::-1].astype(np.float32) / (65535 if im.dtype == np.uint16 else 255)
    if size and (im.shape[1], im.shape[0]) != size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA if im.shape[1] > size[0] else cv2.INTER_CUBIC)
    return im


def stats(im, m=None):
    px = im[m] if m is not None else im.reshape(-1, 3)
    if len(px) < 10:
        return None
    y = luma(px)
    lab = rgb_to_lab(px)
    return dict(y=float(y.mean()), p5=float(np.percentile(y, 5)), p95=float(np.percentile(y, 95)),
                L=float(lab[:, 0].mean()), a=float(lab[:, 1].mean()), b=float(lab[:, 2].mean()),
                rgb=px.mean(0))


def fmt(s):
    return "n/a" if s is None else (f"Y {s['y']:.3f} [{s['p5']:.3f}-{s['p95']:.3f}]  Lab {s['L']:5.1f} {s['a']:5.1f} {s['b']:5.1f}  "
                                    f"RGB {s['rgb'][0]:.3f} {s['rgb'][1]:.3f} {s['rgb'][2]:.3f}")


def cmd_compare(a) -> int:
    info = probe(a.video)
    src = next(f for i, f in enumerate(read_frames(a.video, info["w"], info["h"])) if i == a.frame)
    h, w = src.shape[:2]
    out = load_rgb(a.graded, (w, h))
    ref = load_rgb(a.reference, (w, h))
    yy = np.arange(h)[:, None] / h * np.ones((1, w))
    xx = np.arange(w)[None, :] / w * np.ones((h, 1))
    regions = {"ALL": np.ones((h, w), bool)}
    for i in range(0, 10, 2):
        regions[f"rows {i * 10:02d}-{i * 10 + 20}%"] = (yy >= i / 10) & (yy < (i + 2) / 10)
    regions["left edge"] = xx < 0.12
    regions["right edge"] = xx > 0.88
    regions["center col"] = abs(xx - 0.5) < 0.15
    masks = {}
    for job in ("diver", "fish"):
        d = (a.masks or REPO / "masks" / f"{a.video.stem}__abyss") / f"{job}_full" / f"{a.frame:05d}.png"
        if d.exists():
            m = cv2.imread(str(d), 0) > 140
            regions[f"{job} (mask)"] = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            masks[job] = m
    print(f"{'region':16s}{'':6s}{'graded':80s}\n{'':22s}{'reference'}")
    worst = 0.0
    for name, m in regions.items():
        so, sr = stats(out, m), stats(ref, m)
        print(f"{name:16s}out  {fmt(so)}\n{'':16s}ref  {fmt(sr)}")
        if so and sr and not name.startswith(("diver", "fish")):
            worst = max(worst, abs(so["y"] - sr["y"]))
    print(f"\nmax |dY| over background regions: {worst:.3f}")
    if a.sheet:
        sc = 0.5
        tiles = [cv2.resize(np.clip(x, 0, 1), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) for x in (src, out, ref)]
        sheet = np.concatenate(tiles, 1)
        a.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.sheet), (sheet[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("sheet:", a.sheet)
    return 0


def frame_at(path: Path, idx: int, w: int, h: int) -> np.ndarray:
    for i, f in enumerate(read_frames(path, w, h)):
        if i == idx:
            return f
    raise SystemExit(f"{path}: frame {idx} missing")


def cmd_sheet(a) -> int:
    si, go = probe(a.video), probe(a.graded)
    idx = [int(x) for x in a.frames.split(",")] if a.frames else list(np.linspace(0, si["frames"] - 1, 6).astype(int))
    root = a.masks or REPO / "masks" / f"{a.video.stem}__abyss"
    want = set(idx)
    srcs, outs = {}, {}
    for i, f in enumerate(read_frames(a.video, si["w"], si["h"])):
        if i in want:
            srcs[i] = f
    for i, f in enumerate(read_frames(a.graded, go["w"], go["h"])):
        if i in want:
            outs[i] = f
    cols = []
    for i in idx:
        s = cv2.resize(srcs[i], (go["w"], go["h"]), interpolation=cv2.INTER_AREA)
        o = outs[i].copy()
        for job, col in (("diver", (1, .25, .25)), ("fish", (1, .9, .2))):
            p = root / f"{job}_full" / f"{i:05d}.png"
            if p.exists() and a.overlay:
                m = (cv2.imread(str(p), 0) > 127).astype(np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(o, cs, -1, col, 2)
        t = [cv2.resize(x, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA) for x in (s, o)]
        cols.append(np.concatenate(t, 0))
    sheet = np.concatenate(cols, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), (np.clip(sheet, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet:", a.out, "frames", idx)
    return 0


def ff(path: Path):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    au = [s for s in j["streams"] if s["codec_type"] == "audio"]
    return j, v, au


def cmd_check(a) -> int:
    ji, vi, ai = ff(a.video)
    jo, vo, ao = ff(a.graded)
    n_out = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                                         "stream=nb_read_frames", "-of", "csv=p=0", str(a.graded)]).strip())
    n_in = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                                        "stream=nb_read_frames", "-of", "csv=p=0", str(a.video)]).strip())
    checks = [
        ("codec h264", vo["codec_name"] == "h264"),
        ("pixel format yuv420p", vo["pix_fmt"] == "yuv420p"),
        ("1080p framing (short edge 1080)", min(vo["width"], vo["height"]) == 1080),
        ("same aspect ratio as source", abs(vo["width"] / vo["height"] - vi["width"] / vi["height"]) < 0.002),
        ("same frame count", n_in == n_out),
        ("same frame rate", abs(float(Fraction(vo["avg_frame_rate"])) - float(Fraction(vi["avg_frame_rate"]))) < 0.01),
        ("duration within 0.1 s", abs(float(jo["format"]["duration"]) - float(ji["format"]["duration"])) < 0.1),
        ("audio preserved", bool(ai) == bool(ao) and (not ai or ao[0]["codec_name"] == ai[0]["codec_name"])),
        ("BT.709 tagged", vo.get("color_space") == "bt709"),
    ]
    print(f"output: {vo['width']}x{vo['height']} {vo['codec_name']} {vo['profile']} {vo['pix_fmt']} {vo['avg_frame_rate']} fps, "
          f"{n_out} frames, {float(jo['format']['duration']):.3f}s, {int(jo['format']['size']) / 1e6:.1f} MB, "
          f"audio {ao[0]['codec_name'] if ao else 'none'}")
    ok = True
    for name, good in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        ok &= bool(good)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--video", type=Path, required=True)
    c.add_argument("--frame", type=int, required=True)
    c.add_argument("--graded", type=Path, required=True)
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--masks", type=Path)
    c.add_argument("--sheet", type=Path)
    s = sub.add_parser("sheet")
    s.add_argument("--video", type=Path, required=True)
    s.add_argument("--graded", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--frames", default=None)
    s.add_argument("--masks", type=Path)
    s.add_argument("--overlay", action="store_true", help="draw SAM mask outlines on the graded frames")
    s.add_argument("--scale", type=float, default=0.3)
    k = sub.add_parser("check")
    k.add_argument("--video", type=Path, required=True)
    k.add_argument("--graded", type=Path, required=True)
    a = ap.parse_args()
    return {"compare": cmd_compare, "sheet": cmd_sheet, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
