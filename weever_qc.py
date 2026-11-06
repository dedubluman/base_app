#!/usr/bin/env python3
"""QC sheets for the weever pipeline.

  mask   : 100% crops around the fish with the SAM mask edge (red) and the
           refined alpha 0.5 contour (cyan), plus the alpha itself.
  compare: before | after full frames (downscaled) for chosen frames.
  zoom   : before | after 100% crops of the fish.
  flicker: per-frame stats of the graded video (fish/bg mean luma and
           saturation, mask area) to catch pumping / flicker.

    python weever_qc.py mask --frames 0,100 --out output/qc_mask.jpg
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
STEM = "weever"


def frame(f):
    return cv2.imread(str(REPO / ".cache" / f"frames_{STEM}" / f"{f:05d}.jpg"))


def sam(f):
    p = REPO / "masks" / f"{STEM}__fish" / "fine" / f"{f:05d}.png"
    return cv2.imread(str(p), 0) if p.exists() else None


def alpha(f):
    p = REPO / "masks" / f"{STEM}__fish" / "alpha" / f"{f:05d}.png"
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535 if p.exists() else None


def bbox(m, pad=30):
    ys, xs = np.nonzero(m)
    return max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad


def label(img, t):
    cv2.putText(img, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def hstack_pad(ims, h=None):
    h = h or max(i.shape[0] for i in ims)
    out = []
    for i in ims:
        if i.shape[0] != h:
            i = cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        out.append(i)
    return np.hstack(out)


def mask_sheet(frames, crop=None):
    rows = []
    for f in frames:
        img, m, al = frame(f), sam(f), alpha(f)
        if m is None:
            continue
        y0, y1, x0, x1 = bbox(m > 127)
        if crop:  # sub-crop: "head" / "tail" quarter
            hgt = y1 - y0
            y0, y1 = (y0, y0 + hgt // 3) if crop == "top" else (y1 - hgt // 3, y1)
        c = img[y0:y1, x0:x1].copy()
        cs = c.copy()
        cnts, _ = cv2.findContours((m[y0:y1, x0:x1] > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(cs, cnts, -1, (0, 0, 255), 1)
        tiles = [label(c, f"f{f}"), label(cs, "SAM")]
        if al is not None:
            a = al[y0:y1, x0:x1]
            ca = c.copy()
            cnts2, _ = cv2.findContours((a > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ca, cnts2, -1, (255, 255, 0), 1)
            tiles += [label(ca, "alpha.5"),
                      label(cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), "alpha")]
        rows.append(hstack_pad(tiles))
    w = max(r.shape[1] for r in rows)
    return np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])


def read_video_frame(path, f, fps=30000 / 1001):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{f})",
                          "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mask", "compare", "zoom", "flicker"])
    ap.add_argument("--frames", default="0")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--video", type=Path, default=None, help="graded video (else stills dir)")
    ap.add_argument("--stills", type=Path, default=REPO / "output" / "stills")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fr = [int(x) for x in a.frames.split(",")]

    def graded(f):
        if a.video:
            return read_video_frame(a.video, f)
        return cv2.imread(str(a.stills / f"g_{f:05d}.png"))

    if a.mode == "mask":
        img = mask_sheet(fr, a.crop)
    elif a.mode == "compare":
        cols = []
        for f in fr:
            b, g = frame(f), graded(f)
            pair = np.vstack([label(b, f"before f{f}"), label(g.copy(), "after")])
            cols.append(cv2.resize(pair, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA))
        img = hstack_pad(cols)
    elif a.mode == "zoom":
        rows = []
        for f in fr:
            m = sam(f) > 127
            y0, y1, x0, x1 = bbox(m, 60)
            rows.append(hstack_pad([label(frame(f)[y0:y1, x0:x1].copy(), f"before f{f}"),
                                    label(graded(f)[y0:y1, x0:x1].copy(), "after")]))
        w = max(r.shape[1] for r in rows)
        img = hstack_pad(rows) if len(rows) <= 3 else np.vstack(
            [cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])
        if a.scale != 1:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    else:  # flicker
        cap = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.video), "-f", "rawvideo",
                                "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        H, W = frame(0).shape[:2]
        st = []
        f = 0
        while True:
            buf = cap.stdout.read(H * W * 3)
            if len(buf) < H * W * 3:
                break
            g = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            al = alpha(f)
            m = al > 0.5 if al is not None else np.zeros((H, W), bool)
            hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
            Y = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(float)
            st.append({"f": f, "fishY": float(Y[m].mean()) if m.any() else None,
                       "fishS": float(hsv[..., 1][m].mean()) if m.any() else None,
                       "bgY": float(Y[~m].mean()), "area": int(m.sum())})
            f += 1
        a.out.write_text(json.dumps(st))
        arr = lambda k: np.array([s[k] for s in st if s[k] is not None], float)
        for k in ("fishY", "fishS", "bgY", "area"):
            x = arr(k)
            d = np.abs(np.diff(x))
            print(f"{k}: mean {x.mean():.1f} min {x.min():.1f} max {x.max():.1f} "
                  f"max frame-to-frame jump {d.max():.2f} (p99 {np.percentile(d, 99):.2f})")
        print(f"frames {len(st)}")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(a.out, img.shape)


if __name__ == "__main__":
    main()
