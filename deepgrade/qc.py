"""Quality control: mask stability, output verification, contact sheet.

    python -m deepgrade.qc masks      # temporal stability of the final region weights
    python -m deepgrade.qc verify     # ffprobe source vs graded (res / fps / frames / duration / audio)
    python -m deepgrade.qc sheet      # contact sheet: references + source/graded pairs
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
UP = Path("/root/.claude/uploads/70afcdbd-9d3d-51d2-9790-df796ada4833")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
         "nb_read_packets,duration,color_space,color_primaries,color_transfer,color_range,"
         "sample_rate,channels:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    au = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    return {"video": v, "audio": au, "format_duration": float(d["format"]["duration"])}


def verify(src: Path, out: Path) -> bool:
    a, b = probe(src), probe(out)
    rows = []

    def chk(name, x, y, tol=None):
        ok = (x == y) if tol is None else abs(float(x) - float(y)) <= tol
        rows.append((name, x, y, "PASS" if ok else "FAIL"))

    chk("width", a["video"]["width"], b["video"]["width"])
    chk("height", a["video"]["height"], b["video"]["height"])
    chk("r_frame_rate", a["video"]["r_frame_rate"], b["video"]["r_frame_rate"])
    chk("frames", a["video"]["nb_read_packets"], b["video"]["nb_read_packets"])
    chk("duration_s", a["format_duration"], b["format_duration"], 0.05)
    chk("audio codec", a["audio"]["codec_name"] if a["audio"] else None,
        b["audio"]["codec_name"] if b["audio"] else None)
    chk("audio rate", a["audio"]["sample_rate"] if a["audio"] else None,
        b["audio"]["sample_rate"] if b["audio"] else None)
    chk("audio channels", a["audio"]["channels"] if a["audio"] else None,
        b["audio"]["channels"] if b["audio"] else None)
    rows.append(("pix_fmt", a["video"]["pix_fmt"], b["video"]["pix_fmt"],
                 "PASS" if b["video"]["pix_fmt"] == "yuv420p" else "FAIL"))
    for k in ("color_space", "color_primaries", "color_transfer", "color_range"):
        rows.append((k, a["video"].get(k), b["video"].get(k),
                     "PASS" if b["video"].get(k) in ("bt709", "tv") else "FAIL"))
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"{r[0]:<{w}}  src={str(r[1]):<14} out={str(r[2]):<14} {r[3]}")
    ok = all(r[3] == "PASS" for r in rows)
    print("VERIFY:", "ALL PASS" if ok else "FAILURES")
    return ok


def masks_qc(video: Path) -> None:
    """Temporal stability of the final diver weight.

    Plain consecutive-frame IoU mixes real motion with flicker, so the previous
    mask is first warped into the current frame with DIS optical flow
    (motion-compensated IoU); what is left is jitter / tracking error."""
    from shades.io import iter_frames, probe

    from .build_masks import flow, warp_to

    info = probe(video)
    fin = REPO / "masks" / video.stem / "final"
    W, H = info.width, info.height
    grays = []
    for fr in iter_frames(video, info, scale_w=W // 2):
        g = 0.2126 * fr[..., 0] + 0.7152 * fr[..., 1] + 0.0722 * fr[..., 2]
        grays.append(np.clip(g * 255 + .5, 0, 255).astype(np.uint8))
    prev, rows = None, []
    for t in range(info.n_frames):
        D = cv2.imread(str(fin / "D" / f"{t:05d}.png"), 0).astype(np.float32) / 255
        if prev is not None and prev.max() > 0.5 and D.max() > 0.5:
            mc = warp_to(prev, flow(grays[t], grays[t - 1]), (W, H))
            a, b = mc > 0.5, D > 0.5
            u = np.logical_or(a, b).sum()
            iou_mc = float(np.logical_and(a, b).sum() / u) if u else 1.0
            a0 = prev > 0.5
            u0 = np.logical_or(a0, b).sum()
            iou = float(np.logical_and(a0, b).sum() / u0) if u0 else 1.0
        else:
            iou_mc = iou = 1.0
        rows.append((t, float(D.mean()), iou, iou_mc))
        prev = D
    arr = np.array(rows)
    print(f"frames {len(rows)}  diver area {arr[:,1].min():.4f}..{arr[:,1].max():.4f}")
    print(f"plain IoU          min {arr[:,2].min():.3f}  mean {arr[:,2].mean():.3f}")
    print(f"motion-comp. IoU   min {arr[:,3].min():.3f}  mean {arr[:,3].mean():.3f}  "
          f"p5 {np.percentile(arr[:,3],5):.3f}")
    bad = arr[arr[:, 3] < 0.90]
    print("frames with motion-compensated IoU < 0.90:",
          [(int(r[0]), round(float(r[3]), 3)) for r in bad])
    (REPO / "masks" / video.stem / "qc_masks.json").write_text(json.dumps(
        {"frames": arr[:, 0].astype(int).tolist(), "area": arr[:, 1].tolist(),
         "iou": arr[:, 2].tolist(), "iou_motion_compensated": arr[:, 3].tolist()}))


def sheet(src: Path, out: Path, dst: Path, idx: list[int]) -> None:
    def grab(path: Path, i: int) -> np.ndarray:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                            f"select=eq(n\\,{i}),scale=410:-2", "-frames:v", "1", "-f", "image2pipe",
                            "-vcodec", "png", "-"], capture_output=True)
        return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)

    tiles = []
    for i in idx:
        s, g = grab(src, i), grab(out, i)
        t = np.concatenate([s, g], 1)
        cv2.putText(t, f"frame {i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(t)
    refs = []
    for p in ("dee618b2-image.png", "999f1ff4-image.png"):
        im = cv2.imread(str(UP / p))
        if im is not None:
            refs.append(cv2.resize(im, (410, int(410 * im.shape[0] / im.shape[1]))))
    cols = 2
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles) - len(tiles) % cols, cols)]
    if len(tiles) % cols:
        pad = np.zeros_like(tiles[-1])
        rows.append(np.concatenate([tiles[-1], pad], 1))
    body = np.concatenate(rows, 0)
    if refs:
        h = max(r.shape[0] for r in refs)
        ref_row = np.concatenate([cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in refs]
                                 + [np.zeros((h, body.shape[1] - 410 * len(refs), 3), np.uint8)], 1)
        cv2.putText(ref_row, "reference photographs (targets)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        body = np.concatenate([ref_row, body], 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), body, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet ->", dst, body.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks", "verify", "sheet"])
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--graded", type=Path, default=REPO / "output/player_export_graded.mp4")
    ap.add_argument("--sheet", type=Path, default=REPO / "output/player_export_contact_sheet.jpg")
    ap.add_argument("--frames", default="0,24,48,72,96,120,144,168,184,200,214,229")
    a = ap.parse_args()
    if a.cmd == "masks":
        masks_qc(a.video)
    elif a.cmd == "verify":
        raise SystemExit(0 if verify(a.video, a.graded) else 1)
    else:
        sheet(a.video, a.graded, a.sheet, [int(x) for x in a.frames.split(",")])


if __name__ == "__main__":
    main()
