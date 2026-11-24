"""QC for the graded render: stream checks, reference match, flicker, contact sheet.

    python uw_qc.py --src clip.mov --out output/clip_graded.mp4 --masks masks_uw/clip \
        --ref stand:0:ref1.jpg --ref entry:141:ref2.jpg --ref under:255:ref3.jpg \
        --report output/qc_report.json --sheet output/qc_contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from shades.io import iter_frames, probe


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_name,codec_type,profile,width,height,r_frame_rate,pix_fmt,nb_read_packets,"
         "duration,color_space,bit_rate:format=duration,size", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def stream_checks(src: Path, out: Path) -> dict:
    a, b = ffprobe(src), ffprobe(out)
    vs = next(s for s in a["streams"] if s["codec_type"] == "video")
    vo = next(s for s in b["streams"] if s["codec_type"] == "video")
    ao = [s for s in b["streams"] if s["codec_type"] == "audio"]
    asrc = [s for s in a["streams"] if s["codec_type"] == "audio"]
    checks = {
        "codec_is_h264": vo["codec_name"] == "h264",
        "resolution_1080x1920": (vo["width"], vo["height"]) == (1080, 1920),
        "same_orientation_as_source": (vs["width"] < vs["height"]) == (vo["width"] < vo["height"]),
        "pix_fmt_yuv420p": vo["pix_fmt"] == "yuv420p",
        "frame_count_equal": vs["nb_read_packets"] == vo["nb_read_packets"],
        "fps_equal": vs["r_frame_rate"] == vo["r_frame_rate"],
        "duration_within_50ms": abs(float(a["format"]["duration"]) - float(b["format"]["duration"])) < 0.05,
        "audio_present_like_source": bool(ao) == bool(asrc),
    }
    return {"checks": checks, "src": vs, "out": vo, "audio_out": ao[0] if ao else None,
            "duration_src": a["format"]["duration"], "duration_out": b["format"]["duration"],
            "size_mb": round(int(b["format"]["size"]) / 1e6, 1)}


def de2000(a: np.ndarray, b: np.ndarray) -> float:
    la = rgb2lab(np.clip(a, 0, 1).reshape(-1, 1, 3))
    lb = rgb2lab(np.clip(b, 0, 1).reshape(-1, 1, 3))
    return float(deltaE_ciede2000(la, lb).mean())


def load_frames(path: Path, width: int) -> np.ndarray:
    info = probe(path)
    return np.stack([f for f in iter_frames(path, info, scale_w=width)]).astype(np.float32)


def stats(fr: np.ndarray) -> np.ndarray:
    y = fr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    cb = (fr[..., 2] - y) * 0.5389
    cr = (fr[..., 0] - y) * 0.6350
    return np.stack([y.mean((1, 2)), cb.mean((1, 2)), cr.mean((1, 2))], 1)


def flicker(s: np.ndarray) -> np.ndarray:
    d2 = s[2:] - 2 * s[1:-1] + s[:-2]
    return np.sqrt((d2 ** 2).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--ref", action="append", default=[], help="scene:frame:reference.jpg")
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--sheet", required=True, type=Path)
    ap.add_argument("--frames", default="0,60,110,141,175,205,240,270,305")
    a = ap.parse_args()

    rep = {"streams": stream_checks(a.src, a.out)}
    w = 270
    so, sg = load_frames(a.src, w), load_frames(a.out, w)
    n = min(len(so), len(sg))
    so, sg = so[:n], sg[:n]
    ss, sg_s = stats(so), stats(sg)
    fo, fg = flicker(ss), flicker(sg_s)
    rep["flicker_rms_second_diff"] = {
        "source": dict(zip(("luma", "cb", "cr"), fo.round(5).tolist())),
        "graded": dict(zip(("luma", "cb", "cr"), fg.round(5).tolist())),
        "ratio_graded_over_source": dict(zip(("luma", "cb", "cr"), (fg / np.maximum(fo, 1e-9)).round(2).tolist())),
    }
    # worst single-frame jumps of the graded luma vs the source (added pops)
    jo = np.abs(np.diff(ss[:, 0])); jg = np.abs(np.diff(sg_s[:, 0]))
    rep["max_luma_step"] = {"source": round(float(jo.max()), 4), "graded": round(float(jg.max()), 4),
                            "graded_step_over_source_at_same_frame_max":
                            round(float((jg / np.maximum(jo, 1e-3)).max()), 2)}

    # reference match (blur both like the calibration)
    refm = {}
    for spec in a.ref:
        scene, fi, p = spec.split(":", 2)
        fi = int(fi)
        ref = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        info = probe(a.out)
        g = next(f for i, f in enumerate(iter_frames(a.out, info)) if i == fi)
        g = cv2.resize(g, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        refm[scene] = {"frame": fi, "dE2000_vs_reference": round(
            de2000(cv2.GaussianBlur(g, (0, 0), 2), cv2.GaussianBlur(ref, (0, 0), 2)), 2)}
    rep["reference_match"] = refm
    a.report.write_text(json.dumps(rep, indent=1, default=str))

    # contact sheet: original / graded / SAM subject+water overlay on the original
    subj = np.load(a.masks / "subject_full.npy", mmap_mode="r")
    wat = np.load(a.masks / "water_full.npy", mmap_mode="r")
    cols = []
    for f in (int(x) for x in a.frames.split(",")):
        o8 = (so[f] * 255).astype(np.uint8)[..., ::-1].copy()
        g8 = (sg[f] * 255).astype(np.uint8)[..., ::-1].copy()
        ov = o8.copy()
        for m, colr in ((wat[f], (255, 190, 0)), (subj[f], (0, 255, 0))):
            mm = cv2.resize(np.asarray(m), (ov.shape[1], ov.shape[0])) > 127
            ov[mm] = (0.55 * ov[mm] + 0.45 * np.array(colr)).astype(np.uint8)
        for im in (o8, g8, ov):
            cv2.putText(im, f"{f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cols.append(np.vstack([o8, g8, ov]))
    cv2.imwrite(str(a.sheet), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
