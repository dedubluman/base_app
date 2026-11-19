#!/usr/bin/env python3
"""grade_dive.py <video> — SAM 3.1 diver mask + skin matte + layered underwater grade (v2 / v3).

    . .venv/bin/activate
    python grade_dive.py inputs/dive.mov                 # full run
    python grade_dive.py inputs/dive.mov --sam-stride 1  # SAM on every frame (slow on CPU)
    python grade_dive.py inputs/dive.mov --masks-only    # only SAM + mask post-processing

Writes out/<stem>_graded.mp4, out/<stem>_mask.mp4, out/<stem>_report.json (grade_params.json is the input).
The source file is only ever read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import gradelib as G  # noqa: E402


# --------------------------------------------------------------------------- probe
def probe(video: Path) -> dict:
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    audio = [s for s in j["streams"] if s["codec_type"] == "audio"]
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot = int(v.get("tags", {}).get("rotate", rot) or rot)
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {
        "w": int(v["width"]), "h": int(v["height"]), "fps": Fraction(num, den or 1),
        "pix_fmt": v["pix_fmt"], "nb_frames": int(v.get("nb_frames") or 0),
        "duration": float(v.get("duration") or j["format"]["duration"]),
        "space": v.get("color_space"), "trc": v.get("color_transfer"), "prim": v.get("color_primaries"),
        "range": v.get("color_range"), "rotation": rot, "has_audio": bool(audio),
        "hdr": (v.get("color_transfer") in ("arib-std-b67", "smpte2084")),
    }


def decode_filter(info) -> str:
    """Decode to full-range-correct RGB. HDR (HLG/PQ) is tone-mapped to SDR Rec.709 first."""
    if info["hdr"]:
        return ("zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,"
                "zscale=p=bt709:t=bt709:m=bt709:r=tv,format=rgb48le")
    return "scale=in_color_matrix=bt709:in_range=tv:flags=accurate_rnd+full_chroma_int,format=rgb48le"


def read_frames(video: Path, info):
    w, h = info["w"], info["h"]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-noautorotate", "-i", str(video), "-an",
                          "-vf", decode_filter(info), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                         stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = w * h * 6
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield (np.frombuffer(b, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535.0)
    p.wait()


# --------------------------------------------------------------------------- SAM 3.1 masks
def sam_masks(video: Path, info, name: str, text: str, stride: int, log=print):
    """Runs SAM 3.1 (video tracker, text prompt) on every `stride`-th frame. Returns (T,H,W) bool and kept idx."""
    from shades.config import Masking
    from shades.masking import load_subject_masks, run_sam31

    mdir = REPO / "masks" / (name if stride == 1 else f"{name}__s{stride}")
    src = video
    if stride > 1:
        src = REPO / ".cache" / f"{name}_s{stride}.mp4"
        src.parent.mkdir(exist_ok=True)
        if not src.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-vf",
                            f"select='not(mod(n\\,{stride}))',setpts=N/FRAME_RATE/TB", "-fps_mode", "passthrough",
                            "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p", str(src)], check=True)
    cfg = Masking(text_prompt=text, parts={}, chunk_frames=32, chunk_overlap=4, max_num_objects=4)
    done = mdir / "meta.json"
    if not done.exists():
        run_sam31(src, mdir, cfg, REPO, REPO / ".cache", log=log)
    m, meta = load_subject_masks(mdir)
    return m > 127, stride


def _sdf(m):
    m = m.astype(np.uint8)
    if not m.any():
        return None
    return cv2.distanceTransform(m, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)


def interp_masks(a, b, t):
    """Shape interpolation between two binary masks via signed distance (t in 0..1)."""
    da, db = _sdf(a), _sdf(b)
    if da is None or db is None:
        return (a if t < .5 else b).copy()
    return ((1 - t) * da + t * db) > 0


def postprocess(sam, stride, n, log=print, feather_sigma=2.5, tsmooth=2, max_fill=8, erode_px=2):
    """sam: (T',H,W) bool. -> soft uint8 masks (n,H,W), stats dict."""
    Tp, H, W = sam.shape
    # 1. expand stride (shape interpolation between SAM frames)
    full = np.zeros((n, H, W), bool)
    for i in range(n):
        k, r = divmod(i, stride)
        k = min(k, Tp - 1)
        if r == 0 or k + 1 >= Tp:
            full[i] = sam[k]
        else:
            full[i] = interp_masks(sam[k], sam[k + 1], r / stride)
    # 2. drop tiny blobs
    for i in range(n):
        if full[i].any():
            nl, lab, st, _ = cv2.connectedComponentsWithStats(full[i].astype(np.uint8), connectivity=8)
            if nl > 2:
                keep = [j for j in range(1, nl) if st[j, cv2.CC_STAT_AREA] >= 0.15 * st[1:, cv2.CC_STAT_AREA].max()]
                full[i] = np.isin(lab, keep)
    if erode_px:   # keeps the grade of the subject off the water around the silhouette (no halo)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        for i in range(n):
            full[i] = cv2.erode(full[i].astype(np.uint8), k) > 0
    area = full.reshape(n, -1).sum(1).astype(float)
    # 3. lost-frame detection (zero, or collapsed vs. neighbours) and gap filling
    med = np.array([np.median(area[max(0, i - 7): i + 8][area[max(0, i - 7): i + 8] > 0]) if (area[max(0, i - 7): i + 8] > 0).any() else 0
                    for i in range(n)])
    lost = (area == 0) | (area < 0.3 * med)
    filled, unfilled = [], []
    i = 0
    while i < n:
        if not lost[i]:
            i += 1
            continue
        j = i
        while j < n and lost[j]:
            j += 1
        a, b = i - 1, j
        if a >= 0 and b < n and (j - i) <= max_fill:
            for k in range(i, j):
                full[k] = interp_masks(full[a], full[b], (k - a) / (b - a))
            filled += list(range(i, j))
        else:
            unfilled += list(range(i, j))
        i = j
    # 4. temporal smoothing (±tsmooth) + feather
    w = np.array([1, 2, 3, 2, 1][2 - tsmooth: 3 + tsmooth], np.float32)
    w /= w.sum()
    soft = np.zeros((n, H, W), np.uint8)
    for i in range(n):
        acc = np.zeros((H, W), np.float32)
        for k, wk in zip(range(i - tsmooth, i + tsmooth + 1), w):
            acc += wk * full[min(max(k, 0), n - 1)].astype(np.float32)
        acc = cv2.GaussianBlur(acc, (0, 0), feather_sigma)
        soft[i] = np.clip(acc * 255 + 0.5, 0, 255).astype(np.uint8)
    # 5. QC numbers
    cov = (soft > 127).reshape(n, -1).mean(1) * 100
    iou = []
    for i in range(1, n):
        a_, b_ = full[i - 1], full[i]
        u = (a_ | b_).sum()
        iou.append(float((a_ & b_).sum() / u) if u else 1.0)
    rel = np.abs(np.diff(area)) / np.maximum(area[:-1], 1)
    stats = {
        "coverage_pct_mean": float(cov.mean()), "coverage_pct_min": float(cov.min()), "coverage_pct_max": float(cov.max()),
        "area_jump_rel_p95": float(np.percentile(rel, 95)), "area_jump_rel_max": float(rel.max()),
        "iou_consec_min": float(min(iou)), "iou_consec_p5": float(np.percentile(iou, 5)),
        "empty_frames": int((area == 0).sum()), "lost_detected": int(lost.sum()),
        "filled_frames": len(filled), "unfilled_frames": unfilled[:50], "unfilled_count": len(unfilled),
        "reseeds": 0,
    }
    return soft, stats


# --------------------------------------------------------------------------- IO
def read_gray_video(path: Path, n: int, H: int, W: int) -> np.ndarray:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    out = np.zeros((n, H, W), np.uint8)
    for i in range(n):
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        out[i] = np.frombuffer(b, np.uint8).reshape(H, W)
    p.stdout.close(); p.wait()
    return out


def write_mask_video(soft, info, out: Path):
    n, H, W = soft.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
                          "-framerate", str(info["fps"]), "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                          "-pix_fmt", "gray", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(soft[i].tobytes())
    p.stdin.close()
    p.wait()


def open_encoder(video: Path, info, out: Path, crf: int):
    w, h = info["w"], info["h"]
    a = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
         "-framerate", str(info["fps"]), "-i", "-", "-i", str(video), "-map", "0:v:0"]
    if info["has_audio"]:
        a += ["-map", "1:a:0", "-c:a", "copy"]
    a += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=ed,format=yuv420p",
          "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
          "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
          "-r", str(info["fps"]), "-movflags", "+faststart", str(out)]
    if info["rotation"]:  # re-attach the source's rotation tag (frames are decoded with -noautorotate)
        a[-1:-1] = ["-metadata:s:v:0", f"rotate={-info['rotation'] % 360}"]
    return subprocess.Popen(a, stdin=subprocess.PIPE)


# --------------------------------------------------------------------------- skin matte
def skin_mattes(video, info, soft, S, log=print):
    """Pass 1: colour-qualifier skin matte per frame on the RAW frame, limited to the diver mask; ±2 frame smoothing."""
    n, H, W = soft.shape
    raw = np.zeros((n, H, W), np.uint8)
    t0 = time.time()
    for i, fr in enumerate(read_frames(video, info)):
        if i >= n:
            break
        raw[i] = np.clip(G.skin_matte(fr, soft[i].astype(np.float32) / 255, S) * 255 + 0.5, 0, 255).astype(np.uint8)
        if i % 200 == 0:
            log(f"[skin] {i}/{n} {time.time() - t0:.0f}s")
    w = np.array([1, 2, 3, 2, 1], np.float32) / 9
    out = np.zeros_like(raw)
    for i in range(n):
        acc = np.zeros((H, W), np.float32)
        for k, wk in zip(range(i - 2, i + 3), w):
            acc += wk * raw[min(max(k, 0), n - 1)]
        out[i] = np.minimum(acc, soft[i]).astype(np.uint8)       # never outside the diver mask
    return out


def area_report(soft, skin, limit=0.25, min_px=300):
    m = (soft > 127).reshape(len(soft), -1).sum(1).astype(float)
    s = (skin > 127).reshape(len(skin), -1).sum(1).astype(float)
    ns = np.maximum(m - s, 0)
    rep = {}
    for name, a in (("skin", s), ("nonskin", ns)):
        rel = np.abs(np.diff(a)) / np.maximum(a[:-1], 1)
        bad = [int(i + 1) for i in np.where((rel > limit) & (np.maximum(a[1:], a[:-1]) > min_px))[0]]
        rep[name] = {"area_px_median": float(np.median(a)), "rel_jump_p95": float(np.percentile(rel, 95)), "rel_jump_max": float(rel.max()),
                     "frames_jump_over_25pct": bad[:60], "count": len(bad)}
    return rep


# --------------------------------------------------------------------------- render
def grade_video(video, info, soft, skin, params, out, crf=17, log=print):
    H, W = info["h"], info["w"]
    B = params["bg"]
    luts = G.make_luts(B)
    vig = G.vignette_map(H, W, params["finish"]["vignette"], params["finish"]["vignette_pow"])
    enc = open_encoder(video, info, out, crf)
    t0 = time.time()
    ema = None   # slow chroma anchor of the graded water: follows depth/cast drift without pumping
    for i, fr in enumerate(read_frames(video, info)):
        if i >= soft.shape[0]:
            break
        m = soft[i].astype(np.float32) / 255
        dab = None
        if B.get("anchor_strength", 0) > 0:
            A0 = G.global_grade(cv2.resize(fr, (W // 4, H // 4), interpolation=cv2.INTER_AREA), B, luts)
            keep = cv2.resize(cv2.dilate((m > 0.004).astype(np.uint8), np.ones((31, 31), np.uint8)), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) == 0
            meas = G.to_lab(A0)[..., 1:][keep].mean(0) if keep.any() else (ema if ema is not None else np.zeros(2, np.float32))
            ema = meas if ema is None else (1 - B["anchor_alpha"]) * ema + B["anchor_alpha"] * meas
            dab = np.clip(B["anchor_strength"] * (np.array(B["anchor_ab"], np.float32) - ema), -B["anchor_clip"], B["anchor_clip"])
        o = G.render(fr, m, skin[i].astype(np.float32) / 255 if skin is not None else None, params, vig=vig, dab=dab, luts=luts)
        enc.stdin.write((o * 65535 + 0.5).astype(np.uint16).tobytes())
        if i % 100 == 0:
            log(f"[grade] {i}/{soft.shape[0]} {time.time() - t0:.0f}s")
    enc.stdin.close()
    enc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out")
    ap.add_argument("--params", type=Path, default=REPO / "out" / "grade_params.json")
    ap.add_argument("--text", default="person", help="SAM 3.1 text prompt for the first frame of every chunk")
    ap.add_argument("--sam-stride", type=int, default=2, help="run SAM on every Nth frame, interpolate in between")
    ap.add_argument("--variant", choices=["v2", "v3"], default="v3",
                    help="v2: skin warm correction through a plain luma qualifier; v3: learned colour qualifier skin matte (+ <name>_skin.mp4)")
    ap.add_argument("--remask", action="store_true", help="ignore out/<name>_mask.mp4 and rebuild the diver mask (SAM cache is still reused)")
    ap.add_argument("--masks-only", action="store_true")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    video = a.video.resolve()
    name = a.name or video.stem
    a.out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    print(f"[probe] {info['w']}x{info['h']} {float(info['fps']):.3f}fps {info['pix_fmt']} {info['space']}/{info['trc']} "
          f"rot={info['rotation']} audio={info['has_audio']} hdr={info['hdr']} frames={info['nb_frames']}")
    if info["hdr"]:
        print("[probe] HDR source -> tone-mapped to SDR Rec.709 before grading")
    n = info["nb_frames"] or int(round(info["duration"] * float(info["fps"])))
    H, W = info["h"], info["w"]

    mpath = a.out_dir / f"{name}_mask.mp4"
    if mpath.exists() and not a.remask:
        soft = read_gray_video(mpath, n, H, W)
        print(f"[mask] reusing {mpath.name}")
    else:
        sam, stride = sam_masks(video, info, name, a.text, a.sam_stride)
        soft, mstats = postprocess(sam, stride, n)
        write_mask_video(soft, info, mpath)
        print("[mask]", json.dumps(mstats))
    params = json.loads(a.params.read_text()) if a.params.exists() else json.loads(json.dumps(G.DEFAULT))
    if a.masks_only:
        return
    S = params["skin"]
    if a.variant == "v2":   # plain luma qualifier, no learned hue/chroma gates
        S = dict(S, clf_w=[8.0, 0, 0, 0, 0], clf_b=0.32)
    params = dict(params, skin=S)
    skin = skin_mattes(video, info, soft, S)
    arep = area_report(soft, skin)
    print("[skin]", json.dumps(arep))
    if a.variant == "v3":
        write_mask_video(skin, info, a.out_dir / f"{name}_skin.mp4")
    out = a.out_dir / f"{name}_graded_{a.variant}.mp4"
    grade_video(video, info, soft, skin, params, out, a.crf)
    (a.out_dir / f"{name}_report_{a.variant}.json").write_text(json.dumps({"video": str(video), "variant": a.variant, "area_report": arep}, indent=1))
    print("[done]", out)


if __name__ == "__main__":
    main()
