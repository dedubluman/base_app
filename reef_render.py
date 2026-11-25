#!/usr/bin/env python3
"""Render the reference-matched underwater grade.

  decode (ffmpeg, BT.709 tv -> full-range RGB) -> float32 grade (reef_grade.py) with the SAM 3.1 fish
  node -> TPDF dither -> libx264 High yuv420p BT.709, original AAC copied untouched.

    python reef_render.py --input inputs/reef.mov --masks output/reef_fishmasks.npz \
        --params work/final_params.json --output output/underwater_cinematic_reference_grade_1080p.mp4
"""
import argparse, json, subprocess, time
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from reef_grade import P_DEFAULT, grade, frame_lmed

def probe(path):
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)]))["streams"][0]
    n, d = map(int, j["r_frame_rate"].split("/"))
    return int(j["width"]), int(j["height"]), j["r_frame_rate"], int(j["nb_frames"])

def reader(path, w, h, start=0, end=None):
    vf = "scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=None, help="render only the first N frames (tests)")
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--exp-target", type=float, default=0.43)
    ap.add_argument("--exp-strength", type=float, default=0.4)
    ap.add_argument("--exp-sigma", type=float, default=20.0, help="temporal smoothing of the exposure normaliser (frames)")
    ap.add_argument("--debug-mask-overlay", type=Path, default=None)
    a = ap.parse_args()

    W, H, rfr, nb = probe(a.input)
    P = dict(P_DEFAULT)
    if a.params:
        for k, v in json.loads(a.params.read_text()).items():
            P[k] = tuple(v) if isinstance(v, list) else v
    masks = np.load(a.masks)["masks"] if a.masks else None

    # ---- pass 1: exposure statistics, smoothed in time so the grade never pumps ----
    t0 = time.time()
    small = subprocess.run(["ffmpeg", "-v", "error", "-i", str(a.input), "-vf",
                            "scale=108:192:flags=area:in_range=tv:out_range=pc:in_color_matrix=bt709",
                            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    small = np.frombuffer(small, np.uint8).reshape(-1, 192, 108, 3).astype(np.float32) / 255
    N = len(small) if a.frames is None else min(a.frames, len(small))
    lm = np.array([frame_lmed(f) for f in small])
    lm_s = gaussian_filter1d(lm, a.exp_sigma, mode="nearest")
    lmul = (a.exp_target / lm_s) ** a.exp_strength
    print(f"[pass1] {len(small)} frames; L-median raw {lm.min():.3f}-{lm.max():.3f}, smoothed gain {lmul.min():.3f}-{lmul.max():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass 2: grade + encode ----
    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", rfr, "-i", "-",
           "-i", str(a.input), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
           "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", "-r", rfr, str(a.output)]
    if a.frames:
        enc.insert(-1, "-frames:v"); enc.insert(-1, str(a.frames))
    wr = subprocess.Popen(enc, stdin=subprocess.PIPE)
    rd = reader(a.input, W, H)
    rng = np.random.default_rng(7)
    dbg = None
    t0 = time.time()
    for n in range(N):
        buf = rd.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        x = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
        fm = None
        if masks is not None:
            fm = cv2.resize(masks[min(n, len(masks) - 1)], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        y = grade(x, P, fish=fm, lmul=float(lmul[n]))
        # TPDF dither (1 LSB) so the 8-bit pipe cannot band the smooth teal gradients
        d = (rng.random(y.shape, dtype=np.float32) - rng.random(y.shape, dtype=np.float32))
        o = np.clip(np.rint(y * 255 + d), 0, 255).astype(np.uint8)
        wr.stdin.write(o.tobytes())
        if n % 20 == 0:
            print(f"[render] {n+1}/{N}  {(time.time()-t0)/(n+1):.2f}s/frame", flush=True)
    wr.stdin.close(); wr.wait(); rd.kill()
    print(f"[render] done in {time.time()-t0:.0f}s -> {a.output}", flush=True)

if __name__ == "__main__":
    main()
