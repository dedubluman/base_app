#!/usr/bin/env python3
"""Deterministic, non-generative colour grade of the underwater lionfish clip.

Every output pixel is the source pixel passed through colour operations and
mask-weighted blending. Nothing is synthesised, inpainted, moved or reshaped.

  python grade.py masks --video V --union-dir UNION --out-dir MASKS_FULL
  python grade.py run   --video V --masks MASKS_FULL --reference REF --out OUT.mp4 \
                        --work WORK --frames 10        # first run: 10 frames, then review
All grading parameters live in PARAMS (also written to params.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PARAMS = dict(
    # fixed levels normalisation for the whole clip (8-bit luma, measured on frames 0-29)
    levels_black=56 / 255, levels_white=184 / 255,
    # regions
    feather_subject_px=1.5, feather_fish_px=2.5, feather_bg_px=2.5,
    mask_ema_current=0.6,
    # subject
    reinhard_strength=1.0, subject_ev=0.25, subject_highlight_knee=0.80,
    subject_unsharp_sigma=2.0, subject_unsharp_amount=0.4, subject_red_chroma=0.12,
    # fish: mask disabled -> the small fish are graded as part of the background (they sink into the water colour)
    fish_mask_enabled=False,
    fish_gamma_L=1.15, fish_unsharp_sigma=1.5, fish_unsharp_amount=0.25,
    # background
    bg_hue_rotate_deg=40.0, bg_chroma=0.55, bg_blur_sigma=1.0,
    sand_L=0.87, sand_midtone_contrast=0.90,
    rock_L=0.80, rock_texture_reduce=0.60,
    haze_strength=0.22, haze_lab=(0.55, -0.03, -0.035),
    # global
    vignette_strength=0.5, vignette_falloff_start=0.45, subject_vignette_comp_ev=0.10,
    black_point_lin=0.02, bg_highlight_cap=0.98,
    centroid_ema=0.8,
    # encode
    crf=17, preset="slow", codec="libx264", pix_fmt="yuv420p",
)

# ------------------------------------------------------------------ colour --
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
M1i, M2i = np.linalg.inv(M1), np.linalg.inv(M2)


def srgb_to_lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def lin_to_oklab(c):
    lms = np.cbrt(np.maximum(c @ M1.T, 0))
    return (lms @ M2.T).astype(np.float32)


def oklab_to_lin(lab):
    lms = lab @ M2i.T
    return (lms ** 3 @ M1i.T).astype(np.float32)


def luma709(x):
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# -------------------------------------------------------------------- I/O --
def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(video)],
                         check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), f"{n}/{d}", int(s["nb_frames"])


def read_frames(video, w, h, limit=None):
    """float32 RGB 0..1, BT.709 limited->full range, exactly the coded frames."""
    cmd = ["ffmpeg", "-v", "fatal", "-i", str(video), "-fps_mode", "passthrough"]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += ["-vf", "scale=in_range=tv:in_color_matrix=bt709,format=rgb48le", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 6 * 2)
    size = w * h * 6
    try:
        while True:
            b = p.stdout.read(size)
            if len(b) < size:
                break
            yield np.frombuffer(b, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535.0
    finally:
        if p.poll() is None:
            p.kill()
        p.stdout.close()
        p.wait()


class Encoder:
    def __init__(self, video, out, w, h, fps, P):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
               "-r", fps, "-i", "-", "-i", str(video), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=" + P["pix_fmt"],
               "-c:v", P["codec"], "-preset", P["preset"], "-crf", str(P["crf"]),
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", "-shortest", str(out)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb):
        self.p.stdin.write((np.clip(rgb, 0, 1) * 65535.0 + 0.5).astype(np.uint16).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


# ------------------------------------------------------ per-frame masks ----
def make_masks(video, union_dir, out_dir, stride=3):
    """Stride-3 SAM 3.1 masks -> one lion + one fish alpha per source frame.
    Frames that have a SAM mask are copied unchanged; the two in-between frames are built by
    warping the two neighbouring SAM masks into the frame with dense optical flow (Farneback)
    and averaging them. Documented in meta.json."""
    union_dir, out_dir = Path(union_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h, _, n = probe(video)
    sw, sh = w // 2, h // 2
    grays = [cv2.resize(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (sw, sh),
                        interpolation=cv2.INTER_AREA) for f in read_frames(video, w, h)]
    assert len(grays) == n, (len(grays), n)
    nk = len(list(union_dir.glob("lion_*.png")))

    def load(kind, k):
        p = union_dir / f"{kind}_{k:05d}.png"
        return None if not p.exists() else (cv2.imread(str(p), 0).astype(np.float32) / 255)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    interpolated = []
    for f in range(n):
        k0 = f // stride
        if f % stride == 0 and load("lion", k0) is not None:
            lion, alln = load("lion", k0), load("all", k0)
        else:
            lion = np.zeros((h, w), np.float32)
            alln = np.zeros((h, w), np.float32)
            ws = 0.0
            t = (f - k0 * stride) / stride
            for k, wt in ((k0, 1 - t), (k0 + 1, t)):
                ml, ma = load("lion", k), load("all", k)
                if ml is None or wt <= 0:
                    continue
                fl = cv2.calcOpticalFlowFarneback(grays[f], grays[min(k * stride, n - 1)], None, 0.5, 3, 25, 3, 5, 1.1, 0)
                fl = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
                mx, my = gx + fl[..., 0], gy + fl[..., 1]
                lion += wt * cv2.remap(ml, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                alln += wt * cv2.remap(ma, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                ws += wt
            lion, alln = lion / max(ws, 1e-6), alln / max(ws, 1e-6)
            interpolated.append(f)
        fish = np.clip(alln - lion, 0, 1)
        cv2.imwrite(str(out_dir / f"lion_{f:05d}.png"), (np.clip(lion, 0, 1) * 255 + 0.5).astype(np.uint8))
        cv2.imwrite(str(out_dir / f"fish_{f:05d}.png"), (fish * 255 + 0.5).astype(np.uint8))
    meta = dict(frames=n, width=w, height=h, stride=stride, sam_masks=nk,
                method="SAM 3.1 masks every 3rd frame (copied unchanged); in-between frames = dense optical-flow "
                       "(Farneback) warp of the two neighbouring SAM masks, averaged",
                interpolated_frames=len(interpolated), sam_frames=n - len(interpolated))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


# --------------------------------------------------------------- grade ------
def gauss(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def apply_levels(rgb, P):
    b, w = P["levels_black"], P["levels_white"]
    return np.clip((rgb - b) / (w - b), 0, 1.0)


def unsharp(L, sigma, amount):
    return L + amount * (L - gauss(L, sigma))


def reference_stats(ref_path):
    """Target colour of the lionfish from the reference. The prescribed box (x 28-60 %, y 60-77 %) is
    mostly water (mean hue ~200 deg, teal-grey), which would pull the lionfish towards teal. So the
    statistics use only the lionfish pixels inside that box: warm (a>0.005) or dark (L<0.30)."""
    ref = cv2.cvtColor(cv2.imread(str(ref_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    h, w = ref.shape[:2]
    box = ref[int(0.60 * h):int(0.77 * h), int(0.28 * w):int(0.60 * w)]
    lab = lin_to_oklab(srgb_to_lin(box)).reshape(-1, 3)
    sel = (lab[:, 1] > 0.005) | (lab[:, 0] < 0.30)
    return lab[sel].mean(0), lab[sel].std(0), (box.shape, float(sel.mean()))


def subject_source_stats(video, mask_dir, w, h, P, n=30):
    acc = []
    for i, f in enumerate(read_frames(video, w, h, limit=n)):
        m = cv2.imread(str(Path(mask_dir) / f"lion_{i:05d}.png"), 0) > 200
        lab = lin_to_oklab(srgb_to_lin(apply_levels(f, P)))
        acc.append(lab[m])
    a = np.concatenate(acc)
    return a.mean(0), a.std(0)


def grade_frame(rgb, S, F, B, P, tgt, src, centroid, yy, xx, rockw, sandw, topw):
    h, w = rgb.shape[:2]
    x = apply_levels(rgb, P)
    lin = srgb_to_lin(x)
    lab = lin_to_oklab(lin)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # ---------------- SUBJECT
    ev = 2.0 ** P["subject_ev"]
    lin_s = lin * ev
    k = P["subject_highlight_knee"]
    lin_s = np.where(lin_s > k, k + (lin_s - k) / (1 + (lin_s - k) / (1 - k + 1e-6)), lin_s)  # protect highlights
    lab_s = lin_to_oklab(lin_s)
    mu_t, sd_t = tgt
    mu_s, sd_s = src
    rein = (lab_s - mu_s) / np.maximum(sd_s, 1e-4) * sd_t + mu_t
    lab_s = lab_s + P["reinhard_strength"] * (rein - lab_s)
    lab_s[..., 0] = unsharp(lab_s[..., 0], P["subject_unsharp_sigma"], P["subject_unsharp_amount"])
    hue = np.degrees(np.arctan2(lab_s[..., 2], lab_s[..., 1])) % 360
    wred = np.exp(-0.5 * ((((hue - 40 + 180) % 360) - 180) / 30.0) ** 2)   # reds / oranges (OKLab hue ~ 20-60 deg)
    lab_s[..., 1:] *= (1 + P["subject_red_chroma"] * wred)[..., None]

    # ---------------- FISH
    lab_f = lab.copy()
    lab_f[..., 0] = np.clip(L, 0, 1) ** P["fish_gamma_L"]
    lab_f[..., 0] = unsharp(lab_f[..., 0], P["fish_unsharp_sigma"], P["fish_unsharp_amount"])

    # ---------------- BACKGROUND
    ang = np.radians(P["bg_hue_rotate_deg"])
    ca, sa = np.cos(ang), np.sin(ang)
    lab_b = lab.copy()
    lab_b[..., 1] = (a * ca - b * sa) * P["bg_chroma"]
    lab_b[..., 2] = (a * sa + b * ca) * P["bg_chroma"]
    Lb = gauss(L, P["bg_blur_sigma"])                                    # clarity down (background only)
    # sand (right) darker, midtones flattened; rock wall (left) darker, texture reduced toward the top
    Lb = Lb * (1 - sandw * (1 - P["sand_L"]))
    Lb = 0.45 + (Lb - 0.45) * (1 - sandw * (1 - P["sand_midtone_contrast"]))
    Lb = Lb * (1 - rockw * (1 - P["rock_L"]))
    Lb = Lb + (gauss(Lb, 6.0) - Lb) * (P["rock_texture_reduce"] * rockw * topw)
    lab_b[..., 0] = Lb
    hz = (P["haze_strength"] * smoothstep(0.55, 0.0, yy / h))[..., None]    # depth haze, top half / far background
    lab_b = lab_b * (1 - hz) + np.array(P["haze_lab"], np.float32) * hz

    # ---------------- composite in OKLab, then global
    out_lab = S[..., None] * lab_s + F[..., None] * lab_f + B[..., None] * lab_b
    lin_o = np.clip(oklab_to_lin(out_lab), 0, None)

    # vignette centred on the lionfish centroid; subject compensated +0.1 EV
    R = 0.5 * np.hypot(w, h)
    r = np.hypot(xx - centroid[0], yy - centroid[1]) / R
    vig = 1 - P["vignette_strength"] * smoothstep(P["vignette_falloff_start"], 1.0, r)
    lin_o = lin_o * vig[..., None] * (2.0 ** (P["subject_vignette_comp_ev"] * S))[..., None]
    # black point ~0.02 linear: everything below it is crushed into a short toe that ends at a 0.004
    # linear floor, so blacks are crushed but never clipped to 0. Identical every frame.
    bp = P["black_point_lin"]
    t = smoothstep(0.0, 2 * bp, lin_o)
    lin_o = lin_o * t + 0.004 * (1 - t)
    out = lin_to_srgb(lin_o)
    cap = P["bg_highlight_cap"]                                          # background highlights below 98%
    over = np.maximum(out - 0.90, 0)
    out = np.where(B[..., None] > 0.5, np.minimum(out, 0.90 + over * (cap - 0.90) / np.maximum(over.max(), 1e-6)), out)
    return np.clip(out, 0, 1)


def build_alphas(lion_p, fish_p, P, prev):
    lion = cv2.imread(str(lion_p), 0).astype(np.float32) / 255
    fish = cv2.imread(str(fish_p), 0).astype(np.float32) / 255
    S = np.clip(gauss(lion, P["feather_subject_px"]), 0, 1)
    Fr = np.clip(gauss(fish, P["feather_fish_px"]), 0, 1) * (1 - S)
    if not P["fish_mask_enabled"]:
        Fr = np.zeros_like(Fr)                             # fish stay in BACKGROUND
    if prev is not None:                                   # temporal EMA on alpha only, no trails
        c = P["mask_ema_current"]
        S = c * S + (1 - c) * prev[0]
        Fr = c * Fr + (1 - c) * prev[1]
    Bg = np.clip(1 - S - Fr, 0, 1)
    tot = np.maximum(S + Fr + Bg, 1e-6)                    # normalise: three alphas sum to 1
    return S / tot, Fr / tot, Bg / tot, (S, Fr)


def run(video, mask_dir, ref, out, work, nframes, P):
    w, h, fps, n = probe(video)
    mask_dir, work = Path(mask_dir), Path(work)
    (work / "qa").mkdir(parents=True, exist_ok=True)
    nm = len(list(mask_dir.glob("lion_*.png")))
    if nm != n:
        raise SystemExit(f"STOP: {nm} masks vs {n} frames")
    mu_t, sd_t, bshape = reference_stats(ref)
    mu_s, sd_s = subject_source_stats(video, mask_dir, w, h, P)
    tgt, src = (mu_t, sd_t), (mu_s, sd_s)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rockw = smoothstep(0.42, 0.20, xx / w)                 # static: left wall
    sandw = smoothstep(0.40, 0.60, xx / w) * smoothstep(0.10, 0.30, yy / h)   # static: right sand
    topw = smoothstep(0.85, 0.05, yy / h)                  # texture reduction stronger toward the top
    params = dict(P)
    params.update(reference_box_px=list(bshape[0]), reference_lion_pixel_fraction=bshape[1], reference_mean_oklab=mu_t.tolist(), reference_std_oklab=sd_t.tolist(),
                  subject_src_mean_oklab=mu_s.tolist(), subject_src_std_oklab=sd_s.tolist(),
                  video=str(video), reference=str(ref), frames_processed=nframes, total_frames=n,
                  mask_note=(mask_dir / "meta.json").read_text() if (mask_dir / "meta.json").exists() else "")
    (work / "params.json").write_text(json.dumps(params, indent=1, default=str))
    enc = Encoder(video, out, w, h, fps, P)
    prev, cen = None, None
    outs, srcs, regions = [], [], []
    for i, f in enumerate(read_frames(video, w, h, limit=nframes)):
        S, F, B, prev = build_alphas(mask_dir / f"lion_{i:05d}.png", mask_dir / f"fish_{i:05d}.png", P, prev)
        ys, xs = np.nonzero(S > 0.5)
        c = np.array([xs.mean(), ys.mean()], np.float32) if len(xs) else np.array([w / 2, h / 2], np.float32)
        cen = c if cen is None else P["centroid_ema"] * cen + (1 - P["centroid_ema"]) * c
        g = grade_frame(f, S, F, B, P, tgt, src, cen, yy, xx, rockw, sandw, topw)
        enc.write(g)
        outs.append(g)
        srcs.append(f)
        regions.append((S > 0.5, F > 0.5, B > 0.5))
        print(f"[grade] frame {i + 1}/{nframes}", flush=True)
    enc.close()
    return outs, srcs, regions, (w, h, fps, n)


def qa(outs, srcs, regions, info, work, out_path):
    work = Path(work) / "qa"
    w, h, fps, n = info
    k = len(outs)
    rep = {}
    idx = sorted({0, int(round(0.25 * (k - 1))), int(round(0.5 * (k - 1))), int(round(0.75 * (k - 1))), k - 1})
    for i in idx:
        a = (np.clip(srcs[i], 0, 1) * 255).astype(np.uint8)
        b = (np.clip(outs[i], 0, 1) * 255).astype(np.uint8)
        side = np.hstack([a, b])
        cv2.imwrite(str(work / f"before_after_frame{i:02d}.jpg"), cv2.cvtColor(cv2.resize(side, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # 2 geometry integrity
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                         "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(out_path)],
                        capture_output=True, text=True).stdout
    s = json.loads(pr)["streams"][0]
    rep["geometry"] = dict(out_frames=int(s["nb_read_frames"]), expected_frames=k, out_size=[s["width"], s["height"]],
                           src_size=[w, h], out_fps=s["r_frame_rate"], src_fps=fps)
    cors = []
    for a_, b_, (S_, F_, B_) in zip(srcs, outs, regions):
        ga = cv2.cvtColor((a_ * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        gb = cv2.cvtColor((b_ * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        ea = np.hypot(cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1))
        eb = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
        m = ~cv2.dilate(S_.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        x, y = ea[m] - ea[m].mean(), eb[m] - eb[m].mean()
        cors.append(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())))
    rep["sobel_edge_correlation_outside_subject"] = dict(min=round(min(cors), 4), mean=round(float(np.mean(cors)), 4))
    # 3 flicker
    lm = [float(luma709(o)[B_].mean()) for o, (_, _, B_) in zip(outs, regions)]
    d = np.abs(np.diff(lm)) / np.maximum(np.array(lm[:-1]), 1e-6) * 100
    rep["background_luma_delta_percent"] = dict(max=round(float(d.max()), 3), p95=round(float(np.percentile(d, 95)), 3),
                                                flagged_above_2pct=bool(d.max() > 2.0)) if len(d) else {}
    # 4 clipping
    clip = {}
    for name, ri in (("subject", 0), ("fish", 1), ("background", 2)):
        z = c255 = tot = 0
        for o, r in zip(outs, regions):
            q = (np.clip(o, 0, 1) * 255 + 0.5).astype(np.uint8)[r[ri]]
            z += int((q == 0).all(-1).sum()); c255 += int((q == 255).any(-1).sum()); tot += len(q)
        clip[name] = dict(pct_at_0=round(100 * z / max(tot, 1), 3), pct_at_255=round(100 * c255 / max(tot, 1), 3))
    rep["clipping_percent"] = clip
    rep["clipping_flag_subject_over_1pct"] = bool(clip["subject"]["pct_at_0"] > 1 or clip["subject"]["pct_at_255"] > 1)
    # 5 halo crop: 200x200 along the lionfish edge, 2x zoom
    S_ = regions[min(5, k - 1)][0]
    edge = cv2.morphologyEx(S_.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(edge)
    j = int(np.argmin(xs))                                     # left-most edge point (a fin tip)
    cx, cy = int(np.clip(xs[j] + 60, 100, w - 100)), int(np.clip(ys[j], 100, h - 100))
    sl = (slice(cy - 100, cy + 100), slice(cx - 100, cx + 100))
    ii = min(5, k - 1)
    crop = np.hstack([cv2.resize((srcs[ii][sl] * 255).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                      cv2.resize((np.clip(outs[ii][sl], 0, 1) * 255).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)])
    cv2.imwrite(str(work / "halo_check_2x.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    (work / "qa_summary.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("masks")
    m.add_argument("--video", type=Path, required=True)
    m.add_argument("--union-dir", type=Path, required=True)
    m.add_argument("--out-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--video", type=Path, required=True)
    r.add_argument("--masks", type=Path, required=True)
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "masks":
        make_masks(a.video, a.union_dir, a.out_dir)
    else:
        outs, srcs, regions, info = run(a.video, a.masks, a.reference, a.out, a.work, a.frames, PARAMS)
        qa(outs, srcs, regions, info, a.work, a.out)


if __name__ == "__main__":
    main()
