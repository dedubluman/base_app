#!/usr/bin/env python3
"""Deep-ocean colour grade for an underwater clip, matched to reference stills.

    python deepocean_grade.py \
        --input  ScreenRecording.mov \
        --reference IMG_1697.jpeg IMG_1696.jpeg \
        --output output/clip_deepocean.mp4

What it does
------------
1. **Probe** the input (ffprobe): size, fps, frame count, audio, rotation.
2. **SAM 3.1 masks** of the swimmer (text prompt "person"; the repo's SAM 3.1
   code, CUDA upstream build or the CPU port in ``shades/sam31_cpu.py``).
   SAM runs on every ``--sam-step``-th frame only (a CPU run of every frame
   would take hours); in-between masks are optical-flow warped from the two
   neighbouring keyframes, cross-faded, smoothed over time and feathered.
3. **Learn the look from the reference stills** (graded, so no original is
   needed): the water/seabed of the reference (everything that is not warm
   subject colour) defines a target distribution per channel. A single static
   per-channel quantile-transfer curve maps the video's own water/seabed
   distribution onto it. That removes the green/teal cast, lowers the black
   point and the haze veil, and lands in navy/cobalt, in one global curve, so
   nothing can flicker between frames.
   The warm subject classes (orange shorts, skin) are matched to the
   reference's subject colours in Lab (mean/std per class), inside the SAM
   mask only, so the trunks stay a vibrant orange against the dark water.
   Bright specular pixels (bubbles, glints) are protected from the grade.
4. **Temporal stability**: static curves, mask smoothing along time, and a
   small clamped per-frame exposure stabiliser on the water.
5. **Encode** with FFmpeg: H.264 (libx264), CRF 17, yuv420p, bt709 tags, original
   audio stream copied, scaled so the SHORT edge is 1080 px with the aspect
   ratio preserved exactly (no padding, no bars, no crop).
6. **Verify** the output: codec, size, aspect ratio, frame count, audio,
   black-bar and empty-frame checks and a look-match check vs the reference.

Resolution policy
-----------------
The ratio is kept exactly. A 1232x1078 source (8:7, almost square) cannot
be 1920x1080 without bars or cropping, so it becomes 1234x1080: the short
edge is 1080 and the long edge follows the aspect ratio (rounded to even).
A portrait source would become 1080xN, a 16:9 landscape one 1920x1080.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parent
log = logging.getLogger("deepocean")


class PipelineError(RuntimeError):
    """A user-facing failure (bad input, missing tool, failed stage)."""


# ---------------------------------------------------------------------------
# Probing and ffmpeg I/O
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: Path
    width: int            # displayed size (rotation applied)
    height: int
    fps: Fraction
    n_frames: int
    duration: float
    has_audio: bool
    audio_codec: str | None
    rotation: int

    @property
    def aspect(self) -> float:
        return self.width / self.height


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError as e:
        raise PipelineError(f"required tool not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"{cmd[0]} failed: {e.stderr.strip()[-600:]}") from e


def probe(path: Path) -> VideoInfo:
    if not path.is_file():
        raise PipelineError(f"input not found: {path}")
    d = json.loads(_run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                         "-of", "json", str(path)]))
    vs = [s for s in d["streams"] if s["codec_type"] == "video"]
    if not vs:
        raise PipelineError("input has no video stream")
    v = vs[0]
    aud = [s for s in d["streams"] if s["codec_type"] == "audio"]
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot = rot or int(v.get("tags", {}).get("rotate", 0))
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90:  # ffmpeg auto-rotates on decode
        w, h = h, w
    fps = Fraction(v.get("avg_frame_rate") or v["r_frame_rate"])
    dur = float(v.get("duration") or d["format"].get("duration") or 0)
    n = int(v["nb_frames"]) if v.get("nb_frames", "N/A").isdigit() else 0
    if not n:
        n = int(_run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                      "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                      str(path)]).strip())
    return VideoInfo(path, w, h, fps, n, dur, bool(aud),
                     aud[0]["codec_name"] if aud else None, rot)


def output_size(w: int, h: int, short_edge: int) -> tuple[int, int]:
    """Scale so the short edge == short_edge; long edge follows the aspect
    ratio (rounded to even). Never pads or crops."""
    s = short_edge / min(w, h)
    even = lambda x: max(2, int(round(x / 2)) * 2)
    return (short_edge, even(h * s)) if w < h else (even(w * s), short_edge)


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_frames(path: Path, w: int, h: int, flags: str = "lanczos"):
    """Yield RGB uint8 (h, w, 3) frames, decoded and scaled by ffmpeg.
    ``-fps_mode passthrough`` keeps frame indices identical in every pass."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}:flags={flags}+accurate_rnd+full_chroma_int",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = w * h * 3
    try:
        while (raw := _read_exact(p.stdout, size)) is not None:
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace").strip()
        p.wait()
        if p.returncode not in (0, None) and err:
            log.warning("ffmpeg decode: %s", err[-300:])


class Encoder:
    """FFmpeg H.264 encoder fed with raw RGB frames; audio copied from source."""

    def __init__(self, info: VideoInfo, out: Path, ow: int, oh: int, crf: int,
                 preset: str, log_path: Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}",
               "-framerate", str(info.fps), "-i", "-"]
        if info.has_audio:
            cmd += ["-i", str(info.path)]
        cmd += ["-map", "0:v:0"]
        if info.has_audio:
            cmd += ["-map", "1:a:0"]
        cmd += [
            # RGB -> YCbCr with the BT.709 matrix, limited range (HD standard)
            "-vf", "scale=out_color_matrix=bt709:out_range=tv:"
                   "flags=accurate_rnd+full_chroma_int,format=yuv420p",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            # aq-mode 3 biases bits to dark areas: fewer bands in deep water
            "-x264-params", "aq-mode=3:aq-strength=1.0",
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart"]
        if info.has_audio:
            cmd += (["-c:a", "copy"] if info.audio_codec in ("aac", "mp3", "ac3", "alac")
                    else ["-c:a", "aac", "-b:a", "192k"])
        cmd += [str(out)]
        self._err = open(log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err)
        self.log_path = log_path

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise PipelineError(f"encoder died: {self.log_path.read_text()[-400:]}") from e

    def close(self) -> None:
        self.proc.stdin.close()
        rc = self.proc.wait()
        self._err.close()
        if rc != 0:
            raise PipelineError(f"ffmpeg encode failed ({rc}): "
                                f"{self.log_path.read_text()[-400:]}")


# ---------------------------------------------------------------------------
# SAM 3.1 masks (keyframes) -> dense soft masks
# ---------------------------------------------------------------------------

def key_indices(n: int, step: int) -> list[int]:
    # trailing frames after the last keyframe are flow-warped from it
    return list(range(0, n, step))


def run_sam_keyframes(info: VideoInfo, a: argparse.Namespace, cache: Path) -> tuple[Path, list[int]]:
    """Run SAM 3.1 on every ``step``-th frame (re-using the repo's chunked,
    resumable tracker) and return (mask dir, keyframe indices)."""
    keys = key_indices(info.n_frames, a.sam_step)
    stem = f"{info.path.stem}_k{a.sam_step}"
    mdir = a.masks_dir or (REPO / "masks" / stem)
    meta = mdir / "meta.json"
    if meta.exists() and json.loads(meta.read_text()).get("num_frames") == len(keys):
        log.info("SAM masks cached: %s (%d keyframes)", mdir, len(keys))
        return mdir, keys

    ckpt = REPO / "checkpoints/sam3.1/sam3.1_multiplex.pt"
    if not ckpt.exists() or not (REPO / "sam3/sam3").is_dir():
        raise PipelineError("SAM 3.1 code/weights missing; run scripts/download_sam3.sh "
                            "(needs HF_TOKEN) or pass --masks-dir with cached masks")
    proxy = cache / f"{stem}.mp4"
    if not proxy.exists():
        proxy.parent.mkdir(parents=True, exist_ok=True)
        log.info("building SAM keyframe proxy (%d of %d frames)", len(keys), info.n_frames)
        sel = f"not(mod(n\\,{a.sam_step}))"
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(info.path), "-an", "-fps_mode",
              "passthrough", "-vf", f"select='{sel}',setpts=N/TB", "-fps_mode", "vfr",
              "-c:v", "libx264", "-crf", "3", "-pix_fmt", "yuv420p", str(proxy)])
    sys.path.insert(0, str(REPO))
    from shades.config import Masking
    from shades.masking import run_sam31
    cfg = Masking(text_prompt=a.text_prompt, parts={}, chunk_frames=a.sam_chunk,
                  chunk_overlap=4, max_num_objects=4, detector_batch=2)
    log.info("running SAM 3.1 on %d keyframes (CPU port takes ~15 s/frame)", len(keys))
    run_sam31(proxy, mdir, cfg, REPO, cache, log=log.info)
    return mdir, keys


def load_keyframe_masks(mdir: Path, keys: list[int], sw: int, sh: int) -> dict[int, np.ndarray]:
    out = {}
    for i, k in enumerate(keys):
        m = cv2.imread(str(mdir / "subject" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        out[k] = m
    return out


def dense_masks(gray: np.ndarray, kmasks: dict[int, np.ndarray], a: argparse.Namespace) -> np.ndarray:
    """Per-frame soft mask (n, h, w) float32 in [0, 1] at working resolution.

    Frames between two valid keyframes: both keyframe masks are warped to the
    frame with Farneback optical flow and cross-faded. Then: temporal Gaussian
    smoothing, optional growth and spatial feathering."""
    n, h, w = gray.shape
    min_area = 4e-4 * h * w
    valid = sorted(k for k, m in kmasks.items() if m.sum() > min_area)
    if not valid:
        raise PipelineError("SAM produced no usable subject mask on any keyframe")
    if len(valid) < len(kmasks):
        log.warning("%d keyframes had no subject mask; interpolating across them",
                    len(kmasks) - len(valid))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fl = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
              poly_sigma=1.2, flags=0)

    def warp(src_k: int, t: int) -> np.ndarray:
        if src_k == t:
            return kmasks[src_k]
        flow = cv2.calcOpticalFlowFarneback(gray[t], gray[src_k], None, **fl)
        return cv2.remap(kmasks[src_k], xx + flow[..., 0], yy + flow[..., 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    out = np.zeros((n, h, w), np.float32)
    vk = np.array(valid)
    for t in range(n):
        j = int(np.searchsorted(vk, t))          # first valid keyframe >= t
        if j < len(vk) and vk[j] == t:
            prev = nxt = int(vk[j])
        else:
            prev = int(vk[j - 1]) if j > 0 else None
            nxt = int(vk[j]) if j < len(vk) else None
        if prev is not None and nxt is not None and prev != nxt:
            wgt = (t - prev) / (nxt - prev)
            out[t] = (1 - wgt) * warp(prev, t) + wgt * warp(nxt, t)
        else:
            out[t] = warp(prev if prev is not None else nxt, t)
    if a.mask_grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.mask_grow + 1,) * 2)
        out = np.stack([cv2.dilate(m, k) for m in out])
    if a.mask_smooth > 0:
        out = gaussian_filter1d(out, a.mask_smooth, axis=0, mode="nearest")
    if a.feather > 0:
        out = np.stack([cv2.GaussianBlur(m, (0, 0), a.feather) for m in out])
    return np.clip(out, 0, 1)


def guided_filter(guide: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-aware refinement of a soft mask (He et al.), box-filter version."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)
    mi, mp = box(guide), box(p)
    cov = box(guide * p) - mi * mp
    var = box(guide * guide) - mi * mi
    aa = cov / (var + eps)
    bb = mp - aa * mi
    return box(aa) * guide + box(bb)


# ---------------------------------------------------------------------------
# The look: learned from the reference stills
# ---------------------------------------------------------------------------

@dataclass
class ClassStats:
    mu: np.ndarray   # Lab mean
    sd: np.ndarray   # Lab std
    n: int = 0


@dataclass
class Look:
    lut: np.ndarray                      # (3, 256) float32, output 0..255
    skin_src: ClassStats
    shorts_src: ClassStats
    skin_ref: ClassStats
    shorts_ref: ClassStats
    sat_thr_src: float                   # saturation split skin | shorts (video)
    ref_bg_median: np.ndarray            # for the final look-match check
    ref_bg_p: np.ndarray                 # (3, 3) p1/p50/p99 of reference water
    notes: dict = field(default_factory=dict)


def _bg_rule(rgb: np.ndarray) -> np.ndarray:
    """Water/seabed in a graded reference: blue-dominant pixels."""
    r, b = rgb[..., 0].astype(np.int16), rgb[..., 2].astype(np.int16)
    return b > r + 6


def _warm_rule(rgb: np.ndarray) -> np.ndarray:
    r, b = rgb[..., 0].astype(np.int16), rgb[..., 2].astype(np.int16)
    return (r > b + 10) & (r > 60)


def _sat(rgb: np.ndarray) -> np.ndarray:
    """Class feature separating orange shorts (high) from skin (low): 1 - G/R.
    Hue-like, so sunlit/caustic skin (bright, saturated) is not mistaken for
    the shorts the way a saturation split would."""
    r = np.maximum(rgb[..., 0].astype(np.float32), 1)
    return 1 - np.clip(rgb[..., 1].astype(np.float32) / r, 0, 1)


def _lab(px: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 RGB -> (N, 3) float Lab (L 0..100)."""
    return cv2.cvtColor((px.astype(np.float32) / 255)[:, None, :], cv2.COLOR_RGB2LAB)[:, 0, :]


def _class_stats(px: np.ndarray, sat: np.ndarray, thr: float):
    out = []
    for sel in (sat <= thr, sat > thr):
        lab = _lab(px[sel]) if sel.sum() else np.zeros((0, 3), np.float32)
        out.append(ClassStats(lab.mean(0), lab.std(0) + 1e-3, int(sel.sum())) if len(lab) else None)
    return out


def _otsu(sat: np.ndarray) -> float:
    t, _ = cv2.threshold((sat * 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t / 255


def _subsample(x: np.ndarray, cap: int, rng) -> np.ndarray:
    return x if len(x) <= cap else x[rng.choice(len(x), cap, replace=False)]


def quantile_transfer(src_hist: np.ndarray, ref_hist: np.ndarray) -> np.ndarray:
    """Monotone curve T with CDF_ref(T(x)) = CDF_src(x); histograms are
    smoothed so 8-bit gaps do not turn into stair-steps (banding)."""
    def cdf(h):
        h = gaussian_filter1d(h.astype(np.float64), 1.5, mode="nearest") + 1e-9
        c = np.cumsum(h)
        return (c - h / 2) / c[-1]
    x = np.arange(256, dtype=np.float64)
    t = np.interp(cdf(src_hist), cdf(ref_hist) + 1e-9 * x, x)
    t = np.maximum.accumulate(gaussian_filter1d(t, 1.0, mode="nearest"))
    return t.astype(np.float32)


def learn_look(refs: list[np.ndarray], frames: np.ndarray, masks: np.ndarray,
               seed: int) -> Look:
    """refs: graded reference stills (RGB uint8). frames/masks: the clip at
    working resolution (n,h,w,3 uint8 / n,h,w float)."""
    rng = np.random.default_rng(seed)
    # ---- reference background (water + seabed) distribution
    ref_hist = np.zeros((3, 256))
    ref_bg_px = []
    for im in refs:
        px = im[_bg_rule(im)]
        if len(px) < 1000:
            raise PipelineError("reference has almost no blue water pixels; "
                                "is it really the graded still?")
        for c in range(3):
            ref_hist[c] += np.bincount(px[:, c], minlength=256) / len(px)  # equal weight per still
        ref_bg_px.append(px)
    ref_bg = _subsample(np.concatenate(ref_bg_px), 400_000, rng)
    # ---- video background distribution (outside the SAM mask)
    bg_sel = masks < 0.02
    src_hist = np.stack([np.bincount(frames[..., c][bg_sel], minlength=256)
                         for c in range(3)]).astype(np.float64)
    lut = np.stack([quantile_transfer(src_hist[c], ref_hist[c]) for c in range(3)])

    # ---- warm subject classes (shorts / skin): Lab mean+std transfer
    sub_sel = (masks > 0.6) & _warm_rule(frames)
    spx = _subsample(frames[sub_sel], 600_000, rng)
    if len(spx) < 500:
        raise PipelineError("too few warm subject pixels inside the SAM masks")
    s_sat = _sat(spx)
    thr_s = _otsu(s_sat)
    rwarm = np.concatenate([im[_warm_rule(im)] for im in refs])
    if len(rwarm) < 500:
        raise PipelineError("reference has no warm subject pixels (shorts/skin)")
    rwarm = _subsample(rwarm, 600_000, rng)
    r_sat = _sat(rwarm)
    thr_r = _otsu(r_sat)
    s_skin, s_short = _class_stats(spx, s_sat, thr_s)
    r_skin, r_short = _class_stats(rwarm, r_sat, thr_r)
    pooled_s = ClassStats(_lab(spx).mean(0), _lab(spx).std(0) + 1e-3, len(spx))
    pooled_r = ClassStats(_lab(rwarm).mean(0), _lab(rwarm).std(0) + 1e-3, len(rwarm))
    fix = lambda cs, pooled: cs if (cs is not None and cs.n >= 500) else pooled
    s_skin, s_short, r_skin, r_short = (fix(s_skin, pooled_s), fix(s_short, pooled_s),
                                        fix(r_skin, pooled_r), fix(r_short, pooled_r))
    p = np.percentile(ref_bg, [1, 50, 99], axis=0).T
    look = Look(lut, s_skin, s_short, r_skin, r_short, thr_s, np.median(ref_bg, 0), p,
                notes=dict(sat_thr_src=round(thr_s, 3), sat_thr_ref=round(thr_r, 3),
                           n_ref_bg=int(sum(len(x) for x in ref_bg_px)),
                           n_src_warm=int(sub_sel.sum()),
                           ref_water_p1_50_99=p.round(0).tolist()))
    log.info("look learned: %s", look.notes)
    return look


# ---------------------------------------------------------------------------
# Applying the grade
# ---------------------------------------------------------------------------

def apply_curve(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Float (h,w,3) 0..255 through three 256-entry curves, linear interpolation
    (float in/out: no 8-bit stair-steps in the dark gradients)."""
    x = np.clip(img, 0, 255)
    i0 = np.minimum(x.astype(np.int32), 254)
    f = x - i0
    out = np.empty_like(x)
    for c in range(3):
        l = lut[c]
        out[..., c] = l[i0[..., c]] * (1 - f[..., c]) + l[i0[..., c] + 1] * f[..., c]
    return out


def _smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _lab_transfer(lab, s: ClassStats, t: ClassStats, tame: float = 1.0):
    """Reinhard-style Lab mean/std transfer s -> t.

    Chroma spread is never amplified (green-lit caustic skin would turn
    salmon); lightness contrast may grow a little. ``tame`` < 1 pulls pixels
    further toward the reference mean, and lightness above mean + 2.2 sd is
    soft-limited (over-exposed skin must not blow out against dark water)."""
    ratio = np.clip(t.sd / s.sd, 0.5, 1.0)
    ratio[0] = np.clip(t.sd[0] / s.sd[0], 0.6, 1.2)
    out = (lab - s.mu) * ratio * tame + t.mu
    lim = 2.2 * t.sd[0]
    dl = out[..., 0] - t.mu[0]
    out[..., 0] = t.mu[0] + np.where(dl > 0, lim * np.tanh(dl / lim), dl)
    return out


class Grader:
    def __init__(self, look: Look, a: argparse.Namespace):
        self.look, self.a = look, a

    def subject_colour(self, rgb255: np.ndarray) -> np.ndarray:
        """Warm classes (shorts/skin) -> reference subject colours (Lab)."""
        lk = self.look
        rgb01 = np.clip(rgb255 / 255, 0, 1).astype(np.float32)
        lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2LAB)
        sat = _sat(rgb01)
        s = (1 / (1 + np.exp(-(sat - lk.sat_thr_src) / 0.04)))[..., None].astype(np.float32)
        lab_o = ((1 - s) * _lab_transfer(lab, lk.skin_src, lk.skin_ref, tame=0.7)
                 + s * _lab_transfer(lab, lk.shorts_src, lk.shorts_ref))
        return np.clip(cv2.cvtColor(lab_o.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1) * 255

    def __call__(self, frame: np.ndarray, mask: np.ndarray | None, gain: float,
                 rng: np.random.Generator) -> np.ndarray:
        a, lk = self.a, self.look
        src = np.clip(frame.astype(np.float32) * gain, 0, 255)
        out = apply_curve(src, lk.lut)                          # water / seabed
        if mask is not None and mask.max() > 3e-3:
            ys, xs = np.where(mask > 0.01)
            pad = 6
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
            crop = src[y0:y1, x0:x1]
            warm = self.subject_colour(crop)
            ww = _smoothstep(crop[..., 0] - crop[..., 2], -14, 6)[..., None]
            subj = out[y0:y1, x0:x1] * (1 - ww) + warm * ww      # cool bits (hair, shadow) stay in-water
            m = mask[y0:y1, x0:x1, None]
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m) + subj * m
        if a.highlight_protect > 0:                              # bubbles / glints
            lum = (0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]) / 255
            h = _smoothstep(lum, 0.70, 0.90) * a.highlight_protect
            if mask is not None:   # sunlit skin is graded, not protected
                h = h * (1 - mask)
            h = h[..., None]
            spec = src * np.array([0.93, 0.99, 1.06], np.float32)   # take the green out
            out = out * (1 - h) + spec * h
        out = out + (rng.random(out.shape, np.float32) - rng.random(out.shape, np.float32))
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)       # triangular dither


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_references(paths: list[Path]) -> list[np.ndarray]:
    out = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise PipelineError(f"cannot read reference image: {p}")
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out


def stage_analyse(info: VideoInfo, a, work: Path):
    """Pass 1: working-resolution frames, SAM masks, dense masks, stabiliser gains."""
    sh = a.work_height
    sw = int(round(info.width * sh / info.height))
    log.info("pass 1: decoding %dx%d working frames", sw, sh)
    frames = np.stack(list(read_frames(info.path, sw, sh, "area")))
    n = len(frames)
    if n != info.n_frames:
        log.warning("decoded %d frames, ffprobe said %d; using decoded count", n, info.n_frames)
        info.n_frames = n
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    mdir, keys = run_sam_keyframes(info, a, a.cache)
    km = load_keyframe_masks(mdir, keys, sw, sh)
    masks = dense_masks(gray, km, a)
    cov = masks.mean((1, 2))
    log.info("subject coverage per frame: min %.3f median %.3f max %.3f",
             cov.min(), np.median(cov), cov.max())
    # exposure stabiliser: remove only fast flicker of the water's brightness
    lum = gray.astype(np.float32)
    logm = np.array([np.log(max(np.median(lum[t][masks[t] < 0.02]) if (masks[t] < 0.02).any()
                                else np.median(lum[t]), 1.0)) for t in range(n)])
    gains = np.exp(gaussian_filter1d(logm, a.stab_sigma, mode="nearest") - logm)
    gains = np.clip(gains, 1 - a.stab_clamp, 1 + a.stab_clamp).astype(np.float32)
    log.info("exposure stabiliser: gain range %.3f..%.3f", gains.min(), gains.max())
    return frames, masks, gains


def stage_render(info, a, look, masks, gains, out: Path, ow: int, oh: int, work: Path) -> None:
    sh, sw = masks.shape[1:]
    grader = Grader(look, a)
    rng = np.random.default_rng(a.seed)
    enc = Encoder(info, out, ow, oh, a.crf, a.preset, work / "encode.log")
    prev_dir = work / "preview"
    prev_idx = set(np.linspace(0, info.n_frames - 1, a.preview).astype(int)) if a.preview else set()
    if prev_idx:
        prev_dir.mkdir(parents=True, exist_ok=True)
    t0, n_done = time.time(), 0
    log.info("pass 2: grading + encoding %d frames at %dx%d", info.n_frames, ow, oh)
    try:
        for t, fr in enumerate(read_frames(info.path, ow, oh, "lanczos")):
            if t >= info.n_frames:
                break
            m = None
            if masks[t].max() > 3e-3:
                m = cv2.resize(masks[t], (ow, oh), interpolation=cv2.INTER_LINEAR)
                if a.guided:
                    g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
                    m = np.clip(guided_filter(g, m, a.guided_radius, 2e-4), 0, 1)
            res = grader(fr, m, float(gains[t]), rng)
            enc.write(res)
            if t in prev_idx:
                cv2.imwrite(str(prev_dir / f"{t:05d}.jpg"),
                            cv2.cvtColor(np.hstack([fr, res]), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_done += 1
            if n_done % 50 == 0:
                el = time.time() - t0
                log.info("  %d/%d frames  %.1f fps  eta %.0fs", n_done, info.n_frames,
                         n_done / el, el / n_done * (info.n_frames - n_done))
    except BaseException:
        enc.proc.kill()
        raise
    enc.close()
    if n_done != info.n_frames:
        raise PipelineError(f"rendered {n_done} frames, expected {info.n_frames}")
    log.info("encoded %s (%.1f MB)", out, out.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_output(out: Path, src: VideoInfo, ow: int, oh: int, look: Look | None,
                  ref_px_medians: np.ndarray | None = None) -> dict:
    """Checks on the finished file; returns a report dict (``ok`` overall)."""
    rep: dict = {"file": str(out), "checks": {}}
    chk = rep["checks"]
    o = probe(out)
    v = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=codec_name,pix_fmt,profile", "-of", "json", str(out)]))["streams"][0]
    chk["codec_h264"] = v["codec_name"] == "h264"
    chk["pix_fmt_yuv420p"] = v["pix_fmt"] == "yuv420p"
    chk["container_mp4"] = out.suffix.lower() == ".mp4"
    chk["size_expected"] = (o.width, o.height) == (ow, oh)
    chk["short_edge_1080"] = min(o.width, o.height) == 1080
    chk["aspect_preserved"] = abs(o.aspect / src.aspect - 1) < 0.002
    chk["frame_count_equal"] = o.n_frames == src.n_frames
    chk["fps_equal"] = abs(float(o.fps) - float(src.fps)) < 0.01
    chk["duration_close"] = abs(o.duration - src.duration) < 0.25
    chk["audio_kept"] = (not src.has_audio) or o.has_audio
    rep.update(width=o.width, height=o.height, aspect=round(o.aspect, 5),
               src_aspect=round(src.aspect, 5), frames=o.n_frames, fps=float(o.fps),
               codec=v["codec_name"], pix_fmt=v["pix_fmt"], audio=o.audio_codec)
    # pixel-level checks on every decoded frame
    bars = empty = 0
    meds, n = [], 0
    for fr in read_frames(out, o.width, o.height, "bicubic"):
        n += 1
        y = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        if y.std() < 2.0 or y.mean() < 3.0:
            empty += 1
        strips = [y[:4], y[-4:], y[:, :4], y[:, -4:]]       # 4-px edge strips
        if any(s.max() <= 12 for s in strips):
            bars += 1
        if look is not None and n % 15 == 1:
            px = fr[_bg_rule(fr)]
            if len(px) > 2000:
                meds.append(np.median(px, 0))
    chk["no_empty_frames"] = empty == 0
    chk["no_black_bars"] = bars == 0
    chk["decoded_all_frames"] = n == o.n_frames
    rep.update(frames_decoded=n, frames_empty=empty, frames_with_black_edge=bars)
    if look is not None and meds:
        got = np.median(meds, 0)
        diff = got - look.ref_bg_median
        rep["water_median_rgb"] = got.round(1).tolist()
        rep["reference_water_median_rgb"] = look.ref_bg_median.round(1).tolist()
        chk["water_matches_reference"] = bool(np.abs(diff).max() < 14)
    rep["ok"] = all(chk.values())
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("input / output")
    io.add_argument("--input", type=Path, required=True, help="source video")
    io.add_argument("--reference", type=Path, nargs="+", required=True,
                    help="graded reference still(s) defining the look")
    io.add_argument("--output", type=Path, default=None,
                    help="output .mp4 (default: output/<stem>_deepocean.mp4)")
    io.add_argument("--short-edge", type=int, default=1080,
                    help="short edge of the output; aspect ratio is kept exactly")
    io.add_argument("--crf", type=int, default=17)
    io.add_argument("--preset", default="slow")
    sam = p.add_argument_group("SAM 3.1 masking")
    sam.add_argument("--sam-step", type=int, default=8,
                     help="run SAM on every Nth frame; others are flow-interpolated "
                          "(1 = every frame, ~16 s/frame on 4 CPU cores)")
    sam.add_argument("--sam-chunk", type=int, default=24, help="keyframes per SAM session")
    sam.add_argument("--text-prompt", default="person")
    sam.add_argument("--masks-dir", type=Path, default=None, help="re-use cached SAM masks")
    mk = p.add_argument_group("mask refinement")
    mk.add_argument("--mask-grow", type=int, default=1, help="dilate masks (working-res px)")
    mk.add_argument("--mask-smooth", type=float, default=1.2, help="temporal sigma (frames)")
    mk.add_argument("--feather", type=float, default=1.5, help="edge feather (working-res px)")
    mk.add_argument("--no-guided", dest="guided", action="store_false",
                    help="skip guided-filter edge snapping")
    mk.add_argument("--guided-radius", type=int, default=8)
    gr = p.add_argument_group("grade")
    gr.add_argument("--highlight-protect", type=float, default=0.8,
                    help="0..1 keep specular bubbles/glints out of the grade")
    gr.add_argument("--stab-sigma", type=float, default=5.0, help="exposure stabiliser (frames)")
    gr.add_argument("--stab-clamp", type=float, default=0.06, help="max per-frame gain correction")
    gr.add_argument("--work-height", type=int, default=270, help="analysis resolution height")
    gr.add_argument("--seed", type=int, default=7)
    ru = p.add_argument_group("run control")
    ru.add_argument("--cache", type=Path, default=REPO / ".cache")
    ru.add_argument("--preview", type=int, default=0, help="save N before|after frames")
    ru.add_argument("--skip-verify", action="store_true")
    ru.add_argument("--verify-only", action="store_true", help="only verify an existing --output")
    ru.add_argument("--clean", action="store_true", help="remove intermediate cache on success")
    ru.add_argument("--clean-masks", action="store_true", help="with --clean, also delete SAM masks")
    ru.add_argument("--log-file", type=Path, default=None)
    ru.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    if a.sam_step < 1:
        p.error("--sam-step must be >= 1")
    return a


def setup_logging(a) -> None:
    h = [logging.StreamHandler(sys.stderr)]
    if a.log_file:
        a.log_file.parent.mkdir(parents=True, exist_ok=True)
        h.append(logging.FileHandler(a.log_file))
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, handlers=h,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def run(a: argparse.Namespace) -> int:
    t_start = time.time()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise PipelineError(f"{tool} is not installed / not on PATH")
    info = probe(a.input)
    ow, oh = output_size(info.width, info.height, a.short_edge)
    out = a.output or REPO / "output" / f"{a.input.stem}_deepocean.mp4"
    log.info("input  %s: %dx%d (%.4f) %.3f fps, %d frames, %.2fs, audio=%s rotation=%d",
             a.input.name, info.width, info.height, info.aspect, float(info.fps),
             info.n_frames, info.duration, info.audio_codec, info.rotation)
    log.info("output %dx%d (%.4f): aspect kept (%+.3f%%), no padding, no crop",
             ow, oh, ow / oh, (ow / oh / info.aspect - 1) * 100)
    work = a.cache / f"deepocean_{a.input.stem}"
    work.mkdir(parents=True, exist_ok=True)
    refs = load_references(a.reference)

    look = None
    if not a.verify_only:
        frames, masks, gains = stage_analyse(info, a, work)
        look = learn_look(refs, frames, masks, a.seed)
        (work / "look.json").write_text(json.dumps(look.notes, indent=1))
        del frames
        stage_render(info, a, look, masks, gains, out, ow, oh, work)
    elif not out.exists():
        raise PipelineError(f"--verify-only but {out} does not exist")

    rc = 0
    if not a.skip_verify:
        if look is None:  # verify-only: need the reference stats for the look check
            rb = np.concatenate([r[_bg_rule(r)] for r in refs])
            look = Look(None, None, None, None, None, 0, np.median(rb, 0), None)
        rep = verify_output(out, info, ow, oh, look)
        rep["seconds_total"] = round(time.time() - t_start, 1)
        (out.with_suffix(".report.json")).write_text(json.dumps(rep, indent=1))
        for k, v in rep["checks"].items():
            log.info("  [%s] %s", "PASS" if v else "FAIL", k)
        log.info("verification %s -> %s", "OK" if rep["ok"] else "FAILED",
                 out.with_suffix(".report.json"))
        rc = 0 if rep["ok"] else 2
    if a.clean and rc == 0:
        for d in (work, a.cache / f"sam_frames_{a.input.stem}_k{a.sam_step}"):
            shutil.rmtree(d, ignore_errors=True)
        (a.cache / f"{a.input.stem}_k{a.sam_step}.mp4").unlink(missing_ok=True)
        if a.clean_masks:
            shutil.rmtree(a.masks_dir or REPO / "masks" / f"{a.input.stem}_k{a.sam_step}",
                          ignore_errors=True)
        log.info("intermediate files removed")
    return rc


def main(argv=None) -> int:
    a = parse_args(argv)
    setup_logging(a)
    try:
        return run(a)
    except PipelineError as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        log.error("interrupted (SAM progress is resumable; re-run the same command)")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
