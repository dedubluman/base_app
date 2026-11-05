"""Grade + composite + 9:16 diver-following reframe + Instagram H.264 encode."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from shades.io import iter_frames, probe, read_frames_at

from .grade import DiverGrade, EnvGrade, EnvGrader, grade_diver, work_luma
from .matte import clean, soft_matte, temporal_vote

OUT_W, OUT_H = 1080, 1920


# ---------------------------------------------------------------- light rays
def ray_angle(frames: list[np.ndarray]) -> float:
    """Light-shaft direction (radians from vertical; +ve leans right at the
    bottom) in the open water above the seabed. Projection search: rotate the
    high-passed upper region and pick the angle whose column sums vary most,
    i.e. where the shafts line up. Median over several frames."""
    est = []
    for rgb in frames:
        h, w = rgb.shape[:2]
        Y = work_luma(rgb[: int(h * 0.22)])
        Y = Y - cv2.GaussianBlur(Y, (0, 0), 6)
        Y = Y - cv2.GaussianBlur(Y, (0, 0), 1.0) * 0  # keep fine shafts
        best, best_a = -1.0, 0.0
        for deg in np.arange(-35, 35.5, 1.0):
            M = cv2.getRotationMatrix2D((Y.shape[1] / 2, Y.shape[0] / 2), deg, 1.0)
            r = cv2.warpAffine(Y, M, Y.shape[::-1], flags=cv2.INTER_LINEAR,
                               borderValue=np.nan)
            valid = ~np.isnan(r)
            cnt = valid.sum(0)
            ok = cnt > 0.8 * Y.shape[0]
            if ok.sum() < 50:
                continue
            prof = np.nansum(r, 0)[ok] / cnt[ok]
            v = float(np.var(prof))
            if v > best:
                best, best_a = v, deg
        est.append(best_a)
    # rotating the image by +deg aligns shafts that lean by -deg
    return float(-np.deg2rad(np.median(est)))


def line_kernel(length: int, angle_from_vertical: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx, dy = np.sin(angle_from_vertical), np.cos(angle_from_vertical)
    for t in np.linspace(-c, c, length * 2):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        k[y, x] = 1
    return k / k.sum()


class RayEnhancer:
    """Volumetric emphasis of the light shafts that already exist in the
    water: high-passed highlights smeared along the measured ray direction,
    weighted by open-water depth. Adds no structure of its own."""

    def __init__(self, angle: float, strength: float, length: int = 61):
        self.k = line_kernel(length, angle)
        self.s = strength
        self.prev = None

    def __call__(self, env: np.ndarray, src: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if self.s <= 0:
            return env
        Y = work_luma(src)
        hp = np.clip(Y - cv2.GaussianBlur(Y, (0, 0), 10), 0, None)
        small = cv2.resize(hp, (hp.shape[1] // 2, hp.shape[0] // 2), interpolation=cv2.INTER_AREA)
        r = cv2.filter2D(small, -1, self.k)
        r = cv2.resize(r, (hp.shape[1], hp.shape[0]), interpolation=cv2.INTER_LINEAR)
        if self.prev is not None:  # temporal smoothing: rays never flicker
            r = 0.6 * self.prev + 0.4 * r
        self.prev = r
        tint = np.array([0.55, 0.85, 1.0], np.float32)
        return env + (self.s * r * depth)[..., None] * tint


# -------------------------------------------------------------- reframe path
def crop_path(alphas_cx: np.ndarray, valid: np.ndarray, src_w: int, crop_w: int,
              sigma: float = 18.0) -> np.ndarray:
    """Smooth, camera-operator-like horizontal crop centre that keeps the diver
    framed. Missing frames are interpolated; the path is heavily smoothed and
    clamped to the frame."""
    t = np.arange(len(alphas_cx))
    cx = np.interp(t, t[valid], alphas_cx[valid]) if valid.any() else np.full(len(t), src_w / 2)
    cx = gaussian_filter1d(cx, sigma, mode="nearest")
    return np.clip(cx - crop_w / 2, 0, src_w - crop_w)


# ------------------------------------------------------------------- encoder
class IGEncoder:
    def __init__(self, dst: Path, src_fps: str, crf: int = 16):
        self.rng = np.random.default_rng(7)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{OUT_W}x{OUT_H}",
               "-r", src_fps, "-i", "-",
               "-vf", "fps=30,scale=out_color_matrix=bt709:out_range=tv:"
                      "flags=accurate_rnd+full_chroma_int,format=yuv420p",
               "-an", "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
               "-crf", str(crf), "-maxrate", "20M", "-bufsize", "40M",
               "-x264-params", "aq-mode=3:deblock=-1,-1",
               "-color_primaries", "bt709", "-color_trc", "bt709",
               "-colorspace", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", str(dst)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        n = (self.rng.random(rgb.shape[:2] + (1,), np.float32)
             - self.rng.random(rgb.shape[:2] + (1,), np.float32)) / 255.0
        self.p.stdin.write((np.clip(rgb + n, 0, 1) * 65535 + 0.5).astype(np.uint16).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("encode failed")


# ---------------------------------------------------------------- the render
def clip_tone_range(video: Path, n: int, k: int = 16) -> tuple[float, float]:
    info = probe(video)
    idx = set(np.linspace(0, n - 1, k).astype(int).tolist())
    ys = [work_luma(f)[::2, ::2].ravel() for i, f in enumerate(iter_frames(video, info)) if i in idx]
    lo, hi = np.percentile(np.concatenate(ys), [0.5, 99.7])
    return float(lo), float(hi)


def load_alphas(video: Path, mask_dir: Path, n: int) -> list[np.ndarray]:
    info = probe(video)
    raw = [cv2.imread(str(mask_dir / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127 for i in range(n)]
    raw = temporal_vote([clean(m) for m in raw])
    alphas = []
    for i, f in enumerate(iter_frames(video, info)):
        if i >= n:
            break
        alphas.append(soft_matte(f, raw[i]))
    return alphas


def render(video: Path, out: Path, mask_dir: Path, env: EnvGrade, diver: DiverGrade,
           rays: float = 0.0, frames: list[int] | None = None, preview_dir: Path | None = None,
           log=print) -> dict:
    info = probe(video)
    n, w, h = info.n_frames, info.width, info.height
    lo, hi = clip_tone_range(video, n)
    env.lo, env.hi = lo - 0.03, hi + 0.02
    alphas = load_alphas(video, mask_dir, n)
    area = np.array([a.sum() for a in alphas])
    cx = np.array([(a.sum(0) * np.arange(w)).sum() / max(a.sum(), 1e-6) for a in alphas])
    valid = area > 30
    crop_w = int(round(h * 9 / 16))
    x0s = crop_path(cx, valid, w, crop_w)

    probe_frames = list(read_frames_at(video, info, [0, n // 3, 2 * n // 3, n - 1]).values())
    grader = EnvGrader(env, (h, w))
    angle = ray_angle(probe_frames)
    log(f"[render] light-shaft angle {np.rad2deg(angle):.1f} deg from vertical")
    rayer = RayEnhancer(angle, rays)
    enc = None if frames else IGEncoder(out, str(info.fps))
    want = set(frames or [])
    dstate: dict = {}
    for i, src in enumerate(iter_frames(video, info)):
        depth = grader.depth_map(src)          # stateful: run on every frame
        if frames and i not in want:
            rayer.prev = None
            continue
        e = grader(src, depth)
        e = rayer(e, src, depth)
        a = alphas[i][..., None]
        if valid[i]:
            d = grade_diver(src, e, diver, alphas[i], dstate)
            img = e * (1 - a) + d * a
        else:
            img = e
        x0 = x0s[i]
        xi = int(np.floor(x0))
        fx = float(x0 - xi)
        crop = img[:, xi:xi + crop_w + 1]
        # sub-pixel pan: shift by the fractional part before resampling
        M = np.float32([[1, 0, -fx], [0, 1, 0]])
        crop = cv2.warpAffine(crop, M, (crop_w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        up = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        # gentle output sharpening to offset the 1.78x upscale
        up = up + 0.30 * (up - cv2.GaussianBlur(up, (0, 0), 1.2))
        up = np.clip(up, 0, 1)
        if enc:
            enc.write(up)
        if preview_dir is not None and i in want:
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / f"f{i:04d}.png"),
                        cv2.cvtColor((up * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(preview_dir / f"f{i:04d}_full.png"),
                        cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 50 == 0:
            log(f"[render] frame {i}/{n}")
    if enc:
        enc.close()
    return {"lo": lo, "hi": hi, "ray_angle_rad": angle, "crop_w": crop_w,
            "frames_with_diver": int(valid.sum()), "n": n,
            "env": asdict(env), "diver": asdict(diver), "rays": rays}
