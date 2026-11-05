"""Two-pass SAM 3.1 diver segmentation for small subjects.

Pass 1 (locate): SAM 3.1 video predictor on every Nth frame of the full
frame, text prompt "person" -> coarse diver track.
Pass 2 (refine): a crop window that follows the (smoothed, interpolated)
track is cut from every frame and SAM 3.1 runs on that crop video. SAM
resizes its input to 1008 px, so a ~400 px crop gives the diver ~2.5x more
pixels than the full frame and recovers thin limbs, fins and the snorkel.
Crop-space masks are pasted back into full-frame coordinates.

Both passes are the real SAM 3.1 model (checkpoints/sam3.1/sam3.1_multiplex.pt)
via shades.masking.run_sam31 (upstream predictor, or its mechanical CPU port).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from shades.config import Masking
from shades.masking import run_sam31


def make_subsampled(video: Path, out: Path, step: int) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                    "-vf", f"select='not(mod(n\\,{step}))'", "-fps_mode", "passthrough",
                    "-an", "-c:v", "libx264", "-crf", "8", str(out)], check=True)


def centroids(mdir: Path) -> dict[int, tuple[float, float, float]]:
    """frame -> (cx, cy, area) for non-empty masks."""
    out = {}
    for p in sorted((mdir / "subject").glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
        if m.sum() > 30:
            ys, xs = np.nonzero(m)
            out[int(p.stem)] = (float(xs.mean()), float(ys.mean()), float(m.sum()))
    return out


def interp_track(keys: dict[int, tuple], step: int, n: int, sigma: float) -> np.ndarray:
    """Interpolate keyframe centroids to every frame and smooth -> (n, 2)."""
    ks = sorted(keys)
    fr = np.array([k * step for k in ks], float)
    xs = np.array([keys[k][0] for k in ks])
    ys = np.array([keys[k][1] for k in ks])
    t = np.arange(n)
    tr = np.stack([np.interp(t, fr, xs), np.interp(t, fr, ys)], 1)
    return gaussian_filter1d(tr, sigma, axis=0, mode="nearest")


def write_crop_video(video: Path, out: Path, track: np.ndarray, size: int,
                     wh: tuple[int, int]) -> np.ndarray:
    """Cut a size x size window centred on the track from every frame
    (lossless-ish) and return the per-frame top-left corners."""
    from shades.io import iter_frames, probe

    w, h = wh
    corners = np.stack([np.clip(np.round(track[:, 0] - size / 2), 0, w - size),
                        np.clip(np.round(track[:, 1] - size / 2), 0, h - size)], 1).astype(int)
    info = probe(video)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size}x{size}", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-crf", "4", "-pix_fmt", "yuv444p", str(out)],
        stdin=subprocess.PIPE)
    for i, f in enumerate(iter_frames(video, info)):
        x0, y0 = corners[min(i, len(corners) - 1)]
        c = f[y0:y0 + size, x0:x0 + size]
        proc.stdin.write((np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    return corners


def segment(video: Path, n: int, wh: tuple[int, int], work: Path, repo: Path,
            step: int = 15, crop: int = 400, log=print) -> Path:
    """Run both passes; return the folder of full-frame masks (<i>.png)."""
    cache = repo / ".cache"
    cache.mkdir(exist_ok=True)
    kv = work / f"keyframes{step}.mp4"
    kdir = repo / "masks" / f"{video.stem}_keys{step}"
    if not (kdir / "meta.json").exists():
        make_subsampled(video, kv, step)
        nk = len(range(0, n, step))
        run_sam31(kv, kdir, Masking(chunk_frames=nk, chunk_overlap=0), repo, cache, log=log)
    keys = centroids(kdir)
    log(f"[mask] pass 1: diver found on {len(keys)} keyframes")
    track = interp_track(keys, step, n, sigma=3.0)
    np.save(work / "sam_track.npy", track)

    cv_path = work / "diver_crop.mp4"
    cdir = repo / "masks" / f"{video.stem}_crop{crop}"
    corners = write_crop_video(video, cv_path, track, crop, wh)
    np.save(work / "crop_corners.npy", corners)
    if not (cdir / "meta.json").exists():
        run_sam31(cv_path, cdir, Masking(chunk_frames=32, chunk_overlap=4), repo, cache, log=log)

    full = repo / "masks" / f"{video.stem}_full"
    (full).mkdir(parents=True, exist_ok=True)
    w, h = wh
    for i in range(n):
        m = np.zeros((h, w), np.uint8)
        p = cdir / "subject" / f"{i:05d}.png"
        if p.exists():
            x0, y0 = corners[i]
            m[y0:y0 + crop, x0:x0 + crop] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(full / f"{i:05d}.png"), m)
    (full / "info.json").write_text(json.dumps({"crop": crop, "step": step,
                                                "keyframes_found": len(keys)}))
    return full
