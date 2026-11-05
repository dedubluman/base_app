"""diver_dark — SAM 3.1 diver matte + dark, claustrophobic petrol/teal grade.

Foreground (SAM 3.1 "person" + "selfie stick") keeps the diver: skin is
neutralised to a natural, land-like tone and the shadows are opened; the
stick is kept neutral dark grey. Everything else (and the whole frame after
the diver leaves) gets the dark petrol/teal grade with crushed blacks, tamed
sun highlights and a heavy elliptical vignette.

    python diver_dark_sam.py --input SRC --name person --prompt person --stride 2
    python diver_dark_sam.py --input SRC --name stick --prompt "selfie stick" --stride 2
    python diver_dark.py --input SRC
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

REPO = Path(__file__).resolve().parent
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ mattes --

def load_keys(mdir: Path, stride: int, end: int, h: int, w: int) -> np.ndarray:
    """Keyframe masks (SAM 3.1 at every stride-th frame) -> float [K,H,W]."""
    meta = json.loads((mdir / "meta.json").read_text())
    # keys past the cut are not needed (and may hold junk on the seagrass)
    k = min(meta["num_frames"], (end - 1) // stride + 2)
    keys = np.zeros((k, h, w), np.float32)
    for i in range(k):
        p = mdir / "subject" / f"{i:05d}.png"
        if p.exists():
            keys[i] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
    return keys


def clean(m: np.ndarray, min_blob: int, hole: int) -> np.ndarray:
    """Drop small stray blobs, fill small interior holes."""
    b = (m > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_blob
    b = keep[lab].astype(np.uint8)
    inv = 1 - b
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    for j in range(1, n):
        if j not in border and st[j, cv2.CC_STAT_AREA] < hole:
            b[lab == j] = 1
    return b.astype(np.float32)


def flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense flow such that a(x) ~ b(x + f(x)) (half-res grey, uint8)."""
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)


def warp(m: np.ndarray, f_half: np.ndarray) -> np.ndarray:
    h, w = m.shape
    f = cv2.resize(f_half, (w, h)) * 2.0
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(m, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def build_mattes(greys: list[np.ndarray], keys: np.ndarray, stride: int,
                 end: int, min_blob: int, hole: int) -> np.ndarray:
    """Full-rate soft matte [end,H,W] uint8: keyframes cleaned, in-between
    frames from flow-warped neighbouring keyframes, then a motion-compensated
    3-tap temporal filter against edge chatter."""
    h, w = keys.shape[1:]
    raw = np.zeros((end, h, w), np.float32)
    ck = [clean(k, min_blob, hole) for k in keys]
    for i in range(end):
        a = i // stride
        if i % stride == 0 and a < len(ck):
            raw[i] = ck[a]
            continue
        b = a + 1
        ia, ib = a * stride, b * stride
        if b >= len(ck) or ib >= end:
            raw[i] = warp(ck[a], flow(greys[i], greys[ia])) if a < len(ck) else 0
            continue
        t = (i - ia) / (ib - ia)
        ma = warp(ck[a], flow(greys[i], greys[ia]))
        mb = warp(ck[b], flow(greys[i], greys[ib]))
        raw[i] = (1 - t) * ma + t * mb
    out = np.zeros((end, h, w), np.uint8)
    for i in range(end):
        acc, wsum = raw[i] * 0.5, 0.5
        for j in (i - 1, i + 1):
            if 0 <= j < end:
                acc += 0.25 * warp(raw[j], flow(greys[i], greys[j]))
                wsum += 0.25
        out[i] = np.clip(acc / wsum * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def guided_filter(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.boxFilter(I, -1, k), cv2.boxFilter(p, -1, k)
    a = (cv2.boxFilter(I * p, -1, k) - mI * mp) / (cv2.boxFilter(I * I, -1, k) - mI * mI + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def refine(alpha_u8: np.ndarray, y: np.ndarray, feather: float) -> np.ndarray:
    """Edge-aware snap to this frame's luma, then a small feather (~3 px)."""
    p = alpha_u8.astype(np.float32) / 255.0
    if p.max() == 0:
        return p
    # pull the edge ~1 px inward; the transition band itself is colour-
    # decontaminated in decontaminate() so it shows neither a light nor a dark rim
    p = cv2.erode(p, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    a = guided_filter(y, p, 5, 1e-3)
    a = cv2.GaussianBlur(np.clip(a, 0, 1), (0, 0), feather)
    return np.clip(a, 0, 1)


# ------------------------------------------------------------------- grade --

def to_lin(x):
    return np.power(np.clip(x, 0, None), 2.2)


def to_disp(x):
    return np.power(np.clip(x, 0, None), 1 / 2.2)


def soft_knee(x, knee, ceil):
    """Asymptotic roll-off above `knee` toward `ceil` (no hard clip)."""
    over = np.maximum(x - knee, 0)
    rng = ceil - knee
    return np.where(x > knee, knee + rng * (1 - np.exp(-over / rng)), x)


def vignette(h, w, strength, inner, outer):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    t = np.clip((r - inner) / (outer - inner), 0, 1)
    t = t * t * (3 - 2 * t)
    return 1 - strength * t


PETROL = np.array([0.42, 1.04, 1.12], np.float32)  # unit-luma tint direction
PETROL_HI = np.array([0.16, 1.0, 1.10], np.float32)  # for clipped highlights


def grade_bg(x, vig, cfg):
    # clipped sun / surface glints have no colour left; without help they turn
    # grey and read as pink against the teal. Push them fully into petrol.
    ys = x @ LUMA
    whi = np.clip((ys - 0.55) / 0.45, 0, 1)
    whi = (whi * whi * (3 - 2 * whi))[..., None]
    lin = to_lin(x) * 2.0 ** (-cfg["bg_stops"])
    d = to_disp(lin)
    y = d @ LUMA
    chroma = d - y[..., None]
    pdir = (1 - whi) * (PETROL / (PETROL @ LUMA)) + whi * (PETROL_HI / (PETROL_HI @ LUMA))
    tint = y[..., None] * pdir
    t = cfg["tint"] + (cfg["sun_tint"] - cfg["tint"]) * whi
    d = (1 - t) * d + t * (tint + chroma * cfg["chroma_keep"])
    d = np.clip(d, 0, None)
    # hue-preserving highlight roll-off: compress luma, scale RGB with it
    y = d @ LUMA
    yk = soft_knee(y, cfg["hl_knee"], cfg["hl_ceil"])
    d = d * (yk / np.maximum(y, 1e-5))[..., None]
    m = d.max(-1, keepdims=True)
    d = np.where(m > cfg["hl_ceil"] + 0.1, d * (cfg["hl_ceil"] + 0.1) / np.maximum(m, 1e-6), d)
    # crushed (not clipped) blacks: lowered floor + gentle toe
    d = np.clip(d - cfg["black"], 0, None) / (1 - cfg["black"])
    d = np.where(d < 0.25, 0.25 * (d / 0.25) ** cfg["toe"], d)
    # the sun is a distant light source: let it through most of the vignette
    v = vig[..., None] + (1 - vig[..., None]) * cfg["sun_vig_spare"] * whi
    lin = to_lin(d) * v
    return to_disp(lin)


def skin_gains(stats_lin: np.ndarray, target_lin: np.ndarray, max_gain: float, sigma: float):
    """Per-frame channel gains mapping the diver's mean colour to the target
    skin chroma at equal luma; smoothed over time so the WB never pumps."""
    g = np.ones_like(stats_lin)
    valid = stats_lin[:, 1] > 1e-5
    tl = target_lin / (target_lin @ LUMA)
    for i in np.where(valid)[0]:
        m = stats_lin[i]
        g[i] = np.clip(tl * (m @ LUMA) / np.maximum(m, 1e-5), 1 / max_gain, max_gain)
    if valid.any():
        idx = np.arange(len(g))
        for c in range(3):
            g[:, c] = np.interp(idx, idx[valid], g[valid, c])
        k = int(3 * sigma)
        ker = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
        ker /= ker.sum()
        gp = np.pad(g, ((k, k), (0, 0)), mode="edge")
        g = np.stack([np.convolve(gp[:, c], ker, "valid") for c in range(3)], 1)
    return g.astype(np.float32)


def skin_ramp(y, cfg):
    """Three-point skin colour from luma: red-leaning warm undertone in the
    shadows, natural skin in the mids, lighter warm skin in the highlights.
    All three sit on the vectorscope skin-tone line (~123 deg)."""
    sh, md, hi = (np.array(cfg[k], np.float32) for k in ("skin_shadow", "skin_target", "skin_high"))
    sh, md, hi = sh / (sh @ LUMA), md / (md @ LUMA), hi / (hi @ LUMA)
    t = y[..., None]
    w_sh = np.clip(1 - t / 0.35, 0, 1)
    w_hi = np.clip((t - 0.45) / 0.45, 0, 1)
    w_md = 1 - w_sh - w_hi
    tone = w_sh * sh + w_md * md + w_hi * hi
    return t * (1 + cfg["fg_sat"] * (tone - 1))


def skin_hue(rgb: np.ndarray) -> float:
    """Mean vectorscope hue angle (deg, BT.709 Cb/Cr, 0 = +Cb axis, CCW);
    the skin-tone line sits at ~123 deg."""
    y = rgb @ LUMA
    cb = (rgb[:, 2] - y) / 1.8556
    cr = (rgb[:, 0] - y) / 1.5748
    return float(np.degrees(np.arctan2(cr.mean(), cb.mean())) % 360)


def temporal_luma(ymean, sigma, clip):
    """Skin luma per frame -> Gaussian-smoothed over time; the per-frame
    gain pulls each frame onto the smooth curve (no pumping / caustic
    flicker), while slow real changes in light are kept."""
    valid = ymean > 0
    sm = ymean.copy()
    if valid.sum() > 1:
        idx = np.arange(len(ymean))
        filled = np.interp(idx, idx[valid], ymean[valid])
        k = int(3 * sigma)
        ker = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
        ker /= ker.sum()
        sm = np.convolve(np.pad(filled, k, mode="edge"), ker, "valid")
    g = np.where(valid, sm / np.maximum(ymean, 1e-4), 1.0)
    return sm.astype(np.float32), np.clip(g, *clip).astype(np.float32)


def grade_diver(x, gain, vig, cfg, alpha=None, frame_mean=None, lum_gain=1.0):
    """Natural, land-like skin. Backlit underwater skin has almost no red
    left (red ~1/10 of green in linear light), so channel gains alone only
    amplify noise. Keep the diver's own light and shadow (luma), even out the
    large-scale patchiness from water depth and caustics, gently lift the
    mids, then rebuild the colour from a three-point skin ramp. A little of
    the white-balanced original chroma is kept for local variation."""
    y = (x @ LUMA) * lum_gain                      # temporal luminance match
    if alpha is not None and alpha.max() > 0:
        # large-scale luma inside the body only (normalised masked blur),
        # compressed toward the frame's skin mean: smooths patchy torso /
        # shoulder / face shading, keeps fine detail and edges
        a = alpha.astype(np.float32)
        r = cfg["patch_sigma"]
        low = cv2.GaussianBlur(y * a, (0, 0), r) / np.maximum(cv2.GaussianBlur(a, (0, 0), r), 1e-3)
        low = guided_filter(y, low.astype(np.float32), 8, 4e-3)
        m = frame_mean if frame_mean is not None else float((y * a).sum() / max(a.sum(), 1))
        detail = y - low
        y = m + (low - m) * cfg["patch_keep"] + detail
    y = np.clip(y, 0, None)
    y = to_disp(to_lin(y) * 2.0 ** cfg["fg_stops"])
    y = y ** cfg["mid_gamma"]                      # gentle midtone lift
    y = y + cfg["fg_lift"] * (1 - y) ** 3 * y      # open deep shadows
    y = soft_knee(y, cfg["fg_knee"], cfg["fg_ceil"])
    d = skin_ramp(y, cfg)
    wb = to_disp(to_lin(x) * gain)
    c = wb - (wb @ LUMA)[..., None]
    c = cv2.GaussianBlur(c, (0, 0), 2.0)
    d = np.clip(d + cfg["fg_resid"] * c, 0, 1)
    # a touch of the water's coolness so the skin sits in the scene
    cool = np.array(cfg["cool_tone"], np.float32)
    cool = cool / (cool @ LUMA)
    d = (1 - cfg["fg_cool"]) * d + cfg["fg_cool"] * (d @ LUMA)[..., None] * cool
    # part of the vignette so the diver sits in the same space
    v = 1 - cfg["fg_vig"] * (1 - vig)
    return to_disp(to_lin(d) * v[..., None])


def decontaminate(fg, alpha, sigma=3.0, solid=0.9):
    """Edge colour decontamination: inside the matte's transition band, take
    the foreground colour from the solid body just inside it (normalised
    masked blur) instead of from the band pixel itself, which is half water.
    Prevents both the light rim (bright water recoloured as skin) and the dark
    rim (body edge left in the water grade)."""
    core = (alpha >= solid).astype(np.float32)
    if core.sum() < 50:
        return fg
    num = cv2.GaussianBlur(fg * core[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(core, (0, 0), sigma)[..., None]
    inner = num / np.maximum(den, 1e-4)
    w = np.clip((solid - alpha) / solid, 0, 1)[..., None] * (den > 0.02)
    return fg * (1 - w) + inner * w


def grade_stick(x, cfg):
    y = (x @ LUMA)[..., None]
    d = y * cfg["stick_gain"] + (x - y) * 0.15
    return np.clip(d, 0, 1)


CFG = dict(bg_stops=1.3, tint=0.62, chroma_keep=0.55, hl_knee=0.45, hl_ceil=0.74,
           black=0.025, toe=1.35, vig_strength=0.78, vig_inner=0.30, vig_outer=1.30,
           fg_stops=0.75, fg_lift=0.45, mid_gamma=0.88, fg_sat=0.88, fg_resid=0.10,
           fg_cool=0.05, cool_tone=[0.62, 1.0, 1.06], fg_knee=0.60, fg_ceil=0.84,
           skin_shadow=[0.70, 0.44, 0.36], skin_high=[0.90, 0.76, 0.66],
           patch_sigma=18.0, patch_keep=0.65, lum_sigma=6.0, lum_clip=[0.85, 1.18],
           sun_tint=0.88, sun_vig_spare=0.8, fg_vig=0.45, stick_gain=0.85,
           skin_target=[0.80, 0.62, 0.52], max_gain=6.0, wb_sigma=10.0,
           feather=1.6, min_blob=1500, hole=6000)


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=REPO / "output" / "diver_dark_grade.mp4")
    ap.add_argument("--matte-out", type=Path, default=REPO / "output" / "diver_matte.mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--stick-stride", type=int, default=4)
    ap.add_argument("--head-stride", type=int, default=4)
    ap.add_argument("--end", type=int, default=579,
                    help="first frame after the cut (diver and stick gone)")
    ap.add_argument("--checks", type=Path, default=REPO / "output" / "checks")
    ap.add_argument("--check-frames", default="60,270,480,576,660")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N frames")
    a = ap.parse_args()

    info = probe(a.input)
    h, w = info.height, info.width
    end = min(a.end, info.n_frames)
    stem = a.input.stem
    log(f"[in] {w}x{h} {info.n_frames} frames @ {float(info.fps):.3f}, matte frames 0-{end - 1}")

    # pass 1: half-res greys for flow, full-res luma not kept
    greys = []
    for i, f in enumerate(iter_frames(a.input, info, scale_w=w // 2)):
        if i >= end:
            break
        greys.append(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))

    mattes = {}
    for name, s, blob in (("person", a.stride, CFG["min_blob"]),
                          ("stick", a.stick_stride, CFG["min_blob"]),
                          ("head", a.head_stride, 300)):
        mdir = REPO / "masks" / f"{stem}__{name}_s{s}"
        if name == "head" and not (mdir / "meta.json").exists():
            log("[matte] head: no SAM 3.1 run found, skipped")
            continue
        keys = load_keys(mdir, s, end, h, w)
        mattes[name] = build_mattes(greys, keys, s, end, blob, CFG["hole"])
        log(f"[matte] {name}: {len(keys)} SAM 3.1 keyframes -> {end} frames")
        del keys
    if "head" in mattes:
        # diving mask / snorkel / hair belong to the diver, not to the water
        np.maximum(mattes["person"], mattes.pop("head"), out=mattes["person"])
    del greys

    # pass 2: diver colour stats (linear mean inside the eroded body core)
    target = to_lin(np.array(CFG["skin_target"], np.float32))
    stats = np.zeros((info.n_frames, 3), np.float32)
    ymean = np.zeros(info.n_frames, np.float32)
    ker = np.ones((9, 9), np.uint8)
    for i, f in enumerate(iter_frames(a.input, info)):
        if i >= end:
            break
        core = cv2.erode(((mattes["person"][i] > 200) & (mattes["stick"][i] < 60)).astype(np.uint8), ker)
        if core.sum() > 3000:
            lin = to_lin(f[core > 0])
            # skin only: skip near-black pixels (shorts, strap, deep shadow)
            y = lin @ LUMA
            sel = y > np.percentile(y, 35)
            stats[i] = lin[sel].mean(0)
            ymean[i] = float((f[core > 0] @ LUMA).mean())
    gains = skin_gains(stats, target, CFG["max_gain"], CFG["wb_sigma"])
    ysm, lum_gain = temporal_luma(ymean, CFG["lum_sigma"], CFG["lum_clip"])

    vig = vignette(h, w, CFG["vig_strength"], CFG["vig_inner"], CFG["vig_outer"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.checks.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.input, a.output, info, Encode(crf=16, preset="slow"))
    menc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{w}x{h}", "-r", str(info.fps), "-i", "-",
         "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
         str(a.matte_out)], stdin=subprocess.PIPE)
    checks = {int(c) for c in a.check_frames.split(",")}
    report = {"frames": {}, "cfg": CFG}
    for i, x in enumerate(iter_frames(a.input, info)):
        bg = grade_bg(x, vig, CFG)
        if i < end:
            y = x @ LUMA
            ap_ = refine(mattes["person"][i], y, CFG["feather"])
            as_ = refine(mattes["stick"][i], y, CFG["feather"]) * (1 - ap_)
        else:
            ap_ = as_ = np.zeros((h, w), np.float32)
        out = bg
        if ap_.max() > 0 or as_.max() > 0:
            dv = grade_diver(x, gains[i], vig, CFG, alpha=ap_,
                             frame_mean=float(ysm[i]) if ymean[i] > 0 else None,
                             lum_gain=float(lum_gain[i]))
            dv = decontaminate(dv, ap_)
            st = grade_stick(x, CFG) * (vig[..., None] * 0.5 + 0.5)
            out = bg * (1 - ap_ - as_)[..., None] + dv * ap_[..., None] + st * as_[..., None]
        enc.write(out)
        matte = np.clip(ap_ + as_, 0, 1)
        menc.stdin.write((np.clip(ap_ + as_, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        if i in checks:
            sbs = np.concatenate([x, out], 1)
            cv2.imwrite(str(a.checks / f"f{i:04d}_before_after.jpg"),
                        cv2.cvtColor((np.clip(sbs, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(a.checks / f"f{i:04d}_matte.png"), (matte * 255).astype(np.uint8))
        if i < end:
            rec = {"person_area": int((ap_ > 0.5).sum()),
                   "stick_area": int((as_ > 0.5).sum()),
                   "gain": [round(float(v), 3) for v in gains[i]],
                   "lum_gain": round(float(lum_gain[i]), 4)}
            core = (ap_ > 0.97) & (as_ < 0.03)
            if core.sum() > 3000:
                rec["skin_in_y"] = round(float((x[core] @ LUMA).mean()), 4)
                rec["skin_out_y"] = round(float((out[core] @ LUMA).mean()), 4)
                rec["skin_out_hue"] = round(skin_hue(out[core]), 1)
            report["frames"][i] = rec
        if i % 60 == 0:
            log(f"[render] {i}/{info.n_frames}")
        if a.limit and i + 1 >= a.limit:
            break
    enc.close()
    menc.stdin.close()
    menc.wait()
    (a.output.parent / "diver_dark_report.json").write_text(json.dumps(report, indent=1))
    log(f"[done] {a.output}  {a.matte_out}")


if __name__ == "__main__":
    main()
