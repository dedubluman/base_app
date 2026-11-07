#!/usr/bin/env python3
"""Weever fish: luminous neon subject over a dark, oppressive deep-sea floor.

Two independent grading layers composited with the SAM 3.1 alpha matte
(masks/<stem>__fish/alpha, see weever_matte.py):

FISH  (inside the matte only)
  * skin luminance is split into an illumination base (sun caustics sliding
    over the body) and a texture detail layer. The base is compressed with
    clip-wide constants, so caustics no longer flicker the fish's colour,
    and the detail (mottling, scales, the row of dark spots) is amplified.
  * warm gradient map on that tone: deep maroon -> fiery red -> neon orange
    -> amber -> golden yellow.
  * cool gradient map: cobalt -> electric blue -> cyan -> turquoise.
  * the warm/cool mix is driven by the fish's own anatomy and skin, never by
    invented patterns: the tail fin and fin margins (from the matte's
    geometry) go cobalt/cyan, and the fish's own pale reticulation (positive
    texture detail) turns electric cyan between orange patches, the
    reference's warm/cool interleave. Mixing is done in Oklab with chroma
    carried through, so warm/cool transitions do not go grey.
  * subtle highlight bloom from the fish only (small, low-strength).

ENVIRONMENT (outside the matte)
  * sand chroma removed, re-toned teal-grey / blue-green from black up;
    fixed (clip-constant) tone curve: deep blacks, restrained caustics, no
    exposure pumping.
  * large-radius local contrast keeps sand texture in the dark.
  * soft pool of light that follows the fish (heavily smoothed), plus a
    heavy elliptical vignette and a faint murky haze.

Frames are read and written through ffmpeg pipes (no lossy intermediates).

    python weever_grade.py --input inputs/weever.mov --out output/weever_graded.mp4
    python weever_grade.py ... --frames 0,150,300 --stills output/stills   # QC stills
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent

# ------------------------------------------------------------------ colour --

def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806675902, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)


def lin_to_oklab(c):
    lms = np.cbrt(np.clip(c @ _M1.T, 0, None))
    return lms @ _M2.T


def oklab_to_lin(l):
    lms = l @ np.linalg.inv(_M2).T
    return (lms ** 3) @ np.linalg.inv(_M1).T


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def ramp(stops, t):
    """Piecewise-linear gradient (in Oklab) over t in 0..1. stops: [(pos, hex)]."""
    pos = np.array([s[0] for s in stops], np.float32)
    lab = lin_to_oklab(srgb_to_lin(np.stack([hexc(s[1]) for s in stops])))
    out = np.empty(t.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = np.interp(t, pos, lab[:, k])
    return out


# palette sampled from the reference frame (neon orange / fiery red body,
# electric cyan reticulation, cobalt tail, golden highlights)
WARM = [(0.00, "#140203"), (0.14, "#4A0806"), (0.30, "#B81E06"), (0.46, "#F04A05"),
        (0.62, "#FF7A0A"), (0.78, "#FFA81C"), (0.90, "#FFD04A"), (1.00, "#FFF0A8")]
COOL = [(0.00, "#02041A"), (0.16, "#08125E"), (0.34, "#0F36B8"), (0.52, "#1477E6"),
        (0.70, "#18BDF2"), (0.85, "#5FE6F2"), (1.00, "#D8FBFF")]
# environment: reference background median ~ RGB(15,26,24), teal-green lean
ENV_SHADOW = hexc("#010303")
ENV_MID = hexc("#213A34")
ENV_HIGH = hexc("#8AA89E")


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------- geometry --

class Anatomy:
    """Per-frame body axis (head -> tail) from the matte; orientation is
    kept consistent frame to frame (head stays the end nearest the previous
    head)."""

    def __init__(self):
        self.prev_head = None

    def __call__(self, al):
        m = (al > 0.5).astype(np.uint8)
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        ax = vt[0]
        proj = (pts - c) @ ax
        lo, hi = np.percentile(proj, [0.5, 99.5])
        # width near each end (perpendicular spread)
        perp = (pts - c) @ np.array([-ax[1], ax[0]])
        L = hi - lo
        wa = perp[proj < lo + 0.2 * L].std()
        wb = perp[proj > hi - 0.2 * L].std()
        endA, endB = c + ax * lo, c + ax * hi
        if self.prev_head is None:
            head = endA if wa > wb else endB  # weever head is the broad end
        else:
            head = endA if np.linalg.norm(endA - self.prev_head) < np.linalg.norm(endB - self.prev_head) else endB
        self.prev_head = head
        sgn = 1.0 if np.allclose(head, endA) else -1.0
        return c, ax * sgn, lo, hi, sgn  # direction head->tail


def body_coords(al, roi, anat_out, fin_open=0.8):
    """s: 0 at head -> 1 at tail along the body axis; rim: 1 at the matte
    edge (fin margins), 0 deep inside the body."""
    y0, y1, x0, x1 = roi
    c, d, lo, hi, sgn = anat_out
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    p = (gx - c[0]) * d[0] + (gy - c[1]) * d[1]
    # p runs from -(head extent) to +(tail extent)
    if sgn > 0:
        a_, b_ = lo, hi
    else:
        a_, b_ = -hi, -lo
    s = np.clip((p - a_) / (b_ - a_), 0, 1)
    m = (al[y0:y1, x0:x1] > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    dmax = np.percentile(dist[m > 0], 99) if m.any() else 1
    # thin anatomy (fins, fin rays, tail margins) = mask minus its opening by
    # a disc of ~half the body width: the body outline itself is NOT a rim
    r = max(3, int(round(fin_open * dmax)))
    body = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    fin = ((m > 0) & (body == 0)).astype(np.float32)
    rim = np.clip(cv2.GaussianBlur(fin, (0, 0), 2.0) * 1.3, 0, 1)
    return s, rim, dist


# ----------------------------------------------------------------- grade --

class Grader:
    def __init__(self, P: dict, H: int, W: int, fish_tone: dict, centers: np.ndarray):
        self.P, self.H, self.W = P, H, W
        self.ft = fish_tone
        self.centers = centers
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.gx, self.gy = gx, gy
        nx, ny = (gx - W / 2) / (W / 2), (gy - H / 2) / (H / 2)
        r = np.sqrt((nx * 0.95) ** 2 + (ny * 0.75) ** 2)
        self.vig = (1 - P["vignette"] * smoothstep(0.35, 1.45, r)).astype(np.float32)
        self.anat = Anatomy()
        self.debug = None

    # ---------------- environment ----------------
    def env(self, rgb, f):
        P = self.P
        lin = srgb_to_lin(rgb)
        Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        # large-radius local contrast (texture survives the darkening)
        logY = np.log(Y + 1e-4)
        base = cv2.GaussianBlur(logY, (0, 0), 24)
        logY = base + P["env_detail"] * (logY - base)
        y = np.exp(logY)
        # fixed tone: exposure, then restrain caustic highlights
        y = P["env_exposure"] * y
        y = y / (1 + y / P["env_knee"])
        # pool of light around the fish (smoothed path) + vignette
        cx, cy = self.centers[f]
        d = np.sqrt(((self.gx - cx) / (self.W * 0.75)) ** 2 + ((self.gy - cy) / (self.H * 0.42)) ** 2)
        pool = 1 - P["pool"] * smoothstep(0.15, 1.1, d)
        y = y * pool * self.vig
        # display tone + teal re-tint (luminance-keyed shadow/mid/high)
        t = lin_to_srgb(y)[..., None]
        sh = srgb_to_lin(ENV_SHADOW)
        md = srgb_to_lin(ENV_MID)
        hi = srgb_to_lin(ENV_HIGH)
        tint = np.where(t < 0.35, sh + (md - sh) * (t / 0.35),
                        md + (hi - md) * np.clip((t - 0.35) / 0.65, 0, 1))
        # normalise tint to the target luminance so the curve sets brightness
        tY = tint @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        out = tint * (y / np.maximum(tY, 1e-5))[..., None]
        # keep a breath of the original chroma, rotated cold (not the sand's orange)
        gray = Y[..., None]
        out = out + P["env_keep_chroma"] * (lin - gray) * np.array([0.3, 0.9, 1.0], np.float32) * (y / np.maximum(Y, 1e-4))[..., None]
        # murky haze
        out = out + P["haze"] * srgb_to_lin(ENV_MID) * self.vig[..., None]
        return np.clip(out, 0, None)

    # ---------------- fish ----------------
    def fish(self, rgb, al, f):
        P, ft = self.P, self.ft
        H, W = al.shape
        ys, xs = np.nonzero(al > 0.02)
        y0, y1 = max(0, ys.min() - 24), min(H, ys.max() + 25)
        x0, x1 = max(0, xs.min() - 24), min(W, xs.max() + 25)
        roi = (y0, y1, x0, x1)
        a = al[y0:y1, x0:x1]
        src = rgb[y0:y1, x0:x1]
        lin = srgb_to_lin(src)
        Y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        logY = np.log(Y + 1e-3)
        # edge-preserving denoise: keeps scales / spots, drops codec noise
        logY = cv2.bilateralFilter(logY.astype(np.float32), 0, P["fish_dn_range"], P["fish_dn_space"])
        # masked normalised blur: illumination base from fish pixels only
        w = (a > 0.5).astype(np.float32)
        sig = P["fish_base_sigma"]
        num = cv2.GaussianBlur(logY * w, (0, 0), sig)
        den = cv2.GaussianBlur(w, (0, 0), sig)
        base = num / np.maximum(den, 1e-4)
        detail = logY - base
        # clip-constant normalisation (no per-frame auto exposure)
        bc = (base - ft["base_mid"]) * P["fish_base_keep"]
        tone = ft["base_mid"] + bc + P["fish_detail"] * detail
        t = (tone - ft["lo"]) / (ft["hi"] - ft["lo"])
        t = np.clip(t, 0, 1) ** P["fish_gamma"]

        an = self.anat(al)
        s, rim, dist = body_coords(al, roi, an, P["fin_open"])
        # cool weight: tail fin, fin margins, and the fish's own pale reticulation
        tail = smoothstep(P["tail_start"], P["tail_end"], s)
        cool = np.maximum(tail * P["tail_cool"], rim * P["rim_cool"])
        # band-pass: the fish's own fine pale speckles/reticulation only
        dsm = cv2.GaussianBlur(logY, (0, 0), P["retic_sigma"]) - cv2.GaussianBlur(logY, (0, 0), 4.0 * P["retic_sigma"])
        retic = smoothstep(P["retic_lo"], P["retic_hi"], dsm) * (1 - tail)
        cool = np.clip(cool + retic * P["retic_cool"] * (1 - rim * 0.5), 0, 1)

        warm_lab = ramp(WARM, t)
        tc = np.clip(t * P["cool_lift"], 0, 1)
        cool_lab = ramp(COOL, tc)
        k = cool[..., None]
        L = warm_lab[..., :1] * (1 - k) + cool_lab[..., :1] * k
        # LCh blend; hue travels warm -> yellow -> cyan -> blue (never via
        # magenta), chroma interpolated so the crossover stays vivid
        cw = np.linalg.norm(warm_lab[..., 1:], axis=-1, keepdims=True)
        cc = np.linalg.norm(cool_lab[..., 1:], axis=-1, keepdims=True)
        hw = np.arctan2(warm_lab[..., 2:], warm_lab[..., 1:2])
        hc = np.arctan2(cool_lab[..., 2:], cool_lab[..., 1:2])
        hc = np.where(hc < hw, hc + 2 * np.pi, hc)
        h = hw + (hc - hw) * k
        # mute chroma mid-crossover so the hue sweep does not paint a green band
        C = (cw * (1 - k) + cc * k) * (1 - P["cross_mute"] * np.sin(np.pi * k) ** 2)
        ab = np.concatenate([C * np.cos(h), C * np.sin(h)], -1)
        # rounded-body shading: gentle falloff toward the flanks (cylinder)
        dn = dist / max(float(np.percentile(dist[dist > 0], 98)) if (dist > 0).any() else 1.0, 1.0)
        shade = 1 - P["body_shade"] * (1 - smoothstep(0.0, 0.6, dn))
        L = L * shade[..., None]
        out = oklab_to_lin(np.concatenate([L, ab], -1))
        if self.debug is not None:
            self.debug.update(t=t, cool=cool, rim=rim, s=s, retic=retic)
        # keep a trace of the original luminance structure (photographic)
        out = out * (1 - P["fish_src_mix"]) + P["fish_src_mix"] * lin * np.array([1.2, 0.9, 0.7], np.float32)
        out = np.clip(out * P["fish_exposure"], 0, None)
        return out, roi

    def __call__(self, rgb8, al, f):
        rgb = rgb8.astype(np.float32) / 255
        env = self.env(rgb, f)
        if al is None or al.max() < 0.05:
            return lin_to_srgb(env)
        fish, (y0, y1, x0, x1) = self.fish(rgb, al, f)
        out = env
        a = al[y0:y1, x0:x1, None]
        reg = out[y0:y1, x0:x1]
        reg = reg * (1 - a) + fish * a
        # subtle bloom from the brightest fish highlights only
        hl = fish * a * smoothstep(0.35, 1.0, fish.max(-1, keepdims=True))
        g1 = cv2.GaussianBlur(hl, (0, 0), 5)
        g2 = cv2.GaussianBlur(hl, (0, 0), 18)
        reg = reg + self.P["bloom"] * (0.6 * g1 + 0.4 * g2)
        out[y0:y1, x0:x1] = reg
        o = lin_to_srgb(out)
        # soft shoulder to avoid hard clip on the neon highlights
        o = np.where(o > 0.9, 0.9 + 0.1 * np.tanh((o - 0.9) / 0.1), o)
        return o


DEFAULTS = dict(
    vignette=0.80, pool=0.70, env_exposure=0.080, env_knee=0.06, env_detail=1.5,
    env_keep_chroma=0.10, haze=0.0035,
    fish_base_sigma=9.0, fish_base_keep=0.8, fish_detail=1.3, fish_gamma=1.0,
    fish_src_mix=0.0, fish_exposure=1.0,
    tail_start=0.86, tail_end=0.95, tail_cool=0.95, rim_cool=0.0,
    retic_lo=0.28, retic_hi=0.55, retic_cool=0.75, cool_lift=0.95,
    bloom=0.05, cross_mute=0.6, body_shade=0.16, fish_dn_range=0.18, fish_dn_space=2.5, retic_sigma=1.2, fin_open=0.85, tone_below=1.25, tone_above=1.75,
)


# ------------------------------------------------------------------- io --

def load_alpha(adir: Path, f: int):
    p = adir / f"{f:05d}.png"
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535


def fish_stats(fdir: Path, adir: Path, n: int, sigma: float):
    """Clip-constant fish tone normalisation from ~40 frames."""
    bases, tones = [], []
    for f in range(0, n, max(1, n // 40)):
        al = load_alpha(adir, f)
        if al is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(fdir / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        m = al > 0.5
        Y = srgb_to_lin(rgb) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        logY = np.log(Y + 1e-3)
        w = m.astype(np.float32)
        base = cv2.GaussianBlur(logY * w, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-4)
        bases.append(base[m])
        tones.append(logY[m])
    b = np.concatenate(bases)
    return {"base_mid": float(np.median(b))}


def centers_track(adir: Path, n: int, H: int, W: int):
    c = np.zeros((n, 2), np.float32)
    have = np.zeros(n, bool)
    for f in range(n):
        al = load_alpha(adir, f)
        if al is not None and al.max() > 0.5:
            ys, xs = np.nonzero(al > 0.5)
            c[f] = (xs.mean(), ys.mean())
            have[f] = True
    idx = np.nonzero(have)[0]
    for k in range(2):
        c[:, k] = np.interp(np.arange(n), idx, c[idx, k])
    g = cv2.getGaussianKernel(61, 12).ravel()
    for k in range(2):
        c[:, k] = np.convolve(np.pad(c[:, k], 30, mode="edge"), g, mode="valid")
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frames", default=None, help="comma list -> write stills only")
    ap.add_argument("--stills", type=Path, default=REPO / "output" / "stills")
    ap.add_argument("--set", action="append", default=[], help="PARAM=VALUE")
    ap.add_argument("--crf", type=int, default=12)
    a = ap.parse_args()

    P = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = float(v)
    stem = a.input.stem
    fdir = REPO / ".cache" / f"frames_{stem}"
    adir = REPO / "masks" / f"{stem}__fish" / "alpha"
    n = len(list(fdir.glob("*.jpg")))
    H, W = cv2.imread(str(fdir / "00000.jpg")).shape[:2]
    ft = fish_stats(fdir, adir, n, P["fish_base_sigma"])
    # tone window on the normalised fish log-luminance (clip constant)
    ft["lo"] = ft["base_mid"] - P["tone_below"]
    ft["hi"] = ft["base_mid"] + P["tone_above"]
    centers = centers_track(adir, n, H, W)
    G = Grader(P, H, W, ft, centers)
    print(f"[grade] {n} frames {W}x{H}, fish tone {ft}", flush=True)

    if a.frames:
        a.stills.mkdir(parents=True, exist_ok=True)
        for f in [int(x) for x in a.frames.split(",")]:
            # warm the head/tail orientation tracker from frame 0
            G.anat.prev_head = None
            for g in range(0, f + 1, 10):
                al = load_alpha(adir, g)
                if al is not None:
                    G.anat(al)
            rgb = cv2.cvtColor(cv2.imread(str(fdir / f"{f:05d}.jpg")), cv2.COLOR_BGR2RGB)
            G.debug = {}
            o = G(rgb, load_alpha(adir, f), f)
            if G.debug:
                dbg = np.hstack([G.debug[k] for k in ("t", "cool", "rim", "retic")])
                cv2.imwrite(str(a.stills / f"dbg_{f:05d}.png"), np.round(np.clip(dbg, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(str(a.stills / f"g_{f:05d}.png"),
                        cv2.cvtColor(np.round(np.clip(o, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"[grade] still {f}", flush=True)
        return 0

    # full render: decode source with ffmpeg (auto-rotated, identical to SAM frames)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.input), "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}",
                            "-framerate", "30000/1001", "-i", "-",
                            "-i", str(a.input), "-map", "0:v", "-map", "1:a", "-c:a", "copy",
                            "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
                            "-profile:v", "high", "-pix_fmt", "yuv420p",
                            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                            "-color_range", "tv", "-movflags", "+faststart", "-shortest", str(a.out)],
                           stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    sz = H * W * 3
    for f in range(n):
        buf = dec.stdout.read(sz)
        if len(buf) < sz:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        o = G(rgb, load_alpha(adir, f), f)
        # 16-bit out + tiny dither: no banding in the dark water gradients
        o16 = np.clip(o * 65535 + rng.normal(0, 40, o.shape), 0, 65535).astype("<u2")
        enc.stdin.write(o16.tobytes())
        if f % 50 == 0:
            print(f"[grade] frame {f}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(f"[grade] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
