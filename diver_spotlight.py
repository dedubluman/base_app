#!/usr/bin/env python3
"""Tracked oval spotlight grade for a backlit freediver clip.

    python diver_spotlight.py <video> <outdir> track     # diver track -> <outdir>/track.npy
    python diver_spotlight.py <video> <outdir> preview   # mask_preview.mp4 + compare.jpg
    python diver_spotlight.py <video> <outdir> render    # diver_spotlight.mov

Spec: prompts/diver-oval-mask-grade.md
"""
import cv2, numpy as np, subprocess, sys, json
from scipy.signal import savgol_filter
V, OUTDIR, MODE = sys.argv[1], sys.argv[2], sys.argv[3]   # MODE: track | preview | render
W, H, N = 1232, 1078, 184
P = dict(feather_frac=0.10, pad_x=1.30, pad_y=1.18, stick_ext=0.20,
         lift_stops=1.5, lift_knee=0.50, warm=[1.45, 0.95, 0.62], warm_amt=0.45,
         out_gain=0.50, out_mat=[0.28, 0.74, 0.95], g2b=0.18, out_sat=0.80, crush=0.012, grain=0.8/255)
if len(sys.argv) > 4: P.update(json.loads(sys.argv[4]))

def frames(scale=None, fmt="rgb48le"):
    vf = "scale=in_color_matrix=bt709:in_range=tv" + (f",scale={scale[0]}:{scale[1]}" if scale else "")
    w, h = scale or (W, H); bpp = 6 if fmt == "rgb48le" else 3
    p = subprocess.Popen(["ffmpeg","-v","error","-i",V,"-vf",vf,"-f","rawvideo","-pix_fmt",fmt,"-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w*h*bpp)
        if len(b) < w*h*bpp: break
        a = np.frombuffer(b, np.uint16 if bpp == 6 else np.uint8).reshape(h, w, 3)
        yield a.astype(np.float32) / (65535. if bpp == 6 else 255.)

SW, SH = 308, 270
def seg(s):
    L = s.mean(2); d = cv2.GaussianBlur(L, (0,0), 20) - L
    return cv2.morphologyEx(((d > 0.06) & (L < 0.35)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

if MODE == "track":
    rows = []
    for i, s in enumerate(frames((SW, SH), "rgb24")):
        m = seg(s)
        body = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, lab, st, cen = cv2.connectedComponentsWithStats(body)
        best, ok = None, False
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if cen[k][1] < 0.62*SH and a > 150 and (best is None or a > st[best][4]): best = k
        if best is not None:
            # full-detail extent: original mask pixels within dilated body component
            reg = cv2.dilate((lab == best).astype(np.uint8), np.ones((15,15),np.uint8)) & m
            ys, xs = np.nonzero(reg); ys2 = ys[ys < 0.75*SH]; xs2 = xs[ys < 0.75*SH]
            rows.append([i, xs2.min(), ys2.min(), xs2.max(), ys2.max(), 1]); ok = True
        if not ok: rows.append([i, np.nan, np.nan, np.nan, np.nan, 0])
    r = np.array(rows, float); bad = r[:,5] == 0
    for c in range(1,5):
        v = r[:,c]; idx = np.arange(len(v)); v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    # reject jumps (outliers) vs median filter, then interpolate
    from scipy.ndimage import median_filter
    sx = W/SW; sy = H/SH
    box = r[:,1:5] * [sx, sy, sx, sy]
    med = median_filter(box, size=(9,1), mode="nearest")
    out = np.abs(box - med).max(1) > 120
    for c in range(4):
        v = box[:,c]; idx = np.arange(len(v)); v[out] = np.interp(idx[out], idx[~out], v[~out])
    cx = (box[:,0]+box[:,2])/2; top = box[:,1]; bot = box[:,3]; bw = box[:,2]-box[:,0]
    bot_ext = bot + P["stick_ext"]*H
    cy = (top + bot_ext)/2; ax = bw/2*P["pad_x"]; ay = (bot_ext - top)/2*P["pad_y"]
    sm = lambda v: savgol_filter(v, 31, 2, mode="interp")
    trk = np.stack([sm(cx), sm(cy), sm(np.maximum(ax, 150)), sm(ay)], 1)
    np.save(f"{OUTDIR}/track.npy", trk)
    print("missing:", np.nonzero(bad)[0].tolist(), "outliers:", np.nonzero(out)[0].tolist())
    sys.exit()

trk = np.load(f"{OUTDIR}/track.npy")
def oval(i, scale=4):
    w, h = W//scale, H//scale; m = np.zeros((h, w), np.float32)
    cx, cy, ax, ay = trk[i] / scale
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax*0.95), int(ay*0.95)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0,0), P["feather_frac"]*H/scale)
    return np.clip(cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)[..., None]

def lin(x): return np.power(np.clip(x,0,1), 2.2)
def gam(x): return np.power(np.clip(x,0,1), 1/2.2)

def grade(f, i, rng):
    M = oval(i)
    l = lin(f); Y = l @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # inside: shadow lift in stops, fades out by ~0.35 linear luminance, highlights untouched
    w = np.clip(1 - Y/P["lift_knee"], 0, 1)**1.5
    inside = l * np.power(2.0, P["lift_stops"]*w)[..., None]
    # skin warmth via diver silhouette qualifier (soft)
    small = cv2.resize(f, (SW, SH))
    body = cv2.morphologyEx(seg(small), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    q = cv2.GaussianBlur(cv2.dilate(body.astype(np.float32), np.ones((3,3),np.uint8)), (0,0), 2.0)
    q = np.clip(cv2.resize(q, (W, H)), 0, 1) * w * P["warm_amt"]   # shadows/mids of the diver only
    Yi = (inside @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    warmcol = Yi * np.array(P["warm"], np.float32)
    inside = inside + (warmcol - inside) * q[..., None]
    # outside: darker, navy, desat, crush
    o = l * np.array(P["out_mat"], np.float32)
    o[..., 2] += P["g2b"] * l[..., 1]
    o *= P["out_gain"]
    Yo = (o @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    o = Yo + (o - Yo) * P["out_sat"]
    o = np.clip(o - P["crush"], 0, None) / (1 - P["crush"])
    res = gam(inside) * M + gam(o) * (1 - M)
    return np.clip(res + rng.normal(0, P["grain"], res.shape).astype(np.float32), 0, 1)

rng = np.random.default_rng(0)
if MODE == "preview":
    pw = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r","30000/1001","-i","-",
        "-c:v","libx264","-crf","20","-pix_fmt","yuv420p",f"{OUTDIR}/mask_preview.mp4"], stdin=subprocess.PIPE)
    sel = [0, 45, 90, 135, N-1]; tops, bots = [], []
    for i, f in enumerate(frames()):
        v = (f*255).astype(np.uint8).copy(); cx, cy, ax, ay = trk[i]
        cv2.ellipse(v, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, (255,60,60), 3)
        cv2.circle(v, (int(cx), int(cy)), 8, (255,255,0), -1)
        cv2.putText(v, str(i), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        pw.stdin.write(v.tobytes())
        if i in sel:
            g = grade(f, i, rng)
            tops.append(cv2.resize(f, (W//3, H//3))); bots.append(cv2.resize(g, (W//3, H//3)))
    pw.stdin.close(); pw.wait()
    sheet = np.vstack([np.hstack(tops), np.hstack(bots)])
    cv2.imwrite(f"{OUTDIR}/compare.jpg", cv2.cvtColor((sheet*255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    sys.exit()

if MODE == "render":
    pw = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb48le","-s",f"{W}x{H}","-r","30000/1001","-i","-",
        "-i",V,"-map","0:v","-map","1:a","-c:a","copy",
        "-vf","scale=out_color_matrix=bt709:out_range=tv","-c:v","libx264","-crf","16","-preset","slow","-pix_fmt","yuv420p",
        "-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709","-color_range","tv",
        "-movflags","+faststart",f"{OUTDIR}/diver_spotlight.mov"], stdin=subprocess.PIPE)
    for i, f in enumerate(frames()):
        pw.stdin.write((grade(f, i, rng)*65535 + 0.5).astype(np.uint16).tobytes())
    pw.stdin.close(); pw.wait()
