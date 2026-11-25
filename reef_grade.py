"""Reference-matched underwater grade (OKLab, float32, fixed temporally-stable parameters).

grade(rgb, P, fish=None): rgb float32 HxWx3 gamma-encoded BT.709 (0..1) -> same.
fish: optional float32 HxW soft SAM 3.1 mask (0..1) for the subject node.
"""
import cv2, numpy as np

M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
M2i = np.linalg.inv(M2).astype(np.float32)
M1i = np.linalg.inv(M1).astype(np.float32)

P_DEFAULT = dict(
    wb=(1.18, 0.93, 1.05),      # linear gains: tame green cast
    bp=0.045, wp=0.80,          # OKLab-L black / white point
    con=0.45, gam=1.0,          # S-curve strength, midtone gamma
    clar_big=0.55, clar_sig=26.0, clar_mic=0.35, clar_msig=1.6, clar_lim=0.07,
    teal_amt=0.75, teal_ab=(-0.040, -0.030), teal_lo=0.20, teal_hi=0.58,
    warm_h=62.0, warm_sig=38.0, warm_rot=0.45, warm_c=1.30, warm_c0=0.012, warm_c1=0.05,
    sand_desat=0.65,
    sat=1.08, vib=0.25,
    split_amt=0.5, split_ab=(0.035, 0.050), split_lo=0.35, split_hi=0.80, mid_cool=0.4, mid_ab=(-0.02,-0.018),
    blue_feather=2.2, blue_gain=1.15, blue_lift=0.22, blue_bloom=0.05, blue_sat=0.35,
    fish_clar=0.35, fish_exp=0.03, fish_con=0.12, fish_feather=1.5,
)

def srgb2lin(x): return np.power(np.clip(x, 0, 1), 2.2).astype(np.float32)
def lin2srgb(x): return np.power(np.clip(x, 0, 1), 1 / 2.2).astype(np.float32)
def to_lab(lin):
    lms = np.cbrt(lin @ M1.T)
    return lms @ M2.T
def from_lab(lab):
    lms = (lab @ M2i.T) ** 3
    return lms @ M1i.T
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1);  return t * t * (3 - 2 * t)

def frame_lmed(rgb_small):
    return float(np.median(to_lab(srgb2lin(rgb_small))[..., 0]))

def grade(rgb, P=P_DEFAULT, fish=None, lmul=1.0, blue=None):
    lin = srgb2lin(rgb) * np.array(P["wb"], np.float32)
    lab = to_lab(np.clip(lin, 0, None))
    L, a, b = lab[..., 0] * lmul, lab[..., 1], lab[..., 2]
    # --- tone: black point, white point, S-curve ---
    Ln = np.clip((L - P["bp"]) / (P["wp"] - P["bp"]), 0, 1.15)
    Ln = np.power(Ln, P["gam"])
    s = Ln * Ln * (3 - 2 * np.clip(Ln, 0, 1))
    Ln = Ln * (1 - P["con"]) + s * P["con"] + np.clip(Ln - 1, 0, None) * (1 - P["con"])
    # --- clarity (luminance only, limited => no halos) ---
    big = cv2.GaussianBlur(Ln, (0, 0), P["clar_sig"]); mic = cv2.GaussianBlur(Ln, (0, 0), P["clar_msig"])
    lim = P["clar_lim"]
    d_big = lim * np.tanh((Ln - big) / lim); d_mic = lim * np.tanh((Ln - mic) / lim)
    fish_w = None
    if fish is not None:
        fish_w = cv2.GaussianBlur(fish.astype(np.float32), (0, 0), P["fish_feather"])[..., None][..., 0]
    cl = P["clar_big"] + (0 if fish_w is None else 0)
    Ln = Ln + P["clar_big"] * d_big + P["clar_mic"] * d_mic
    if fish_w is not None:
        Ln = Ln + fish_w * (P["fish_clar"] * d_mic + P["fish_exp"])
        # local S-curve for stripe separation inside the fish only
        sf = Ln * Ln * (3 - 2 * np.clip(Ln, 0, 1))
        Ln = Ln * (1 - fish_w * P["fish_con"]) + sf * fish_w * P["fish_con"]
    Lo = np.clip(Ln, 0, 1.3)
    # highlight shoulder: roll off above 0.82 so bright rock never clips
    k0 = 0.82
    Lo = np.where(Lo > k0, k0 + (1 - k0) * (1 - np.exp(-(Lo - k0) / (1 - k0))) * 0.97, Lo)
    # --- blue-fish node: soft SAM 3.1 mask -> brighter, richer, gently lifted off the rock ---
    wb = None
    if blue is not None:
        wb = cv2.GaussianBlur(blue.astype(np.float32), (0, 0), P["blue_feather"])
        wb = np.clip(wb * P["blue_gain"], 0, 1)
        bloom = cv2.GaussianBlur(wb, (0, 0), P["blue_feather"] * 3.5)
        Lo = Lo + wb * P["blue_lift"] * (1.05 - Lo) + bloom * P["blue_bloom"]
    # --- chroma / hue work ---
    C = np.hypot(a, b); h = np.degrees(np.arctan2(b, a)) % 360
    # warm rocks: hue pull toward amber + chroma lift, weighted by hue proximity and existing chroma
    dh = (h - P["warm_h"] + 180) % 360 - 180
    ww = np.exp(-0.5 * (dh / P["warm_sig"]) ** 2) * sstep(P["warm_c0"], P["warm_c1"], C) * sstep(0.12, 0.35, Lo)
    h2 = h - dh * P["warm_rot"] * ww
    C2 = C * (1 + (P["warm_c"] - 1) * ww)
    # sand / green-ish neutrals: desaturate low-chroma yellow-green
    sg = np.exp(-0.5 * ((((h - 115 + 180) % 360) - 180) / 45) ** 2) * (1 - sstep(0.02, 0.06, C))
    C2 = C2 * (1 - P["sand_desat"] * sg)
    # global saturation with vibrance (protects already saturated colours e.g. blue chromis)
    C2 = C2 * (P["sat"] + P["vib"] * (1 - sstep(0.03, 0.14, C2)))
    if wb is not None:
        C2 = C2 * (1 + wb * P["blue_sat"])
    a2 = C2 * np.cos(np.radians(h2)); b2 = C2 * np.sin(np.radians(h2))
    # --- cinematic shadows -> petrol teal / cyan blue ---
    ws = (1 - sstep(P["teal_lo"], P["teal_hi"], Lo)) * P["teal_amt"]
    ta, tb = P["teal_ab"]
    keep = 1 - 0.5 * ws
    a2 = a2 * keep + ta * ws * (0.4 + 0.6 * (1 - Lo / P["teal_hi"]).clip(0, 1))
    b2 = b2 * keep + tb * ws * (0.4 + 0.6 * (1 - Lo / P["teal_hi"]).clip(0, 1))
    # --- split-tone: cool midtones, amber highlights (sun-lit surfaces) ---
    wh = sstep(P["split_lo"], P["split_hi"], Lo) * P["split_amt"]
    wm = (1 - sstep(0.15, 0.55, Lo)) * sstep(0.04, 0.25, Lo) * P["mid_cool"]
    a2 = a2 + P["split_ab"][0] * wh + P["mid_ab"][0] * wm
    b2 = b2 + P["split_ab"][1] * wh + P["mid_ab"][1] * wm
    out = from_lab(np.stack([Lo, a2, b2], -1))
    out = np.clip(out, 0, 1)
    return lin2srgb(out)
