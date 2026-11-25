#!/usr/bin/env python3
"""Combine the diver alpha and SAM 3.1 part masks into per-frame soft class weights.

    python build_classes.py --diver masks_refined/player_export --out masks_classes/player_export

Classes inside the diver: shorts, other (fins, hair, dive mask / snorkel gear), skin (the rest).
Output per frame: classes_{i:05d}.npz with float16 arrays w_skin, w_shorts (0..1). 'other' is simply
the remainder of the diver alpha: it receives NO diver-specific correction (environment grade only).
The weights are blurred class indicators normalised to the diver alpha, so they sum to <= alpha and
there is no uncovered seam between skin and shorts."""
import argparse
from pathlib import Path
import cv2, numpy as np
from shades.io import probe, iter_frames

REPO = Path(__file__).resolve().parent
SRC = REPO / "inputs/player_export.mov"


def rd(d, i):
    m = cv2.imread(str(Path(d) / f"{i:05d}.png"), 0)
    return None if m is None else m.astype(np.float32) / 255


def hair_qc(frames, hair_dir, diver_dir):
    """Drop tracked hair masks that jump in area/position or sit on bright body parts (drift onto skin)."""
    L = np.array([.2126, .7152, .0722], np.float32); st = {}
    for i, f in enumerate(frames):
        m = rd(hair_dir, i)
        if m is None or (m > .5).sum() < 60: continue
        mb = m > .5; ys, xs = np.nonzero(mb)
        st[i] = dict(area=mb.sum(), c=np.array([xs.mean(), ys.mean()]), y=float(np.median((f @ L)[mb])))
    bad = set()
    for i, s_ in st.items():
        nb = [st[j] for j in range(i - 8, i + 9) if j != i and j in st]
        if len(nb) < 3:
            continue
        ma = np.median([n["area"] for n in nb]); mc = np.median([n["c"] for n in nb], 0)
        if not .45 * ma <= s_["area"] <= 1.8 * ma or np.linalg.norm(s_["c"] - mc) > 35 or s_["y"] > 0.33:
            bad.add(i)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diver", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma", type=float, default=1.1)
    ap.add_argument("--shorts-grow", type=int, default=1)
    ap.add_argument("--gate-lo", type=float, default=0.12); ap.add_argument("--gate-hi", type=float, default=0.22)
    ap.add_argument("--dark-y", type=float, default=0.12, help="src luma below which an unsegmented diver pixel counts as gear")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    info = probe(SRC)
    P = {k: REPO / f"masks/player_export__{k}/subject" for k in ("shorts", "fins", "hair")}
    frames = list(iter_frames(SRC, info))
    bad_hair = hair_qc(frames, P["hair"], a.diver)
    print("hair frames dropped by QC:", sorted(bad_hair), flush=True)
    for i, f in enumerate(frames):
        dv = rd(a.diver, i)
        db = dv > 0.5
        z = np.zeros_like(dv)
        sh = (rd(P["shorts"], i) if rd(P["shorts"], i) is not None else z) > 0.5
        fi = (rd(P["fins"], i) if rd(P["fins"], i) is not None else z) > 0.5
        ha = (rd(P["hair"], i) if rd(P["hair"], i) is not None else z) > 0.5
        if i in bad_hair: ha = np.zeros_like(ha)
        sh &= db
        y = f @ np.array([.2126, .7152, .0722], np.float32)
        hsv = cv2.cvtColor((np.clip(f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        # gear fallback: very dark diver pixels that no SAM part claimed (dive mask, snorkel, fin edges)
        gear = db & ~sh & (y < a.dark_y)
        gear = cv2.morphologyEx(gear.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        other = (fi | ha | gear)
        other = cv2.dilate(other.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # keep warm skin correction off gear edges
        if a.shorts_grow:
            shd = cv2.dilate(sh.astype(np.uint8), np.ones((2 * a.shorts_grow + 1,) * 2, np.uint8)) > 0
            sh = shd & db & ~other
        sh &= ~other
        skin = db & ~sh & ~other
        den = cv2.GaussianBlur(db.astype(np.float32), (0, 0), a.sigma) + 1e-4
        ws = np.clip(cv2.GaussianBlur(skin.astype(np.float32), (0, 0), a.sigma) / den, 0, 1) * dv
        # soft dark-gear gate: very dark pixels (fins/mask/snorkel SAM missed) get no skin warmth
        yb = cv2.GaussianBlur(y, (0, 0), 1.2)
        t = np.clip((yb - a.gate_lo) / (a.gate_hi - a.gate_lo), 0, 1)
        ws = ws * (t * t * (3 - 2 * t))
        wh = np.clip(cv2.GaussianBlur(sh.astype(np.float32), (0, 0), a.sigma) / den, 0, 1) * dv
        np.savez_compressed(a.out / f"classes_{i:05d}.npz", skin=ws.astype(np.float16), shorts=wh.astype(np.float16),
                            skin_core=cv2.erode(skin.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        if i % 50 == 0: print("class frame", i, flush=True)


if __name__ == "__main__":
    main()
