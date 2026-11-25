#!/usr/bin/env python3
"""Combine the diver alpha and SAM 3.1 part masks into per-frame soft class weights.

    python build_classes.py --diver masks_refined/player_export --out masks_classes/player_export

Classes inside the diver: shorts, other (fins, hair, gear), skin (the rest).
Output per frame: classes_{i:05d}.npz with float16 w_skin, w_shorts (0..1) and the eroded skin core.
'other' is simply the remainder of the diver alpha: it receives NO diver-specific correction (environment grade only).
The weights are blurred class indicators normalised to the diver alpha, so skin/shorts transitions have no gap.

SAM 3.1 part sources (masks/player_export__<part>/subject):
  shorts  prompt "shorts"  (diver_parts.py + part_track.py to fill frames the detector missed)
  fins    prompt "flipper" (diver_parts.py: one fin per chunk)  +  fins2 (fins_both.py: every instance, both fins)
  hair    prompt "hair"    (diver_parts.py + part_track.py). The tracker drifted onto the thigh/knee in the ranges listed in
          --hair-drop (found in the per-frame audit: area jumps ~100 px -> ~1300 px -> ~4000 px, mask sitting on bare leg);
          those frames carry no hair class, i.e. the head there is graded like any non-skin dark object only if SAM saw it.
"""
import argparse
from pathlib import Path
import cv2, numpy as np
from shades.io import probe, iter_frames

REPO = Path(__file__).resolve().parent
SRC = REPO / "inputs/player_export.mov"


def rd(d, i):
    m = cv2.imread(str(Path(d) / f"{i:05d}.png"), 0)
    return None if m is None else m.astype(np.float32) / 255


def ranges(s):
    out = set()
    for part in filter(None, s.split(",")):
        a, b = part.split(":"); out |= set(range(int(a), int(b) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diver", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigma", type=float, default=1.1)
    ap.add_argument("--shorts-grow", type=int, default=1)
    ap.add_argument("--gear-grow", type=int, default=3, help="px the gear (fin/hair) mask is grown so skin warmth never touches it")
    ap.add_argument("--edge-band", type=int, default=3); ap.add_argument("--edge-bright", type=float, default=0.28)
    ap.add_argument("--hair-drop", default="255:335,364:391")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    info = probe(SRC)
    P = {k: REPO / f"masks/player_export__{k}/subject" for k in ("shorts", "fins", "fins2", "hair")}
    drop = ranges(a.hair_drop)
    kg = np.ones((2 * a.gear_grow + 1,) * 2, np.uint8)
    for i, f in enumerate(iter_frames(SRC, info)):
        dv = rd(a.diver, i); db = dv > 0.5; z = np.zeros(db.shape, bool)
        g = lambda k: (lambda m: z if m is None else m > 0.5)(rd(P[k], i))
        sh = g("shorts") & db
        other = g("fins") | g("fins2") | (g("hair") & db if i not in drop else z)
        other = cv2.dilate(other.astype(np.uint8), kg) > 0
        # Edge band: a fin/hair/gear pixel within a few px of the mask boundary that is bright is bare leg/arm that the
        # SAM mask edge slightly over-covers (fins are dark). Treat it as skin so no bright, environment-graded (cyan)
        # sliver survives along the boundary.
        yb = cv2.GaussianBlur(f @ np.array([.2126, .7152, .0722], np.float32), (0, 0), 0.8)
        core = cv2.erode(other.astype(np.uint8), np.ones((2 * a.edge_band + 1,) * 2, np.uint8)) > 0
        skin_pre = db & ~sh & ~other                       # skin as SAM sees it: only leg/arm pixels next to it may flip
        near_skin = cv2.dilate(skin_pre.astype(np.uint8), np.ones((2 * a.edge_band + 1,) * 2, np.uint8)) > 0
        other = core | (other & ~core & ~((yb >= a.edge_bright) & near_skin))
        if a.shorts_grow:
            sh = (cv2.dilate(sh.astype(np.uint8), np.ones((2 * a.shorts_grow + 1,) * 2, np.uint8)) > 0) & db
        sh &= ~other
        skin = db & ~sh & ~other
        den = cv2.GaussianBlur(db.astype(np.float32), (0, 0), a.sigma) + 1e-4
        ws = np.clip(cv2.GaussianBlur(skin.astype(np.float32), (0, 0), a.sigma) / den, 0, 1) * dv
        wh = np.clip(cv2.GaussianBlur(sh.astype(np.float32), (0, 0), a.sigma) / den, 0, 1) * dv
        np.savez_compressed(a.out / f"classes_{i:05d}.npz", skin=ws.astype(np.float16), shorts=wh.astype(np.float16),
                            skin_core=cv2.erode(skin.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        if i % 50 == 0: print("class frame", i, flush=True)


if __name__ == "__main__":
    main()
