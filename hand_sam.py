#!/usr/bin/env python3
"""SKIN positive points on the hand beyond the watch (preview f0 (265,440) -> f14 (340,440),
source (528,878) -> (678,878)), propagated forward and backward with SAM 3.1. A single positive
click spreads over the whole body, so each click gets a ring of negative clicks around the hand."""
import pickle
from pathlib import Path
import numpy as np
from diver_fix import SamX, run_window

REPO = Path(__file__).resolve().parent
frames = REPO / ".cache" / "sam_frames_diver_bwd"
sam = SamX(frames, REPO / ".cache")


def ring(cx, cy, r=70):
    return [(int(cx + r * np.cos(a)), int(cy + r * np.sin(a)), 0) for a in np.radians(range(0, 360, 45))]


ev = {(0, "SKIN"): [(528, 878, 1)] + ring(528, 878),
      (14, "SKIN"): [(678, 878, 1)] + ring(678, 878)}
res = run_window(sam, 0, 14, ev, 7, "hand")            # seed in the middle -> both directions
hand = {f: m for f, m in res["SKIN"].items()}
print("hand areas:", {f: int(m.sum()) for f, m in sorted(hand.items())}, flush=True)
pickle.dump(hand, open(REPO / "masks" / "diver__cls" / "hand_sam.pkl", "wb"))
