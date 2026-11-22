"""RIFE v4.26 frame interpolation wrapper (CPU/GPU), arbitrary timestep."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .rife_ifnet import IFNet

WEIGHTS = Path(__file__).resolve().parents[1] / "checkpoints" / "rife" / "flownet_v4.26.pkl"


class Rife:
    def __init__(self, weights: Path = WEIGHTS, device: str | None = None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = IFNet()
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.net.load_state_dict(sd, strict=False)
        self.net.eval().to(self.dev)

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray, t: float, scale: float = 1.0,
                 return_flow: bool = False):
        """a, b: float32 HxWx3 in [0,1] (RGB). Returns frame at fractional time t in (0,1)."""
        h, w = a.shape[:2]
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(self.dev)
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(self.dev)
        ps = max(128, int(128 / scale))
        ph, pw = ((h - 1) // ps + 1) * ps, ((w - 1) // ps + 1) * ps
        pad = (0, pw - w, 0, ph - h)
        ta, tb = F.pad(ta, pad, mode="replicate"), F.pad(tb, pad, mode="replicate")
        sl = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow_list, mask, merged = self.net(torch.cat((ta, tb), 1), t, sl)
        out = merged[-1][0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        if return_flow:
            fl = flow_list[-1][0, :, :h, :w].cpu().numpy()
            m = torch.sigmoid(mask)[0, 0, :h, :w].cpu().numpy()
            return out, fl, m
        return out
