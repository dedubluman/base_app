"""Thin CPU wrapper around the vendored RIFE v4.26 IFNet: interpolate at an arbitrary timestep."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .rife_ifnet import IFNet

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints/rife/flownet_v4.26.pkl"


class RIFE:
    def __init__(self, ckpt: Path = CKPT, threads: int = 0):
        if threads:
            torch.set_num_threads(threads)
        self.net = IFNet().eval()
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)   # tensors only, no pickled code
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        res = self.net.load_state_dict(sd, strict=False)
        self.missing, self.unexpected = list(res.missing_keys), list(res.unexpected_keys)
        if self.missing:
            raise RuntimeError(f"RIFE weights incomplete, missing {self.missing[:5]}")

    @staticmethod
    def to_tensor(rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].float()

    @torch.inference_mode()
    def __call__(self, a: np.ndarray, b: np.ndarray, t: float, scale: float = 1.0, with_flow: bool = False):
        """a, b: float32 HxWx3 RGB in 0..1 -> interpolated frame at time t in (0,1)."""
        x0, x1 = self.to_tensor(a), self.to_tensor(b)
        h, w = x0.shape[2:]
        ph, pw = ((h - 1) // 64 + 1) * 64, ((w - 1) // 64 + 1) * 64
        pad = (0, pw - w, 0, ph - h)
        x = torch.cat((F.pad(x0, pad, mode="replicate"), F.pad(x1, pad, mode="replicate")), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]   # v4.26 has five IFBlocks
        flow_list, mask, merged = self.net(x, timestep=float(t), scale_list=scale_list)
        out = merged[4][:, :, :h, :w].clamp(0, 1)[0].permute(1, 2, 0).numpy()
        if with_flow:
            fl = flow_list[-1][:, :, :h, :w][0].numpy()      # 4 channels: flow t->0 (xy), t->1 (xy)
            return out, fl, mask[:, :, :h, :w][0, 0].numpy()
        return out
