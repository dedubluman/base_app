"""RAFT-large optical-flow frame interpolation with forward/backward occlusion checks.

Independent second implementation used to cross-check RIFE and as a fallback."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def _bwarp(img, flow):
    n, c, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=img.dtype), torch.arange(w, dtype=img.dtype), indexing="ij")
    gx = (xs[None] + flow[:, 0]) / (w - 1) * 2 - 1
    gy = (ys[None] + flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


class RaftInterp:
    def __init__(self, flow_scale: float = 0.5, iters: int = 16):
        self.net = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).eval()
        self.s, self.iters = flow_scale, iters
        self._cache = {}

    @torch.no_grad()
    def flows(self, a, b):
        """Bidirectional flow at full resolution (cached on array ids)."""
        h, w = a.shape[:2]
        ta = torch.from_numpy(a).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b).permute(2, 0, 1)[None]
        hh, ww = int(round(h * self.s / 8)) * 8, int(round(w * self.s / 8)) * 8
        ra = F.interpolate(ta, (hh, ww), mode="bilinear", align_corners=False) * 2 - 1
        rb = F.interpolate(tb, (hh, ww), mode="bilinear", align_corners=False) * 2 - 1
        f01 = self.net(ra, rb, num_flow_updates=self.iters)[-1]
        f10 = self.net(rb, ra, num_flow_updates=self.iters)[-1]
        up = lambda f: F.interpolate(f, (h, w), mode="bilinear", align_corners=False) * torch.tensor(
            [w / ww, h / hh]).view(1, 2, 1, 1)
        return up(f01), up(f10)

    @torch.no_grad()
    def __call__(self, a, b, t, flows=None):
        h, w = a.shape[:2]
        ta = torch.from_numpy(a).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b).permute(2, 0, 1)[None]
        f01, f10 = flows if flows is not None else self.flows(a, b)
        # occlusion: flow consistency
        e01 = (f01 + _bwarp(f10, f01)).norm(dim=1, keepdim=True)
        e10 = (f10 + _bwarp(f01, f10)).norm(dim=1, keepdim=True)
        m01 = f01.norm(dim=1, keepdim=True)
        m10 = f10.norm(dim=1, keepdim=True)
        occ0 = (e01 > 0.05 * (m01 + m10) + 1.5).float()   # pixel of a not visible in b
        occ1 = (e10 > 0.05 * (m01 + m10) + 1.5).float()
        # intermediate flow approximation (Super SloMo)
        ft0 = -(1 - t) * t * f01 + t * t * f10
        ft1 = (1 - t) ** 2 * f01 - t * (1 - t) * f10
        w0 = _bwarp(ta, ft0)
        w1 = _bwarp(tb, ft1)
        v0 = 1 - _bwarp(occ0, ft0)
        v1 = 1 - _bwarp(occ1, ft1)
        wt0, wt1 = (1 - t) * v0, t * v1
        norm = wt0 + wt1
        blend = (1 - t) * w0 + t * w1
        out = torch.where(norm > 1e-3, (wt0 * w0 + wt1 * w1) / norm.clamp_min(1e-3), blend)
        return out[0].clamp(0, 1).permute(1, 2, 0).numpy()
