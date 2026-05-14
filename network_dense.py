"""
Dense-grid thickness predictor — per-point CNN + Fourier encoding,
reshaped directly to 2D feature map, decoded by U-Net.

No Transformer: with dense regular grids, U-Net provides spatial context.
Much faster than O(N^2) attention for 41x41=1681-point grids.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fourier features ──────────────────────────────────────────────────────

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_freqs=16):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(-1, 1).expand(-1, in_dim).clone())
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, x):
        x_proj = x.unsqueeze(-2) * self.freqs * math.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


# ── 1D CNN signal encoder ─────────────────────────────────────────────────

class SignalEncoder(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, 9, 2, 4), nn.BatchNorm1d(24), nn.ReLU(inplace=True),
            nn.Conv1d(24, 48, 7, 2, 3), nn.BatchNorm1d(48), nn.ReLU(inplace=True),
            nn.Conv1d(48, d_model, 5, 2, 2), nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, 3, 2, 1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Geometry encoder ──────────────────────────────────────────────────────

class GeometryEncoder(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.ff_abs = FourierFeatures(2, 10)       # → 40
        self.ff_rel = FourierFeatures(2, 10)       # → 40
        self.ff_dist = FourierFeatures(1, 6)       # → 12
        raw_dim = 40 + 40 + 12 + 2                 # +2 for sin/cos angle
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_model), nn.LayerNorm(d_model),
            nn.ReLU(inplace=True), nn.Linear(d_model, d_model),
        )

    def forward(self, pos, src_pos):
        abs_f = self.ff_abs(pos)
        rel = pos - src_pos.unsqueeze(1)
        rel_f = self.ff_rel(rel)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        dist_f = self.ff_dist(dist)
        angle = torch.atan2(rel[..., 1:2], rel[..., 0:1] + 1e-8)
        x = torch.cat([abs_f, rel_f, dist_f, angle.sin(), angle.cos()], dim=-1)
        return self.proj(x)


# ── U-Net decoder ─────────────────────────────────────────────────────────

class ThinUNet(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.bn = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bn(F.max_pool2d(e2, 2))
        up1 = self.up1(b)
        # Crop skip connection to match upsampled spatial dims (handles odd sizes)
        if up1.shape[2:] != e2.shape[2:]:
            e2 = e2[:, :, :up1.shape[2], :up1.shape[3]]
        d1 = self.dec1(torch.cat([up1, e2], dim=1))
        up2 = self.up2(d1)
        if up2.shape[2:] != e1.shape[2:]:
            e1 = e1[:, :, :up2.shape[2], :up2.shape[3]]
        d2 = self.dec2(torch.cat([up2, e1], dim=1))
        return self.head(d2)


# ── Full model ────────────────────────────────────────────────────────────

class DenseThicknessPredictor(nn.Module):
    """Dense regular-grid thickness prediction → full 2D thickness map.

    Input:
      signals:    (B, N, T)  — N = grid_h * grid_w, on a regular grid
      positions:  (B, N, 2)  — grid coordinates (regular spacing)
      source_pos: (B, 2)     — excitation source position

    Output:
      thickness_map: (B, 1, grid_h, grid_w) — predicted thickness
    """

    def __init__(self, d_model=64, grid_h=41, grid_w=41, out_h=32, out_w=32):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_h = out_h
        self.out_w = out_w
        self.d_model = d_model

        self.signal_encoder = SignalEncoder(d_model)
        self.geo_encoder = GeometryEncoder(d_model)
        self.decoder = ThinUNet(d_model)

        # If output resolution differs from input grid, add a resizing conv
        if out_h != grid_h or out_w != grid_w:
            self.resize = nn.Upsample(size=(out_h, out_w), mode="bilinear",
                                      align_corners=False)
        else:
            self.resize = nn.Identity()

    def forward(self, signals, positions, source_pos):
        B, N, T = signals.shape

        # 1. Per-point encoding (independent, no attention)
        x = signals.view(B * N, 1, T)
        sig_feat = self.signal_encoder(x).view(B, N, -1)      # (B, N, D)
        geo_feat = self.geo_encoder(positions, source_pos)      # (B, N, D)
        feat = sig_feat + geo_feat                              # (B, N, D)

        # 2. Reshape to 2D feature map (regular grid → image)
        feat_map = feat.view(B, self.grid_h, self.grid_w, self.d_model)
        feat_map = feat_map.permute(0, 3, 1, 2)                 # (B, D, H, W)

        # 3. U-Net decode to thickness map
        out = self.decoder(feat_map)                            # (B, 1, H, W)
        out = self.resize(out)                                  # (B, 1, out_h, out_w)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
