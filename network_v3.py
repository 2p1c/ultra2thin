"""
v3 Hybrid: v2's flexible encoder + v1's U-Net decoder.

  signals [B,N,T]  positions [B,N,2]  source [B,2]  pad_mask [B,N]
       │                 │                │               │
  ① 1D CNN          ② Fourier(4ch)       │               │
  sig_feat            geo_feat           │               │
       └─────────┬─────────┘              │               │
           ③ feat = sig + geo             │               │
                 │                        │               │
           ④ Transformer ◄───────────────┘               │
             点间自注意力 [B,N,D]                          │
                 │                                        │
           ⑤ GridScatter ◄── positions                    │
             高斯散布 → [B,D,32,32]                        │
                 │
           ⑥ ThinUNet
             enc→bn→dec+skip → [B,1,32,32]
                 │
           厚度图 [B,32,32]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fourier features ────────────────────────────────────────────────────

class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_freqs=16):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs.view(-1, 1).expand(-1, in_dim).clone())
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, x):
        x_proj = x.unsqueeze(-2) * self.freqs * math.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


# ── 1D CNN signal encoder ──────────────────────────────────────────────

class SignalEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, 2, 4), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, d_model, 3, 2, 1),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Geometry encoder (Fourier 4-channel) ────────────────────────────────

class GeometryEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.ff_abs = FourierFeatures(2, 12)      # → 48
        self.ff_rel = FourierFeatures(2, 12)      # → 48
        self.ff_dist = FourierFeatures(1, 8)      # → 16
        self.proj = nn.Sequential(
            nn.Linear(114, d_model), nn.LayerNorm(d_model),
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


# ── Transformer ─────────────────────────────────────────────────────────

class PointTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 3,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        if mask is not None:
            return self.transformer(x, src_key_padding_mask=mask)
        return self.transformer(x)


# ── Grid scatter ────────────────────────────────────────────────────────

class GridScatter(nn.Module):
    def __init__(self, grid_h=32, grid_w=32, sigma=3.0):
        super().__init__()
        gy, gx = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32),
            torch.arange(grid_w, dtype=torch.float32), indexing='ij')
        self.register_buffer('grid_y', gy.clone())
        self.register_buffer('grid_x', gx.clone())
        self.sigma = sigma
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, features, positions):
        B, N, D = features.shape
        device = features.device
        rx = positions[:, :, 0].to(device)
        ry = positions[:, :, 1].to(device)
        dx = self.grid_x.unsqueeze(0).unsqueeze(0) - rx.view(B, N, 1, 1)
        dy = self.grid_y.unsqueeze(0).unsqueeze(0) - ry.view(B, N, 1, 1)
        w = torch.exp(-(dx**2 + dy**2) / (2 * self.sigma**2))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        H, W = self.grid_h, self.grid_w
        out = torch.zeros(B, D, H, W, device=device)
        for n in range(N):
            out += features[:, n, :].unsqueeze(-1).unsqueeze(-1) * w[:, n].unsqueeze(1)
        return out


# ── U-Net decoder ───────────────────────────────────────────────────────

class ThinUNet(nn.Module):
    def __init__(self, in_ch=128):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.bn = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bn(F.max_pool2d(e2, 2))
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.head(d2)


# ── Full v3 model ───────────────────────────────────────────────────────

class HybridThicknessPredictor(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.signal_encoder = SignalEncoder(d_model)
        self.geo_encoder = GeometryEncoder(d_model)
        self.transformer = PointTransformer(d_model, n_heads, n_layers)
        self.scatter = GridScatter(32, 32, sigma=3.0)
        self.decoder = ThinUNet(d_model)

    def forward(self, signals, positions, source_pos, pad_mask=None):
        B, N, T = signals.shape
        x = signals.view(B * N, 1, T)
        sig_feat = self.signal_encoder(x).view(B, N, -1)
        geo_feat = self.geo_encoder(positions, source_pos)
        feat = sig_feat + geo_feat
        feat = self.transformer(feat, pad_mask)
        feat_map = self.scatter(feat, positions)
        return self.decoder(feat_map).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
