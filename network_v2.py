"""
Point-wise thickness prediction with flexible grid + source.

Each measurement point i has:
  signal_i  — time-domain scattered signal  [T]
  pos_i     — absolute position on plate     [2]
  src       — excitation source position     [2]

Encoding per point:
  CNN(signal_i)  +  Fourier(abs_pos)  +  Fourier(rel_to_src)
  + Fourier(dist_to_src)  +  Direction(sin,cos)

Self-attention across all points → per-point thickness.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fourier feature encoding (NeRF-style) ───────────────────────────────

class FourierFeatures(nn.Module):
    """Encode scalar or vector input with sin/cos at multiple frequencies."""

    def __init__(self, in_dim, num_freqs=16, learned=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.out_dim = in_dim * 2 * num_freqs
        if learned:
            self.freqs = nn.Parameter(torch.randn(num_freqs, in_dim) * 0.5)
        else:
            freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
            self.register_buffer('freqs', freqs.view(-1, 1).expand(-1, in_dim).clone())

    def forward(self, x):
        """x: (..., in_dim) → (..., out_dim)"""
        x_proj = x.unsqueeze(-2) * self.freqs * math.pi  # (..., num_freqs, in_dim)
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


# ── Per-point signal encoder ─────────────────────────────────────────────

class SignalEncoder(nn.Module):
    """1D CNN: raw 256-sample signal → feature vector."""

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
        """x: (B*N, 1, T) → (B*N, D)"""
        return self.net(x).squeeze(-1)


# ── Rich per-point geometry encoder ──────────────────────────────────────

class GeometryEncoder(nn.Module):
    """Encode measurement point position with full geometric context.

    Encodes:
      - absolute position on plate
      - relative position to source (vector)
      - distance to source (scalar)
      - direction from source (sin/cos of angle)
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.ff_abs = FourierFeatures(2, num_freqs=12)    # → 48 dims
        self.ff_rel = FourierFeatures(2, num_freqs=12)    # → 48 dims
        self.ff_dist = FourierFeatures(1, num_freqs=8)    # → 16 dims
        # Total raw: 48 + 48 + 16 + 2(angle) = 114
        self.proj = nn.Sequential(
            nn.Linear(114, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos, src_pos):
        """pos: (B, N, 2), src_pos: (B, 2) → (B, N, D)"""
        # Absolute position
        abs_feat = self.ff_abs(pos)                       # (B, N, 48)

        # Relative to source
        rel = pos - src_pos.unsqueeze(1)                  # (B, N, 2)
        rel_feat = self.ff_rel(rel)                       # (B, N, 48)

        # Distance to source
        dist = torch.norm(rel, dim=-1, keepdim=True)      # (B, N, 1)
        dist_feat = self.ff_dist(dist)                    # (B, N, 16)

        # Direction (sin/cos of angle from source to point)
        angle = torch.atan2(rel[..., 1:2], rel[..., 0:1] + 1e-8)  # (B, N, 1)

        combined = torch.cat([abs_feat, rel_feat, dist_feat,
                              angle.sin(), angle.cos()], dim=-1)
        return self.proj(combined)                        # (B, N, D)


# ── Transformer with flexible-length support ────────────────────────────

class PointTransformer(nn.Module):
    """Self-attention across measurement points. Supports variable N via masking."""

    def __init__(self, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 3,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        """x: (B, N, D), mask: (B, N) True for padded positions"""
        if mask is not None:
            return self.transformer(x, src_key_padding_mask=mask)
        return self.transformer(x)


# ── Full model ───────────────────────────────────────────────────────────

class PointThicknessPredictor(nn.Module):
    """Flexible-grid thickness prediction from per-point signals.

    Input:
      signals:       (B, N, T)  — scattered signals at each measurement point
      positions:     (B, N, 2)  — measurement point grid coordinates
      source_pos:    (B, 2)     — excitation source position
      pad_mask:      (B, N)     — True where position is padding (optional)

    Output:
      thickness:     (B, N)     — predicted thickness [mm] at each point
    """

    def __init__(self, n_time=256, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.signal_encoder = SignalEncoder(d_model)
        self.geo_encoder = GeometryEncoder(d_model)
        self.transformer = PointTransformer(d_model, n_heads, n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, signals, positions, source_pos, pad_mask=None):
        B, N, T = signals.shape

        # 1. Encode each point's signal
        x = signals.view(B * N, 1, T)
        sig_feat = self.signal_encoder(x).view(B, N, -1)   # (B, N, D)

        # 2. Encode geometry (position + source context)
        geo_feat = self.geo_encoder(positions, source_pos)   # (B, N, D)

        # 3. Combine signal and geometry features
        feat = sig_feat + geo_feat                            # (B, N, D)

        # 4. Self-attention across measurement points
        feat = self.transformer(feat, pad_mask)               # (B, N, D)

        # 5. Per-point thickness prediction
        thickness = self.head(feat).squeeze(-1)               # (B, N)

        return thickness


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
