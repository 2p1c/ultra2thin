"""
Point-wise thickness prediction along a 1D line — for 2D COMSOL cross-section data.

Adapted from network_v2.py.  Only the GeometryEncoder changes (1D positions
instead of 2D grid coords).  SignalEncoder, PointTransformer, and MLP head
are identical to v2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_freqs=16):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs.view(-1, 1).expand(-1, in_dim).clone())
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, x):
        x_proj = x.unsqueeze(-2) * self.freqs * math.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


class SignalEncoder(nn.Module):
    """1D CNN: raw 256-sample signal → d_model-dim feature.  Identical to v2."""

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


class GeometryEncoder1D(nn.Module):
    """Encode 1D measurement-point position with source-relative geometry.

    Features (all 1D):
      - absolute position on the line       Fourier(1, 12频) → 24D
      - relative position to source         Fourier(1, 12频) → 24D
      - distance to source (scalar)         Fourier(1, 8频)  → 16D
      - sign of relative position (L/R)                      →  1D
      ─────────────────────────────────────────────────────────────
      Total 65D → Linear projection → d_model
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.ff_abs = FourierFeatures(1, 12)     # → 24
        self.ff_rel = FourierFeatures(1, 12)     # → 24
        self.ff_dist = FourierFeatures(1, 8)     # → 16
        raw_dim = 24 + 24 + 16 + 1
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos, src_pos):
        """
        Args:
            pos:     (B, N, 1) — measurement point x-coordinates
            src_pos: (B, 1)    — source x-coordinate
        Returns:
            geo_feat: (B, N, d_model)
        """
        abs_f = self.ff_abs(pos)                           # (B, N, 24)
        rel = pos - src_pos.unsqueeze(1)                    # (B, N, 1)
        rel_f = self.ff_rel(rel)                            # (B, N, 24)
        dist = torch.abs(rel)                               # (B, N, 1)
        dist_f = self.ff_dist(dist)                         # (B, N, 16)
        side = torch.sign(rel + 1e-8)                       # (B, N, 1): +1 right, -1 left
        x = torch.cat([abs_f, rel_f, dist_f, side], dim=-1) # (B, N, 65)
        return self.proj(x)


class PointTransformer(nn.Module):
    """Self-attention across measurement points.  Identical to v2."""

    def __init__(self, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 3,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, mask=None):
        if mask is not None:
            return self.transformer(x, src_key_padding_mask=mask)
        return self.transformer(x)


class ConvDecoder1D(nn.Module):
    """1D CNN decoder — spatial smoothing across the measurement line."""

    def __init__(self, in_ch=128, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Conv1d(hid, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Conv1d(hid, hid // 2, 5, padding=2), nn.BatchNorm1d(hid // 2), nn.ReLU(inplace=True),
            nn.Conv1d(hid // 2, 1, 5, padding=2),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B, N)


class LineThicknessPredictor(nn.Module):
    """1D thickness prediction with optional 1D CNN decoder.

    Two modes:
      dense=True  → fixed N-point grid, Transformer → 1D CNN decoder
      dense=False → variable N, Transformer → per-point MLP head (v2 style)

    Input:
      signals:       (B, N, T)  — scattered signals at each measurement point
      positions:     (B, N, 1)  — measurement point x-coordinates
      source_pos:    (B, 1)     — excitation source x-position
      pad_mask:      (B, N)     — True where position is padding (optional)

    Output:
      thickness:     (B, N)     — predicted thickness [mm] at each point
    """

    def __init__(self, n_time=256, d_model=128, n_heads=4, n_layers=4,
                 dense=True, decoder_hid=128, use_transformer=True):
        super().__init__()
        self.dense = dense
        self.use_transformer = use_transformer
        self.signal_encoder = SignalEncoder(d_model)
        self.geo_encoder = GeometryEncoder1D(d_model)

        if use_transformer:
            self.transformer = PointTransformer(d_model, n_heads, n_layers)
        else:
            self.transformer = None

        if dense:
            self.decoder = ConvDecoder1D(d_model, decoder_hid)
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Linear(d_model // 2, 1),
            )
            self.decoder = None

    def forward(self, signals, positions, source_pos, pad_mask=None):
        B, N, T = signals.shape

        x = signals.view(B * N, 1, T)
        sig_feat = self.signal_encoder(x).view(B, N, -1)
        geo_feat = self.geo_encoder(positions, source_pos)
        feat = sig_feat + geo_feat

        if self.use_transformer:
            feat = self.transformer(feat, pad_mask)       # (B, N, D)

        if self.dense:
            feat = feat.permute(0, 2, 1)                  # (B, D, N) for Conv1d
            return self.decoder(feat)                      # (B, N)
        else:
            return self.head(feat).squeeze(-1)             # (B, N)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
