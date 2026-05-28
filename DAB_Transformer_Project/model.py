"""
model.py — DAB_Transformer (tối ưu: dropout, deeper, cải thiện CNN)

  [M1] CNN encoder  (3 lớp conv, stride tổng = 32, thêm dropout)
  [M-aug] SpecAugment (train only)
  [M2] Positional encoding
  [M3] DAB blocks   (NUM_LAYERS=4, có dropout)
  [M4] Classifier → logits CTC
"""
import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint
from config import Config
from augment import FeatureSpecAugment


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class DAB_Block(nn.Module):
    """
    DAB block với dropout — quan trọng để chống overfit.
    Pre-Norm (norm trước attention) ổn định hơn Post-Norm cho model nhỏ.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)   # LayerNorm thay GroupNorm: ổn định hơn với seq
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Gate: dynamic attention bias
        self.gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),  # depthwise
            nn.Conv1d(d_model, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # FFN mở rộng x4 với dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        # Pre-Norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # Gating
        g = self.gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * g

        # Pre-Norm FFN
        x_norm = self.norm2(x)
        x = x + self.dropout2(self.ffn(x_norm))
        return x, g


class DAB_Transformer(nn.Module):
    def __init__(self, num_classes, d_model, nhead, num_layers):
        super().__init__()
        drop = Config.DROPOUT

        # [M1] CNN encoder: 3 lớp — stride 8 × 4 × 1 = 32 tổng (giữ tương thích utils)
        # Lớp 3 (kernel=5, stride=1) tăng receptive field không thêm stride
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=8, padding=5),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(drop * 0.5),                                   # nhẹ ở đầu
            nn.Conv1d(64, d_model, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2),  # [MỚI]
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        self.spec_augment = FeatureSpecAugment()
        self.pos_encoder  = PositionalEncoding(d_model, dropout=drop)
        self.blocks = nn.ModuleList(
            [DAB_Block(d_model, nhead, dropout=drop) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1)).transpose(1, 2)   # [B, T, D]
        x = self.spec_augment(x)
        x = self.pos_encoder(x)

        all_gates = []
        for b in self.blocks:
            if Config.USE_GRADIENT_CHECKPOINT and self.training:
                x, g = checkpoint(b, x, use_reentrant=False)
            else:
                x, g = b(x)
            all_gates.append(g)

        logits = self.classifier(x)
        return logits, all_gates