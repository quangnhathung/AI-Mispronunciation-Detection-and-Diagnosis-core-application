from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(kernel_size // 2, kernel_size // 2),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        time_strides: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if time_strides is None:
            time_strides = [2, 2, 2]

        self.total_time_stride = 1
        self.output_channels = channels[-1]

        blocks = []
        in_ch = input_channels
        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, time_strides)):
            blocks.append(
                ConvBlock(in_ch, out_ch, kernel_size=k, stride=s, dropout=dropout)
            )
            self.total_time_stride *= s
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape
        x = self.blocks(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * F)
        return x


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is not None:
            total_length = x.size(1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=total_length
            )
        else:
            x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class CTCProjectionHead(nn.Module):
    def __init__(self, input_size: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(input_size, vocab_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.projection(x), dim=-1)


class CNNBiLSTMCTC(BaseModel):
    def __init__(
        self,
        input_dim: int = 80,
        vocab_size: int = 42,
        cnn_channels: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        cnn_strides: List[int] = None,
        cnn_dropout: float = 0.2,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 4,
        rnn_dropout: float = 0.3,
        rnn_bidirectional: bool = True,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3, 3, 3]
        if cnn_strides is None:
            cnn_strides = [2, 2, 2]

        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.time_stride = 1
        for s in cnn_strides:
            self.time_stride *= s

        self.cnn_encoder = CNNEncoder(
            input_channels=1,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            time_strides=cnn_strides,
            dropout=cnn_dropout,
        )

        cnn_output_size = cnn_channels[-1] * (input_dim // (2 ** len(cnn_strides)))
        if cnn_output_size == 0:
            cnn_output_size = cnn_channels[-1] * 1

        self.rnn_encoder = BiLSTMEncoder(
            input_size=cnn_output_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )

        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.projection = CTCProjectionHead(rnn_output_size, vocab_size)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, F, T = x.shape
        x = x.unsqueeze(1)

        x = self.cnn_encoder(x)

        if lengths is not None:
            lengths = torch.div(lengths, self.time_stride, rounding_mode="floor")
            lengths = torch.clamp(lengths, min=1)

        x = self.rnn_encoder(x, lengths)
        x = self.projection(x)
        return x

    def get_log_probs(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(x, lengths)

    def get_feat_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(audio_lengths, self.time_stride, rounding_mode="floor").clamp(min=1)
