from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        e = torch.tanh(self.proj(x))
        a = torch.softmax(self.score(e).squeeze(-1), dim=1)  # [B, T]
        ctx = torch.sum(x * a.unsqueeze(-1), dim=1)  # [B, C]
        return ctx, a


class CNNBiLSTMAttnRegressor(nn.Module):
    def __init__(self, dropout: float = 0.2, lstm_hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.bilstm1 = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm2 = nn.LSTM(
            input_size=2 * lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn = AttentionPool(2 * lstm_hidden)
        self.head = nn.Linear(2 * lstm_hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T]
        x = x.unsqueeze(1)  # [B,1,T]
        x = self.cnn(x)  # [B,C,T]
        x = x.transpose(1, 2)  # [B,T,C]
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        ctx, attn = self.attn(x)
        y = self.head(ctx)
        return y, attn
