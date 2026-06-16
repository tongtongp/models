from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class DiagonalSSM(nn.Module):
    """纯 PyTorch 的对角 SSM 退化实现（Python 3.13 兼容兜底）。"""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(d_model))
        self.b = nn.Parameter(torch.randn(d_model))
        self.c = nn.Parameter(torch.randn(d_model))
        self.d = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        bsz, steps, dim = x.shape
        state = torch.zeros(bsz, dim, device=x.device, dtype=x.dtype)
        outputs = []
        a = torch.tanh(self.a)
        for t in range(steps):
            state = a * state + self.b * x[:, t, :]
            y = self.c * state + self.d * x[:, t, :]
            outputs.append(y)
        out = torch.stack(outputs, dim=1)
        return self.dropout(out)


class DiseaseMamba(nn.Module):
    """Mamba(SSM) 主干 + 过程特征 MLP 融合，输出双目标增量。"""

    def __init__(
        self,
        seq_dim: int,
        tab_dim: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, d_model)

        if Mamba is not None:
            self.mamba_layers = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(n_layers)]
            )
        else:
            self.mamba_layers = nn.ModuleList(
                [DiagonalSSM(d_model=d_model, dropout=dropout) for _ in range(n_layers)]
            )
        self.norm = nn.LayerNorm(d_model)

        self.tab_net = nn.Sequential(
            nn.Linear(tab_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(32, 2)

    def forward(self, seq_input: torch.Tensor, tab_input: torch.Tensor) -> torch.Tensor:
        x = self.seq_proj(seq_input)
        for layer in self.mamba_layers:
            x = x + layer(x)
        x = self.norm(x)
        seq_feat = x[:, -1, :]
        tab_feat = self.tab_net(tab_input)
        fused = torch.cat([seq_feat, tab_feat], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)


def ensure_mamba_available() -> bool:
    return Mamba is not None
