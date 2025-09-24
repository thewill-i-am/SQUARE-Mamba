"""Lightweight CPU-only fallback for the mamba_ssm package.

This replacement provides a minimal nn.Module with the same constructor and
forward signature as the real library so that the training scripts can run on
machines without CUDA or the compiled selective scan kernels. The
implementation simply applies a small GRU followed by a linear projection to
match the expected output shape.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        hidden_size = max(d_model, d_state)
        self.expand = expand
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor, inference_params: object = None) -> torch.Tensor:
        output, _ = self.rnn(hidden_states)
        return self.proj(output)

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: Optional[torch.dtype] = None, **_: object
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        hidden_size = self.rnn.hidden_size
        conv_state = torch.zeros(batch_size, hidden_size, max(1, self.expand), device=device, dtype=dtype or self.proj.weight.dtype)
        ssm_state = torch.zeros(batch_size, hidden_size, max(1, self.expand), device=device, dtype=dtype or self.proj.weight.dtype)
        return conv_state, ssm_state

