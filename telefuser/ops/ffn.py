"""Feed-forward network implementations."""

from __future__ import annotations

import torch
import torch.nn as nn

from .activations import GEGLU, GELU, ApproximateGELU, LinearActivation, SwiGLU


class FeedForward(nn.Module):
    """Standard feed-forward network with configurable activation.

    Args:
        dim: Input/output dimension.
        dim_out: Output dimension (defaults to dim).
        mult: Hidden dim multiplier (default: 4).
        dropout: Dropout probability.
        activation_fn: Activation function name.
        final_dropout: Apply dropout after output projection.
        inner_dim: Explicit hidden dimension (overrides mult).
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Select activation-based projection (handles internal dimension expansion)
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            # GEGLU outputs 2/3 of inner_dim after gating, so we compensate
            act_fn = GEGLU(dim, inner_dim * 2 // 3, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim * 2 // 3, bias=bias)
        elif activation_fn == "swiglu":
            # SwiGLU also uses gating, same compensation as GEGLU
            act_fn = SwiGLU(dim, inner_dim * 2 // 3, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
