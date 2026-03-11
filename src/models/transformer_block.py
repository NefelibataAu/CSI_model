"""Basic Transformer block: Multi-Head Self-Attention + Feed-Forward Network.

Architecture (Pre-LN, widely used for stable training)::

    x → LayerNorm → MultiheadAttention → + → LayerNorm → FFN → +
    └──────────────────────────────────────┘   └─────────────────┘

"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single Transformer encoder block (Pre-LN).

    Parameters
    ----------
    d_model:
        Token embedding dimension.
    n_heads:
        Number of attention heads.
    d_ff:
        Feed-forward hidden dimension (default ``4 * d_model``).
    dropout:
        Dropout rate applied after attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor ``[B, T, d_model]``.
        key_padding_mask:
            Optional boolean mask ``[B, T]``; ``True`` means ignore.

        Returns
        -------
        torch.Tensor
            Output tensor ``[B, T, d_model]``.
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerBlock` layers.

    Parameters
    ----------
    d_model:
        Token embedding dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of stacked blocks.
    d_ff:
        Feed-forward hidden dimension (default ``4 * d_model``).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward through all blocks.

        Parameters
        ----------
        x:
            ``[B, T, d_model]``
        key_padding_mask:
            Optional ``[B, T]`` boolean mask.

        Returns
        -------
        torch.Tensor
            ``[B, T, d_model]``
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)
