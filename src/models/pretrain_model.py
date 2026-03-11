"""BERT-style masked pretraining model for CSI.

Architecture::

    H (complex) → CSITokenizer → [MASK] → TransformerEncoder
                                        ↓
                              reconstruction head (Linear)
                                        ↓
                              reconstructed raw tokens (Re/Im)
                                        ↓
                              MSE loss on masked positions only

"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.tokenizer import CSITokenizer
from src.models.transformer_block import TransformerEncoder


class MaskedCSIPretrainModel(nn.Module):
    """BERT-style masked pretraining model.

    Predicts the raw Re/Im features of randomly masked antenna-position
    tokens from the context of unmasked tokens.

    Parameters
    ----------
    n_sc, n_r, n_t:
        CSI dimensions (subcarriers, receive, transmit antennas).
    d_model:
        Transformer hidden dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of Transformer blocks.
    d_ff:
        FFN hidden dim (default ``4 * d_model``).
    dropout:
        Dropout rate.
    mask_ratio:
        Fraction of tokens to mask.
    """

    def __init__(
        self,
        n_sc: int,
        n_r: int,
        n_t: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.tokenizer = CSITokenizer(
            n_sc=n_sc,
            n_r=n_r,
            n_t=n_t,
            d_model=d_model,
            mask_ratio=mask_ratio,
            learnable_mask=True,
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        # Reconstruction head: d_model -> N_sc*2 (raw Re/Im token)
        self.recon_head = nn.Linear(d_model, n_sc * 2)

    def forward(
        self, H: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run masked pretraining forward pass.

        Parameters
        ----------
        H:
            Complex CSI batch ``[B, N_sc, N_r, N_t]``.

        Returns
        -------
        dict with keys:

        - ``"loss"``: scalar MSE loss on masked token positions.
        - ``"pred_raw"``: reconstructed raw tokens ``[B, N_tokens, N_sc*2]``.
        - ``"target_raw"``: target raw tokens ``[B, N_tokens, N_sc*2]``.
        - ``"mask"``: boolean mask ``[B, N_tokens]``.
        """
        tok_out = self.tokenizer(H, apply_mask=True)
        tokens = tok_out["tokens"]            # [B, T, d_model] (unmasked)
        masked_tokens = tok_out["masked_tokens"]  # [B, T, d_model]
        mask = tok_out["mask"]                # [B, T] bool
        raw_target = tok_out["raw_tokens"]   # [B, T, N_sc*2]

        # Encode the masked sequence
        encoded = self.encoder(masked_tokens)  # [B, T, d_model]

        # Reconstruct raw token features
        pred_raw = self.recon_head(encoded)   # [B, T, N_sc*2]

        # MSE loss only on masked positions
        if mask.any():
            loss = nn.functional.mse_loss(
                pred_raw[mask], raw_target[mask]
            )
        else:
            loss = torch.tensor(0.0, device=H.device, requires_grad=True)

        return {
            "loss": loss,
            "pred_raw": pred_raw,
            "target_raw": raw_target,
            "mask": mask,
        }
