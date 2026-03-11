"""Downstream CSI compression & reconstruction model.

Architecture::

    H (complex) → CSITokenizer → TransformerEncoder → mean-pool
                                                     ↓
                                               bottleneck (linear)
                                                     ↓ z  [B, latent_dim]
                                               expand + TransformerDecoder
                                                     ↓
                                              CSITokenizer.detokenize
                                                     ↓
                                              Ĥ (complex)  [B, N_sc, N_r, N_t]

Training loss: NMSE on complex CSI.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.tokenizer import CSITokenizer
from src.models.transformer_block import TransformerEncoder
from src.metrics.nmse import nmse_loss


class CSICompressionModel(nn.Module):
    """Encoder–Bottleneck–Decoder model for CSI compression/reconstruction.

    Parameters
    ----------
    n_sc, n_r, n_t:
        CSI dimensions.
    d_model:
        Transformer hidden dimension.
    n_heads:
        Number of attention heads.
    n_layers_enc:
        Number of encoder Transformer blocks.
    n_layers_dec:
        Number of decoder Transformer blocks.
    d_ff:
        FFN hidden dim (default ``4 * d_model``).
    latent_dim:
        Bottleneck dimension (controls compression ratio).
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        n_sc: int,
        n_r: int,
        n_t: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_enc: int = 4,
        n_layers_dec: int = 4,
        d_ff: int | None = None,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_sc = n_sc
        self.n_r = n_r
        self.n_t = n_t
        self.n_tokens = n_r * n_t
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Shared tokenizer (no masking needed for finetune)
        self.tokenizer = CSITokenizer(
            n_sc=n_sc, n_r=n_r, n_t=n_t, d_model=d_model,
            mask_ratio=0.0, learnable_mask=False,
        )

        # Encoder Transformer
        self.encoder = TransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers_enc,
            d_ff=d_ff, dropout=dropout,
        )

        # Bottleneck: pool → compress
        self.bottleneck_enc = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Tanh(),
        )

        # Expand latent back to sequence
        self.bottleneck_dec = nn.Linear(latent_dim, d_model * self.n_tokens)

        # Decoder Transformer
        self.decoder = TransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers_dec,
            d_ff=d_ff, dropout=dropout,
        )

    def encode(self, H: torch.Tensor) -> torch.Tensor:
        """Compress complex CSI to latent vector.

        Parameters
        ----------
        H:
            Complex CSI ``[B, N_sc, N_r, N_t]``.

        Returns
        -------
        torch.Tensor
            Latent vector ``[B, latent_dim]``.
        """
        tokens = self.tokenizer.tokenize(H)          # [B, T, d_model]
        enc = self.encoder(tokens)                   # [B, T, d_model]
        pooled = enc.mean(dim=1)                     # [B, d_model]
        z = self.bottleneck_enc(pooled)              # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct complex CSI from latent vector.

        Parameters
        ----------
        z:
            Latent vector ``[B, latent_dim]``.

        Returns
        -------
        torch.Tensor
            Reconstructed complex CSI ``[B, N_sc, N_r, N_t]``.
        """
        B = z.shape[0]
        expanded = self.bottleneck_dec(z)                    # [B, T*d_model]
        tokens = expanded.view(B, self.n_tokens, self.d_model)  # [B, T, d_model]
        dec = self.decoder(tokens)                           # [B, T, d_model]
        H_hat = self.tokenizer.detokenize(dec)               # [B, N_sc, N_r, N_t]
        return H_hat

    def forward(
        self, H: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass: compress and reconstruct.

        Parameters
        ----------
        H:
            Complex CSI batch ``[B, N_sc, N_r, N_t]``.

        Returns
        -------
        dict with keys:

        - ``"loss"``: NMSE loss (scalar).
        - ``"H_hat"``: reconstructed complex CSI ``[B, N_sc, N_r, N_t]``.
        - ``"z"``: latent code ``[B, latent_dim]``.
        """
        z = self.encode(H)
        H_hat = self.decode(z)
        loss = nmse_loss(H_hat, H)
        return {"loss": loss, "H_hat": H_hat, "z": z}

    def load_pretrained_encoder(
        self, pretrained_state: dict, strict: bool = False
    ) -> None:
        """Load encoder + tokenizer weights from a pretrained checkpoint.

        Parameters
        ----------
        pretrained_state:
            State dict from a :class:`~src.models.pretrain_model.MaskedCSIPretrainModel`.
        strict:
            Whether to enforce strict key matching.
        """
        # Map pretrained keys to compression model
        own_state = self.state_dict()
        loaded = 0
        for name, param in pretrained_state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"[load_pretrained_encoder] Loaded {loaded} parameter tensors.")
