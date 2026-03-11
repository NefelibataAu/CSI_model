"""CSI Tokenizer — converts a complex CSI batch into a sequence of tokens.

Antenna-position tokenization
------------------------------
Given a batch  ``H`` of shape ``[B, N_sc, N_r, N_t]`` (complex64):

1. **Flatten** the antenna dimensions: reshape to ``[B, N_r*N_t, N_sc]``.
2. **Real/Imag split**: concatenate Re and Im along the feature axis →
   ``[B, N_r*N_t, N_sc*2]`` (float32).
3. **Linear projection**: project each token from ``N_sc*2`` to ``d_model``
   via a learned linear layer → ``[B, N_r*N_t, d_model]``.

The inverse (detokenize) maps ``[B, N_r*N_t, d_model]`` back to
``[B, N_sc, N_r, N_t]`` complex.

Random masking (BERT-style)
----------------------------
``apply_mask`` sets a random subset of token positions to zero (or a
learned mask embedding) and returns the mask boolean tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CSITokenizer(nn.Module):
    """Tokenize a batch of complex CSI tensors.

    Parameters
    ----------
    n_sc:
        Number of subcarriers.
    n_r:
        Number of receive antennas.
    n_t:
        Number of transmit antennas.
    d_model:
        Model hidden dimension (projection output size).
    mask_ratio:
        Fraction of tokens to mask during pretraining (default 0.15).
    learnable_mask:
        If ``True``, replace masked tokens with a learned ``[MASK]``
        embedding rather than zeros.
    """

    def __init__(
        self,
        n_sc: int,
        n_r: int,
        n_t: int,
        d_model: int,
        mask_ratio: float = 0.15,
        learnable_mask: bool = True,
    ) -> None:
        super().__init__()
        self.n_sc = n_sc
        self.n_r = n_r
        self.n_t = n_t
        self.n_tokens = n_r * n_t
        self.token_dim = n_sc * 2  # Re + Im
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Linear projection: token_dim -> d_model
        self.proj = nn.Linear(self.token_dim, d_model)
        # Inverse projection: d_model -> token_dim
        self.inv_proj = nn.Linear(d_model, self.token_dim)
        # Learnable [MASK] token
        if learnable_mask:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.register_parameter("mask_token", None)

    # ------------------------------------------------------------------
    # Core tokenisation helpers (no learned parameters)
    # ------------------------------------------------------------------

    @staticmethod
    def csi_to_tokens_raw(H: torch.Tensor) -> torch.Tensor:
        """Convert complex CSI to raw float tokens (before projection).

        Parameters
        ----------
        H:
            Complex CSI batch, shape ``[B, N_sc, N_r, N_t]``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, N_r*N_t, N_sc*2]``.
        """
        B, N_sc, N_r, N_t = H.shape
        # [B, N_sc, N_r, N_t] -> [B, N_r*N_t, N_sc]
        H_perm = H.permute(0, 2, 3, 1).reshape(B, N_r * N_t, N_sc)
        # Re/Im split -> [B, N_r*N_t, N_sc*2]
        tokens_real = torch.cat([H_perm.real, H_perm.imag], dim=-1)
        return tokens_real

    @staticmethod
    def tokens_raw_to_csi(tokens: torch.Tensor, n_sc: int, n_r: int, n_t: int) -> torch.Tensor:
        """Inverse of :meth:`csi_to_tokens_raw`.

        Parameters
        ----------
        tokens:
            Float tensor of shape ``[B, N_r*N_t, N_sc*2]``.
        n_sc, n_r, n_t:
            Original CSI dimensions.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``[B, N_sc, N_r, N_t]``.
        """
        B, _, _ = tokens.shape
        real = tokens[..., :n_sc]  # [B, N_r*N_t, N_sc]
        imag = tokens[..., n_sc:]  # [B, N_r*N_t, N_sc]
        H_complex = torch.complex(real, imag)  # [B, N_r*N_t, N_sc]
        # [B, N_r*N_t, N_sc] -> [B, N_r, N_t, N_sc] -> [B, N_sc, N_r, N_t]
        H_complex = H_complex.reshape(B, n_r, n_t, n_sc).permute(0, 3, 1, 2)
        return H_complex

    # ------------------------------------------------------------------
    # Forward (tokenise + project)
    # ------------------------------------------------------------------

    def tokenize(self, H: torch.Tensor) -> torch.Tensor:
        """Tokenize complex CSI to projected embeddings.

        Parameters
        ----------
        H:
            Complex CSI batch, shape ``[B, N_sc, N_r, N_t]``.

        Returns
        -------
        torch.Tensor
            Float embedding tensor of shape ``[B, N_r*N_t, d_model]``.
        """
        raw = self.csi_to_tokens_raw(H)          # [B, n_tokens, token_dim]
        return self.proj(raw)                    # [B, n_tokens, d_model]

    def detokenize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reconstruct complex CSI from embeddings.

        Parameters
        ----------
        embeddings:
            Float tensor of shape ``[B, N_r*N_t, d_model]``.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``[B, N_sc, N_r, N_t]``.
        """
        raw = self.inv_proj(embeddings)          # [B, n_tokens, token_dim]
        return self.tokens_raw_to_csi(raw, self.n_sc, self.n_r, self.n_t)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def apply_mask(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random BERT-style masking to projected token embeddings.

        Parameters
        ----------
        tokens:
            Float tensor of shape ``[B, N_tokens, d_model]``.

        Returns
        -------
        masked_tokens:
            Tokens with masked positions replaced (zeros or mask embedding).
            Shape ``[B, N_tokens, d_model]``.
        mask:
            Boolean tensor of shape ``[B, N_tokens]``.  ``True`` at masked
            positions.
        """
        B, T, D = tokens.shape
        # Sample mask indices per sample
        noise = torch.rand(B, T, device=tokens.device)
        mask = noise < self.mask_ratio  # [B, T] bool

        if self.mask_token is not None:
            mask_emb = self.mask_token.expand(B, T, D)
            masked_tokens = torch.where(mask.unsqueeze(-1), mask_emb, tokens)
        else:
            masked_tokens = tokens * (~mask).unsqueeze(-1).float()

        return masked_tokens, mask

    def forward(
        self, H: torch.Tensor, apply_mask: bool = False
    ) -> dict[str, torch.Tensor]:
        """Tokenize and optionally mask.

        Parameters
        ----------
        H:
            Complex CSI batch ``[B, N_sc, N_r, N_t]``.
        apply_mask:
            Whether to apply random masking.

        Returns
        -------
        dict with keys:

        - ``"tokens"``: projected embeddings ``[B, N_tokens, d_model]``
        - ``"raw_tokens"``: raw Re/Im features ``[B, N_tokens, N_sc*2]``
        - ``"masked_tokens"`` (only when ``apply_mask=True``)
        - ``"mask"`` (only when ``apply_mask=True``): bool ``[B, N_tokens]``
        """
        raw = self.csi_to_tokens_raw(H)
        tokens = self.proj(raw)
        out: dict[str, torch.Tensor] = {"tokens": tokens, "raw_tokens": raw}
        if apply_mask:
            masked_tokens, mask = self.apply_mask(tokens)
            out["masked_tokens"] = masked_tokens
            out["mask"] = mask
        return out
