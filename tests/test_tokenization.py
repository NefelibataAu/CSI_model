"""Tests verifying that CSI tokenization is a perfect round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from src.data.tokenizer import CSITokenizer

B = 4
N_SC = 16
N_R = 4
N_T = 2
D_MODEL = 64


def make_csi() -> torch.Tensor:
    real = torch.randn(B, N_SC, N_R, N_T)
    imag = torch.randn(B, N_SC, N_R, N_T)
    return torch.complex(real, imag)


class TestRawRoundTrip:
    """The static helpers :meth:`csi_to_tokens_raw` /
    :meth:`tokens_raw_to_csi` must form a perfect bijection."""

    def test_round_trip_real(self):
        H = make_csi()
        raw = CSITokenizer.csi_to_tokens_raw(H)
        H_rec = CSITokenizer.tokens_raw_to_csi(raw, N_SC, N_R, N_T)
        assert torch.allclose(H_rec.real, H.real, atol=1e-5)

    def test_round_trip_imag(self):
        H = make_csi()
        raw = CSITokenizer.csi_to_tokens_raw(H)
        H_rec = CSITokenizer.tokens_raw_to_csi(raw, N_SC, N_R, N_T)
        assert torch.allclose(H_rec.imag, H.imag, atol=1e-5)

    def test_output_is_complex(self):
        H = make_csi()
        raw = CSITokenizer.csi_to_tokens_raw(H)
        H_rec = CSITokenizer.tokens_raw_to_csi(raw, N_SC, N_R, N_T)
        assert H_rec.is_complex()

    def test_output_shape(self):
        H = make_csi()
        raw = CSITokenizer.csi_to_tokens_raw(H)
        H_rec = CSITokenizer.tokens_raw_to_csi(raw, N_SC, N_R, N_T)
        assert H_rec.shape == H.shape


class TestProjectionRoundTrip:
    """When the projection layers are identity-like (just after init), the
    tokenizer should at least preserve tensor shapes and types correctly."""

    def setup_method(self):
        self.tok = CSITokenizer(N_SC, N_R, N_T, D_MODEL, mask_ratio=0.0)

    def test_tokenize_then_detokenize_shape(self):
        H = make_csi()
        tokens = self.tok.tokenize(H)
        H_hat = self.tok.detokenize(tokens)
        assert H_hat.shape == H.shape

    def test_detokenized_is_complex(self):
        H = make_csi()
        tokens = self.tok.tokenize(H)
        H_hat = self.tok.detokenize(tokens)
        assert H_hat.is_complex()

    def test_trained_round_trip_converges(self):
        """After a few gradient steps the tokenizer should learn to
        reconstruct CSI approximately (sanity-check that gradients flow)."""
        torch.manual_seed(0)
        tok = CSITokenizer(N_SC, N_R, N_T, D_MODEL, mask_ratio=0.0)
        opt = torch.optim.Adam(tok.parameters(), lr=1e-2)

        H = make_csi()
        raw_target = CSITokenizer.csi_to_tokens_raw(H)

        initial_loss = None
        for _ in range(20):
            opt.zero_grad()
            tokens = tok.tokenize(H)
            raw_pred = tok.inv_proj(tokens)
            loss = torch.nn.functional.mse_loss(raw_pred, raw_target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step()

        assert loss.item() < initial_loss, "Loss should decrease after training"


class TestMaskingProperties:
    """Basic sanity checks for the masking mechanism."""

    def setup_method(self):
        self.tok = CSITokenizer(N_SC, N_R, N_T, D_MODEL, mask_ratio=0.5)

    def test_masked_fraction_approximate(self):
        """With mask_ratio=0.5 and large N, the fraction should be ≈0.5."""
        H = make_csi()
        tokens = self.tok.tokenize(H)
        _, mask = self.tok.apply_mask(tokens)
        fraction = mask.float().mean().item()
        # Allow ±0.15 tolerance given small sample size
        assert abs(fraction - 0.5) < 0.15

    def test_unmasked_tokens_unchanged(self):
        """Unmasked positions must be identical to the original tokens."""
        H = make_csi()
        tokens = self.tok.tokenize(H)
        masked_tokens, mask = self.tok.apply_mask(tokens)
        unmasked = ~mask
        if unmasked.any():
            assert torch.allclose(
                masked_tokens[unmasked], tokens[unmasked], atol=1e-6
            )
