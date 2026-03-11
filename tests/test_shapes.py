"""Tests for tensor shape contracts across the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from src.data.tokenizer import CSITokenizer
from src.models.transformer_block import TransformerBlock, TransformerEncoder
from src.models.pretrain_model import MaskedCSIPretrainModel
from src.models.compression_model import CSICompressionModel
from src.metrics.nmse import nmse, nmse_db

# ── Shared test dimensions ──────────────────────────────────────────────────
B = 4
N_SC = 16
N_R = 2
N_T = 2
D_MODEL = 64
N_HEADS = 4
N_TOKENS = N_R * N_T


def make_csi() -> torch.Tensor:
    """Create a random complex CSI batch [B, N_sc, N_r, N_t]."""
    real = torch.randn(B, N_SC, N_R, N_T)
    imag = torch.randn(B, N_SC, N_R, N_T)
    return torch.complex(real, imag)


# ── Tokenizer shapes ────────────────────────────────────────────────────────

class TestTokenizerShapes:
    def setup_method(self):
        self.tok = CSITokenizer(N_SC, N_R, N_T, D_MODEL, mask_ratio=0.15)
        self.H = make_csi()

    def test_raw_token_shape(self):
        raw = CSITokenizer.csi_to_tokens_raw(self.H)
        assert raw.shape == (B, N_TOKENS, N_SC * 2)

    def test_projected_token_shape(self):
        tokens = self.tok.tokenize(self.H)
        assert tokens.shape == (B, N_TOKENS, D_MODEL)

    def test_masked_token_shape(self):
        tokens = self.tok.tokenize(self.H)
        masked, mask = self.tok.apply_mask(tokens)
        assert masked.shape == tokens.shape
        assert mask.shape == (B, N_TOKENS)
        assert mask.dtype == torch.bool

    def test_forward_no_mask(self):
        out = self.tok(self.H, apply_mask=False)
        assert "tokens" in out
        assert "raw_tokens" in out
        assert "masked_tokens" not in out

    def test_forward_with_mask(self):
        out = self.tok(self.H, apply_mask=True)
        assert "masked_tokens" in out
        assert "mask" in out


# ── Transformer shapes ──────────────────────────────────────────────────────

class TestTransformerShapes:
    def test_block_output_shape(self):
        block = TransformerBlock(D_MODEL, N_HEADS)
        x = torch.randn(B, N_TOKENS, D_MODEL)
        out = block(x)
        assert out.shape == (B, N_TOKENS, D_MODEL)

    def test_encoder_output_shape(self):
        enc = TransformerEncoder(D_MODEL, N_HEADS, n_layers=2)
        x = torch.randn(B, N_TOKENS, D_MODEL)
        out = enc(x)
        assert out.shape == (B, N_TOKENS, D_MODEL)


# ── Pretrain model shapes ───────────────────────────────────────────────────

class TestPretrainModelShapes:
    def setup_method(self):
        self.model = MaskedCSIPretrainModel(
            n_sc=N_SC, n_r=N_R, n_t=N_T,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
        )
        self.H = make_csi()

    def test_forward_output_keys(self):
        out = self.model(self.H)
        for key in ("loss", "pred_raw", "target_raw", "mask"):
            assert key in out

    def test_loss_is_scalar(self):
        out = self.model(self.H)
        assert out["loss"].shape == ()

    def test_pred_raw_shape(self):
        out = self.model(self.H)
        assert out["pred_raw"].shape == (B, N_TOKENS, N_SC * 2)


# ── Compression model shapes ────────────────────────────────────────────────

class TestCompressionModelShapes:
    def setup_method(self):
        self.model = CSICompressionModel(
            n_sc=N_SC, n_r=N_R, n_t=N_T,
            d_model=D_MODEL, n_heads=N_HEADS,
            n_layers_enc=2, n_layers_dec=2,
            latent_dim=32,
        )
        self.H = make_csi()

    def test_encode_shape(self):
        z = self.model.encode(self.H)
        assert z.shape == (B, 32)

    def test_decode_shape(self):
        z = self.model.encode(self.H)
        H_hat = self.model.decode(z)
        assert H_hat.shape == (B, N_SC, N_R, N_T)
        assert H_hat.is_complex()

    def test_forward_output_keys(self):
        out = self.model(self.H)
        for key in ("loss", "H_hat", "z"):
            assert key in out

    def test_loss_is_scalar(self):
        out = self.model(self.H)
        assert out["loss"].shape == ()


# ── NMSE metric ──────────────────────────────────────────────────────────────

class TestNMSE:
    def test_perfect_reconstruction_is_zero(self):
        H = make_csi()
        val = nmse(H, H)
        assert val.item() < 1e-6

    def test_nmse_is_non_negative(self):
        H = make_csi()
        H_hat = H + 0.1 * make_csi()
        val = nmse(H_hat, H)
        assert val.item() >= 0.0

    def test_nmse_db_shape(self):
        H = make_csi()
        H_hat = H + 0.1 * make_csi()
        db = nmse_db(H_hat, H)
        assert db.shape == ()
