"""NMSE (Normalised Mean Square Error) metrics for complex CSI."""

from __future__ import annotations

import torch


def nmse(H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Compute NMSE between reconstructed and true complex CSI.

    .. math::

        \\text{NMSE} = \\frac{\\|H - \\hat{H}\\|_F^2}{\\|H\\|_F^2}

    averaged over the batch.

    Parameters
    ----------
    H_hat:
        Reconstructed CSI (complex or real), arbitrary shape
        ``[B, ...]``.
    H:
        Ground-truth CSI, same shape as ``H_hat``.

    Returns
    -------
    torch.Tensor
        Scalar NMSE value (dimensionless, ≥ 0).
    """
    # Flatten all dims except batch
    B = H.shape[0]
    diff = (H_hat - H).reshape(B, -1)
    target = H.reshape(B, -1)

    if H.is_complex():
        num = (diff.abs() ** 2).sum(dim=-1)
        den = (target.abs() ** 2).sum(dim=-1)
    else:
        num = (diff ** 2).sum(dim=-1)
        den = (target ** 2).sum(dim=-1)

    return (num / (den + 1e-12)).mean()


def nmse_db(H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """NMSE in dB: ``10 * log10(NMSE)``."""
    return 10.0 * torch.log10(nmse(H_hat, H) + 1e-12)


def nmse_loss(H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Differentiable NMSE loss suitable for gradient-based optimization.

    Equivalent to :func:`nmse` but guaranteed to be differentiable
    with respect to ``H_hat``.
    """
    return nmse(H_hat, H)
