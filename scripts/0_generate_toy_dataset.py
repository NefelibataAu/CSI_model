#!/usr/bin/env python3
"""Generate a small synthetic complex CSI dataset and save to data/.

Usage
-----
    python scripts/0_generate_toy_dataset.py

Output
------
    data/csi_toy.npz   — numpy archive with key "csi", shape [N, N_sc, N_r, N_t] complex64
    data/csi_toy.pt    — same data as a PyTorch complex64 tensor

The channel is modelled as i.i.d. complex Gaussian (Rayleigh flat fading)
per antenna pair, independent across subcarriers — suitable as a toy dataset
to verify the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def generate_rayleigh_csi(
    n_samples: int = 2000,
    n_sc: int = 32,
    n_r: int = 4,
    n_t: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Generate i.i.d. complex Gaussian CSI.

    Returns
    -------
    np.ndarray
        Complex64 array of shape ``[N, N_sc, N_r, N_t]``.
    """
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n_samples, n_sc, n_r, n_t)).astype(np.float32)
    imag = rng.standard_normal((n_samples, n_sc, n_r, n_t)).astype(np.float32)
    csi = (real + 1j * imag) / np.sqrt(2.0)
    return csi.astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy CSI dataset")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--n_sc", type=int, default=32, help="Number of subcarriers")
    parser.add_argument("--n_r", type=int, default=4, help="Receive antennas")
    parser.add_argument("--n_t", type=int, default=4, help="Transmit antennas")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.n_samples} samples  "
        f"[N_sc={args.n_sc}, N_r={args.n_r}, N_t={args.n_t}] ..."
    )
    csi = generate_rayleigh_csi(
        n_samples=args.n_samples,
        n_sc=args.n_sc,
        n_r=args.n_r,
        n_t=args.n_t,
        seed=args.seed,
    )
    print(f"  Shape: {csi.shape}  dtype: {csi.dtype}")

    npz_path = out_dir / "csi_toy.npz"
    np.savez(npz_path, csi=csi)
    print(f"  Saved → {npz_path}")

    pt_path = out_dir / "csi_toy.pt"
    torch.save(torch.from_numpy(csi), pt_path)
    print(f"  Saved → {pt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
