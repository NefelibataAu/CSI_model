#!/usr/bin/env python3
"""Finetune the CSI compression & reconstruction model.

Usage
-----
    # Without pretrained weights
    python scripts/2_finetune_compression.py --config configs/finetune.yaml

    # With pretrained backbone
    python scripts/2_finetune_compression.py \\
        --config configs/finetune.yaml \\
        --pretrained checkpoints/pretrain/pretrain_epoch0100.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.data.csi_dataset import CSIDataset
from src.models.compression_model import CSICompressionModel
from src.train.finetune_loop import finetune


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI Compression Finetuning")
    parser.add_argument(
        "--config", type=str, default="configs/finetune.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Optional path to pretrained backbone checkpoint (.pt)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ Data
    dataset = CSIDataset(
        path=cfg["data"]["path"],
        npz_key=cfg["data"].get("npz_key", "csi"),
    )
    n_val = max(1, int(len(dataset) * cfg["data"].get("val_split", 0.1)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )
    print(f"Dataset: {len(dataset)} samples  "
          f"(train={n_train}, val={n_val})")

    # ---------------------------------------------------------------- Model
    csi_cfg = cfg["csi"]
    mdl_cfg = cfg["model"]
    model = CSICompressionModel(
        n_sc=csi_cfg["n_sc"],
        n_r=csi_cfg["n_r"],
        n_t=csi_cfg["n_t"],
        d_model=mdl_cfg["d_model"],
        n_heads=mdl_cfg["n_heads"],
        n_layers_enc=mdl_cfg.get("n_layers_enc", 4),
        n_layers_dec=mdl_cfg.get("n_layers_dec", 4),
        d_ff=mdl_cfg.get("d_ff"),
        latent_dim=mdl_cfg.get("latent_dim", 128),
        dropout=mdl_cfg.get("dropout", 0.1),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Load pretrained weights if provided
    pretrained_path = args.pretrained or cfg.get("pretrained", {}).get("checkpoint")
    if pretrained_path is not None:
        print(f"Loading pretrained weights from: {pretrained_path}")
        state = torch.load(pretrained_path, map_location="cpu")
        model.load_pretrained_encoder(state)

    # --------------------------------------------------------------- Training
    tr_cfg = cfg["train"]
    ck_cfg = cfg["checkpoint"]
    history = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=tr_cfg["n_epochs"],
        lr=tr_cfg.get("lr", 5e-5),
        weight_decay=tr_cfg.get("weight_decay", 1e-2),
        use_amp=tr_cfg.get("use_amp", False),
        save_dir=ck_cfg.get("save_dir"),
        save_every=ck_cfg.get("save_every", 10),
        device_str=tr_cfg.get("device", "cpu"),
    )

    print(
        f"\nFinetuning complete.  "
        f"Final train NMSE: {history['train_nmse_db'][-1]:.2f} dB  "
        f"Final val NMSE: {history['val_nmse_db'][-1]:.2f} dB"
    )


if __name__ == "__main__":
    main()
