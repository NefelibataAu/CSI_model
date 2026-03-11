#!/usr/bin/env python3
"""Run BERT-style masked CSI pretraining.

Usage
-----
    python scripts/1_pretrain.py --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.data.csi_dataset import CSIDataset
from src.models.pretrain_model import MaskedCSIPretrainModel
from src.train.pretrain_loop import pretrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked CSI Pretraining")
    parser.add_argument(
        "--config", type=str, default="configs/pretrain.yaml",
        help="Path to YAML config file",
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
    train_set, _ = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
    )
    print(f"Dataset: {len(dataset)} samples  "
          f"(train={n_train}, val={n_val})")

    # ---------------------------------------------------------------- Model
    csi_cfg = cfg["csi"]
    mdl_cfg = cfg["model"]
    model = MaskedCSIPretrainModel(
        n_sc=csi_cfg["n_sc"],
        n_r=csi_cfg["n_r"],
        n_t=csi_cfg["n_t"],
        d_model=mdl_cfg["d_model"],
        n_heads=mdl_cfg["n_heads"],
        n_layers=mdl_cfg["n_layers"],
        d_ff=mdl_cfg.get("d_ff"),
        dropout=mdl_cfg.get("dropout", 0.1),
        mask_ratio=mdl_cfg.get("mask_ratio", 0.15),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --------------------------------------------------------------- Training
    tr_cfg = cfg["train"]
    ck_cfg = cfg["checkpoint"]
    history = pretrain(
        model=model,
        loader=train_loader,
        n_epochs=tr_cfg["n_epochs"],
        lr=tr_cfg.get("lr", 1e-4),
        weight_decay=tr_cfg.get("weight_decay", 1e-2),
        use_amp=tr_cfg.get("use_amp", False),
        save_dir=ck_cfg.get("save_dir"),
        save_every=ck_cfg.get("save_every", 5),
        device_str=tr_cfg.get("device", "cpu"),
    )

    print(f"\nPretraining complete.  Final loss: {history[-1]:.6f}")


if __name__ == "__main__":
    main()
