"""Pretraining loop for masked CSI self-supervised learning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def pretrain_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Train the masked pretraining model for one epoch.

    Parameters
    ----------
    model:
        :class:`~src.models.pretrain_model.MaskedCSIPretrainModel`.
    loader:
        DataLoader yielding complex CSI batches ``[B, N_sc, N_r, N_t]``.
    optimizer:
        PyTorch optimiser.
    device:
        Target device.
    scaler:
        Optional AMP GradScaler (pass ``None`` to disable AMP).

    Returns
    -------
    float
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="pretrain", leave=False):
        H = batch.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(H)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(H)
            loss = out["loss"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def pretrain(
    model: nn.Module,
    loader: DataLoader,
    n_epochs: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    use_amp: bool = False,
    save_dir: Optional[str] = None,
    save_every: int = 5,
    device_str: str = "cpu",
) -> list[float]:
    """Full pretraining loop.

    Parameters
    ----------
    model:
        The masked pretraining model.
    loader:
        Training DataLoader.
    n_epochs:
        Total number of epochs.
    lr:
        Learning rate.
    weight_decay:
        AdamW weight decay.
    use_amp:
        Enable automatic mixed precision (GPU only).
    save_dir:
        Directory to save checkpoints; disabled if ``None``.
    save_every:
        Save checkpoint every this many epochs.
    device_str:
        Device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    list[float]
        Per-epoch training loss history.
    """
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    history: list[float] = []
    for epoch in range(1, n_epochs + 1):
        loss = pretrain_one_epoch(model, loader, optimizer, device, scaler)
        scheduler.step()
        history.append(loss)
        print(f"[Pretrain] Epoch {epoch}/{n_epochs}  loss={loss:.6f}")

        if save_dir is not None and epoch % save_every == 0:
            ckpt_path = Path(save_dir) / f"pretrain_epoch{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    return history
