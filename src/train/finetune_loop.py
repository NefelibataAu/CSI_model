"""Finetuning loop for CSI compression & reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.nmse import nmse_db


def finetune_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, float]:
    """Train the compression model for one epoch.

    Returns
    -------
    tuple[float, float]
        ``(mean_loss, mean_nmse_db)`` over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_nmse_db = 0.0

    for batch in tqdm(loader, desc="finetune", leave=False):
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
        with torch.no_grad():
            total_nmse_db += nmse_db(out["H_hat"], H).item()

    n = max(len(loader), 1)
    return total_loss / n, total_nmse_db / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the compression model.

    Returns
    -------
    tuple[float, float]
        ``(mean_loss, mean_nmse_db)`` over the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_nmse_db = 0.0

    for batch in loader:
        H = batch.to(device)
        out = model(H)
        total_loss += out["loss"].item()
        total_nmse_db += nmse_db(out["H_hat"], H).item()

    n = max(len(loader), 1)
    return total_loss / n, total_nmse_db / n


def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    use_amp: bool = False,
    save_dir: Optional[str] = None,
    save_every: int = 10,
    device_str: str = "cpu",
) -> dict[str, list]:
    """Full finetuning loop.

    Parameters
    ----------
    model:
        :class:`~src.models.compression_model.CSICompressionModel`.
    train_loader:
        Training DataLoader.
    val_loader:
        Optional validation DataLoader.
    n_epochs:
        Total training epochs.
    lr:
        Learning rate.
    weight_decay:
        AdamW weight decay.
    use_amp:
        Enable AMP (GPU only).
    save_dir:
        Checkpoint save directory.
    save_every:
        Checkpoint interval.
    device_str:
        Device string.

    Returns
    -------
    dict
        History dict with keys ``"train_loss"``, ``"train_nmse_db"``,
        and optionally ``"val_loss"``, ``"val_nmse_db"``.
    """
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {
        "train_loss": [],
        "train_nmse_db": [],
        "val_loss": [],
        "val_nmse_db": [],
    }

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_nmse = finetune_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_nmse_db"].append(tr_nmse)

        log = f"[Finetune] Epoch {epoch}/{n_epochs}  loss={tr_loss:.6f}  NMSE={tr_nmse:.2f} dB"

        if val_loader is not None:
            val_loss, val_nmse = evaluate(model, val_loader, device)
            history["val_loss"].append(val_loss)
            history["val_nmse_db"].append(val_nmse)
            log += f"  val_loss={val_loss:.6f}  val_NMSE={val_nmse:.2f} dB"

        print(log)

        if save_dir is not None and epoch % save_every == 0:
            ckpt_path = Path(save_dir) / f"finetune_epoch{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    return history
