"""CSI Dataset — loads complex CSI tensors from .npz or .pt files.

Expected data layout
--------------------
Each sample is a complex-valued array of shape ``[N_sc, N_r, N_t]``.
The file must contain either:
  * a ``.pt`` file exported with ``torch.save(tensor, path)`` where the
    tensor has shape ``[N_samples, N_sc, N_r, N_t]`` (complex64/complex128),
  * or an ``.npz`` file with a key ``"csi"`` containing the same shape.

The DataLoader will then return batches of shape ``[B, N_sc, N_r, N_t]``
(complex64 torch tensors).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    """Load CSI data from a ``.npz`` or ``.pt`` file.

    Parameters
    ----------
    path:
        Path to the data file.
    npz_key:
        Key inside the ``.npz`` archive that holds the CSI array.
        Ignored for ``.pt`` files.
    """

    def __init__(self, path: Union[str, Path], npz_key: str = "csi") -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu")
            if not torch.is_tensor(data):
                raise ValueError(".pt file must contain a single tensor")
        elif path.suffix == ".npz":
            archive = np.load(path)
            if npz_key not in archive:
                raise KeyError(
                    f"Key '{npz_key}' not found in {path}. "
                    f"Available keys: {list(archive.keys())}"
                )
            data = torch.from_numpy(archive[npz_key])
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix!r}")

        # Ensure complex64
        if not data.is_complex():
            raise ValueError(
                "Loaded tensor must be complex (complex64 / complex128). "
                f"Got dtype={data.dtype}"
            )
        self._data = data.to(torch.complex64)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]  # shape: [N_sc, N_r, N_t]

    @property
    def shape(self):
        return tuple(self._data.shape)
