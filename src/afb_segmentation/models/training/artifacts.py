"""Checkpoint and training-history file helpers.

The functions write the existing PyTorch state dictionaries and epoch rows to
the paths supplied by the training workflows.
"""

import csv
from typing import Any, Dict

import torch


def save_checkpoint(state: dict, path: str) -> None:
    """Serialize a training state dictionary with ``torch.save``.

    Parameters
    ----------
    state : dict
        Checkpoint contents to serialize.
    path : str
        Destination path for the checkpoint file.
    """
    torch.save(state, path)


def write_history(history: list[Dict[str, Any]], path: str) -> None:
    """Write epoch history rows to a CSV file, replacing any existing file.

    Parameters
    ----------
    history : list[Dict[str, Any]]
        Epoch result rows. The first row defines the CSV column order.
    path : str
        Destination path for the CSV file.

    Notes
    -----
    An empty history is a no-op. An existing file at ``path`` is overwritten.
    """
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


__all__ = ["save_checkpoint", "write_history"]
