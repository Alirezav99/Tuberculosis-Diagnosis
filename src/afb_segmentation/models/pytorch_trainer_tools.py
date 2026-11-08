"""Compatibility imports for PyTorch segmentation training helpers.

The epoch loops and instance metrics are maintained in
``training.segmentation_epochs`` and ``metrics.instance_segmentation``.
"""

from .loss.segmentation import CompoundLoss
from .metrics.instance_segmentation import (
    compute_batch_instance_metrics,
    compute_instance_segmentation_metrics,
)
from .training.segmentation_epochs import eval_one_epoch, train_one_epoch

__all__ = [
    "CompoundLoss",
    "compute_batch_instance_metrics",
    "compute_instance_segmentation_metrics",
    "eval_one_epoch",
    "train_one_epoch",
]
