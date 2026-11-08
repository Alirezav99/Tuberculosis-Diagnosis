"""Compatibility imports for the historical segmentation-loss module.

The loss implementations are maintained in ``loss.segmentation``.
"""

from .loss.segmentation import (
    BoundaryLoss,
    CompoundLoss,
    DiceBCELoss,
    DiceLoss,
    FocalLoss,
)

__all__ = ["BoundaryLoss", "CompoundLoss", "DiceBCELoss", "DiceLoss", "FocalLoss"]
