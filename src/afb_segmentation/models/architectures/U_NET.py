"""Compatibility imports for the original U-Net architecture module.

The implementation is maintained in ``architectures.unet``; this module keeps
the historical import path available.
"""

from .unet import (
    AttentionGate,
    AttentionUNetMaskPredictor,
    AttentionUp,
    DoubleConv,
    Down,
    OutConv,
    UNetMaskPredictor,
    Up,
)

__all__ = [
    "AttentionGate",
    "AttentionUNetMaskPredictor",
    "AttentionUp",
    "DoubleConv",
    "Down",
    "OutConv",
    "UNetMaskPredictor",
    "Up",
]
