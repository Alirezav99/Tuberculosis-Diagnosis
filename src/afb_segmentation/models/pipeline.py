"""Compatibility import for the YOLO-plus-U-Net inference pipeline.

The implementation is maintained in ``inference.pipeline``.
"""

from .inference.pipeline import (
    CONFIG_TRAIN_PATH,
    DATA_PATH,
    MODEL_PATH,
    RUNS_PATH,
    Pipeline,
)

__all__ = ["CONFIG_TRAIN_PATH", "DATA_PATH", "MODEL_PATH", "Pipeline", "RUNS_PATH"]
