"""Metrics for semantic and instance-level segmentation evaluation.

The package currently provides connected-component instance metrics.
"""

from .instance_segmentation import (
    compute_batch_instance_metrics,
    compute_instance_segmentation_metrics,
)

__all__ = ["compute_batch_instance_metrics", "compute_instance_segmentation_metrics"]
