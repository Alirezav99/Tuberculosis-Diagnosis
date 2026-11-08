"""Compatibility facade for the visualization subpackage."""

from .visualization.histograms import (
    plot_histogram_bacill_area,
    plot_professional_histogram,
)
from .visualization.images import show_images_side_by_side
from .visualization.masks import plot_mask_spatial_distribution
from .visualization.model_metrics import (
    plot_evaluation_metrics,
    plot_training_hist_models,
)

__all__ = [
    "plot_evaluation_metrics",
    "plot_histogram_bacill_area",
    "plot_mask_spatial_distribution",
    "plot_professional_histogram",
    "plot_training_hist_models",
    "show_images_side_by_side",
]
