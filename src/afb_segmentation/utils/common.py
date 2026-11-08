"""Compatibility facade for utilities split into focused modules.

Use the canonical modules for new imports: ``config``, ``devices``,
``image_filtering``, ``mask_analysis``, and ``model_results``.
"""

from . import model
from .config import read_config
from .devices import select_device
from .image_filtering import (
    create_distance_matrix,
    distance_based_filtering,
    filter_image_background,
)
from .mask_analysis import count_white_pixels_in_mask_tiles, logger
from .model_results import gather_models

__all__ = [
    "count_white_pixels_in_mask_tiles",
    "create_distance_matrix",
    "distance_based_filtering",
    "filter_image_background",
    "gather_models",
    "logger",
    "model",
    "read_config",
    "select_device",
]
