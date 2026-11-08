"""Compatibility imports for the DDS3 preparation API.

Implementation lives in :mod:`afb_segmentation.data_pipeline.preparation`.
"""

from afb_segmentation.data_pipeline.augmentation import (
    dataset_augmentation as dds3_augmentor,
)
from afb_segmentation.data_pipeline.augmentation import (
    spatial_bias as dds3_spatial_bias,
)
from afb_segmentation.data_pipeline.conversion import images as image_conversion
from afb_segmentation.data_pipeline.preparation import files as file_management
from afb_segmentation.data_pipeline.preparation.bmp import convert_bmp_to_png
from afb_segmentation.data_pipeline.preparation.dataset_manager import (
    ROOT_DATA,
    ROOT_SNAPSHOT_DATA,
    DatasetManager,
    init_dds3,
)
from afb_segmentation.data_pipeline.preparation.restructure import restructure_dataset
from afb_segmentation.data_pipeline.preparation.yolo_export import create_yolo_dataset

if __name__ == "__main__":
    init_dds3()
