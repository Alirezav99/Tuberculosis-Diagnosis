"""PyTorch dataset for paired DDS3 images and segmentation masks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from afb_segmentation.utils.image_filtering import filter_image_background


class CustomDatasetTBDD3(Dataset):
    """Load DDS3 images after YOLO-based background filtering with GT masks.

    Image paths are collected from the split's ``images`` directory; each
    target mask is looked up by the same filename under ``masks``. The YOLO
    prediction is used by ``filter_image_background`` to prepare the input,
    while the returned target is loaded from the ground-truth mask file.
    """

    def __init__(self, resource_folder: str, yolo, size: int):
        """Set the split directory, YOLO model, and image-filtering size.

        Parameters
        ----------
        resource_folder : str
            Split directory containing ``images`` and ``masks`` subdirectories.
        yolo
            Model passed to the background-filtering helper.
        size : int
            Size argument passed to ``filter_image_background``.
        """
        self.images_dir = Path(resource_folder) / "images"
        self.masks_dir = Path(resource_folder) / "masks"
        self.image_paths = list(self.images_dir.glob("*.png"))
        self.yolo = yolo
        self.size = size

    def __len__(self) -> int:
        """Return the number of image paths discovered at initialization.

        Returns
        -------
        int
            Number of image files found when the dataset was initialized.
        """
        return len(self.image_paths)

    def __getitem__(self, idx) -> tuple[torch.FloatTensor, torch.IntTensor]:
        """Return one filtered image tensor and its filename-matched GT mask.

        The image is converted to RGB and passed through YOLO background
        filtering. The grayscale target mask is converted to integer values
        after division by 255; its dimensions are not changed here.

        Parameters
        ----------
        idx : int
            Index of the image and matching mask to load.

        Returns
        -------
        tuple[torch.FloatTensor, torch.IntTensor]
            Filtered image tensor and ground-truth mask tensor.
        """
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        filtered_img = filter_image_background(
            image=image, mask=mask, yolo_model=self.yolo, size=self.size
        )
        return filtered_img, torch.IntTensor(mask / 255)
