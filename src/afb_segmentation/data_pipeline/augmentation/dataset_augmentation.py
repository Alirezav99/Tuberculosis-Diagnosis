"""Create augmented copies of DDS3 image-mask pairs."""

import glob
import logging
import os
from typing import Tuple

import dask
import torch
from dask.diagnostics import ProgressBar
from PIL import Image

from afb_segmentation.data_pipeline.augmentation import transforms
from afb_segmentation.data_pipeline.conversion import image_tiles


def augment_image(
    img: Image.Image, mask: Image.Image, shuffle_tiles: bool
) -> Tuple[Image.Image, Image.Image]:
    """
    Augment an image using tensor-based tile processing.

    Parameters
    ----------
    img : Image.Image
        Input image
    mask : Image.Image
        Input mask
    shuffle_tiles : bool
        Whether to shuffle tiles after augmentation

    Returns
    -------
    Tuple[Image.Image, Image.Image]
        Augmented image and mask
    """
    img_tiles, mask_tiles = image_tiles.convert_image_to_tensor_tiles(img, mask)

    aug_img_tiles, aug_mask_tiles = transforms.augment_tiles_tensor(
        img_tiles, mask_tiles
    )

    if shuffle_tiles:
        num_tiles = aug_img_tiles.shape[0]
        shuffle_indices = torch.randperm(num_tiles)
        aug_img_tiles = aug_img_tiles[shuffle_indices]
        aug_mask_tiles = aug_mask_tiles[shuffle_indices]

    new_img, new_mask = image_tiles.convert_tensor_tiles_to_image(
        aug_img_tiles, aug_mask_tiles
    )

    return new_img, new_mask


def _augment_and_save(
    img_path: str,
    mask_path: str,
    dst_img: str,
    dst_msk: str,
    shuffle_tiles: bool,
) -> None:
    """
    Load, augment and save a single image-mask pair.

    Parameters
    ----------
    img_path : str
        Path to input image
    mask_path : str
        Path to input mask
    dst_img : str
        Path to save augmented image
    dst_msk : str
        Path to save augmented mask
    shuffle_tiles : bool
        Whether to shuffle tiles during augmentation
    """
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    new_img, new_mask = augment_image(img, mask, shuffle_tiles)

    new_img.save(dst_img)
    new_mask.save(dst_msk)


def augment_dds3(
    folder2augment: str = "train",
    shuffle_tiles: bool = False,
) -> None:
    """Write one augmented copy for each image in a restructured DDS3 split.

    Source image and mask lists are collected separately with ``glob`` and
    paired by list index. Augmented files are written into the same split under
    generated ``MSC<number>.png`` names. Work is dispatched to Dask's process
    scheduler; torchvision and optional tile shuffling use random operations
    without a seed set here.

    Parameters
    ----------
    folder2augment : str, optional
        Split name beneath ``ROOT_DATA/dds3-restructured``.
    shuffle_tiles : bool, optional
        Whether to randomly permute tiles before reconstruction.

    Notes
    -----
    Adds image and mask files to the selected split's ``images`` and
    ``masks`` directories.
    """

    src_data = os.getenv("ROOT_DATA")
    path_data = os.path.join(src_data, "dds3-restructured", folder2augment)
    train_images = glob.glob(os.path.join(path_data, "images", "*.png"))
    train_masks = glob.glob(os.path.join(path_data, "masks", "*.png"))
    size_data = len(train_images)

    logging.info(f"Start to add augmentations for {folder2augment}.")

    tasks = []
    for i in range(size_data):
        dst_img = os.path.join(path_data, "images", f"MSC{size_data + i}.png")
        dst_msk = os.path.join(path_data, "masks", f"MSC{size_data + i}.png")
        task = dask.delayed(_augment_and_save)(
            train_images[i],
            train_masks[i],
            dst_img,
            dst_msk,
            shuffle_tiles,
        )
        tasks.append(task)

    with ProgressBar():
        dask.compute(*tasks, scheduler="processes")

    logging.info(f"Augmentation finished.")
