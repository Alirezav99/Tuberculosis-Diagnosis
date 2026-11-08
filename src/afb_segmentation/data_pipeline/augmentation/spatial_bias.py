"""Apply paired spatial rolls to DDS3 training images and masks."""

import os
import re
from pathlib import Path
from typing import Tuple

import dask
import numpy as np
import torch
from dask.diagnostics import ProgressBar
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm


def roll_image(
    img: Tensor | PngImageFile, msk: Tensor | PngImageFile, shift_size: int = 20
) -> Tuple[PngImageFile, PngImageFile]:
    """
    Apply spatial transformation by rolling an image and its corresponding mask.

    Applies the same stochastic row and column roll to both inputs. Candidate
    shifts are sampled using Torch and NumPy; this function does not set a seed.

    Parameters
    ----------
    img : Tensor or Image.Image
        Input image to be transformed.
    msk : Tensor or Image.Image
        Corresponding mask/label image to be transformed with the same shift.
    shift_size : int, default=20
        Positive scale used to derive the standard deviation (``shift_size / 3``)
        of candidate shifts.

    Returns
    -------
    Tuple[Image.Image, Image.Image]
        A tuple containing the rolled image and mask as PIL Images.

    Raises
    ------
    ValueError
        If shift_size is not positive.
    """
    if shift_size <= 0:
        raise ValueError("shift_size must be positive")

    to_pil = ToPILImage()
    to_tensor = ToTensor()

    img_tensor = to_tensor(img) if isinstance(img, Image.Image) else img
    msk_tensor = to_tensor(msk) if isinstance(msk, Image.Image) else msk

    std = shift_size / 3
    rand_numbers = torch.normal(mean=0, std=std, size=(1, 1000))[0]
    rand_numbers = torch.where(
        rand_numbers < 0, rand_numbers + 3 * std, rand_numbers - 3 * std
    ).numpy()
    x_shift, y_shift = np.random.choice(rand_numbers, (2,))
    x_shift, y_shift = int(x_shift), int(y_shift)

    img_tensor = to_pil(
        torch.roll(img_tensor, shifts=(0, y_shift, x_shift), dims=(0, 1, 2))
    )
    msk_tensor = to_pil(
        torch.roll(msk_tensor, shifts=(0, y_shift, x_shift), dims=(0, 1, 2))
    )

    return (img_tensor, msk_tensor)


def _roll_and_save(img_path: Path, mask_path: Path, shift_size: int) -> None:
    """
    Helper function to roll and save a single image-mask pair.

    Parameters
    ----------
    img_path : Path
        Path to the image file.
    mask_path : Path
        Path to the mask file.
    shift_size : int
        Positive scale used to derive the candidate-shift standard deviation.

    Raises
    ------
    IOError
        If files cannot be opened or saved.
    """
    with Image.open(img_path) as img, Image.open(mask_path) as msk:
        img_rolled, msk_rolled = roll_image(img, msk, shift_size=shift_size)
        img_rolled.save(img_path)
        msk_rolled.save(mask_path)


def remove_dds3_spatial_bias(n_iters=6, shift_size=20):
    """
    Apply spatial bias removal to all images and masks in the DDS3 training dataset.

    Iterates through all PNG files matching the pattern MSC{number}.png in the
    dds3-restructured/train/images directory and applies random spatial shifts
    to both images and their corresponding masks. The transformations are applied
    in parallel using Dask for efficiency.

    Uses the ROOT_DATA environment variable to locate the dataset directory.
    Only processes files that have corresponding masks in the masks directory.
    The rolled images and masks overwrite the original files.

    Parameters
    ----------
    n_iters : int, default=6
        Number of times to process the discovered image list. Each iteration
        applies another random roll to the same source files.
    shift_size : int, default=20
        Positive scale used to derive the candidate-shift standard deviation.

    Returns
    -------
    None
        Matching images and masks are overwritten in place.

    Raises
    ------
    EnvironmentError
        If ROOT_DATA environment variable is not set.
    FileNotFoundError
        If the dataset directories don't exist.
    ValueError
        If shift_size is not positive.
    """
    if shift_size <= 0:
        raise ValueError("shift_size must be positive")

    src_data = os.getenv("ROOT_DATA")
    if not src_data:
        raise EnvironmentError("ROOT_DATA environment variable is not set")

    data_root = Path(src_data)
    images_dir = data_root / "dds3-restructured" / "train" / "images"
    masks_dir = data_root / "dds3-restructured" / "train" / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    pattern = re.compile(r"MSC(\d+)\.png")
    image_files = [
        f for f in images_dir.iterdir() if f.is_file() and pattern.fullmatch(f.name)
    ]

    for i in range(n_iters):
        print(f"Starting iteration {i + 1}/{n_iters} for spatial bias removal...")

        tasks = []
        processed_count = 0
        for img_path in tqdm(image_files, desc="Preparing rolling tasks"):
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                task = dask.delayed(_roll_and_save)(img_path, mask_path, shift_size)
                tasks.append(task)
                processed_count += 1

        with ProgressBar():
            dask.compute(*tasks, scheduler="processes")
            print(f"Successfully processed {processed_count} image-mask pairs.")

        print(f"Completed iteration {i + 1}/{n_iters}.\n")
