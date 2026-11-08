"""Build paired torchvision transforms for image-mask tile augmentation.

The transform composition is used by offline DDS3 augmentation. The paired
image and mask tiles are passed through torchvision v2 transforms, then mask
values are thresholded by ``augment_tiles_tensor`` before reconstruction.
"""

from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import ToPILImage, v2

to_pil = ToPILImage()


def create_augmentation_compose(
    p_hflip: float = 0.5,
    p_vfilp: float = 0.5,
    sharpness: float = 3,
    rotation_low: int = 0,
    rotation_high: int = 0,
    scale_high: float = 1.7,
    scale_low: float = 1,
    contrast_low: float = 0.85,
    contrast_high: float = 1.3,
    bright_low: float = 0.85,
    bright_high: float = 1.2,
    saturation_low: float = 0.9,
    saturation_high: float = 1.3,
    hue_low: float = -0.05,
    hue_high: float = 0.05,
    kernel_blur: int = 1,
):
    """Build the configured random torchvision v2 transform composition.

    Parameters
    ----------
    p_hflip : float, optional
        Probability of applying horizontal flip.
    p_vfilp : float, optional
        Probability of applying vertical flip.
    sharpness : float, optional
        Sharpness factor for RandomAdjustSharpness.
    rotation_low : int, optional
        Lower bound for random rotation in degrees.
    rotation_high : int, optional
        Upper bound for random rotation in degrees.
    scale_high : float, optional
        Upper bound for random scaling.
    scale_low : float, optional
        Lower bound for random scaling.
    contrast_low : float, optional
        Lower bound for random contrast adjustmen.
    contrast_high : float, optional
        Upper bound for random contrast adjustment.
    bright_low : float, optional
        Lower bound for random brightness adjustment.
    bright_high : float, optional
        Upper bound for random brightness adjustment.
    saturation_low : float, optional
        Lower bound for random saturation adjustment.
    saturation_high : float, optional
        Upper bound for random saturation adjustment.
    hue_low : float, optional
        Lower bound for random hue adjustment.
    hue_high : float, optional
        Upper bound for random hue adjustment.
    kernel_blur : int, optional
        Kernel size for Gaussian blur.

    Returns
    -------
    torchvision.transforms.v2.Compose
        A composition containing the configured random flips, affine,
        appearance, and blur transforms.
    """

    list_transforms = [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p_hflip),
        v2.RandomVerticalFlip(p_vfilp),
        v2.RandomApply(
            [
                v2.RandomAdjustSharpness(sharpness_factor=sharpness),
            ],
            p=0.5,
        ),
        v2.RandomAffine(
            degrees=(rotation_low, rotation_high),
            scale=(scale_low, scale_high),
        ),
        v2.ColorJitter(
            contrast=(contrast_low, contrast_high),
            brightness=(bright_low, bright_high),
            saturation=(saturation_low, saturation_high),
            hue=(hue_low, hue_high),
        ),
        v2.RandomApply(
            [
                v2.GaussianBlur(kernel_size=kernel_blur),
            ],
            p=0.3,
        ),
    ]

    return v2.Compose(list_transforms)


def augment_tiles_tensor(
    img_tiles: Tensor, mask_tiles: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Apply augmentations to tensor tiles in batch.

    Parameters
    ----------
    img_tiles : Tensor
        Image tiles with shape (num_tiles, channels, tile_h, tile_w)
    mask_tiles : Tensor
        Mask tiles with shape (num_tiles, 1, tile_h, tile_w)

    Returns
    -------
    Tuple[Tensor, Tensor]
        Augmented image and mask tiles. Mask tile values are thresholded to
        either 0 or 255 before they are returned.
    """
    transforms = create_augmentation_compose()

    augmented_imgs = []
    augmented_masks = []

    for i in range(img_tiles.shape[0]):
        img_tile = to_pil(img_tiles[i])
        mask_tile = to_pil(mask_tiles[i])

        aug_img, aug_mask = transforms(img_tile, mask_tile)

        aug_mask = torch.where(
            aug_mask > 100,
            torch.tensor(255, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8),
        )

        augmented_imgs.append(aug_img.unsqueeze(0))
        augmented_masks.append(aug_mask.unsqueeze(0))

    augmented_imgs = torch.cat(augmented_imgs, dim=0)
    augmented_masks = torch.cat(augmented_masks, dim=0)

    return augmented_imgs, augmented_masks
