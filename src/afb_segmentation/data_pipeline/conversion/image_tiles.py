"""Convert images and masks to and from fixed-size tiles."""

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor


def convert_image_to_mosaics(
    image: PngImageFile, mask: PngImageFile, mos_siz: List[int] = [40, 40]
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Crop an image and its mask into corresponding fixed-size tiles.

    Parameters
    ----------
    image : PngImageFile
        The input image to divide into mosaics.
    mask : PngImageFile
        A mask image cropped with the same coordinates as ``image``.
    mos_siz : List[int], default=[40, 40]
        The tile size as ``[height, width]``.

    Returns
    -------
    Tuple[List[Image.Image], List[Image.Image]]
        Two lists in matching row-major order: image tiles and mask tiles.
        Tiles are cropped at each grid position; the mask is not used to omit
        empty tiles.
    """
    img_w, img_h = image.size
    mos_h, mos_w = mos_siz
    mosaics_images = []
    mosaics_masks = []

    for y in range(0, img_h, mos_h):
        for x in range(0, img_w, mos_w):
            box = (x, y, x + mos_w, y + mos_h)
            tile_mask = mask.crop(box)
            tile_image = image.crop(box)

            mosaics_images.append(tile_image)
            mosaics_masks.append(tile_mask)

    return (mosaics_images, mosaics_masks)


def convert_tiles_to_image(tiles, masks, tiles_per_row=10, tiles_per_col=10):
    """Place image and mask tiles into a fixed row-major grid.

    Parameters
    ----------
    tiles : list
        List of image tiles. Each tile can be a PIL image, NumPy array, or
        tensor; all tiles should have the same size.
    masks : list
        List of mask tiles in the same order as ``tiles``.
    tiles_per_row : int, optional
        Number of grid columns (default is 10).
    tiles_per_col : int, optional
        Number of grid rows (default is 10).

    Returns
    -------
    tuple of PIL.Image
        The combined image and mask as PIL images. Tiles are pasted in list
        order; unused grid cells retain the new image's default fill.
    """
    if isinstance(tiles[0], Tensor):
        to_pil = ToPILImage()
        tiles = [to_pil(tile) for tile in tiles]
    elif isinstance(tiles[0], np.ndarray):
        tiles = [Image.fromarray(tile) for tile in tiles]
    if isinstance(masks[0], Tensor):
        to_pil = ToPILImage()
        masks = [to_pil(mask) for mask in masks]
    elif isinstance(masks[0], np.ndarray):
        masks = [Image.fromarray(mask) for mask in masks]
    tile_w, tile_h = tiles[0].size
    big_img = Image.new("RGB", (tile_w * tiles_per_row, tile_h * tiles_per_col))
    mask_mode = masks[0].mode if hasattr(masks[0], "mode") else "L"
    big_mask = Image.new(mask_mode, (tile_w * tiles_per_row, tile_h * tiles_per_col))
    for idx, (tile, mask) in enumerate(zip(tiles, masks)):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        big_img.paste(tile, (col * tile_w, row * tile_h))
        big_mask.paste(mask, (col * tile_w, row * tile_h))
    return big_img, big_mask


def convert_image_to_tensor_tiles(
    image: Image.Image, mask: Image.Image, tile_size: Tuple[int, int] = (40, 40)
) -> Tuple[Tensor, Tensor]:
    """
    Convert PIL Images to tensor tiles using unfold operations.

    Parameters
    ----------
    image : PIL.Image
        Input image to convert to tiles.
    mask : PIL.Image
        Input mask to convert to tiles.
    tile_size : tuple[int, int], optional
        Size of each tile as ``(height, width)``; defaults to ``(40, 40)``.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Image tiles and mask tiles as tensors with shape (num_tiles, channels, tile_h, tile_w)
    """
    to_tensor = ToTensor()

    img_tensor = to_tensor(image)
    mask_tensor = to_tensor(mask)

    tile_h, tile_w = tile_size

    img_tiles = img_tensor.unfold(1, tile_h, tile_h).unfold(2, tile_w, tile_w)
    mask_tiles = mask_tensor.unfold(1, tile_h, tile_h).unfold(2, tile_w, tile_w)

    try:
        C, n_tiles_h, n_tiles_w, tile_h, tile_w = img_tiles.shape
        img_tiles = (
            img_tiles.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, tile_h, tile_w)
        )
        C_mask = mask_tiles.shape[0]
        mask_tiles = (
            mask_tiles.permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(-1, C_mask, tile_h, tile_w)
            .mean(dim=1, keepdim=True)
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"[convert_image_to_tensor_tiles] Shape mismatch or data corruption detected. img_tiles.shape: {getattr(img_tiles, 'shape', None)}, mask_tiles.shape: {getattr(mask_tiles, 'shape', None)}. Original error: {e}"
        )

    return img_tiles, mask_tiles


def convert_tensor_tiles_to_image(
    img_tiles: Tensor, mask_tiles: Tensor, grid_size: Tuple[int, int] = (10, 10)
) -> Tuple[Image.Image, Image.Image]:
    """
    Convert tensor tiles back to PIL Images.

    Parameters
    ----------
    img_tiles : Tensor
        Image tiles with shape ``(num_tiles, channels, tile_h, tile_w)``.
    mask_tiles : Tensor
        Mask tiles with shape ``(num_tiles, 1, tile_h, tile_w)``.
    grid_size : tuple[int, int], optional
        Grid dimensions ``(n_tiles_h, n_tiles_w)``; defaults to ``(10, 10)``.

    Returns
    -------
    Tuple[Image.Image, Image.Image]
        Reconstructed image and mask as PIL Images
    """
    to_pil = ToPILImage()
    n_tiles_h, n_tiles_w = grid_size
    num_tiles, C, tile_h, tile_w = img_tiles.shape

    img_grid = img_tiles.view(n_tiles_h, n_tiles_w, C, tile_h, tile_w)
    mask_grid = mask_tiles.view(n_tiles_h, n_tiles_w, 1, tile_h, tile_w)

    img_grid = img_grid.permute(2, 0, 3, 1, 4).contiguous()
    img_full = img_grid.view(C, n_tiles_h * tile_h, n_tiles_w * tile_w)

    mask_grid = mask_grid.permute(2, 0, 3, 1, 4).contiguous()
    mask_full = mask_grid.view(1, n_tiles_h * tile_h, n_tiles_w * tile_w)

    img_pil = to_pil(img_full)
    mask_pil = to_pil(mask_full.squeeze(0))

    return img_pil, mask_pil
