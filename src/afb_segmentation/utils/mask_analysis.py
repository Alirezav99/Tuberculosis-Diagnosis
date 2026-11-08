"""Analyze foreground pixel distribution in segmentation masks."""

import logging
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("afb_segmentation.utils.common")


def count_white_pixels_in_mask_tiles(
    masks_dir: str, grid_size: int = 10
) -> pd.DataFrame:
    """Count value-255 pixels in an evenly divided grid for each PNG mask.

    Parameters
    ----------
    masks_dir : str
        Directory containing PNG mask files.
    grid_size : int, default=10
        Number of rows and columns in the grid.

    Returns
    -------
    pandas.DataFrame
        One row per successfully loaded mask. Columns contain ``mask_name``
        and pixel counts for grid cells named ``(row,col)``. Rows are sorted
        by the numeric portion of each mask filename.

    Raises
    ------
    FileNotFoundError
        If ``masks_dir`` does not exist.
    ValueError
        If the directory contains no PNG files.
    """
    logger.info(f"Starting white pixel count analysis for masks in: {masks_dir}")
    logger.info(f"Using grid size: {grid_size}x{grid_size}")

    if not os.path.exists(masks_dir):
        logger.error(f"Directory does not exist: {masks_dir}")
        raise FileNotFoundError(f"Directory does not exist: {masks_dir}")

    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

    if not mask_files:
        logger.error(f"No PNG files found in directory: {masks_dir}")
        raise ValueError(f"No PNG files found in directory: {masks_dir}")

    logger.info(f"Found {len(mask_files)} mask files to process")

    tile_cols = [
        f"({row+1},{col+1})" for row in range(grid_size) for col in range(grid_size)
    ]

    data = []
    for mask_name in tqdm(mask_files, desc="Processing masks", unit="mask"):
        logger.debug(f"Processing mask: {mask_name}")
        mask_path = os.path.join(masks_dir, mask_name)

        try:
            img = Image.open(mask_path).convert("L")
            arr = np.array(img)
            logger.debug(f"Loaded mask {mask_name} with shape: {arr.shape}")
        except Exception as e:
            logger.error(f"Failed to load mask {mask_name}: {e}")
            continue

        h, w = arr.shape
        tile_h, tile_w = h // grid_size, w // grid_size
        logger.debug(f"Tile dimensions: {tile_h}x{tile_w}")

        tile_counts = []
        for row in range(grid_size):
            for col in range(grid_size):
                y0, y1 = row * tile_h, (row + 1) * tile_h if row < grid_size - 1 else h
                x0, x1 = col * tile_w, (col + 1) * tile_w if col < grid_size - 1 else w
                tile = arr[y0:y1, x0:x1]
                white_pixels = np.sum(tile == 255)
                tile_counts.append(white_pixels)

        data.append([mask_name] + tile_counts)
        logger.debug(f"Completed processing {mask_name}")

    logger.info(f"Successfully processed {len(data)} masks")

    df = pd.DataFrame(data, columns=["mask_name"] + tile_cols)
    df["mask_id"] = df["mask_name"].apply(lambda x: int(x[3:-4]))
    df.sort_values(by="mask_id", inplace=True)

    logger.info("DataFrame created and sorted by mask ID")
    return df.reset_index(drop=True).drop(columns=["mask_id"])
