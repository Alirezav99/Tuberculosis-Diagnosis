"""Calculate mask heatmaps and spatial entropy for dataset analysis."""

import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_spatial_entropy(
    image: np.ndarray,
    kernel_size: int = 5,
    stride: int = 5,
) -> float:
    """Calculate normalized Shannon entropy for an image heatmap.

    The image is partitioned into cells according to ``kernel_size`` and
    ``stride``. Cell intensity sums form the probability distribution used for
    entropy calculation.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional probability heatmap.
    kernel_size : int, default=5
        Size of each region whose intensity is summed.
    stride : int, default=5
        Distance between successive regions.

    Returns
    -------
    float
        Entropy normalized to the range from zero to one.
    """
    cell_intensities = []

    for i in range(0, image.shape[0] - kernel_size + 1, stride):
        for j in range(0, image.shape[1] - kernel_size + 1, stride):
            local_region = image[i : i + kernel_size, j : j + kernel_size]
            local_sum = np.sum(local_region)
            cell_intensities.append(local_sum)

    cell_intensities = np.array(cell_intensities)
    total_intensity = cell_intensities.sum()

    if total_intensity == 0:
        return 0.0

    prob_distribution = cell_intensities / total_intensity

    entropy_value = 0.0
    for p in prob_distribution:
        if p > 0:
            entropy_value -= p * np.log2(p)

    num_cells = len(cell_intensities)
    max_entropy = np.log2(num_cells) if num_cells > 1 else 1.0
    normalized_entropy = entropy_value / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy


def get_mean_patch_heatmap(mask_dir: str) -> np.ndarray:
    """Return the normalized mean pixel distribution across PNG masks.

    Parameters
    ----------
    mask_dir : str
        Directory containing mask PNG files.

    Returns
    -------
    numpy.ndarray
        Probability heatmap with the dimensions of an input mask.

    Raises
    ------
    ValueError
        If ``mask_dir`` contains no PNG masks.
    """
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not mask_files:
        raise ValueError(f"No mask files found in {mask_dir}")

    sum_mask = None
    count = 0

    for mask_path in tqdm(mask_files, desc="Processing patches for mean heatmap"):
        img = Image.open(mask_path).convert("L")
        mask = np.array(img, dtype=np.float32)
        mask = mask / 255.0

        if sum_mask is None:
            sum_mask = np.zeros_like(mask, dtype=np.float32)

        sum_mask += mask
        count += 1

    mean_patch = sum_mask / count

    total = mean_patch.sum()
    if total > 0:
        mean_patch = mean_patch / total
    else:
        mean_patch = np.ones_like(mean_patch) / mean_patch.size

    return mean_patch
