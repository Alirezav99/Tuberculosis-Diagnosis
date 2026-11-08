"""Plot mask pixel distributions across dataset splits."""

import glob
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def plot_mask_spatial_distribution(
    type_set: Literal["train", "validation", "test"],
    title: str,
    colorbar_title: str = "Probability",
    x_title: str = "Width",
    y_title: str = "Height",
    save_fig: bool = False,
    file_name: str = "mask_distribution",
) -> None:
    """Plot the spatial distribution of mask pixels across a dataset split.

    Parameters
    ----------
    type_set : {'train', 'validation', 'test'}
        Dataset split whose masks are aggregated.
    title : str
        Figure title.
    colorbar_title : str, default='Probability'
        Colorbar label.
    x_title : str, default='Width'
        Horizontal axis label.
    y_title : str, default='Height'
        Vertical axis label.
    save_fig : bool, default=False
        If true, save the figure as ``<file_name>.png``; otherwise display it.
    file_name : str, default='mask_distribution'
        Output filename stem used when ``save_fig`` is true.

    Returns
    -------
    None

    Notes
    -----
    The mask directory is resolved from ``ROOT_DATA`` and the selected split.
    """

    root_data = os.getenv("ROOT_DATA")
    print(root_data, type_set)
    mask_dir = os.path.join(root_data, "dds3-restructured", type_set, "masks")
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    sum_mask = None
    count = 0
    for mask_path in tqdm(mask_files, desc="Processing masks"):

        img = Image.open(mask_path).convert("L")
        mask = np.array(img)
        mask_bin = (mask > 0).astype(np.float32)

        if sum_mask is None:
            sum_mask = np.zeros_like(mask_bin, dtype=np.float32)

        sum_mask += mask_bin
        count += 1

    prob_map = sum_mask / count

    plt.figure(figsize=(7, 7), dpi=600)
    plt.imshow(prob_map, cmap="viridis", interpolation="nearest", aspect="auto")
    plt.colorbar(label=colorbar_title, fraction=0.046, pad=0.04)
    plt.title(title, fontsize=14)
    plt.xlabel(x_title, fontsize=12)
    plt.ylabel(y_title, fontsize=12)

    height, width = prob_map.shape
    x_ticks = np.linspace(0, width, 11)
    y_ticks = np.linspace(0, height, 11)
    plt.xticks(x_ticks, labels=[f"{int(x)}" for x in x_ticks])
    plt.yticks(y_ticks, labels=[f"{int(y)}" for y in y_ticks])

    plt.grid(visible=True, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
    contours = plt.contour(
        prob_map, levels=5, colors="black", linewidths=0.7, alpha=0.5
    )
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"{file_name}.png", dpi=600, bbox_inches="tight")
    else:
        plt.show()
