"""Measure training-mask spatial entropy before and after bias transforms.

Run with ``python -m experiments.analysis.spatial_bias`` from the repository
root. The workflow resets and prepares the working DDS3 dataset, creates
augmented training samples, applies in-place spatial rolls, and records
normalized mask entropy at each stage. Running the analysis changes the
working dataset and requires configured data paths and a writable output
location.
"""

import os

from dotenv import load_dotenv

from experiments.analysis.spatial_entropy import (
    calculate_spatial_entropy,
    get_mean_patch_heatmap,
)

load_dotenv()

from dask.distributed import Client, LocalCluster

from afb_segmentation.data_pipeline.augmentation import (
    dataset_augmentation,
    spatial_bias,
)
from afb_segmentation.data_pipeline.preparation import bmp, restructure
from afb_segmentation.data_pipeline.preparation.dataset_manager import DatasetManager

ROOT_DATA = os.getenv("ROOT_DATA", "artifacts/dataset")
mask_dir = os.path.join(ROOT_DATA, "dds3-restructured", "train", "masks")


def analyze_spatial_bias_reduction(
    mask_dir: str,
    output_csv: str = "experiments/spatial_entropy_analysis.csv",
    n_bias_removal_iters: int = 6,
    n_workers: int = 32,
) -> None:
    """
    Analyze the effect of augmentation and spatial bias removal on spatial entropy.

    This function:
    1. Resets the dataset to its original state
    2. Calculates entropy for the original dataset
    3. Applies augmentation and recalculates entropy
    4. Iteratively removes spatial bias and tracks entropy changes
    5. Saves all results to a CSV file

    Parameters
    ----------
    mask_dir : str
        Path to directory containing mask PNG files.
    output_csv : str, optional
        Path to output CSV file, by default "experiments/spatial_entropy_analysis.csv".
    n_bias_removal_iters : int, optional
        Number of spatial bias removal iterations, by default 6.
    n_workers : int, optional
        Number of Dask workers for augmentation, by default 32.

    Side effects
    ------------
    Resets and rebuilds the working DDS3 data, writes augmented training files,
    and overwrites training image/mask pairs during each spatial-bias iteration.
    The CSV parent directory must already exist.
    """
    import pandas as pd

    results = []

    print("=" * 60)
    print("Step 1: Resetting dataset to original state")
    print("=" * 60)

    DatasetManager.reset_dataset()
    bmp.convert_bmp_to_png()
    restructure.restructure_dataset()

    print("\nCalculating entropy for original dataset...")
    heat_map = get_mean_patch_heatmap(mask_dir)
    entropy = calculate_spatial_entropy(heat_map)
    results.append({"step": "original", "iteration": 0, "entropy": entropy})
    print(f"Original dataset entropy: {entropy:.6f}")

    print("\n" + "=" * 60)
    print("Step 2: Applying augmentation")
    print("=" * 60)
    cluster = LocalCluster(n_workers=n_workers, processes=True)
    with Client(cluster) as client:
        dataset_augmentation.augment_dds3(
            folder2augment="train",
            shuffle_tiles=True,
        )

    print("\nCalculating entropy after augmentation...")
    heat_map = get_mean_patch_heatmap(mask_dir)
    entropy = calculate_spatial_entropy(heat_map)
    results.append({"step": "augmentation", "iteration": 0, "entropy": entropy})
    print(f"Post-augmentation entropy: {entropy:.6f}")

    print("\n" + "=" * 60)
    print("Step 3: Iterative spatial bias removal")
    print("=" * 60)
    for i in range(1, n_bias_removal_iters + 1):
        print(f"\nBias removal iteration {i}/{n_bias_removal_iters}...")
        spatial_bias.remove_dds3_spatial_bias(n_iters=1)

        heat_map = get_mean_patch_heatmap(mask_dir)
        entropy = calculate_spatial_entropy(heat_map)
        results.append({"step": "bias_removal", "iteration": i, "entropy": entropy})
        print(f"Entropy after iteration {i}: {entropy:.6f}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_csv}")
    print("=" * 60)
    print("\nSummary:")
    print(df.to_string(index=False))
    print(
        f"\nEntropy change: {results[0]['entropy']:.6f} → {results[-1]['entropy']:.6f}"
    )
    print(f"Total improvement: {(results[-1]['entropy'] - results[0]['entropy']):.6f}")


if __name__ == "__main__":
    analyze_spatial_bias_reduction(
        mask_dir=os.path.join(ROOT_DATA, "dds3-restructured", "train", "masks"),
        output_csv="experiments/spatial_entropy_analysis.csv",
        n_bias_removal_iters=6,
        n_workers=32,
    )
