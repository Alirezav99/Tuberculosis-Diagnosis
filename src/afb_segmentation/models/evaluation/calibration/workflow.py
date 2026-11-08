"""Calibration evaluation workflows for segmentation models.

The generic workflow evaluates a supplied model and loader; the U-Net
workflow resolves the configured test scenario, checkpoint, and output paths.
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import CalibrationAnalyzer, CalibrationMetrics
from .reporting import plot_reliability_diagram, save_calibration_results


def evaluate_segmentation_calibration(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    n_bins: int = 15,
    mc_dropout: bool = False,
    n_mc_samples: int = 10,
    plot: bool = True,
) -> CalibrationMetrics:
    """Convenience function to evaluate and save calibration results.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device for inference
    output_dir : str
        Directory to save results
    n_bins : int
        Number of bins for calibration
    mc_dropout : bool
        Whether to use MC Dropout
    n_mc_samples : int
        Number of MC samples if using dropout
    plot : bool
        Whether to generate reliability diagram

    Returns
    -------
    result : CalibrationMetrics
        CalibrationMetrics object with all computed metrics
    """
    analyzer = CalibrationAnalyzer(n_bins=n_bins)

    print("Computing calibration metrics...")
    metrics, detailed = analyzer.evaluate_model_calibration(
        model=model,
        data_loader=test_loader,
        device=device,
        mc_dropout=mc_dropout,
        n_mc_samples=n_mc_samples,
    )

    print("\nCalibration Analysis Results:")
    print("-" * 40)
    print(f"  Expected Calibration Error (ECE): {metrics.ece:.4f}")
    print(f"  Maximum Calibration Error (MCE):  {metrics.mce:.4f}")
    print(f"  Adaptive Calibration Error (ACE): {metrics.ace:.4f}")
    print(f"  Brier Score:                      {metrics.brier_score:.4f}")
    print(f"  Negative Log-Likelihood:          {metrics.nll:.4f}")
    print(f"  Overconfidence Error:             {metrics.overconfidence_error:.4f}")
    print(f"  Underconfidence Error:            {metrics.underconfidence_error:.4f}")

    save_calibration_results(metrics, detailed, output_dir)

    if plot:
        plot_path = os.path.join(output_dir, "reliability_diagram.png")
        plot_reliability_diagram(metrics, plot_path)

    return metrics


def evaluate_unet_calibration(augmentation: bool = True) -> CalibrationMetrics:
    """Evaluate calibration of trained UNet models on test set.

    Parameters
    ----------
    augmentation : bool
        Whether to use the model trained with augmentation

    Returns
    -------
    result : CalibrationMetrics
        CalibrationMetrics for the evaluated model
    """
    from dotenv import load_dotenv

    from ...training.unet_components import get_dataloaders, get_model
    from ...training.unet_config import load_config

    load_dotenv()

    project_root = os.getenv("ROOT_PROJECT")
    models_root = os.getenv("ROOT_MODELS")
    data_root = os.getenv("ROOT_DATA")

    cfg = load_config(project_root)

    subdir = "with_augmentation" if augmentation else "without_augmentation"
    weights_dir = os.path.join(models_root, "unet", subdir, "weights")
    results_dir = os.path.join(models_root, "unet", subdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    root_test = os.path.join(data_root, "dds3-restructured/test")
    test_loader = get_dataloaders(
        root_test, cfg, shuffle=False, augmentation=augmentation
    )

    model = get_model(cfg)
    best_model_path = os.path.join(weights_dir, "best.pt")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model not found at {best_model_path}")

    print(f"Loading model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)

    print(f"\nEvaluating calibration for UNet ({subdir})")
    print("=" * 60)

    metrics = evaluate_segmentation_calibration(
        model=model,
        test_loader=test_loader,
        device=cfg.device,
        output_dir=results_dir,
        n_bins=15,
        mc_dropout=True,
        n_mc_samples=cfg.mc_dropout_samples,
        plot=True,
    )

    return metrics


def main() -> None:
    """Evaluate calibration for both configured U-Net data scenarios."""
    print("Calibration Analysis Module")
    print("=" * 60)

    for augmentation in [False, True]:
        scenario = "with_augmentation" if augmentation else "without_augmentation"
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {scenario}")
        print("=" * 60)

        try:
            metrics = evaluate_unet_calibration(augmentation=augmentation)
            print(f"\n{scenario} - Final Results:")
            print(f"  ECE: {metrics.ece:.4f}")
            print(f"  MCE: {metrics.mce:.4f}")
            print(f"  Brier Score: {metrics.brier_score:.4f}")
            print(f"  NLL: {metrics.nll:.4f}")
        except FileNotFoundError as e:
            print(f"Skipping {scenario}: {e}")
        except Exception as e:
            print(f"Error evaluating {scenario}: {e}")

    print("\n" + "=" * 60)
    print("Calibration evaluation complete!")
    print("=" * 60)


__all__ = [
    "evaluate_segmentation_calibration",
    "evaluate_unet_calibration",
    "main",
]
