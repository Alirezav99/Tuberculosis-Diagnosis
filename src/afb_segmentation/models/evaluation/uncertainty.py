"""Monte Carlo dropout evaluation for segmentation models.

The module enables dropout layers during repeated inference, aggregates pixel
metrics and uncertainty statistics, and writes aggregate and per-image CSV
results.
"""

import csv
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def enable_dropout(model: nn.Module) -> None:
    """Put Dropout and Dropout2d modules in training mode for MC dropout.

    Parameters
    ----------
    model : nn.Module
        Model whose dropout modules will be enabled.

    Notes
    -----
    Other modules retain their current mode; call ``model.eval()`` first when
    evaluation behavior is desired for non-dropout layers.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


def mc_dropout_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform repeated stochastic forward passes with dropout enabled.

    Parameters
    ----------
    model : nn.Module
        Neural network that returns per-pixel probabilities.
    inputs : torch.Tensor
        Input batch with shape ``(B, C, H, W)``.
    n_samples : int
        Number of stochastic forward passes to average.
    device : torch.device
        Device on which to run the model.

    Returns
    -------
    mean_prediction : torch.Tensor
        Mean prediction across the stochastic forward passes.
    uncertainty : torch.Tensor
        Per-pixel standard deviation across the passes.
    """

    model.eval()
    enable_dropout(model)
    inputs = inputs.to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(inputs)
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)

    return mean_pred.squeeze(0), uncertainty.squeeze(0)


def evaluate_with_uncertainty(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    n_samples: int,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Evaluate pixel metrics and MC dropout uncertainty over a test loader.

    Parameters
    ----------
    model : nn.Module
        Trained model
    device : torch.device
        Device on which to run the model.
    test_loader : DataLoader
        Test data loader
    n_samples : int
        Number of MC Dropout samples

    Returns
    -------
    aggregate_metrics : Dict[str, Any]
        Aggregate test metrics. Segmentation metrics threshold the mean
        prediction at 0.5.
    per_sample_results : list[Dict[str, Any]]
        One uncertainty statistics row for each image in loader order.
        Confidence uses an uncertainty threshold of 0.2.
    """
    from tqdm import tqdm

    model.to(device)
    model.eval()
    enable_dropout(model)

    per_sample_results = []
    all_mean_preds = []
    all_labels = []
    all_uncertainties = []

    sample_idx = 0
    for inputs, labels in tqdm(test_loader, desc="MC Dropout Evaluation"):
        inputs = inputs.float().to(device)
        labels = labels.float().unsqueeze(1).to(device)

        mean_pred, uncertainty = mc_dropout_inference(
            model=model,
            inputs=inputs,
            n_samples=n_samples,
            device=device,
        )

        all_mean_preds.append(mean_pred)
        all_labels.append(labels)
        all_uncertainties.append(uncertainty)

        batch_size = inputs.shape[0]
        for i in range(batch_size):
            pred_i = mean_pred[i]
            unc_i = uncertainty[i]
            label_i = labels[i]

            binary_pred = (pred_i > 0.5).float()

            correct = (binary_pred == label_i).float()
            accuracy = correct.mean().item()

            mean_uncertainty = unc_i.mean().item()
            max_uncertainty = unc_i.max().item()
            min_uncertainty = unc_i.min().item()
            std_uncertainty = unc_i.std().item()

            fg_mask = label_i > 0.5
            bg_mask = label_i <= 0.5

            fg_uncertainty = unc_i[fg_mask].mean().item()
            bg_uncertainty = unc_i[bg_mask].mean().item()

            confidence_threshold = 0.2
            confident_ratio = (unc_i < confidence_threshold).float().mean().item()

            eps = 1e-7
            pred_clamped = torch.clamp(pred_i, eps, 1 - eps)
            entropy = -(
                pred_clamped * torch.log(pred_clamped)
                + (1 - pred_clamped) * torch.log(1 - pred_clamped)
            )
            mean_entropy = entropy.mean().item()

            per_sample_results.append(
                {
                    "sample_idx": sample_idx,
                    "accuracy": accuracy,
                    "mean_uncertainty": mean_uncertainty,
                    "max_uncertainty": max_uncertainty,
                    "min_uncertainty": min_uncertainty,
                    "std_uncertainty": std_uncertainty,
                    "foreground_uncertainty": fg_uncertainty,
                    "background_uncertainty": bg_uncertainty,
                    "confident_ratio": confident_ratio,
                    "mean_entropy": mean_entropy,
                    "num_foreground_pixels": fg_mask.sum().item(),
                    "num_background_pixels": bg_mask.sum().item(),
                }
            )
            sample_idx += 1

    all_mean_preds = torch.cat(all_mean_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)

    binary_preds = (all_mean_preds > 0.5).float()

    tp = ((binary_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((binary_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((binary_preds == 0) & (all_labels == 1)).sum().item()
    tn = ((binary_preds == 0) & (all_labels == 0)).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    aggregate_metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_mean_uncertainty": all_uncertainties.mean().item(),
        "test_std_uncertainty": all_uncertainties.std().item(),
        "test_max_uncertainty": all_uncertainties.max().item(),
        "test_confident_ratio": (all_uncertainties < 0.2).float().mean().item(),
        "mc_dropout_samples": n_samples,
        "total_samples": sample_idx,
    }

    return aggregate_metrics, per_sample_results


def write_uncertainty_results(
    aggregate_metrics: Dict[str, Any],
    per_sample_results: list[Dict[str, Any]],
    results_dir: str,
) -> None:
    """Write aggregate and per-image uncertainty metrics to CSV files.

    Parameters
    ----------
    aggregate_metrics : Dict[str, Any]
        Aggregate metric values written as one CSV row.
    per_sample_results : list[Dict[str, Any]]
        Per-image metric rows. The first row defines the CSV column order.
    results_dir : str
        Existing directory where the CSV files are written.

    Notes
    -----
    Always writes ``uncertainty_aggregate.csv``. Writes
    ``uncertainty_per_sample.csv`` only when per-sample rows are present. The
    files are created in ``results_dir`` (which must already exist).
    """

    aggregate_path = os.path.join(results_dir, "uncertainty_aggregate.csv")
    with open(aggregate_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate_metrics.keys()))
        writer.writeheader()
        writer.writerow(aggregate_metrics)
    print(f"Aggregate uncertainty metrics saved to: {aggregate_path}")

    if per_sample_results:
        per_sample_path = os.path.join(results_dir, "uncertainty_per_sample.csv")
        keys = list(per_sample_results[0].keys())
        with open(per_sample_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in per_sample_results:
                writer.writerow(row)
        print(f"Per-sample uncertainty results saved to: {per_sample_path}")


__all__ = [
    "enable_dropout",
    "evaluate_with_uncertainty",
    "mc_dropout_inference",
    "write_uncertainty_results",
]
