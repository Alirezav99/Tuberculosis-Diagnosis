"""Pixel-wise calibration calculations for segmentation predictions.

The analyzer treats each pixel as a sample and computes expected, maximum,
and adaptive calibration errors together with Brier score, negative
log-likelihood, and confidence errors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class CalibrationMetrics:
    """Container for calibration scores and reliability-bin statistics.

    Attributes
    ----------
    ece : float
        Expected Calibration Error.
    mce : float
        Maximum Calibration Error.
    brier_score : float
        Mean squared probability error.
    nll : float
        Negative log-likelihood.
    ace : float
        Adaptive Calibration Error.
    overconfidence_error : float
        Average error from predictions more confident than their accuracy.
    underconfidence_error : float
        Average error from predictions less confident than their accuracy.
    bin_accuracies : List[float]
        Accuracy values for reliability bins.
    bin_confidences : List[float]
        Mean confidence values for reliability bins.
    bin_counts : List[int]
        Number of pixel predictions in each bin.
    bin_edges : List[float]
        Probability boundaries for the bins.
    """

    ece: float
    mce: float
    brier_score: float
    nll: float
    ace: float
    overconfidence_error: float
    underconfidence_error: float

    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)
    bin_edges: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scores and the bin count to a serialization dictionary.

        Returns
        -------
        result : Dict[str, Any]
            Scalar calibration scores and the number of reliability bins.
        """
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "nll": self.nll,
            "ace": self.ace,
            "overconfidence_error": self.overconfidence_error,
            "underconfidence_error": self.underconfidence_error,
            "n_bins": len(self.bin_accuracies),
        }


class CalibrationAnalyzer:
    """Analyzer for model calibration in segmentation tasks.

    Computes calibration metrics on a per-pixel basis for segmentation models,
    treating each pixel prediction as an independent sample.

    Attributes
    ----------
    n_bins
        Number of bins for ECE/MCE computation
    adaptive_bins
        Whether to use adaptive binning (equal samples per bin)
    """

    def __init__(
        self,
        n_bins: int = 15,
        adaptive_bins: bool = False,
        eps: float = 1e-7,
    ):
        """Initialize the CalibrationAnalyzer.

        Parameters
        ----------
        n_bins : int
            Number of bins for calibration error computation
        adaptive_bins : bool
            If True, use adaptive binning with equal samples per bin
        eps : float
            Small epsilon for numerical stability
        """
        self.n_bins = n_bins
        self.adaptive_bins = adaptive_bins
        self.eps = eps

    def compute_ece(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, List[float], List[float], List[int], List[float]]:
        """Compute Expected Calibration Error (ECE).

        ECE = Σ (|B_b| / n) * |acc(B_b) - conf(B_b)|

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities, flattened (N,)
        labels : torch.Tensor
            Ground truth labels, flattened (N,)

        Returns
        -------
        ece : float
            Expected Calibration Error
        bin_accuracies : List[float]
            Accuracy in each bin
        bin_confidences : List[float]
            Mean confidence in each bin
        bin_counts : List[int]
            Number of samples in each bin
        bin_edges : List[float]
            Bin boundaries
        """
        predictions = predictions.flatten().cpu().numpy()
        labels = labels.flatten().cpu().numpy()

        n_samples = len(predictions)

        if self.adaptive_bins:

            sorted_indices = np.argsort(predictions)
            bin_size = n_samples // self.n_bins
            bin_edges = [0.0]

            for i in range(1, self.n_bins):
                idx = min(i * bin_size, n_samples - 1)
                bin_edges.append(predictions[sorted_indices[idx]])
            bin_edges.append(1.0)
            bin_edges = np.array(bin_edges)
        else:

            bin_edges = np.linspace(0, 1, self.n_bins + 1)

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0

        for i in range(self.n_bins):

            if i == self.n_bins - 1:

                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
            else:
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

            bin_count = mask.sum()
            bin_counts.append(int(bin_count))

            if bin_count > 0:

                bin_preds = predictions[mask]
                bin_labels = labels[mask]

                binary_preds = (bin_preds > 0.5).astype(float)
                accuracy = (binary_preds == bin_labels).mean()

                confidence = np.where(bin_preds > 0.5, bin_preds, 1 - bin_preds).mean()

                bin_accuracies.append(float(accuracy))
                bin_confidences.append(float(confidence))

                ece += (bin_count / n_samples) * abs(accuracy - confidence)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)

        return ece, bin_accuracies, bin_confidences, bin_counts, bin_edges.tolist()

    def compute_mce(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute Maximum Calibration Error (MCE).

        MCE = max_b |acc(B_b) - conf(B_b)|

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities
        labels : torch.Tensor
            Ground truth labels

        Returns
        -------
        mce : float
            Maximum Calibration Error
        """
        _, bin_accuracies, bin_confidences, bin_counts, _ = self.compute_ece(
            predictions, labels
        )

        mce = 0.0
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            if count > 0:
                mce = max(mce, abs(acc - conf))

        return mce

    def compute_brier_score(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute Brier Score.

        BS = (1/n) * Σ (p_i - y_i)²

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities
        labels : torch.Tensor
            Ground truth labels

        Returns
        -------
        brier_score : float
            Mean squared error between predictions and labels
        """
        predictions = predictions.flatten().float()
        labels = labels.flatten().float()

        brier = ((predictions - labels) ** 2).mean().item()
        return brier

    def compute_nll(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute Negative Log-Likelihood.

        NLL = -(1/n) * Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities
        labels : torch.Tensor
            Ground truth labels

        Returns
        -------
        nll : float
            Negative log-likelihood
        """
        predictions = predictions.flatten().float()
        labels = labels.flatten().float()

        predictions = torch.clamp(predictions, self.eps, 1 - self.eps)

        nll = (
            -(
                labels * torch.log(predictions)
                + (1 - labels) * torch.log(1 - predictions)
            )
            .mean()
            .item()
        )

        return nll

    def compute_overconfidence_error(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute Overconfidence and Underconfidence Errors.

        Overconfidence: when confidence > accuracy (model too sure)
        Underconfidence: when confidence < accuracy (model too uncertain)

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities
        labels : torch.Tensor
            Ground truth labels

        Returns
        -------
        overconfidence_error : float
            Average overconfidence
        underconfidence_error : float
            Average underconfidence
        """
        predictions = predictions.flatten().cpu().numpy()
        labels = labels.flatten().cpu().numpy()

        n_samples = len(predictions)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)

        overconfidence = 0.0
        underconfidence = 0.0

        for i in range(self.n_bins):
            if i == self.n_bins - 1:
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
            else:
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

            bin_count = mask.sum()
            if bin_count > 0:
                bin_preds = predictions[mask]
                bin_labels = labels[mask]

                binary_preds = (bin_preds > 0.5).astype(float)
                accuracy = (binary_preds == bin_labels).mean()
                confidence = np.where(bin_preds > 0.5, bin_preds, 1 - bin_preds).mean()

                weight = bin_count / n_samples
                gap = confidence - accuracy

                if gap > 0:
                    overconfidence += weight * gap
                else:
                    underconfidence += weight * abs(gap)

        return overconfidence, underconfidence

    def compute_ace(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute Adaptive Calibration Error (ACE).

        Like ECE but with adaptive binning (equal samples per bin).

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities
        labels : torch.Tensor
            Ground truth labels

        Returns
        -------
        ace : float
            Adaptive Calibration Error
        """
        original_adaptive = self.adaptive_bins
        self.adaptive_bins = True
        ace, _, _, _, _ = self.compute_ece(predictions, labels)
        self.adaptive_bins = original_adaptive
        return ace

    def compute_calibration_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> CalibrationMetrics:
        """Compute all calibration metrics.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted probabilities (any shape, will be flattened)
        labels : torch.Tensor
            Ground truth labels (same shape as predictions)

        Returns
        -------
        result : CalibrationMetrics
            CalibrationMetrics containing all computed metrics
        """

        ece, bin_accs, bin_confs, bin_counts, bin_edges = self.compute_ece(
            predictions, labels
        )

        mce = self.compute_mce(predictions, labels)
        brier = self.compute_brier_score(predictions, labels)
        nll = self.compute_nll(predictions, labels)
        ace = self.compute_ace(predictions, labels)
        overconf, underconf = self.compute_overconfidence_error(predictions, labels)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            nll=nll,
            ace=ace,
            overconfidence_error=overconf,
            underconfidence_error=underconf,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
            bin_edges=bin_edges,
        )

    def evaluate_model_calibration(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        mc_dropout: bool = False,
        n_mc_samples: int = 10,
    ) -> Tuple[CalibrationMetrics, Dict[str, Any]]:
        """Evaluate model calibration on a dataset.

        Parameters
        ----------
        model : nn.Module
            Trained segmentation model
        data_loader : DataLoader
            DataLoader for evaluation data
        device : torch.device
            Device to run inference on
        mc_dropout : bool
            Whether to use MC Dropout for predictions
        n_mc_samples : int
            Number of MC Dropout samples if enabled

        Returns
        -------
        metrics : CalibrationMetrics
            CalibrationMetrics for the entire dataset
        detailed_results : Dict[str, Any]
            Dictionary with additional analysis
        """
        model.to(device)
        model.eval()

        if mc_dropout:

            for module in model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    module.train()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Calibration Evaluation"):
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                if len(labels.shape) == 3:
                    labels = labels.unsqueeze(1)

                if mc_dropout:

                    mc_preds = []
                    for _ in range(n_mc_samples):
                        pred = model(inputs)
                        mc_preds.append(pred)
                    predictions = torch.stack(mc_preds, dim=0).mean(dim=0)
                else:
                    predictions = model(inputs)

                if predictions.min() < 0 or predictions.max() > 1:
                    predictions = torch.sigmoid(predictions)

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.compute_calibration_metrics(all_predictions, all_labels)

        detailed_results = {
            "total_pixels": all_predictions.numel(),
            "positive_ratio": (all_labels > 0.5).float().mean().item(),
            "mean_prediction": all_predictions.mean().item(),
            "std_prediction": all_predictions.std().item(),
            "mc_dropout_enabled": mc_dropout,
            "n_mc_samples": n_mc_samples if mc_dropout else 0,
        }

        return metrics, detailed_results


__all__ = ["CalibrationAnalyzer", "CalibrationMetrics"]
