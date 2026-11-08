"""CSV serialization and reliability plots for calibration results.

The functions consume the metric container and detailed per-bin statistics
produced by ``evaluation.calibration.metrics``.
"""

import csv
import os
from typing import Any, Dict, Optional

from .metrics import CalibrationMetrics


def save_calibration_results(
    metrics: CalibrationMetrics,
    detailed_results: Dict[str, Any],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Save calibration results to CSV files.

    Notes
    -----
    - calibration_metrics.csv: Aggregate calibration metrics
    - calibration_bins.csv: Per-bin statistics for reliability diagram

    Parameters
    ----------
    metrics : CalibrationMetrics
        CalibrationMetrics object
    detailed_results : Dict[str, Any]
        Additional analysis results
    output_dir : str
        Directory to save results
    prefix : str
        Optional prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    prefix_str = f"{prefix}_" if prefix else ""

    metrics_path = os.path.join(output_dir, f"{prefix_str}calibration_metrics.csv")
    combined = {**metrics.to_dict(), **detailed_results}

    with open(metrics_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(combined.keys()))
        writer.writeheader()
        writer.writerow(combined)
    print(f"Calibration metrics saved to: {metrics_path}")

    bins_path = os.path.join(output_dir, f"{prefix_str}calibration_bins.csv")
    bin_data = []
    for i, (acc, conf, count) in enumerate(
        zip(metrics.bin_accuracies, metrics.bin_confidences, metrics.bin_counts)
    ):
        bin_data.append(
            {
                "bin_index": i,
                "bin_lower": metrics.bin_edges[i] if i < len(metrics.bin_edges) else 0,
                "bin_upper": (
                    metrics.bin_edges[i + 1] if i + 1 < len(metrics.bin_edges) else 1
                ),
                "accuracy": acc,
                "confidence": conf,
                "count": count,
                "gap": abs(acc - conf) if count > 0 else 0,
            }
        )

    with open(bins_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bin_data[0].keys()))
        writer.writeheader()
        writer.writerows(bin_data)
    print(f"Calibration bins saved to: {bins_path}")


def plot_reliability_diagram(
    metrics: CalibrationMetrics,
    output_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot reliability diagram from calibration metrics.

    Parameters
    ----------
    metrics : CalibrationMetrics
        CalibrationMetrics with bin data
    output_path : Optional[str]
        Path to save the plot (optional)
    title : str
        Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping reliability diagram plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bin_centers = [
        (metrics.bin_edges[i] + metrics.bin_edges[i + 1]) / 2
        for i in range(len(metrics.bin_accuracies))
    ]

    valid_mask = [c > 0 for c in metrics.bin_counts]
    valid_centers = [c for c, v in zip(bin_centers, valid_mask) if v]
    valid_accs = [a for a, v in zip(metrics.bin_accuracies, valid_mask) if v]
    valid_confs = [c for c, v in zip(metrics.bin_confidences, valid_mask) if v]

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    ax1.bar(
        valid_centers,
        valid_accs,
        width=1.0 / len(metrics.bin_accuracies),
        alpha=0.7,
        label="Model",
        edgecolor="black",
    )

    for center, acc, conf in zip(valid_centers, valid_accs, valid_confs):
        if conf > acc:
            ax1.fill_between(
                [center - 0.03, center + 0.03],
                [acc, acc],
                [conf, conf],
                color="red",
                alpha=0.3,
            )

    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{title}\nECE: {metrics.ece:.4f}, MCE: {metrics.mce:.4f}")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(
        bin_centers,
        metrics.bin_counts,
        width=1.0 / len(metrics.bin_counts),
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Reliability diagram saved to: {output_path}")
    else:
        plt.show()

    plt.close()


__all__ = ["plot_reliability_diagram", "save_calibration_results"]
