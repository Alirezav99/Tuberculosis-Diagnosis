"""Compatibility imports and command-line entry point for calibration analysis.

The implementation lives in ``evaluation.calibration``. This module retains
the historical import path and delegates script execution to the canonical
workflow.
"""

from .evaluation.calibration.metrics import CalibrationAnalyzer, CalibrationMetrics
from .evaluation.calibration.reporting import (
    plot_reliability_diagram,
    save_calibration_results,
)
from .evaluation.calibration.workflow import (
    evaluate_segmentation_calibration,
    evaluate_unet_calibration,
    main,
)

__all__ = [
    "CalibrationAnalyzer",
    "CalibrationMetrics",
    "evaluate_segmentation_calibration",
    "evaluate_unet_calibration",
    "main",
    "plot_reliability_diagram",
    "save_calibration_results",
]

if __name__ == "__main__":
    main()
