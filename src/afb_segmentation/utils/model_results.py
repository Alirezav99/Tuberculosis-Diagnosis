"""Load trained YOLO models and their recorded metrics."""

import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def gather_models(dir_models: str):
    """Collect YOLO checkpoints and their recorded train/test CSV data.

    Parameters
    ----------
    dir_models : str
        Directory containing model result subdirectories.

    Returns
    -------
    dict
        Mapping from successfully loaded model directory names to a YOLO
        object and parsed training and test history dictionaries. Entries
        whose model or CSV files fail to load are skipped.
    """
    models = {}
    for model_name in os.listdir(dir_models):
        model_dir = Path(dir_models) / model_name / "train/weights/best.pt"
        try:
            models[model_name] = {
                "model": YOLO(model_dir),
                "train_history": pd.read_csv(
                    Path(dir_models) / model_name / "train/results.csv"
                ).to_dict(),
                "results_test": pd.read_csv(
                    Path(dir_models) / model_name / "val/results.csv"
                ).to_dict(),
            }
        except:
            pass

    return models
