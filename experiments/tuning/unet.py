"""Run an Optuna search for the mask-refinement Attention U-Net.

Run with ``python -m experiments.tuning.unet`` from the repository root. The
workflow loads a YOLO checkpoint, constructs loaders by splitting the
restructured validation set, runs three training epochs per trial, and writes
completed trial rows to ``unet_trials.csv``. Configuration paths and the
selected device are resolved at module import.
"""

import csv
import os

import optuna
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from ultralytics import YOLO

from afb_segmentation.data_pipeline.load.dataset import CustomDatasetTBDD3
from afb_segmentation.models.architectures.unet import AttentionUNetMaskPredictor
from afb_segmentation.models.loss.segmentation import CompoundLoss
from afb_segmentation.models.training.segmentation_epochs import (
    eval_one_epoch,
    train_one_epoch,
)

load_dotenv()

ROOT_PROJECT = os.getenv("ROOT_PROJECT")
ROOT_MODELS = os.getenv("ROOT_MODELS")
DATA_PATH = os.getenv("ROOT_DATA")
VAL_PATH = f"{DATA_PATH}/dds3-restructured/validation"
SIZE = 400
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
batch_size = 10

root_best_model = os.path.join(
    ROOT_MODELS, "yolo11l-seg - True", "train", "weights", "best.pt"
)

direction = "maximize"
metric_to_optimize = "f1"
n_trials = 3

keys = [
    "lr",
    "dropout",
    "optimizer",
    "label_smoothing",
    "depth",
    "scale_factor",
    "dice_weight",
    "focal_weight",
    "boundary_weight",
    "focal_alpha",
    "focal_gamma",
]


def objective(trial: optuna.trial.Trial) -> float:
    """Train one sampled UNet configuration and return its validation F1.

    The training and validation loaders are initialized in the script entry
    point and read from module scope. Trial rows are appended to
    ``trial_results`` for later CSV output.
    """
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=False)
    dropout = trial.suggest_float("dropout", 0.2, 0.5, log=False)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, log=False)
    depth = trial.suggest_int("depth", 2, 3)
    scale_factor = trial.suggest_categorical(
        "scale_factor",
        [
            1,
            2,
        ],
    )

    dice_weight = trial.suggest_float("dice_weight", 0.3, 0.8, log=False)
    focal_weight = trial.suggest_float("focal_weight", 0.1, 0.5, log=False)
    boundary_weight = trial.suggest_float("boundary_weight", 0.1, 0.3, log=False)
    focal_alpha = trial.suggest_float("focal_alpha", 0.6, 0.8, log=False)
    focal_gamma = trial.suggest_float("focal_gamma", 1.5, 2.5, log=False)

    loss_fn = CompoundLoss(
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        boundary_weight=boundary_weight,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        use_boundary=False,
    )

    model = AttentionUNetMaskPredictor(
        in_channels=3,
        out_channels=1,
        dropout=dropout,
        base_channels=64,
        depth=depth,
        channel_factor=2,
        scale_factor=scale_factor,
    )

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(3):
        metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            device=DEVICE,
            train_loader=train_loader,
            optimizer=optimizer,
            label_smoothing=label_smoothing,
        )

    val_metrics = eval_one_epoch(
        model=model,
        loss_fn=loss_fn,
        device=DEVICE,
        eval_loader=val_loader,
    )
    print("\n===========================\n")

    trial_results.append(
        {
            "lr": lr,
            "dropout": dropout,
            "optimizer": optimizer_name,
            "label_smoothing": label_smoothing,
            "depth": depth,
            "scale_factor": scale_factor,
            "dice_weight": dice_weight,
            "focal_weight": focal_weight,
            "boundary_weight": boundary_weight,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            **(val_metrics),
        }
    )

    return val_metrics[metric_to_optimize]


if __name__ == "__main__":
    root_model = root_best_model
    yolo = YOLO(root_model)

    train_dataset = CustomDatasetTBDD3(
        resource_folder=VAL_PATH,
        yolo=yolo,
        size=SIZE,
    )

    train_dataset, val_dataset = random_split(
        train_dataset, [len(train_dataset) - 500, 500]
    )
    _, train_dataset = random_split(train_dataset, [len(train_dataset) - 500, 500])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    trial_results = []
    results_path = os.path.join(ROOT_PROJECT, "experiments/")
    os.makedirs(results_path, exist_ok=True)

    try:
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    except KeyboardInterrupt:
        print("")

    if trial_results:
        csv_file = os.path.join(results_path, "unet_trials.csv")
        all_keys = set(keys)
        for row in trial_results:
            all_keys.update(row.keys())
        all_keys = list(all_keys)

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in trial_results:
                writer.writerow(row)
        print(f"Trial results saved to {csv_file}")
