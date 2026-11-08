"""Configuration schema and loading for the U-Net workflow.

The loader resolves numeric settings and the execution device from the
project model configuration.
"""

import os
from dataclasses import dataclass

import torch

from afb_segmentation.utils.config import read_config
from afb_segmentation.utils.devices import select_device


@dataclass
class UnetConfig:
    """Resolved UNet training, architecture, loss, and evaluation settings.

    The configuration is populated from the ``unet`` section of
    ``models/config.yml``. Device selection is resolved to a ``torch.device``;
    the YOLO checkpoint paths are used to prepare the corresponding dataset
    scenario.

    Attributes
    ----------
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size used by the loaders.
    size : int
        Image size used by the dataset.
    device : torch.device
        Device selected for model execution.
    lr : float
        Optimizer learning rate.
    dropout : float
        Dropout probability in the U-Net blocks.
    optimizer : str
        Optimizer name used to select Adam or SGD.
    label_smoothing : float
        Smoothing amount applied to training labels for loss calculation.
    depth : int
        Number of U-Net downsampling and upsampling stages.
    scale_factor : int
        Channel divisor applied when constructing the U-Net.
    dice_weight : float
        Weight assigned to Dice loss.
    focal_weight : float
        Weight assigned to focal loss.
    boundary_weight : float
        Configured boundary loss weight; the training loss includes this term
        when the weight is positive.
    focal_alpha : float
        Positive-class weight used by focal loss.
    focal_gamma : float
        Focusing parameter used by focal loss.
    yolo_path_with_augmentation : str
        YOLO checkpoint for the augmented dataset scenario.
    yolo_path_without_augmentation : str
        YOLO checkpoint for the unaugmented dataset scenario.
    scheduler_patience : int
        Plateau epochs before reducing the learning rate.
    scheduler_factor : float
        Learning-rate multiplier used by the scheduler.
    scheduler_min_lr : float
        Minimum learning rate allowed by the scheduler.
    early_stopping_patience : int
        Number of non-improving epochs allowed by the trainer.
    mc_dropout_samples : int
        Stochastic passes used for uncertainty evaluation.
    """

    epochs: int
    batch_size: int
    size: int
    device: torch.device
    lr: float
    dropout: float
    optimizer: str
    label_smoothing: float
    depth: int
    scale_factor: int
    focal_weight: float
    dice_weight: float
    boundary_weight: float
    focal_alpha: float
    focal_gamma: float
    yolo_path_with_augmentation: str
    yolo_path_without_augmentation: str
    scheduler_patience: int
    scheduler_factor: float
    scheduler_min_lr: float
    early_stopping_patience: int
    mc_dropout_samples: int


def load_config(project_root: str) -> UnetConfig:
    """Load UNet settings from the project's model configuration file.

    Parameters
    ----------
    project_root : str
        Repository root containing
        ``src/afb_segmentation/models/config.yml``.

    Returns
    -------
    config : UnetConfig
        The ``unet`` settings with numeric values converted and the device
        selected as a ``torch.device``.
    """
    config_path = os.path.join(project_root, "src/afb_segmentation/models/config.yml")
    cfg = read_config(config_path).get("unet", {})

    device = select_device(cfg.get("device", ""))

    return UnetConfig(
        epochs=int(cfg.get("epochs")),
        batch_size=int(cfg["batch_size"]),
        size=int(cfg["size"]),
        device=device,
        lr=float(cfg["lr"]),
        dropout=float(cfg["dropout"]),
        optimizer=cfg["optimizer"],
        label_smoothing=float(cfg["label_smoothing"]),
        depth=int(cfg["depth"]),
        scale_factor=int(cfg["scale_factor"]),
        dice_weight=float(cfg["dice_weight"]),
        focal_weight=float(cfg["focal_weight"]),
        boundary_weight=float(cfg["boundary_weight"]),
        focal_alpha=float(cfg["focal_alpha"]),
        focal_gamma=float(cfg["focal_gamma"]),
        yolo_path_with_augmentation=cfg["yolo_path_with_augmentation"],
        yolo_path_without_augmentation=cfg["yolo_path_without_augmentation"],
        scheduler_patience=int(cfg["scheduler_patience"]),
        scheduler_factor=float(cfg["scheduler_factor"]),
        scheduler_min_lr=float(cfg["scheduler_min_lr"]),
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        mc_dropout_samples=int(cfg["mc_dropout_samples"]),
    )


__all__ = ["UnetConfig", "load_config"]
