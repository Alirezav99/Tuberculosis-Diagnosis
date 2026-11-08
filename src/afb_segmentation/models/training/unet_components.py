"""U-Net model, loss, optimizer, and data-loader construction.

Loader creation selects the configured prepared dataset scenario through
``DatasetManager`` before wrapping the split in ``CustomDatasetTBDD3``.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from ultralytics import YOLO

from afb_segmentation.data_pipeline.load.dataset import CustomDatasetTBDD3
from afb_segmentation.data_pipeline.preparation.dataset_manager import DatasetManager

from ..architectures import unet as U_NET
from ..loss.segmentation import CompoundLoss
from .unet_config import UnetConfig


def get_dataloaders(
    data_root: str, cfg, shuffle: bool, augmentation: bool
) -> DataLoader:
    """Create a loader for one DDS3 split and select its dataset scenario.

    Parameters
    ----------
    data_root : str
        Split directory containing ``images`` and ``masks``.
        cfg : UnetConfig
        Resolved UNet configuration.
    shuffle : bool
        Whether the returned loader shuffles sample order.
    augmentation : bool
        Selects the prepared dataset snapshot and matching YOLO
        checkpoint path.

    Returns
    -------
    loader : DataLoader
        A PyTorch data loader wrapping ``CustomDatasetTBDD3``.
    Notes
    -----
        Selecting a scenario calls ``DatasetManager.levelup_dataset`` or
        ``DatasetManager.reset_dataset`` before constructing the loader.
    """
    if augmentation:
        DatasetManager.levelup_dataset()
        yolo = YOLO(cfg.yolo_path_with_augmentation)
    else:
        DatasetManager.reset_dataset()
        yolo = YOLO(cfg.yolo_path_without_augmentation)

    dataset = CustomDatasetTBDD3(resource_folder=data_root, yolo=yolo, size=cfg.size)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)

    return train_loader


def get_model(cfg: UnetConfig) -> nn.Module:
    """Build the configured attention U-Net mask predictor.

    Parameters
    ----------
    cfg : UnetConfig
        Resolved model and training configuration.

    Returns
    -------
    model : nn.Module
        Attention U-Net that returns per-pixel sigmoid probabilities.
    """
    model = U_NET.AttentionUNetMaskPredictor(
        in_channels=3,
        out_channels=1,
        dropout=cfg.dropout,
        base_channels=64,
        depth=cfg.depth,
        channel_factor=2,
        scale_factor=cfg.scale_factor,
    )
    return model


def get_loss_fn(cfg: UnetConfig) -> nn.Module:
    """Build the configured compound loss used by the trainer.

    Parameters
    ----------
    cfg : UnetConfig
        Resolved Dice, focal, and boundary loss weights and focal parameters.

    Returns
    -------
    loss_fn : nn.Module
        Compound Dice/focal/boundary loss.

    Notes
    -----
    The component weights and focal parameters come from ``cfg``. Boundary
    loss is enabled in the constructed ``CompoundLoss``.
    """
    return CompoundLoss(
        dice_weight=cfg.dice_weight,
        focal_weight=cfg.focal_weight,
        boundary_weight=cfg.boundary_weight,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        use_boundary=True,
    )


def get_optimizer(model: nn.Module, cfg: UnetConfig) -> torch.optim.Optimizer:
    """Create the configured optimizer for ``model``.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters will be optimized.
    cfg : UnetConfig
        Configuration supplying the optimizer name and learning rate.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        Adam when configured by name; otherwise SGD with momentum 0.9.

    Notes
    -----
    Adam is selected when ``cfg.optimizer`` is case-insensitively ``"adam"``;
    every other value selects SGD with momentum 0.9.
    """
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)


__all__ = ["get_dataloaders", "get_loss_fn", "get_model", "get_optimizer"]
