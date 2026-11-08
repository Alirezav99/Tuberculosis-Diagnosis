"""Compatibility imports and command-line entry point for U-Net training.

The implementation is maintained in ``training.unet_trainer``.
"""

from .evaluation.uncertainty import (
    enable_dropout,
    evaluate_with_uncertainty,
    mc_dropout_inference,
    write_uncertainty_results,
)
from .evaluation.unet import evaluate_trained_models_on_test
from .training.artifacts import save_checkpoint, write_history
from .training.unet_components import (
    get_dataloaders,
    get_loss_fn,
    get_model,
    get_optimizer,
)
from .training.unet_config import UnetConfig, load_config
from .training.unet_trainer import UnetTrainer, main

__all__ = [
    "UnetConfig",
    "UnetTrainer",
    "enable_dropout",
    "evaluate_trained_models_on_test",
    "evaluate_with_uncertainty",
    "get_dataloaders",
    "get_loss_fn",
    "get_model",
    "get_optimizer",
    "load_config",
    "main",
    "mc_dropout_inference",
    "save_checkpoint",
    "write_history",
    "write_uncertainty_results",
]

if __name__ == "__main__":
    main()
