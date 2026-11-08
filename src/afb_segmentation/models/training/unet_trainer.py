"""Training orchestration for the attention U-Net refinement model.

The trainer builds scenario-specific loaders, saves checkpoints and history,
and runs Monte Carlo dropout evaluation after training.
"""

import os
from typing import Any, Dict, Tuple

import torch
from dotenv import load_dotenv

from ..evaluation.uncertainty import (
    evaluate_with_uncertainty,
    write_uncertainty_results,
)
from .artifacts import save_checkpoint, write_history
from .segmentation_epochs import eval_one_epoch, train_one_epoch
from .unet_components import get_dataloaders, get_loss_fn, get_model, get_optimizer
from .unet_config import load_config

load_dotenv()


class UnetTrainer:
    """Own the configured UNet training, checkpoint, and uncertainty workflow.

    Construction creates result/weight directories, prepares train,
    validation, and test loaders for the selected augmentation scenario, and
    initializes the model, loss, optimizer, and learning-rate scheduler.
    """

    def __init__(self, augmentation: bool = True) -> None:
        """Initialize the trainer for augmented or unaugmented data.

        Parameters
        ----------
        augmentation : bool
            Select the prepared dataset and associated YOLO
            checkpoint scenario.

        Environment
        -----------
            Reads ``ROOT_PROJECT``, ``ROOT_MODELS``, and ``ROOT_DATA``.
        """
        self.project_root = os.getenv("ROOT_PROJECT")
        self.models_root = os.getenv("ROOT_MODELS")
        self.data_root = os.getenv("ROOT_DATA")
        self.cfg = load_config(self.project_root)

        subdir = "with_augmentation" if augmentation else "without_augmentation"
        self.results_dir = os.path.join(self.models_root, "unet", subdir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.weights_dir = os.path.join(self.models_root, "unet", subdir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        root_train = os.path.join(self.data_root, "dds3-restructured/train")
        root_validation = os.path.join(self.data_root, "dds3-restructured/validation")
        root_test = os.path.join(self.data_root, "dds3-restructured/test")

        self.train_loader = get_dataloaders(
            root_train, self.cfg, shuffle=True, augmentation=augmentation
        )
        self.val_loader = get_dataloaders(
            root_validation, self.cfg, shuffle=False, augmentation=augmentation
        )
        self.test_loader = get_dataloaders(
            root_test, self.cfg, shuffle=False, augmentation=augmentation
        )

        self.model = get_model(self.cfg)
        self.loss_fn = get_loss_fn(self.cfg)
        self.optimizer = get_optimizer(self.model, self.cfg)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.cfg.scheduler_patience,
            factor=self.cfg.scheduler_factor,
            min_lr=self.cfg.scheduler_min_lr,
        )

        self.history: list[Dict[str, Any]] = []

    def run(self) -> None:
        """Train by epoch, evaluate on validation data, and save checkpoints.

        The method writes ``last.pt``, epoch-specific checkpoints, and training
        history under the scenario's output directories. Best-checkpoint
        tracking currently compares validation F1 with ``<``; therefore a
        lower validation F1 is treated as an improvement by this implementation.
        """
        best_metric = float("inf")
        metric_name = "f1"

        epochs_no_improve = 0
        for epoch in range(1, self.cfg.epochs + 1):
            print(f"Epoch {epoch}/{self.cfg.epochs}")

            train_metrics = train_one_epoch(
                model=self.model,
                loss_fn=self.loss_fn,
                device=self.cfg.device,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                label_smoothing=self.cfg.label_smoothing,
            )

            val_metrics = eval_one_epoch(
                model=self.model,
                loss_fn=self.loss_fn,
                device=self.cfg.device,
                eval_loader=self.val_loader,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "train_f1": train_metrics.get("f1"),
                "val_loss": val_metrics.get("loss"),
                "val_f1": val_metrics.get("f1"),
            }

            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.history.append(row)

            ckpt = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            if hasattr(self, "scheduler") and self.scheduler is not None:
                try:
                    ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
                except Exception:
                    pass

            save_checkpoint(
                ckpt, os.path.join(self.weights_dir, f"epoch_{epoch:03d}_last.pt")
            )
            save_checkpoint(ckpt, os.path.join(self.weights_dir, "last.pt"))

            current_metric = val_metrics.get(metric_name)
            is_better = current_metric < best_metric
            if is_better:
                best_metric = current_metric
                save_checkpoint(
                    ckpt, os.path.join(self.weights_dir, f"epoch_{epoch:03d}_best.pt")
                )
                save_checkpoint(ckpt, os.path.join(self.weights_dir, "best.pt"))
                epochs_no_improve = 0
            else:
                epochs_no_improve = locals().get("epochs_no_improve", 0) + 1

            if (
                current_metric is not None
                and hasattr(self, "scheduler")
                and self.scheduler is not None
            ):
                try:
                    self.scheduler.step(current_metric)
                except Exception:
                    pass

            if epochs_no_improve > self.cfg.early_stopping_patience:
                print(
                    "Early stopping: no improvement in "
                    f"{epochs_no_improve} epochs "
                    f"(patience={self.cfg.early_stopping_patience})"
                )
                break

            write_history(
                self.history, os.path.join(self.results_dir, "train_history.csv")
            )

            print(f"Epoch {epoch} finished. val_{metric_name}={current_metric}")

        print("\n" + "=" * 60)
        print("Running Monte Carlo Dropout Uncertainty Evaluation on Test Set")
        print(f"Number of MC samples: {self.cfg.mc_dropout_samples}")
        print("=" * 60)

    def evaluate_uncertainty(self) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
        """Evaluate model with MC Dropout uncertainty on test set.

        Notes
        -----
        1. Loads the best model checkpoint
        2. Enables dropout during inference
        3. Runs multiple forward passes per sample
        4. Computes uncertainty statistics
        5. Saves aggregate and per-sample results to CSV files in
           ``results_dir``

        Returns
        -------
        aggregate_metrics : Dict[str, Any]
            Aggregate test metrics.
        per_sample_results : list[Dict[str, Any]]
            One uncertainty statistics row for each image in test-loader order.
        """

        best_model_path = os.path.join(self.weights_dir, "best.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.cfg.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Warning: best.pt not found, using current model state")

        aggregate_metrics, per_sample_results = evaluate_with_uncertainty(
            model=self.model,
            device=self.cfg.device,
            test_loader=self.test_loader,
            n_samples=self.cfg.mc_dropout_samples,
        )

        print("\nUncertainty Evaluation Results:")
        print("-" * 40)
        for key, value in aggregate_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        write_uncertainty_results(
            aggregate_metrics=aggregate_metrics,
            per_sample_results=per_sample_results,
            results_dir=self.results_dir,
        )

        return aggregate_metrics, per_sample_results


def main() -> None:
    """Run the configured augmented and unaugmented U-Net workflows."""
    for augmentation in [False, True]:
        print(f"Starting UNet training. Augmentation={augmentation}")
        trainer = UnetTrainer(augmentation=augmentation)
        trainer.run()
        trainer.evaluate_uncertainty()

    from ..evaluation.unet import evaluate_trained_models_on_test

    evaluate_trained_models_on_test()


__all__ = ["UnetTrainer", "main"]
